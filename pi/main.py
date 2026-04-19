"""Top-level asyncio orchestrator. Wires every service together."""

from __future__ import annotations

import asyncio
import logging
from time import time

import uvicorn

from pi.config import (
    HAPTIC_RATE_LIMIT_MS,
    MEMORY_WATCH_INTERVAL_S,
    WEBAPP_HOST,
    WEBAPP_PORT,
    YOLO_BACKEND,
)
from pi.models import (
    DistanceReading,
    FallEvent,
    HapticCommand,
    HealthState,
    IMUSample,
    VoiceTurn,
)
from pi.services.camera import CameraService
from pi.services.fusion import FusionEngine, FusionState
from pi.services.narrator import NarratorService
from pi.services.serial_bridge import SerialBridge
from pi.services.voice import VoicePipeline, make_assistant_turn, make_user_turn
from pi.services.webapp_server import WebappServer
from pi.services.yolo import YoloService
from pi.services.yolo_ei import YoloEIService
from pi.services.yolo_qaihub import YoloQAIHubService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("belt")


class Belt:
    def __init__(self) -> None:
        self.serial = SerialBridge()
        self.camera = CameraService()
        self.yolo = self._make_yolo()
        self.voice = VoicePipeline()
        self.webapp = WebappServer(
            self.camera,
            detections_supplier=lambda: self.yolo.get_latest().boxes,
            health_supplier=self._health_snapshot,
        )
        self.fusion_state = FusionState()
        self.fusion = FusionEngine(self.fusion_state, self._on_haptic)
        self.narrator = NarratorService(
            self.yolo, self.voice, self._on_haptic, broadcast=self.webapp.broadcast,
        )
        self.health = HealthState()

    def _make_yolo(self):
        if YOLO_BACKEND == "ei":
            log.info("yolo backend: edge-impulse (NPU via QNN .eim)")
            return YoloEIService(self.camera)
        if YOLO_BACKEND == "qaihub":
            log.info("yolo backend: qai-hub yolo26 (LiteRT, optional HTP NPU)")
            return YoloQAIHubService(self.camera)
        log.info("yolo backend: cpu (ultralytics)")
        return YoloService(self.camera)

    async def run(self) -> None:
        await self.camera.start()
        await self.yolo.start()
        await self.serial.start()
        await self.fusion.start()
        await self.narrator.start()
        asyncio.create_task(self._serial_event_loop())
        asyncio.create_task(self._fusion_input_loop())
        asyncio.create_task(self._detections_broadcast_loop())
        asyncio.create_task(self._health_loop())
        await self._serve_webapp()

    async def _serve_webapp(self) -> None:
        config = uvicorn.Config(
            self.webapp.app, host=WEBAPP_HOST, port=WEBAPP_PORT, log_level="warning"
        )
        await uvicorn.Server(config).serve()

    async def _serial_event_loop(self) -> None:
        while True:
            msg = await self.serial.events.get()
            t = msg.get("t")
            if t == "imu":
                await self.webapp.broadcast(IMUSample(
                    ax=msg["ax"], ay=msg["ay"], az=msg["az"],
                    gx=msg.get("gx", 0), gy=msg.get("gy", 0), gz=msg.get("gz", 0),
                    source=msg.get("source", "m5"),
                ))
            elif t == "distance":
                front_cm = int(msg.get("front_cm", -1))
                back_cm = int(msg.get("back_cm", -1))
                self.fusion_state.distance = DistanceReading(a_cm=front_cm, b_cm=back_cm)
                mm = self.fusion_state.distance.min_mm
                # UI expects F/B/L/R fields in mm; -1 collapses to null on the
                # client side via the bulk-update path. Keep `mm` for legacy.
                await self.webapp.broadcast({
                    "t": "distance",
                    "mm": mm,
                    "front_cm": front_cm,
                    "back_cm": back_cm,
                    "F": front_cm * 10 if front_cm >= 0 else None,
                    "B": back_cm * 10 if back_cm >= 0 else None,
                })
            elif t == "fall":
                event = FallEvent(severity=msg.get("severity", "hard"), az_peak=msg.get("az_peak", 0.0))
                await self.webapp.broadcast(event)
                cmd = self.fusion.trigger_fall_sos()
                await self._on_haptic(cmd)
                asyncio.get_running_loop().run_in_executor(
                    None, self.voice.speak, "Are you okay? Say yes or no."
                )
            elif t == "button":
                await self.webapp.broadcast({"t": "button", "down": bool(msg.get("down"))})
            elif t == "ack":
                log.info("M5 ack: %s", msg.get("what"))
            elif t == "err":
                log.warning("M5 err: %s", msg.get("what"))
            elif t == "_audio_done":
                await self._handle_voice_turn(msg["audio"])
            elif t == "_audio_timeout":
                log.warning("audio capture timed out")
            elif t == "hello":
                log.info("M5 hello: fw=%s caps=%s", msg.get("fw"), msg.get("caps"))

    async def _fusion_input_loop(self) -> None:
        while True:
            self.fusion_state.detections = self.yolo.get_latest()
            await asyncio.sleep(0.05)

    async def _detections_broadcast_loop(self) -> None:
        """Push the latest YOLO detections to webapp clients at ~5 Hz so the
        UI's detection summary stays live even though boxes are baked into
        the MJPEG overlay server-side."""
        last_ts = 0
        while True:
            await asyncio.sleep(0.2)
            frame = self.yolo.get_latest()
            if frame.ts_ms == last_ts:
                continue
            last_ts = frame.ts_ms
            boxes = [
                {"cls": b.cls, "conf": b.conf, "x": b.x, "y": b.y, "w": b.w, "h": b.h}
                for b in frame.boxes
            ]
            await self.webapp.broadcast({"t": "detections", "boxes": boxes, "ts_ms": frame.ts_ms})

    async def _handle_voice_turn(self, audio: bytes) -> None:
        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(None, self.voice.transcribe, audio)
        if not transcript:
            return
        await self.webapp.broadcast(make_user_turn(transcript))
        det = self.yolo.get_latest()
        intent = await loop.run_in_executor(None, self.voice.derive_intent, transcript, det)
        self.fusion_state.intent = intent
        if intent.reply:
            await self.webapp.broadcast(make_assistant_turn(intent.reply))
            loop.run_in_executor(None, self.voice.speak, intent.reply)

    async def _on_haptic(self, cmd: HapticCommand) -> None:
        self.serial.send(cmd)
        await self.webapp.broadcast({
            "t": "haptic", "dir": cmd.dir, "intensity": cmd.intensity,
            "pattern": cmd.pattern, "duration_ms": cmd.duration_ms,
        })

    def _health_snapshot(self) -> dict:
        return {
            "serial_ok": self.health.serial_ok,
            "yolo_fps": self.health.yolo_fps,
            "camera_fps": round(self.camera.fps, 2),
            "stream_fps": round(self.webapp._mjpeg_fps_overlay, 2),
            "warmup_ok": self.health.warmup_ok,
            "ollama_alive": self.health.ollama_alive,
            "last_pong_age_ms": self.health.last_pong_age_ms,
        }

    async def _health_loop(self) -> None:
        while True:
            self.health.serial_ok = self.serial.healthy
            self.health.yolo_fps = round(self.yolo.fps, 2)
            self.health.last_pong_age_ms = self.serial.last_pong_age_ms
            await self.webapp.broadcast({"t": "health", **self._health_snapshot()})
            await asyncio.sleep(1.0)


async def main() -> None:
    belt = Belt()
    await belt.run()


if __name__ == "__main__":
    asyncio.run(main())
