"""Top-level asyncio orchestrator. Wires every service together."""

from __future__ import annotations

import asyncio
import logging
from time import time
from typing import cast

import uvicorn

from pi.config import (
    HAPTIC_RATE_LIMIT_MS,
    MEMORY_WATCH_INTERVAL_S,
    WEBAPP_HOST,
    WEBAPP_PORT,
    YOLO_BACKEND,
)
import math

from pi.models import (
    DistanceReading,
    FallEvent,
    HapticCommand,
    HealthState,
    IMUSample,
    Pattern,
    VoiceTurn,
)
from pi.services.camera import CameraService
from pi.services.fusion import FusionEngine, FusionState
from pi.services.narrator import NarratorService
from pi.services.pi_mic_recorder import PiMicRecorder
from pi.services.serial_bridge import SerialBridge
from pi.services.voice import VoicePipeline, make_assistant_turn, make_user_turn
from pi.services.webapp_server import WebappServer
from pi.services.yolo import YoloService
from pi.services.yolo_ei import YoloEIService
from pi.services.yolo_qaihub import YoloQAIHubService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
log = logging.getLogger("belt")


_err_counts: dict[str, int] = {}


class Belt:
    def __init__(self) -> None:
        mic = PiMicRecorder()
        pi_mic_ok = PiMicRecorder.available() and self._test_pi_mic(mic)
        if not pi_mic_ok:
            log.info("Pi mic unavailable or silent; BtnA will use M5 onboard mic (AUDIO_ON/OFF)")
        self.serial = SerialBridge(
            ptt_recorder=mic if pi_mic_ok else None,
        )
        self.camera = CameraService()
        self.yolo = self._make_yolo()
        self.voice = VoicePipeline()
        self._demo_mode = False
        self.webapp = WebappServer(
            self.camera,
            detections_supplier=lambda: self.yolo.get_latest().boxes,
            health_supplier=self._health_snapshot,
            on_browser_audio=self._on_browser_audio,
            on_demo_command=self._on_demo_command,
        )
        self.fusion_state = FusionState()
        self.fusion = FusionEngine(
            self.fusion_state,
            self._on_haptic,
            paused_supplier=lambda: self._demo_mode,
        )
        self.narrator = NarratorService(
            self.yolo,
            self.voice,
            self._on_haptic,
            distance_supplier=lambda: self.fusion_state.distance,
            broadcast=self.webapp.broadcast,
            automation_active_supplier=lambda: not self._demo_mode,
        )
        self.health = HealthState()

    @staticmethod
    def _test_pi_mic(mic: "PiMicRecorder") -> bool:
        """Quick 0.5 s capture test; returns False if arecord produces no audio."""
        if not mic.start():
            return False
        import time
        time.sleep(0.5)
        pcm = mic.stop()
        if len(pcm) < 100:
            return False
        import numpy as np
        samples = np.frombuffer(pcm, dtype=np.int16)
        return int(np.max(np.abs(samples))) > 0

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
        asyncio.create_task(self._motor_selftest())
        await self._serve_webapp()

    async def _motor_selftest(self) -> None:
        """Boot-time pulse of each belt motor so wiring can be heard/felt.

        Motors are indexed 0..3 on the Modulino I2C chain. Whichever buzzes
        (or doesn't) tells you which address is actually live — this is the
        fastest way to catch the "left motor always fires, others silent"
        class of problem. Gated on the serial link being up.
        """
        await asyncio.sleep(2.0)
        if not self.serial.healthy:
            log.warning("motor self-test skipped: serial not healthy")
            return
        log.info("motor self-test: pulsing motors 0..3 (L F R B)")
        for i, label in enumerate(("LEFT", "FRONT", "RIGHT", "BACK")):
            self.serial.send_raw(f"M,{i},30,250")
            log.info("  self-test motor %d (%s)", i, label)
            await asyncio.sleep(0.6)
        log.info("motor self-test: done")

    async def _serve_webapp(self) -> None:
        config = uvicorn.Config(
            self.webapp.app, host=WEBAPP_HOST, port=WEBAPP_PORT, log_level="warning"
        )
        await uvicorn.Server(config).serve()

    async def _serial_event_loop(self) -> None:
        while True:
            msg = await self.serial.events.get()
            t = msg.get("t")

            # Bench mode: only IMU (+ low-level serial diagnostics) from the M5.
            if self._demo_mode and t not in ("imu", "ack", "err", "hello"):
                continue

            if t == "imu":
                await self.webapp.broadcast(IMUSample(
                    ax=msg["ax"], ay=msg["ay"], az=msg["az"],
                    gx=msg.get("gx", 0), gy=msg.get("gy", 0), gz=msg.get("gz", 0),
                    source=msg.get("source", "m5"),
                ))
            elif t == "distance":
                left_cm = int(msg.get("left_cm", -1))
                right_cm = int(msg.get("right_cm", -1))
                self.fusion_state.distance = DistanceReading(a_cm=left_cm, b_cm=right_cm)
                mm = self.fusion_state.distance.min_mm

                # Derive forward from the 45° sensors: cos(45°) ≈ 0.707
                COS45 = math.cos(math.radians(45))
                valid = [v for v in (left_cm, right_cm) if v >= 0]
                front_cm = int(min(valid) * COS45) if valid else -1

                # Synthetic rear: slow drift 150-250 cm (no real sensor)
                t_sec = time()
                back_cm = int(200 + 50 * math.sin(t_sec * 0.3))

                left_mm = left_cm * 10 if left_cm >= 0 else None
                right_mm = right_cm * 10 if right_cm >= 0 else None
                front_mm = front_cm * 10 if front_cm >= 0 else None
                back_mm = back_cm * 10

                await self.webapp.broadcast({
                    "t": "distance",
                    "mm": mm,
                    "front_cm": front_cm,
                    "back_cm": back_cm,
                    "F": front_mm,
                    "B": back_mm,
                    "L": left_mm,
                    "R": right_mm,
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
                what = msg.get("what", "")
                _err_counts[what] = _err_counts.get(what, 0) + 1
                n = _err_counts[what]
                if n <= 3 or n % 100 == 0:
                    log.warning("M5 err: %s (x%d)", what, n)
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

    async def _on_browser_audio(self, pcm_bytes: bytes) -> None:
        log.info("browser audio received: %d bytes", len(pcm_bytes))
        await self._handle_voice_turn(pcm_bytes)

    async def _broadcast_demo_distances(
        self, l_mm: int, r_mm: int, f_mm: int, b_mm: int, *, quiet: bool = False
    ) -> None:
        """Push synthetic proximity to all dashboards (mm). Updates fusion sample for UI only."""
        l_mm = max(0, int(l_mm))
        r_mm = max(0, int(r_mm))
        f_mm = max(0, int(f_mm))
        b_mm = max(0, int(b_mm))
        self.fusion_state.distance = DistanceReading(a_cm=l_mm // 10, b_cm=r_mm // 10)
        mm = min(l_mm, r_mm, f_mm, b_mm)
        payload = {
            "t": "distance",
            "mm": mm,
            "front_cm": f_mm // 10,
            "back_cm": b_mm // 10,
            "F": float(f_mm),
            "B": float(b_mm),
            "L": float(l_mm),
            "R": float(r_mm),
        }
        await self.webapp.broadcast(payload)
        if not quiet:
            await self.webapp.broadcast({"t": "ack", "what": "demo_distance"})

    async def _on_demo_command(self, msg: dict) -> None:
        t = msg.get("t", "")
        try:
            if t == "demo_mode":
                self._demo_mode = msg.get("mode") == "demo"
                if not self._demo_mode:
                    self.fusion_state.distance = None
                log.info("haptic bench: %s", "ON (M5 US off, IMU on)" if self._demo_mode else "OFF (live belt)")
                await self.webapp.broadcast({"t": "ack", "what": f"mode={'bench' if self._demo_mode else 'live'}"})
                if self._demo_mode:
                    await self._broadcast_demo_distances(1500, 1500, 1500, 2000, quiet=True)

            elif t == "demo_motor":
                idx = int(msg.get("idx", 0))
                pw = int(msg.get("power", 30))
                ms = int(msg.get("ms", 300))
                self.serial.send_raw(f"M,{idx},{pw},{ms}")
                await self.webapp.broadcast({"t": "ack", "what": f"motor {idx} pw={pw} ms={ms}"})

            elif t == "demo_motor_all":
                pw = int(msg.get("power", 30))
                ms = int(msg.get("ms", 300))
                self.serial.send_raw(f"MA,{pw},{ms}")
                await self.webapp.broadcast({"t": "ack", "what": f"motor_all pw={pw} ms={ms}"})

            elif t == "demo_motor_stop":
                self.serial.send_raw("STOP")
                await self.webapp.broadcast({"t": "ack", "what": "motor_stop"})

            elif t == "demo_haptic":
                direction = str(msg.get("dir", "F")).upper()
                intensity = int(msg.get("intensity", 180))
                ms = int(msg.get("ms", 300))
                pat = str(msg.get("pattern", "pulse")).lower()
                if pat not in ("pulse", "sos", "solid", "ramp"):
                    pat = "pulse"
                cmd = HapticCommand(
                    dir=direction,
                    intensity=intensity,
                    pattern=cast(Pattern, pat),
                    duration_ms=ms,
                )
                await self._on_haptic(cmd)
                await self.webapp.broadcast({"t": "ack", "what": f"haptic {direction} {pat} i={intensity} ms={ms}"})

            elif t == "demo_distance":
                quiet = bool(msg.get("quiet"))
                await self._broadcast_demo_distances(
                    int(msg.get("L", 1500)),
                    int(msg.get("R", 1500)),
                    int(msg.get("F", 1500)),
                    int(msg.get("B", 2000)),
                    quiet=quiet,
                )

            elif t == "demo_distance_live":
                self._demo_mode = False
                self.fusion_state.distance = None
                await self.webapp.broadcast({"t": "ack", "what": "live_belt"})

            elif t == "demo_tts":
                text = str(msg.get("text", "")).strip()
                if text:
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(None, self.voice.speak, text)
                    await self.webapp.broadcast({"t": "ack", "what": f"tts: {text[:40]}"})

            elif t == "demo_voice_inject":
                role = msg.get("role", "user")
                text = str(msg.get("text", "")).strip()
                if text:
                    await self.webapp.broadcast({"t": "voice", "role": role, "text": text})
                    await self.webapp.broadcast({"t": "ack", "what": f"voice_inject ({role})"})

            elif t == "demo_nav":
                await self.webapp.broadcast({
                    "t": "demo_nav",
                    "bearing": msg.get("bearing", 0),
                    "threat": msg.get("threat", ""),
                    "zone": msg.get("zone", ""),
                })
                await self.webapp.broadcast({"t": "ack", "what": "nav_override"})

            else:
                await self.webapp.broadcast({"t": "err", "what": f"unknown demo cmd: {t}"})

        except Exception as e:
            log.warning("demo command error: %s", e)
            await self.webapp.broadcast({"t": "err", "what": str(e)})

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
