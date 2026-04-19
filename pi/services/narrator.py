"""Reactive narrator: watches YOLO output for newly-arrived stable objects
and asks Gemma to choose a (speak, haptic) reaction.

Design notes
------------
- Gemma 270M is text-only, so we send it a *symbolic* scene description
  (class + normalized position + confidence) rather than raw pixels. YOLO
  has already done perception; Gemma reasons.
- Single-flight: while a Gemma call is in flight, additional 'new object'
  events are dropped (Gemma is ~5s/call, queueing would cause stale advice).
- Per-class cooldown stops the narrator from re-announcing the same object
  every time it briefly leaves and re-enters the frame.
- Stable-frame debounce stops false alarms from one-frame YOLO flickers.

Activation: enabled by default. Set ``BELT_NARRATOR=0`` to disable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from time import monotonic, perf_counter, time
from typing import Awaitable, Callable

from pi.config import LOCAL_LLM_HOST, LOCAL_LLM_MODEL
from pi.models import Detection, DetectionFrame, HapticCommand
from pi.services.voice import VoicePipeline
from pi.services.yolo import YoloService

log = logging.getLogger(__name__)

POLL_INTERVAL_S = 0.5
STABLE_FRAMES = 2
PER_CLASS_COOLDOWN_S = 30.0
GEMMA_TIMEOUT_S = 25.0

VALID_DIRS = {"F", "B", "L", "R", "ALL", "NONE"}
VALID_PATTERNS = {"solid", "pulse", "ramp", "sos"}


HapticEmitter = Callable[[HapticCommand], Awaitable[None]]


class NarratorService:
    def __init__(
        self,
        yolo: YoloService,
        voice: VoicePipeline,
        emit_haptic: HapticEmitter,
        broadcast: Callable[[dict], Awaitable[None]] | None = None,
    ) -> None:
        self._yolo = yolo
        self._voice = voice
        self._emit_haptic = emit_haptic
        self._broadcast = broadcast or (lambda _msg: _noop())
        self._stable_count: dict[str, int] = {}
        self._last_announced_at: dict[str, float] = {}
        self._inflight = False
        self._stop = asyncio.Event()
        self._enabled = os.environ.get("BELT_NARRATOR", "1") == "1"
        self._latencies_ms: list[int] = []

    async def start(self) -> None:
        if not self._enabled:
            log.info("narrator disabled (BELT_NARRATOR=0)")
            return
        log.info("narrator: model=%s poll=%.1fs stable=%d cooldown=%.0fs",
                 LOCAL_LLM_MODEL, POLL_INTERVAL_S, STABLE_FRAMES, PER_CLASS_COOLDOWN_S)
        asyncio.create_task(self._warmup())
        asyncio.create_task(self._loop())

    async def _warmup(self) -> None:
        """Pre-load the model so the first real narration isn't a cold start."""
        try:
            loop = asyncio.get_running_loop()
            t0 = perf_counter()
            await loop.run_in_executor(
                None,
                self._call_gemma,
                Detection(cls="chair", conf=0.9, x=0.5, y=0.5, w=0.3, h=0.4),
                DetectionFrame(boxes=[]),
            )
            log.info("narrator: warmup ok (%.0fms)", (perf_counter() - t0) * 1000)
        except Exception as e:
            log.warning("narrator: warmup failed: %s", e)

    def stop(self) -> None:
        self._stop.set()

    async def _loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(POLL_INTERVAL_S)
            if self._inflight:
                continue
            scene = self._yolo.get_latest()
            new_classes = self._tick_debounce(scene)
            if not new_classes:
                continue
            cls = new_classes[0]
            new_obj = next((b for b in scene.boxes if b.cls == cls), None)
            if new_obj is None:
                continue
            self._inflight = True
            asyncio.create_task(self._react(new_obj, scene))

    def _tick_debounce(self, scene: DetectionFrame) -> list[str]:
        now_s = monotonic()
        present = {b.cls for b in scene.boxes}
        for cls in list(self._stable_count.keys()):
            if cls not in present:
                self._stable_count[cls] = 0
        new_stable: list[str] = []
        for cls in present:
            self._stable_count[cls] = self._stable_count.get(cls, 0) + 1
            if self._stable_count[cls] != STABLE_FRAMES:
                continue
            last = self._last_announced_at.get(cls, 0.0)
            if now_s - last < PER_CLASS_COOLDOWN_S:
                continue
            new_stable.append(cls)
            self._last_announced_at[cls] = now_s
        return new_stable

    async def _react(self, new_obj: Detection, scene: DetectionFrame) -> None:
        try:
            t0 = perf_counter()
            loop = asyncio.get_running_loop()
            decision = await asyncio.wait_for(
                loop.run_in_executor(None, self._call_gemma, new_obj, scene),
                timeout=GEMMA_TIMEOUT_S,
            )
            ms = (perf_counter() - t0) * 1000
            if decision is None:
                log.info("narrator: %s -> no decision (%.0fms)", new_obj.cls, ms)
                return
            self._latencies_ms.append(int(ms))
            self._latencies_ms = self._latencies_ms[-20:]
            avg = sum(self._latencies_ms) / len(self._latencies_ms)
            if decision.get("skip"):
                log.info("narrator: %s -> skip (%.0fms, avg %.0fms)", new_obj.cls, ms, avg)
                return
            log.info("narrator: %s -> %s (%.0fms, avg %.0fms)", new_obj.cls, decision, ms, avg)
            await self._broadcast({"t": "narrator", "obj": new_obj.cls, "decision": decision,
                                   "latency_ms": int(ms), "ts_ms": int(time() * 1000)})
            speak = (decision.get("speak") or "").strip()
            if speak:
                loop.run_in_executor(None, self._voice.speak, speak)
            haptic = decision.get("haptic") or {}
            cmd = self._build_haptic(haptic)
            if cmd is not None:
                await self._emit_haptic(cmd)
        except asyncio.TimeoutError:
            log.warning("narrator: gemma timeout for %s", new_obj.cls)
        except Exception as e:
            log.warning("narrator: react failed for %s: %s", new_obj.cls, e)
        finally:
            self._inflight = False

    def _call_gemma(self, new_obj: Detection, scene: DetectionFrame) -> dict | None:
        import ollama  # type: ignore[import-not-found]

        position_hint = self._position_hint(new_obj)
        system = (
            'You announce new objects to a blind user wearing a haptic belt. '
            'Reply with one JSON object. Use direction "L" or "R" to tell them to turn '
            'when an obstacle is close ahead. Use "NONE" when speech alone is enough.'
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "New: chair ahead, close."},
            {"role": "assistant", "content": '{"speak":"Chair ahead, step left.","haptic":{"dir":"L","intensity":160},"skip":false}'},
            {"role": "user", "content": f"New: {new_obj.cls} {position_hint}."},
        ]
        client = ollama.Client(host=LOCAL_LLM_HOST)
        resp = client.chat(
            model=LOCAL_LLM_MODEL,
            messages=messages,
            format="json",
            keep_alive="30m",
            options={
                "temperature": 0.2,
                "num_predict": 40,
                "num_ctx": 512,
                "num_thread": 6,
            },
        )
        raw = resp["message"]["content"]
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("narrator: bad JSON from gemma: %r", raw[:200])
            return None
        speak = (decision.get("speak") or "").strip()
        if not speak or "<" in speak or "|" in speak or len(speak) > 80:
            log.warning("narrator: garbage speak %r, skipping", speak[:80])
            return None
        return decision

    @staticmethod
    def _position_hint(d: Detection) -> str:
        h = "left" if d.x < 0.4 else ("right" if d.x > 0.6 else "ahead")
        size = "close" if d.h > 0.4 else ("medium" if d.h > 0.2 else "far")
        return f"{h}, {size}"

    @staticmethod
    def _build_haptic(haptic: dict) -> HapticCommand | None:
        if not isinstance(haptic, dict):
            return None
        dir_ = str(haptic.get("dir", "NONE")).upper()
        if dir_ not in VALID_DIRS or dir_ == "NONE":
            return None
        pattern = str(haptic.get("pattern", "pulse")).lower()
        if pattern not in VALID_PATTERNS:
            pattern = "pulse"
        intensity = int(max(0, min(200, haptic.get("intensity", 120))))
        duration_ms = int(max(100, min(800, haptic.get("duration_ms", 300))))
        return HapticCommand(dir=dir_, intensity=intensity, pattern=pattern, duration_ms=duration_ms)


async def _noop() -> None:
    return None
