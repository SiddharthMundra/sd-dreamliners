"""Navigation fusion engine. Two-loop architecture (A7).

Fast loop @ 50Hz: Distance-only obstacle avoidance, the safety floor.
Slow loop @ ~5Hz: full priority ladder with YOLO and voice intent.

Each input carries `ts_ms`; stale inputs are ignored (CQ2).
A small rate-limiter prevents motor spam.
"""

from __future__ import annotations

import asyncio
import logging
from time import time
from typing import Awaitable, Callable

from pi.config import (
    DISTANCE_FRESH_MS,
    DISTANCE_THRESHOLD_MM,
    FUSION_FAST_HZ,
    FUSION_SLOW_HZ,
    HAPTIC_RATE_LIMIT_MS,
    MAX_BUZZ_MS,
    STALE_INPUT_MS,
    YOLO_FRESH_MS,
)
from pi.models import (
    DetectionFrame,
    Direction,
    DistanceReading,
    HapticCommand,
    IntentResult,
)

log = logging.getLogger(__name__)


HapticEmitter = Callable[[HapticCommand], Awaitable[None] | None]


class FusionState:
    """Mutable inputs the fusion loops sample each tick."""

    def __init__(self) -> None:
        self.distance: DistanceReading | None = None
        self.detections: DetectionFrame = DetectionFrame(boxes=[])
        self.intent: IntentResult = IntentResult(intent="IDLE")
        self.fall_active_until_ms: int = 0


def _now_ms() -> int:
    return int(time() * 1000)


class FusionEngine:
    def __init__(self, state: FusionState, emit: HapticEmitter) -> None:
        self._state = state
        self._emit = emit
        self._last_emit_per_dir_ms: dict[Direction, int] = {}

    async def start(self) -> None:
        asyncio.create_task(self._fast_loop())
        asyncio.create_task(self._slow_loop())

    async def _fast_loop(self) -> None:
        period = 1.0 / FUSION_FAST_HZ
        while True:
            await asyncio.sleep(period)
            cmd = self._evaluate_distance_only()
            if cmd is not None:
                await self._maybe_emit(cmd)

    async def _slow_loop(self) -> None:
        period = 1.0 / FUSION_SLOW_HZ
        while True:
            await asyncio.sleep(period)
            cmd = self._evaluate_full_ladder()
            if cmd is not None:
                await self._maybe_emit(cmd)

    def _evaluate_distance_only(self) -> HapticCommand | None:
        d = self._state.distance
        if d is None or _now_ms() - d.ts_ms > DISTANCE_FRESH_MS:
            return None
        if d.mm >= DISTANCE_THRESHOLD_MM:
            return None
        intensity = self._distance_intensity(d.mm)
        return HapticCommand(dir="F", intensity=intensity, pattern="pulse", duration_ms=200)

    def _evaluate_full_ladder(self) -> HapticCommand | None:
        if _now_ms() < self._state.fall_active_until_ms:
            return None  # SOS already commanded; let it play out

        d = self._state.distance
        det = self._state.detections
        intent = self._state.intent

        # Priority 2 already covered by fast loop. Skip here to avoid double-emit.
        # Priority 3: YOLO obstacle in path -> haptic on opposite side
        if det.boxes and _now_ms() - det.ts_ms <= YOLO_FRESH_MS:
            in_path = self._closest_in_path(det)
            if in_path is not None:
                turn_dir: Direction = "R" if in_path.x < 0.5 else "L"
                return HapticCommand(dir=turn_dir, intensity=180, pattern="pulse", duration_ms=300)

        # Priority 4: voice SEEKING mode
        if intent.intent == "SEEKING" and intent.target:
            target_box = self._find_target(det, intent.target)
            if target_box is not None:
                point_dir: Direction = self._direction_to(target_box.x)
                return HapticCommand(dir=point_dir, intensity=200, pattern="solid", duration_ms=400)
            return HapticCommand(dir="ALL", intensity=80, pattern="ramp", duration_ms=600)

        return None

    def trigger_fall_sos(self) -> HapticCommand:
        self._state.fall_active_until_ms = _now_ms() + 1500
        return HapticCommand(dir="ALL", intensity=255, pattern="sos", duration_ms=1500)

    @staticmethod
    def _distance_intensity(mm: int) -> int:
        # 0mm -> 255, threshold mm -> 80, linear inverse
        ratio = max(0.0, min(1.0, 1 - mm / DISTANCE_THRESHOLD_MM))
        return int(80 + ratio * (255 - 80))

    @staticmethod
    def _closest_in_path(det: DetectionFrame):
        # In-path = box center in middle third horizontally and bbox tall enough
        # (proxy for "near"). Pick the largest such box.
        candidates = [b for b in det.boxes if 0.33 < b.x < 0.66 and b.h > 0.25]
        if not candidates:
            return None
        return max(candidates, key=lambda b: b.h)

    @staticmethod
    def _find_target(det: DetectionFrame, target: str):
        target_lc = target.lower()
        matches = [b for b in det.boxes if target_lc in b.cls.lower()]
        return max(matches, key=lambda b: b.conf) if matches else None

    @staticmethod
    def _direction_to(x: float) -> Direction:
        if x < 0.4:
            return "L"
        if x > 0.6:
            return "R"
        return "F"

    async def _maybe_emit(self, cmd: HapticCommand) -> None:
        last = self._last_emit_per_dir_ms.get(cmd.dir, 0)
        if _now_ms() - last < HAPTIC_RATE_LIMIT_MS:
            return
        if cmd.duration_ms > MAX_BUZZ_MS:
            cmd.duration_ms = MAX_BUZZ_MS
        self._last_emit_per_dir_ms[cmd.dir] = _now_ms()
        result = self._emit(cmd)
        if asyncio.iscoroutine(result):
            await result
