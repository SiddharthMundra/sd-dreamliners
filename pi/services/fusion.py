"""Navigation fusion engine — safety floor only.

This layer is deliberately *dumb and fast*: any time a fresh ultrasonic pair
reports something closer than ``DISTANCE_THRESHOLD_MM``, buzz a short warning
on the corresponding side (or the center motor if both pairs are close). All
reasoning (which direction to turn, what to say) lives in the Gemma-driven
navigator in ``pi/services/narrator.py``.

The emit callback is rate-limited per-direction so the motors don't chatter.
"""

from __future__ import annotations

import asyncio
import logging
from time import time
from typing import Awaitable, Callable, Optional

from pi.config import (
    DISTANCE_FRESH_MS,
    DISTANCE_THRESHOLD_MM,
    FUSION_FAST_HZ,
    HAPTIC_RATE_LIMIT_MS,
    MAX_BUZZ_MS,
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
    """Mutable inputs the fusion loop samples each tick."""

    def __init__(self) -> None:
        self.distance: DistanceReading | None = None
        self.detections: DetectionFrame = DetectionFrame(boxes=[])
        self.intent: IntentResult = IntentResult(intent="IDLE")
        self.fall_active_until_ms: int = 0


def _now_ms() -> int:
    return int(time() * 1000)


def _angle_to_direction(angle_deg: float) -> Direction:
    """Map an obstacle's bearing (degrees from forward, +right/-left) to a belt motor.

    Rules, normalized to (-180, 180]:
      |a| <= 45     → F  (obstacle ahead, buzz front)
      |a| >= 135    → B  (obstacle behind, buzz back)
      -135 < a < -45 → R  (obstacle on the LEFT   → push user RIGHT)
      45 <   a < 135 → L  (obstacle on the RIGHT  → push user LEFT)
    """
    a = ((angle_deg + 180.0) % 360.0) - 180.0
    abs_a = abs(a)
    if abs_a <= 45.0:
        return "F"
    if abs_a >= 135.0:
        return "B"
    return "R" if a < 0 else "L"


class FusionEngine:
    def __init__(
        self,
        state: FusionState,
        emit: HapticEmitter,
        paused_supplier: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._state = state
        self._emit = emit
        self._paused_supplier = paused_supplier
        self._last_emit_per_dir_ms: dict[str, int] = {}

    async def start(self) -> None:
        asyncio.create_task(self._loop())

    async def _loop(self) -> None:
        period = 1.0 / FUSION_FAST_HZ
        while True:
            await asyncio.sleep(period)
            cmd = self._evaluate()
            if cmd is not None:
                await self._maybe_emit(cmd)

    def _evaluate(self) -> HapticCommand | None:
        if self._paused_supplier is not None and self._paused_supplier():
            return None
        if _now_ms() < self._state.fall_active_until_ms:
            return None
        d = self._state.distance
        if d is None or _now_ms() - d.ts_ms > DISTANCE_FRESH_MS:
            return None
        threshold_cm = DISTANCE_THRESHOLD_MM // 10
        a_close = 0 <= d.a_cm < threshold_cm
        b_close = 0 <= d.b_cm < threshold_cm
        if not (a_close or b_close):
            return None
        # Fuse the two echo pins → one haptic. With the current wiring
        # (pair A = left @ -45°, pair B = right @ +45°):
        #   - left-only trip     → F motor (obstacle on forward-left arc)
        #   - right-only trip    → F motor (obstacle on forward-right arc)
        #   - both trip at once  → ALL motors ("boxed in")
        if a_close and b_close:
            closest_cm = min(d.a_cm, d.b_cm)
            direction: Direction = "ALL"
        else:
            closest_cm = d.a_cm if a_close else d.b_cm
            closer_angle = d.a_angle_deg if a_close else d.b_angle_deg
            direction = _angle_to_direction(closer_angle)
        intensity = self._distance_intensity(closest_cm * 10)
        return HapticCommand(
            dir=direction, intensity=intensity, pattern="pulse", duration_ms=150
        )

    def trigger_fall_sos(self) -> HapticCommand:
        self._state.fall_active_until_ms = _now_ms() + 1500
        return HapticCommand(dir="ALL", intensity=255, pattern="sos", duration_ms=1500)

    @staticmethod
    def _distance_intensity(mm: int) -> int:
        # 0mm -> 255, threshold mm -> 80, linear inverse.
        ratio = max(0.0, min(1.0, 1 - mm / max(1, DISTANCE_THRESHOLD_MM)))
        return int(80 + ratio * (255 - 80))

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
