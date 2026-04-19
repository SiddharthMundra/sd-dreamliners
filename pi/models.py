"""Shared dataclasses crossing service boundaries (CQ3).

The wire format here speaks the real M5 firmware protocol
(``m5firmwarestarkhacks/src/main.cpp``) — plain-text, newline-terminated,
500000 baud serial. We do *not* change the firmware; we only drive its
existing endpoints. See ``pi/services/serial_bridge.py`` for the parser.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Literal

from pi.config import (
    MOTOR_BACK_IDX,
    MOTOR_FRONT_IDX,
    MOTOR_LEFT_IDX,
    MOTOR_RIGHT_IDX,
    PROTOCOL_VERSION,
    US_PAIR_A_ANGLE_DEG,
    US_PAIR_A_ROLE,
    US_PAIR_B_ANGLE_DEG,
    US_PAIR_B_ROLE,
)

Direction = Literal["F", "B", "L", "R", "ALL", "NONE"]
Pattern = Literal["solid", "pulse", "ramp", "sos"]
Intent = Literal["IDLE", "SEEKING", "WHATDOYOUSEE", "STATUS", "EMERGENCY"]
FallSeverity = Literal["soft", "hard"]

# Which motor index (0..3 on the Modulino chain at 0x38..0x3B) lights up for
# each logical direction. Configurable in pi/config.py.
_DIR_TO_MOTOR: dict[str, int] = {
    "L": MOTOR_LEFT_IDX,
    "F": MOTOR_FRONT_IDX,
    "R": MOTOR_RIGHT_IDX,
    "B": MOTOR_BACK_IDX,
}


def _now_ms() -> int:
    return int(time() * 1000)


@dataclass
class HapticCommand:
    """Emitted by fusion / navigator; carried to the M5 as ``M,i,p,ms`` lines.

    ``intensity`` is kept on the 0..255 scale the fusion code already uses;
    the wire encoder clamps it down to the M5 firmware's 0..50 power range
    (the Modulino Vibro driver saturates above that).
    """

    dir: Direction
    intensity: int
    pattern: Pattern
    duration_ms: int
    v: int = PROTOCOL_VERSION

    def to_wire_lines(self) -> list[bytes]:
        """Encode this command as one-or-more M5 serial lines."""
        power = _intensity_to_power(self.intensity)
        dur_ms = max(0, min(2000, int(self.duration_ms)))
        if dur_ms == 0 or power == 0:
            return [b"STOP\n"]
        if self.dir == "ALL":
            return [f"MA,{power},{dur_ms}\n".encode()]
        if self.dir == "NONE":
            return []
        motor = _DIR_TO_MOTOR.get(self.dir)
        if motor is None:
            return []
        if self.pattern == "sos":
            # 3 short buzzes on the chosen motor — the firmware doesn't
            # have a pattern engine, so we compose it from short pulses.
            return [f"M,{motor},{power},{max(120, dur_ms // 3)}\n".encode()] * 3
        return [f"M,{motor},{power},{dur_ms}\n".encode()]


def _intensity_to_power(intensity: int) -> int:
    """Map 0..255 UI intensity to the Modulino 0..50 power scale."""
    if intensity <= 0:
        return 0
    scaled = int(round(intensity * 50 / 255))
    return max(1, min(50, scaled))


@dataclass
class Detection:
    cls: str
    conf: float
    x: float  # normalized [0,1] across image width
    y: float
    w: float
    h: float

    def angle_deg(self, hfov_deg: float = 70.0) -> float:
        """Horizontal offset of the box center in degrees from forward.

        Positive = to the right, negative = to the left. Uses the camera's
        horizontal FOV so YOLO's pixel space lines up with the ultrasonic
        angular labels.
        """
        return (self.x - 0.5) * hfov_deg


@dataclass
class DetectionFrame:
    boxes: list[Detection]
    ts_ms: int = field(default_factory=_now_ms)


@dataclass
class DistanceReading:
    """Two echo-pair readings the M5 firmware reports as ``D,<a>,<b>``.

    The firmware groups its 4 ultrasonic sensors into 2 pairs (pair A =
    sensors 1&2, pair B = sensors 3&4, closest of each pair wins). The
    *role* + *angle* of each pair is configurable via env vars so the
    Pi-side fusion knows where each reading points in space.
    """

    a_cm: int = -1
    b_cm: int = -1
    a_role: str = US_PAIR_A_ROLE
    a_angle_deg: float = US_PAIR_A_ANGLE_DEG
    b_role: str = US_PAIR_B_ROLE
    b_angle_deg: float = US_PAIR_B_ANGLE_DEG
    ts_ms: int = field(default_factory=_now_ms)

    @property
    def min_cm(self) -> int:
        vals = [v for v in (self.a_cm, self.b_cm) if v >= 0]
        return min(vals) if vals else -1

    @property
    def min_mm(self) -> int:
        cm = self.min_cm
        return cm * 10 if cm >= 0 else -1

    def roles_cm(self) -> dict[str, int]:
        out: dict[str, int] = {}
        if self.a_cm >= 0:
            out[self.a_role] = self.a_cm
        if self.b_cm >= 0:
            out[self.b_role] = self.b_cm
        return out


@dataclass
class IMUSample:
    ax: float
    ay: float
    az: float
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    source: str = "m5"
    ts_ms: int = field(default_factory=_now_ms)


@dataclass
class FallEvent:
    severity: FallSeverity = "hard"
    az_peak: float = 0.0
    ts_ms: int = field(default_factory=_now_ms)


@dataclass
class VoiceTurn:
    role: Literal["user", "assistant"]
    text: str
    ts_ms: int = field(default_factory=_now_ms)


@dataclass
class IntentResult:
    intent: Intent
    target: str | None = None
    reply: str = ""


@dataclass
class HealthState:
    serial_ok: bool = False
    yolo_fps: float = 0.0
    pi_cpu: float = 0.0
    warmup_ok: bool = False
    ollama_alive: bool = True
    last_pong_age_ms: int = 0
