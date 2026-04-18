"""Shared dataclasses crossing service boundaries (CQ3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Literal

from pi.config import PROTOCOL_VERSION

Direction = Literal["F", "B", "L", "R", "ALL"]
Pattern = Literal["solid", "pulse", "ramp", "sos"]
Intent = Literal["IDLE", "SEEKING", "WHATDOYOUSEE", "STATUS", "EMERGENCY"]
FallSeverity = Literal["soft", "hard"]


def _now_ms() -> int:
    return int(time() * 1000)


@dataclass
class HapticCommand:
    dir: Direction
    intensity: int
    pattern: Pattern
    duration_ms: int
    v: int = PROTOCOL_VERSION

    def to_wire(self) -> dict:
        return {
            "t": "haptic",
            "dir": self.dir,
            "intensity": int(max(0, min(255, self.intensity))),
            "pattern": self.pattern,
            "duration_ms": int(self.duration_ms),
            "v": self.v,
        }


@dataclass
class Detection:
    cls: str
    conf: float
    x: float
    y: float
    w: float
    h: float


@dataclass
class DetectionFrame:
    boxes: list[Detection]
    ts_ms: int = field(default_factory=_now_ms)


@dataclass
class DistanceReading:
    mm: int
    ts_ms: int = field(default_factory=_now_ms)


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
    severity: FallSeverity
    az_peak: float
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
