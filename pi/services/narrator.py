"""Gemma-driven haptic navigator.

Replaces the old "one-shot per new class" narrator. At ``NAV_POLL_INTERVAL_S``
we fuse the current YOLO detections with the latest ultrasonic pair
readings into a single compact scene description and ask Gemma to choose:

  - ``dir``:     one of F, B, L, R, NONE  (where to *push the user away from*
                 the obstacle — i.e. obstacle on the left -> dir=R)
  - ``urgency``: 0..255 (mapped to motor intensity / duration)
  - ``speak``:   short spoken hint (or empty string for silent steer)

The prompt is strict JSON-schema-style and the user message is a
structured, tokens-per-line summary so the 270M model can do basic
spatial reasoning. We rate-limit per side and drop the call if one is
still in flight — navigation must never queue stale advice.

If Gemma times out / returns garbage, we fall back to a deterministic
rule: obstacle's angle < 0 -> haptic R, > 0 -> haptic L, centered -> F.
"""

from __future__ import annotations

import asyncio
import json
import logging
from time import monotonic, perf_counter, time
from typing import Awaitable, Callable, Optional

from pi.config import (
    CAMERA_HFOV_DEG,
    LOCAL_LLM_HOST,
    LOCAL_LLM_MODEL,
    NAV_COOLDOWN_MS,
    NAV_LLM_TIMEOUT_S,
    NAV_OBSTACLE_CM,
    NAV_POLL_INTERVAL_S,
)
from pi.models import Detection, DetectionFrame, DistanceReading, HapticCommand
from pi.services.voice import VoicePipeline
from pi.services.yolo import YoloService

log = logging.getLogger(__name__)

VALID_DIRS = {"F", "B", "L", "R", "NONE"}
_NEAR_CLASSES = {
    "person", "chair", "couch", "bench", "bicycle", "motorcycle", "car",
    "dining table", "table", "dog", "cat", "stairs", "door", "suitcase",
}


HapticEmitter = Callable[[HapticCommand], Awaitable[None]]


class NarratorService:
    def __init__(
        self,
        yolo: YoloService,
        voice: VoicePipeline,
        emit_haptic: HapticEmitter,
        distance_supplier: Callable[[], DistanceReading | None] | None = None,
        broadcast: Callable[[dict], Awaitable[None]] | None = None,
        automation_active_supplier: Optional[Callable[[], bool]] = None,
    ) -> None:
        self._yolo = yolo
        self._voice = voice
        self._emit_haptic = emit_haptic
        self._distance_supplier = distance_supplier or (lambda: None)
        self._broadcast = broadcast or (lambda _msg: _noop())
        self._automation_active_supplier = automation_active_supplier
        self._inflight = False
        self._stop = asyncio.Event()
        self._last_emit_ms = 0
        self._last_spoken_cls: str = ""
        self._last_spoken_at_ms = 0

    async def start(self) -> None:
        log.info(
            "navigator: deterministic mode, poll=%.2fs obstacle=%dcm",
            NAV_POLL_INTERVAL_S, NAV_OBSTACLE_CM,
        )
        asyncio.create_task(self._loop())

    def stop(self) -> None:
        self._stop.set()

    async def _warmup(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            t0 = perf_counter()
            await loop.run_in_executor(
                None,
                self._call_gemma,
                _dummy_scene_summary(),
            )
            log.info("navigator: warmup ok (%.0fms)", (perf_counter() - t0) * 1000)
        except Exception as e:
            log.warning("navigator: warmup failed: %s", e)

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.sleep(NAV_POLL_INTERVAL_S)
            except asyncio.CancelledError:
                return
            if self._automation_active_supplier is not None and not self._automation_active_supplier():
                continue
            if self._inflight:
                continue
            scene = self._yolo.get_latest()
            distance = self._distance_supplier()
            if not self._has_obstacle(scene, distance):
                continue
            if _now_ms() - self._last_emit_ms < NAV_COOLDOWN_MS:
                continue
            self._inflight = True
            asyncio.create_task(self._react(scene, distance))

    @staticmethod
    def _has_obstacle(scene: DetectionFrame, dist: DistanceReading | None) -> bool:
        if dist is not None:
            cm = dist.min_cm
            if 0 <= cm < NAV_OBSTACLE_CM:
                return True
        return any(
            (b.cls in _NEAR_CLASSES and b.h > 0.2)
            for b in scene.boxes
        )

    async def _react(self, scene: DetectionFrame, dist: DistanceReading | None) -> None:
        loop = asyncio.get_running_loop()
        try:
            decision = _fallback_decision(scene, dist)
        finally:
            self._inflight = False

        cmd = _build_haptic(decision)
        if cmd is None:
            return
        await self._broadcast({
            "t": "navigator",
            "decision": decision,
            "ts_ms": int(time() * 1000),
        })
        self._last_emit_ms = _now_ms()
        await self._emit_haptic(cmd)
        speak = (decision.get("speak") or "").strip()
        if speak and self._should_speak(speak, scene):
            loop.run_in_executor(None, self._voice.speak, speak)

    def _should_speak(self, speak: str, scene: DetectionFrame) -> bool:
        # Per-class speech cooldown so the narrator doesn't repeat itself
        # every 600ms. "class" is approximated by the first detected obstacle.
        cls = next((b.cls for b in scene.boxes if b.cls in _NEAR_CLASSES), "obstacle")
        now = _now_ms()
        if cls == self._last_spoken_cls and now - self._last_spoken_at_ms < 10000:
            return False
        self._last_spoken_cls = cls
        self._last_spoken_at_ms = now
        return True

    def _call_gemma(self, summary: str) -> dict | None:
        import ollama  # type: ignore[import-not-found]

        system = (
            "You steer a blind user wearing a 4-motor haptic belt.\n"
            "The belt has 4 motors: LEFT, FRONT, RIGHT, BACK.\n"
            "The belt has 2 ultrasonic sensors: one facing FRONT (angle 0) and\n"
            "one facing REAR (angle 180). The camera reports YOLO boxes with\n"
            "their own angle (negative = to the user's LEFT, positive = RIGHT).\n"
            "Read the SCENE and reply with ONE JSON object, nothing else:\n"
            '  {"dir":"L|R|F|B|NONE","urgency":0-255,"speak":"<<=10 words>"}\n'
            "Rules (choose exactly one dir):\n"
            " - ultrasonic FRONT close (<120cm) and no strong side cue -> dir=F.\n"
            " - ultrasonic REAR  close (<120cm) -> dir=B (alert: something behind).\n"
            " - YOLO obstacle at angle < -20 (on user's LEFT)  -> dir=R (step right).\n"
            " - YOLO obstacle at angle > +20 (on user's RIGHT) -> dir=L (step left).\n"
            " - YOLO obstacle near 0 and close                 -> dir=F.\n"
            " - Both front AND rear close                      -> dir=F (stop).\n"
            " - Nothing close -> dir=NONE, urgency=0, speak=\"\".\n"
            "urgency scales with proximity: <40cm => 240, <80cm => 200, <120cm => 160."
        )
        client = ollama.Client(host=LOCAL_LLM_HOST)
        resp = client.chat(
            model=LOCAL_LLM_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": _FEW_SHOT_Q1},
                {"role": "assistant", "content": _FEW_SHOT_A1},
                {"role": "user", "content": _FEW_SHOT_Q2},
                {"role": "assistant", "content": _FEW_SHOT_A2},
                {"role": "user", "content": _FEW_SHOT_Q3},
                {"role": "assistant", "content": _FEW_SHOT_A3},
                {"role": "user", "content": _FEW_SHOT_Q4},
                {"role": "assistant", "content": _FEW_SHOT_A4},
                {"role": "user", "content": summary},
            ],
            format="json",
            keep_alive="30m",
            options={"temperature": 0.1, "num_predict": 60, "num_ctx": 768, "num_thread": 2},
        )
        raw = resp["message"]["content"]
        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("navigator: bad JSON from gemma: %r", raw[:200])
            return None
        if not isinstance(decision, dict):
            return None
        if str(decision.get("dir", "")).upper() not in VALID_DIRS:
            return None
        return decision


def _now_ms() -> int:
    return int(time() * 1000)


def _build_scene_summary(scene: DetectionFrame, dist: DistanceReading | None) -> str:
    lines: list[str] = ["SCENE:"]
    if dist is not None and _now_ms() - dist.ts_ms < 1500:
        if dist.a_cm >= 0:
            lines.append(
                f"- ultrasonic {dist.a_role} @ {dist.a_angle_deg:+.0f}deg: {dist.a_cm}cm"
            )
        if dist.b_cm >= 0:
            lines.append(
                f"- ultrasonic {dist.b_role} @ {dist.b_angle_deg:+.0f}deg: {dist.b_cm}cm"
            )
    else:
        lines.append("- ultrasonic: stale")
    if scene.boxes:
        ranked = sorted(scene.boxes, key=lambda b: -b.h)[:4]
        for b in ranked:
            ang = b.angle_deg(CAMERA_HFOV_DEG)
            proximity = _proximity_label(b.h)
            lines.append(
                f"- yolo {b.cls} @ {ang:+.0f}deg ({proximity}, conf={b.conf:.2f})"
            )
    else:
        lines.append("- yolo: no detections")
    return "\n".join(lines)


def _proximity_label(h: float) -> str:
    if h > 0.45:
        return "very close"
    if h > 0.25:
        return "close"
    if h > 0.12:
        return "medium"
    return "far"


def _dummy_scene_summary() -> str:
    return (
        "SCENE:\n"
        "- ultrasonic front @ 0deg: 180cm\n"
        "- ultrasonic rear  @ 180deg: 200cm\n"
        "- yolo: no detections"
    )


# Few-shot exemplars — keep their wording tightly aligned with the live
# scene summary that _build_scene_summary emits, because gemma3:270m leans
# heavily on surface patterns.
_FEW_SHOT_Q1 = (
    "SCENE:\n"
    "- ultrasonic front @ 0deg: 260cm\n"
    "- ultrasonic rear  @ 180deg: 240cm\n"
    "- yolo person @ -45deg (close, conf=0.88)"
)
_FEW_SHOT_A1 = '{"dir":"R","urgency":210,"speak":"Person on your left, step right."}'
_FEW_SHOT_Q2 = (
    "SCENE:\n"
    "- ultrasonic front @ 0deg: 35cm\n"
    "- ultrasonic rear  @ 180deg: 280cm\n"
    "- yolo: no detections"
)
_FEW_SHOT_A2 = '{"dir":"F","urgency":240,"speak":"Obstacle directly ahead, stop."}'
_FEW_SHOT_Q3 = (
    "SCENE:\n"
    "- ultrasonic front @ 0deg: 300cm\n"
    "- ultrasonic rear  @ 180deg: 40cm\n"
    "- yolo: no detections"
)
_FEW_SHOT_A3 = '{"dir":"B","urgency":220,"speak":"Something close behind you."}'
_FEW_SHOT_Q4 = (
    "SCENE:\n"
    "- ultrasonic front @ 0deg: 220cm\n"
    "- ultrasonic rear  @ 180deg: 230cm\n"
    "- yolo: no detections"
)
_FEW_SHOT_A4 = '{"dir":"NONE","urgency":0,"speak":""}'


def _fallback_decision(scene: DetectionFrame, dist: DistanceReading | None) -> dict:
    """Deterministic direction picker used when Gemma is slow/unsure.

    Priority order — closest real obstacle wins:
      1. Both ultrasonic pairs close          → F (stop, boxed in)
      2. Single ultrasonic pair close         → map its angle to F/B/L/R
      3. YOLO box (if no ultrasonic trigger)  → image x-position → F/L/R (never B)
    """
    a_close = dist is not None and 0 <= dist.a_cm < NAV_OBSTACLE_CM
    b_close = dist is not None and 0 <= dist.b_cm < NAV_OBSTACLE_CM

    if a_close and b_close and dist is not None:
        cm = min(dist.a_cm, dist.b_cm)
        urgency = _cm_to_urgency(cm)
        # "Boxed in" — tell user to stop (front buzz, highest urgency).
        return {
            "dir": "F",
            "urgency": max(200, min(255, urgency)),
            "speak": "Stop, obstacles front and back.",
        }
    if a_close and dist is not None:
        urgency = _cm_to_urgency(dist.a_cm)
        direction = _angle_to_direction(dist.a_angle_deg)
        return {
            "dir": direction,
            "urgency": max(80, min(255, urgency)),
            "speak": _fallback_phrase(direction, "obstacle"),
        }
    if b_close and dist is not None:
        urgency = _cm_to_urgency(dist.b_cm)
        direction = _angle_to_direction(dist.b_angle_deg)
        return {
            "dir": direction,
            "urgency": max(80, min(255, urgency)),
            "speak": _fallback_phrase(direction, "obstacle"),
        }

    # No ultrasonic trigger → fall back to the camera. YOLO's FOV is narrow
    # (~75°) so treating its horizontal extremes as "far left / far right"
    # of the belt gives the user a useful L/R cue even when the camera
    # never sees ±90°. A thin central band stays F so we don't chatter.
    if scene.boxes:
        near = max(scene.boxes, key=lambda b: b.h)
        if near.h > 0.2:
            urgency = int(80 + near.h * 400)
            if near.x < 0.35:
                direction = "R"
            elif near.x > 0.65:
                direction = "L"
            else:
                direction = "F"
            return {
                "dir": direction,
                "urgency": max(80, min(255, urgency)),
                "speak": _fallback_phrase(direction, near.cls),
            }

    return {"dir": "NONE", "urgency": 0, "speak": ""}


def _angle_to_direction(angle_deg: float) -> str:
    """Shares its ruleset with pi/services/fusion._angle_to_direction."""
    a = ((angle_deg + 180.0) % 360.0) - 180.0
    abs_a = abs(a)
    if abs_a <= 45.0:
        return "F"
    if abs_a >= 135.0:
        return "B"
    return "R" if a < 0 else "L"


def _fallback_phrase(direction: str, label: str) -> str:
    if direction == "F":
        return f"{label} directly ahead."
    if direction == "B":
        return f"{label} close behind you."
    if direction == "R":
        return f"{label} on your left, step right."
    if direction == "L":
        return f"{label} on your right, step left."
    return ""


def _cm_to_urgency(cm: int) -> int:
    if cm <= 0:
        return 255
    if cm >= NAV_OBSTACLE_CM:
        return 80
    ratio = 1 - cm / NAV_OBSTACLE_CM
    return int(80 + ratio * (255 - 80))


def _build_haptic(decision: dict) -> HapticCommand | None:
    dir_ = str(decision.get("dir", "NONE")).upper()
    if dir_ not in VALID_DIRS or dir_ == "NONE":
        return None
    urgency = int(decision.get("urgency", 160))
    urgency = max(0, min(255, urgency))
    if urgency < 40:
        return None
    duration = 200 + int((urgency / 255) * 400)
    return HapticCommand(dir=dir_, intensity=urgency, pattern="pulse", duration_ms=duration)


async def _noop() -> None:
    return None
