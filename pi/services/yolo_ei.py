"""Edge Impulse YOLO inference service (Qualcomm QNN / Hexagon NPU).

Drop-in replacement for `pi.services.yolo.YoloService` that runs the model
through an Edge Impulse `.eim` runner. When the `.eim` is built in Studio
with deployment target "Linux (AARCH64 with Qualcomm QNN)", the runner
loads the Hexagon NPU at process start and inference happens off-CPU.

Same public API as `YoloService` so `pi/main.py` can swap them with one
env var (`BELT_YOLO_BACKEND=ei`):

    start() / stop() / warmup() / get_latest() -> DetectionFrame
    fps property

If the `.eim` file is missing or fails to start, the service logs and
exits its worker thread without crashing the rest of the belt — fusion
just sees an empty `DetectionFrame` and degrades gracefully.
"""

from __future__ import annotations

import logging
import os
import threading
from time import monotonic, time

from pi.config import EI_MODEL_PATH, YOLO_CONF
from pi.models import Detection, DetectionFrame
from pi.services.camera import CameraService

log = logging.getLogger(__name__)


class YoloEIService:
    def __init__(self, camera: CameraService) -> None:
        self._camera = camera
        self._latest: DetectionFrame = DetectionFrame(boxes=[])
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fps_actual = 0.0
        self._runner = None
        self._model_dim: tuple[int, int] = (0, 0)
        self._resize_mode: str = "squash"

    @property
    def fps(self) -> float:
        return self._fps_actual

    def get_latest(self) -> DetectionFrame:
        with self._lock:
            return self._latest

    async def start(self) -> None:
        threading.Thread(target=self._infer_loop, daemon=True, name="yolo-ei").start()

    def stop(self) -> None:
        self._stop.set()
        if self._runner is not None:
            try:
                self._runner.stop()
            except Exception:
                pass
            self._runner = None

    def warmup(self) -> None:
        """Spawn the EI runner subprocess and run one dummy inference."""
        runner = self._init_runner()
        if runner is None:
            return
        self._runner = runner
        try:
            import numpy as np

            w, h = self._model_dim
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            features, _ = runner.get_features_from_image_auto_studio_settings(dummy)
            runner.classify(features)
            log.info("yolo-ei warm (model=%s dim=%dx%d)", EI_MODEL_PATH, w, h)
        except Exception as exc:
            log.warning("yolo-ei warmup failed: %s", exc)

    def _init_runner(self):
        if not os.path.exists(EI_MODEL_PATH):
            log.error(
                "yolo-ei: model file not found at %s. "
                "Deploy from Edge Impulse Studio with target "
                "'Linux (AARCH64 with Qualcomm QNN)' and copy the .eim there. "
                "See README.md > Hardware acceleration.",
                EI_MODEL_PATH,
            )
            return None
        try:
            from edge_impulse_linux.image import ImageImpulseRunner
        except ImportError:
            log.error(
                "yolo-ei: edge_impulse_linux not installed. "
                "Run: pip3 install --break-system-packages --user edge_impulse_linux"
            )
            return None

        try:
            runner = ImageImpulseRunner(EI_MODEL_PATH)
            info = runner.init()
        except Exception as exc:
            log.error("yolo-ei: failed to start runner: %s", exc)
            return None

        params = info.get("model_parameters", {})
        w = int(params.get("image_input_width", 0))
        h = int(params.get("image_input_height", 0))
        if w == 0 or h == 0:
            log.error("yolo-ei: model is not an image model (input %dx%d)", w, h)
            runner.stop()
            return None

        self._model_dim = (w, h)
        self._resize_mode = params.get("image_resize_mode") or "squash"
        log.info(
            "yolo-ei: loaded %s (input=%dx%d, labels=%d, resize=%s)",
            EI_MODEL_PATH, w, h, len(params.get("labels", [])), self._resize_mode,
        )
        return runner

    def _infer_loop(self) -> None:
        if self._runner is None:
            self._runner = self._init_runner()
        runner = self._runner
        if runner is None:
            log.warning("yolo-ei: no runner, detections disabled")
            return

        last_ts = 0
        frame_count = 0
        window_start = monotonic()

        while not self._stop.is_set():
            frame_bgr, ts_ms = self._camera.get_latest_bgr()
            if frame_bgr is None or ts_ms == last_ts:
                self._stop.wait(0.02)
                continue
            last_ts = ts_ms

            try:
                import cv2

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                features, _ = runner.get_features_from_image_auto_studio_settings(frame_rgb)
                res = runner.classify(features)
            except Exception as exc:
                log.warning("yolo-ei: classify failed: %s", exc)
                self._stop.wait(0.1)
                continue

            boxes = self._extract_boxes(res)
            with self._lock:
                self._latest = DetectionFrame(boxes=boxes, ts_ms=int(time() * 1000))

            frame_count += 1
            elapsed = monotonic() - window_start
            if elapsed >= 1.0:
                self._fps_actual = frame_count / elapsed
                frame_count = 0
                window_start = monotonic()

    def _extract_boxes(self, res: dict) -> list[Detection]:
        """EI returns bounding boxes in *model input* pixel coords. We
        normalize to [0, 1] so downstream code (fusion, webapp overlay)
        sees the same shape it gets from the Ultralytics path."""
        result = (res or {}).get("result") or {}
        bboxes = result.get("bounding_boxes") or []
        mw, mh = self._model_dim
        if mw == 0 or mh == 0:
            return []

        out: list[Detection] = []
        for b in bboxes:
            conf = float(b.get("value", 0.0))
            if conf < YOLO_CONF:
                continue
            x = float(b.get("x", 0))
            y = float(b.get("y", 0))
            w = float(b.get("width", 0))
            h = float(b.get("height", 0))
            cx = (x + w / 2.0) / mw
            cy = (y + h / 2.0) / mh
            bw = w / mw
            bh = h / mh
            out.append(Detection(
                cls=str(b.get("label", "?")),
                conf=conf, x=cx, y=cy, w=bw, h=bh,
            ))
        return out
