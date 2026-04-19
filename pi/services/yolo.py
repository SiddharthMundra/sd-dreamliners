"""YOLOv8n inference service.

Runs in a worker thread so the asyncio loop stays responsive. Latest
detections are cached and exposed via `get_latest()`. Consumers (fusion,
webapp) read snapshots; nothing blocks on inference.
"""

from __future__ import annotations

import logging
import threading
from time import monotonic, time

from pi.config import YOLO_CONF, YOLO_CPU_THREADS, YOLO_IMGSZ, YOLO_WEIGHTS
from pi.models import Detection, DetectionFrame
from pi.services.camera import CameraService

log = logging.getLogger(__name__)


class YoloService:
    def __init__(self, camera: CameraService) -> None:
        self._camera = camera
        self._latest: DetectionFrame = DetectionFrame(boxes=[])
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fps_actual = 0.0

    @property
    def fps(self) -> float:
        return self._fps_actual

    def get_latest(self) -> DetectionFrame:
        with self._lock:
            return self._latest

    async def start(self) -> None:
        threading.Thread(target=self._infer_loop, daemon=True, name="yolo").start()

    def stop(self) -> None:
        self._stop.set()

    def warmup(self) -> None:
        """Run one inference on a black image to load weights into memory."""
        import numpy as np
        import torch
        from ultralytics import YOLO  # type: ignore[import-not-found]

        # Cap CPU thread pool so YOLO bursts don't starve the MJPEG encoder
        # and other asyncio work. Pi has 8 cores; leave 4 for everything else.
        torch.set_num_threads(YOLO_CPU_THREADS)
        torch.set_num_interop_threads(1)

        model = YOLO(YOLO_WEIGHTS)
        dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
        model(dummy, imgsz=YOLO_IMGSZ, verbose=False)
        self._model = model
        log.info("yolo warm (imgsz=%d threads=%d)", YOLO_IMGSZ, YOLO_CPU_THREADS)

    def _infer_loop(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError:
            log.warning("ultralytics not installed, yolo disabled")
            return

        if not hasattr(self, "_model"):
            self._model = YOLO(YOLO_WEIGHTS)

        last_ts = 0
        frame_count = 0
        window_start = monotonic()
        names = self._model.names

        while not self._stop.is_set():
            frame, ts_ms = self._camera.get_latest_bgr()
            if frame is None or ts_ms == last_ts:
                self._stop.wait(0.02)
                continue
            last_ts = ts_ms

            results = self._model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
            boxes = self._extract_boxes(results, names, frame.shape)
            with self._lock:
                self._latest = DetectionFrame(boxes=boxes, ts_ms=int(time() * 1000))

            frame_count += 1
            elapsed = monotonic() - window_start
            if elapsed >= 1.0:
                self._fps_actual = frame_count / elapsed
                frame_count = 0
                window_start = monotonic()

    @staticmethod
    def _extract_boxes(results, names, shape) -> list[Detection]:
        h, w = shape[0], shape[1]
        out: list[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                out.append(Detection(cls=names[cls_id], conf=conf, x=cx, y=cy, w=bw, h=bh))
        return out
