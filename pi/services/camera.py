"""Camera service.

Three capture backends, tried in order:
  1. GStreamer + qtiqmmfsrc  — Rubik Pi 5 (Qualcomm CamSS). Verified working.
  2. OpenCV V4L2             — generic USB webcam fallback.
  3. picamera2               — Raspberry Pi OS path (kept for portability).

Frames live in a single-slot ring buffer so consumers always see the latest
frame without blocking the producer.
"""

from __future__ import annotations

import logging
import os
import threading
from time import monotonic, time
from typing import Optional

import numpy as np

from pi.config import CAMERA_FPS, CAMERA_H, CAMERA_W

log = logging.getLogger(__name__)


def _gstreamer_pipeline(src_w: int = 1280, src_h: int = 720, camera: int = 0) -> str:
    """Qualcomm QMMF -> RGB at the configured target size."""
    return (
        f"qtiqmmfsrc camera={camera} ! "
        f"video/x-raw,format=NV12,width={src_w},height={src_h},framerate=30/1 ! "
        f"videoscale ! video/x-raw,width={CAMERA_W},height={CAMERA_H} ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 max-buffers=2 sync=false"
    )


class CameraService:
    def __init__(self) -> None:
        self._frame: Optional[np.ndarray] = None  # stored as BGR
        self._frame_ts_ms: int = 0
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._fps_actual = 0.0
        self._backend = ""

    @property
    def fps(self) -> float:
        return self._fps_actual

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def frame_age_ms(self) -> int:
        if self._frame_ts_ms == 0:
            return 10_000
        return int(time() * 1000) - self._frame_ts_ms

    def get_latest_bgr(self) -> tuple[Optional[np.ndarray], int]:
        with self._frame_lock:
            if self._frame is None:
                return None, 0
            return self._frame, self._frame_ts_ms

    async def start(self) -> None:
        threading.Thread(target=self._capture_loop, daemon=True, name="camera").start()

    def stop(self) -> None:
        self._stop.set()

    def _capture_loop(self) -> None:
        cap, backend = self._open_capture()
        if cap is None:
            log.error("camera: no backend opened")
            return
        self._backend = backend
        log.info("camera: %s @ %dx%d target %d fps", backend, CAMERA_W, CAMERA_H, CAMERA_FPS)

        target_dt = 1.0 / CAMERA_FPS
        frames = 0
        window_start = monotonic()
        import cv2  # type: ignore[import-not-found]

        while not self._stop.is_set():
            t0 = monotonic()
            ok, frame = cap.read()
            if not ok or frame is None:
                self._stop.wait(0.05)
                continue
            if frame.shape[1] != CAMERA_W or frame.shape[0] != CAMERA_H:
                frame = cv2.resize(frame, (CAMERA_W, CAMERA_H), interpolation=cv2.INTER_AREA)
            ts_ms = int(time() * 1000)
            with self._frame_lock:
                self._frame = frame
                self._frame_ts_ms = ts_ms
            frames += 1
            elapsed = monotonic() - window_start
            if elapsed >= 1.0:
                self._fps_actual = frames / elapsed
                frames = 0
                window_start = monotonic()
            sleep_for = target_dt - (monotonic() - t0)
            if sleep_for > 0:
                self._stop.wait(sleep_for)
        cap.release()

    def _open_capture(self):
        backend_pref = os.environ.get("BELT_CAMERA_BACKEND", "auto").lower()
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:
            log.error("opencv (cv2) not installed")
            return None, ""

        if backend_pref in ("auto", "gstreamer"):
            cap = cv2.VideoCapture(_gstreamer_pipeline(), cv2.CAP_GSTREAMER)
            if cap.isOpened():
                return cap, "gstreamer+qtiqmmfsrc"
            log.warning("gstreamer pipeline failed, trying V4L2")

        if backend_pref in ("auto", "v4l2"):
            for idx in (0, 1, 32, 33):
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
                    return cap, f"v4l2:/dev/video{idx}"
            log.warning("V4L2 indices 0/1/32/33 all failed")

        if backend_pref in ("auto", "picamera2"):
            try:
                from picamera2 import Picamera2  # type: ignore[import-not-found]

                cam = Picamera2()
                cam.configure(cam.create_preview_configuration(
                    main={"size": (CAMERA_W, CAMERA_H), "format": "RGB888"},
                ))
                cam.start()
                return _Picamera2Adapter(cam), "picamera2"
            except Exception as e:
                log.warning("picamera2 unavailable: %s", e)

        return None, ""


class _Picamera2Adapter:
    """Make picamera2 quack like cv2.VideoCapture."""

    def __init__(self, cam) -> None:
        self._cam = cam

    def read(self):
        import cv2  # type: ignore[import-not-found]

        try:
            arr = self._cam.capture_array("main")
            return True, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return False, None

    def release(self) -> None:
        try:
            self._cam.stop()
        except Exception:
            pass


def encode_jpeg(frame_bgr: np.ndarray, quality: int = 75) -> bytes:
    """Synchronous JPEG encode (cheap enough at 320x320, no executor needed)."""
    import cv2  # type: ignore[import-not-found]

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


def draw_overlay(frame_bgr: np.ndarray, detections) -> np.ndarray:
    """Bake YOLO boxes into the frame for /mjpeg-overlay and snapshots."""
    import cv2  # type: ignore[import-not-found]

    img = frame_bgr.copy()
    h, w = img.shape[:2]
    for d in detections:
        x = int((d.x - d.w / 2) * w)
        y = int((d.y - d.h / 2) * h)
        bw = int(d.w * w)
        bh = int(d.h * h)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 157, 91), 2)
        label = f"{d.cls} {int(d.conf * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), (255, 157, 91), -1)
        cv2.putText(img, label, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (10, 10, 10), 1, cv2.LINE_AA)
    return img
