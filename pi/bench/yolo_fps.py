"""Phase 1 exit gate: YOLO throughput benchmark.

Pass criterion: p95 inference < 333 ms (= 3 FPS floor).
Fail -> flip to laptop YOLO offload, do not continue with on-Pi vision.

Usage:
    python -m pi.bench.yolo_fps [--frames 200] [--source live|black]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
from time import perf_counter, sleep

import numpy as np

from pi.config import YOLO_CONF, YOLO_IMGSZ, YOLO_WEIGHTS


def _start_camera_blocking():
    """Start the camera capture thread synchronously and wait for first frame."""
    from pi.services.camera import CameraService

    cam = CameraService()
    threading.Thread(target=cam._capture_loop, daemon=True, name="bench-camera").start()
    for _ in range(50):
        frame, _ = cam.get_latest_bgr()
        if frame is not None:
            return cam
        sleep(0.1)
    raise RuntimeError("camera produced no frames within 5s")


def _live_frames(n: int):
    cam = _start_camera_blocking()
    print(f"camera ready: backend={cam.backend} fps={cam.fps:.1f}", flush=True)
    yielded = 0
    last_ts = 0
    while yielded < n:
        frame, ts = cam.get_latest_bgr()
        if frame is None or ts == last_ts:
            sleep(0.02)
            continue
        last_ts = ts
        yielded += 1
        yield frame


def _black_frames(n: int):
    for _ in range(n):
        yield np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--source", choices=("live", "black"), default="live")
    args = parser.parse_args()

    from ultralytics import YOLO  # type: ignore[import-not-found]

    print(f"loading {YOLO_WEIGHTS} ...", flush=True)
    model = YOLO(YOLO_WEIGHTS)
    print("warming up ...", flush=True)
    model(np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8), imgsz=YOLO_IMGSZ, verbose=False)

    source = _live_frames(args.frames) if args.source == "live" else _black_frames(args.frames)
    times_ms: list[float] = []
    print(f"benchmarking {args.frames} frames ({args.source}) ...", flush=True)
    for i, frame in enumerate(source, 1):
        t0 = perf_counter()
        model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
        times_ms.append((perf_counter() - t0) * 1000)
        if i % 25 == 0:
            recent = times_ms[-25:]
            print(f"  {i}/{args.frames}  recent_avg={sum(recent)/len(recent):.1f}ms", flush=True)

    p50 = statistics.median(times_ms)
    p95 = sorted(times_ms)[int(len(times_ms) * 0.95)]
    mean = sum(times_ms) / len(times_ms)
    fps = 1000 / mean

    print()
    print(f"frames    : {len(times_ms)}")
    print(f"mean ms   : {mean:.1f}")
    print(f"p50 ms    : {p50:.1f}")
    print(f"p95 ms    : {p95:.1f}")
    print(f"mean FPS  : {fps:.2f}")
    print()
    if p95 < 333:
        print("VERDICT: PASS  (p95 < 333ms = 3 FPS floor). Run YOLO on Pi.")
        sys.exit(0)
    else:
        print("VERDICT: FAIL  (p95 >= 333ms). Flip to laptop YOLO offload.")
        sys.exit(1)


if __name__ == "__main__":
    main()
