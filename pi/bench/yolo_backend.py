"""Benchmark YOLO inference latency for the two backends.

Usage on the Pi:
    python3 -m pi.bench.yolo_backend cpu
    BELT_EI_MODEL=~/sd-dreamliners/models/yolo.eim python3 -m pi.bench.yolo_backend ei

Runs 30 inferences on a synthetic frame and prints median + p95 ms per
inference. The point is to compare the CPU path (Ultralytics + PyTorch)
against the Edge Impulse runner (NPU when the .eim is QNN-built).

If you see EI median noticeably lower than CPU, the NPU is firing.
If they're the same or EI is slower, the .eim was built with the
wrong target (CPU/TFLite instead of QNN).
"""

from __future__ import annotations

import statistics
import sys
from time import perf_counter

import numpy as np


def bench_cpu(iters: int = 30) -> list[float]:
    from ultralytics import YOLO  # type: ignore[import-not-found]

    from pi.config import YOLO_CONF, YOLO_IMGSZ, YOLO_WEIGHTS

    model = YOLO(YOLO_WEIGHTS)
    frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)

    times: list[float] = []
    for _ in range(iters):
        t0 = perf_counter()
        model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)
        times.append((perf_counter() - t0) * 1000)
    return times


def bench_ei(iters: int = 30) -> list[float]:
    import os

    from edge_impulse_linux.image import ImageImpulseRunner

    from pi.config import EI_MODEL_PATH

    if not os.path.exists(EI_MODEL_PATH):
        raise SystemExit(
            f"No .eim at {EI_MODEL_PATH}. Deploy from EI Studio first."
        )

    runner = ImageImpulseRunner(EI_MODEL_PATH)
    info = runner.init()
    w = info["model_parameters"]["image_input_width"]
    h = info["model_parameters"]["image_input_height"]
    print(f"  ei model: {w}x{h}, labels={len(info['model_parameters']['labels'])}")

    frame = np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8)
    features, _ = runner.get_features_from_image_auto_studio_settings(frame)
    runner.classify(features)

    times: list[float] = []
    try:
        for _ in range(iters):
            features, _ = runner.get_features_from_image_auto_studio_settings(frame)
            t0 = perf_counter()
            runner.classify(features)
            times.append((perf_counter() - t0) * 1000)
    finally:
        runner.stop()
    return times


def report(name: str, times: list[float]) -> None:
    times_sorted = sorted(times)
    p50 = statistics.median(times_sorted)
    p95 = times_sorted[int(0.95 * len(times_sorted)) - 1]
    fps = 1000.0 / p50 if p50 > 0 else 0.0
    print(f"{name}: median {p50:.1f} ms  p95 {p95:.1f} ms  =>  {fps:.1f} fps")


def main() -> None:
    backend = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    print(f"benchmarking yolo backend: {backend}")
    if backend == "cpu":
        report("cpu  ", bench_cpu())
    elif backend == "ei":
        report("ei   ", bench_ei())
    else:
        raise SystemExit("backend must be 'cpu' or 'ei'")


if __name__ == "__main__":
    main()
