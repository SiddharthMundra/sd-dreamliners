"""Qualcomm AI Hub YOLO inference service (LiteRT runtime).

Runs the `yolo26_det` model exported via qai-hub-models, traced with
`include_postprocessing=False` and `split_output=True`, then converted
through onnx2tf to TFLite. Inference is served by `ai_edge_litert`
(Google's LiteRT runtime) which beats Ultralytics+PyTorch on this board
by 4-5x even on CPU because the graph is leaner and there is no Python
torch overhead per frame.

Backends, picked by `BELT_QAIHUB_BACKEND`:
  - `cpu` (default): LiteRT XNNPACK on the QCS6490 Cortex-A78. ~44 fps
    fp32 at 320x320. Best end-to-end latency on this Pi today because
    the QNN HTP delegate falls back to CPU for several yolo26 ops, and
    the round-trip through the delegate ends up slower than running on
    XNNPACK directly.
  - `htp`: load /usr/lib/libQnnTFLiteDelegate.so so the supported subgraph
    actually runs on the Hexagon NPU. Useful when you want NPU residency
    (frees CPU for STT / Ollama). ~35 fps fp32 with partial CPU fallback.

Drop-in replacement for `YoloService` / `YoloEIService`: same public API
(`start`, `stop`, `warmup`, `get_latest`, `fps`).
"""

from __future__ import annotations

import logging
import os
import threading
from time import monotonic, time
from typing import Optional

import numpy as np

from pi.config import QAIHUB_BACKEND, QAIHUB_HTP_DELEGATE, QAIHUB_MODEL_PATH, YOLO_CONF
from pi.models import Detection, DetectionFrame
from pi.services.camera import CameraService

log = logging.getLogger(__name__)


# COCO80 class names (yolo26_det is COCO-trained, same labels as yolov8n).
COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)

NMS_IOU = 0.45
MAX_DETECTIONS = 50


class YoloQAIHubService:
    def __init__(self, camera: CameraService) -> None:
        self._camera = camera
        self._latest: DetectionFrame = DetectionFrame(boxes=[])
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._fps_actual = 0.0
        self._interpreter = None
        self._input_idx = 0
        self._output_box_idx = 0
        self._output_score_idx = 0
        self._input_dim = 320  # square
        self._input_dtype = np.float32
        self._input_scale = 1.0
        self._input_zero = 0
        self._box_scale = 1.0
        self._box_zero = 0
        self._box_dtype = np.float32
        self._score_scale = 1.0
        self._score_zero = 0
        self._score_dtype = np.float32

    @property
    def fps(self) -> float:
        return self._fps_actual

    def get_latest(self) -> DetectionFrame:
        with self._lock:
            return self._latest

    async def start(self) -> None:
        threading.Thread(target=self._infer_loop, daemon=True, name="yolo-qaihub").start()

    def stop(self) -> None:
        self._stop.set()

    def warmup(self) -> None:
        if self._interpreter is None:
            self._init_interpreter()
        if self._interpreter is None:
            return
        dummy = np.zeros((self._input_dim, self._input_dim, 3), dtype=np.uint8)
        for _ in range(2):
            self._infer_once(dummy)
        log.info(
            "yolo-qaihub warm (model=%s backend=%s in=%dx%d dtype=%s)",
            os.path.basename(QAIHUB_MODEL_PATH),
            QAIHUB_BACKEND, self._input_dim, self._input_dim,
            self._input_dtype.__name__,
        )

    def _init_interpreter(self) -> None:
        if not os.path.exists(QAIHUB_MODEL_PATH):
            log.error("yolo-qaihub: model not found: %s", QAIHUB_MODEL_PATH)
            return
        try:
            from ai_edge_litert.interpreter import Interpreter, load_delegate
        except ImportError:
            log.error("yolo-qaihub: ai_edge_litert not installed")
            return

        delegates: list = []
        if QAIHUB_BACKEND == "htp":
            if not os.path.exists(QAIHUB_HTP_DELEGATE):
                log.warning(
                    "yolo-qaihub: HTP delegate missing at %s, falling back to CPU",
                    QAIHUB_HTP_DELEGATE,
                )
            else:
                try:
                    delegates.append(load_delegate(QAIHUB_HTP_DELEGATE, {"backend_type": "htp"}))
                    log.info("yolo-qaihub: HTP delegate loaded (NPU)")
                except Exception as exc:
                    log.warning("yolo-qaihub: HTP delegate failed (%s), falling back to CPU", exc)

        try:
            it = Interpreter(
                model_path=QAIHUB_MODEL_PATH,
                experimental_delegates=delegates,
                num_threads=4,
            )
            it.allocate_tensors()
        except Exception as exc:
            log.error("yolo-qaihub: interpreter init failed: %s", exc)
            return

        in_d = it.get_input_details()[0]
        outs = it.get_output_details()
        # Outputs are named boxes_quantized_output / scores_quantized_output
        # but order can differ between fp32 / int8 builds. Identify by shape:
        # boxes is [1, 4, N], scores is [1, 80, N].
        box_d = next(d for d in outs if d["shape"][1] == 4)
        score_d = next(d for d in outs if d["shape"][1] == 80)

        self._interpreter = it
        self._input_idx = in_d["index"]
        self._output_box_idx = box_d["index"]
        self._output_score_idx = score_d["index"]
        self._input_dim = int(in_d["shape"][1])
        self._input_dtype = np.dtype(in_d["dtype"]).type
        self._input_scale, self._input_zero = self._qparams(in_d)
        self._box_dtype = np.dtype(box_d["dtype"]).type
        self._box_scale, self._box_zero = self._qparams(box_d)
        self._score_dtype = np.dtype(score_d["dtype"]).type
        self._score_scale, self._score_zero = self._qparams(score_d)

    @staticmethod
    def _qparams(detail: dict) -> tuple[float, int]:
        qp = detail.get("quantization_parameters") or {}
        scales = qp.get("scales")
        zeros = qp.get("zero_points")
        if scales is None or len(scales) == 0:
            return 1.0, 0
        return float(scales[0]), int(zeros[0])

    def _preprocess(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox BGR frame to a square model input. Returns the tensor
        plus the scale + pad applied so we can map boxes back later."""
        import cv2  # type: ignore[import-not-found]

        src_h, src_w = frame_bgr.shape[:2]
        dim = self._input_dim
        scale = min(dim / src_w, dim / src_h)
        new_w = int(round(src_w * scale))
        new_h = int(round(src_h * scale))
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((dim, dim, 3), 114, dtype=np.uint8)
        pad_x = (dim - new_w) // 2
        pad_y = (dim - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        if self._input_dtype == np.float32:
            tensor = (rgb.astype(np.float32) / 255.0)[None]
        else:
            # Quantized int8 input: typically scale=1/128, zero=0.
            arr = rgb.astype(np.float32) / 255.0
            arr = arr / max(self._input_scale, 1e-9) + self._input_zero
            tensor = np.clip(arr, -128, 127).astype(self._input_dtype)[None]
        return tensor, scale, pad_x, pad_y

    def _infer_once(self, frame_bgr: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray, float, int, int]]:
        if self._interpreter is None:
            return None
        tensor, scale, pad_x, pad_y = self._preprocess(frame_bgr)
        self._interpreter.set_tensor(self._input_idx, tensor)
        self._interpreter.invoke()
        boxes = self._interpreter.get_tensor(self._output_box_idx)
        scores = self._interpreter.get_tensor(self._output_score_idx)
        if self._box_dtype != np.float32:
            boxes = (boxes.astype(np.float32) - self._box_zero) * self._box_scale
        if self._score_dtype != np.float32:
            scores = (scores.astype(np.float32) - self._score_zero) * self._score_scale
        return boxes, scores, scale, pad_x, pad_y

    def _decode(
        self, boxes_raw: np.ndarray, scores_raw: np.ndarray,
        scale: float, pad_x: int, pad_y: int, src_w: int, src_h: int,
    ) -> list[Detection]:
        """boxes_raw: [1,4,N] xywh in model-input pixel space.
        scores_raw: [1,80,N] already sigmoid'd."""
        # [4, N] and [80, N]
        b = boxes_raw[0]
        s = scores_raw[0]
        # max class per anchor (vectorized)
        cls_id = s.argmax(axis=0)
        cls_conf = s[cls_id, np.arange(s.shape[1])]
        keep = cls_conf >= YOLO_CONF
        if not np.any(keep):
            return []
        b = b[:, keep]
        cls_conf = cls_conf[keep]
        cls_id = cls_id[keep]

        cx, cy, bw, bh = b[0], b[1], b[2], b[3]
        # to xyxy in model-input space
        x1 = cx - bw / 2.0
        y1 = cy - bh / 2.0
        # cv2.dnn.NMSBoxes wants xywh ints
        rects = np.stack([x1, y1, bw, bh], axis=1).astype(np.float32).tolist()
        confs = cls_conf.astype(np.float32).tolist()

        import cv2  # type: ignore[import-not-found]

        idxs = cv2.dnn.NMSBoxes(rects, confs, YOLO_CONF, NMS_IOU)
        if len(idxs) == 0:
            return []
        if hasattr(idxs, "flatten"):
            idxs = idxs.flatten().tolist()

        dim = self._input_dim
        out: list[Detection] = []
        for i in idxs[:MAX_DETECTIONS]:
            # undo letterbox: subtract pad, divide by scale, normalize by src
            ux = (cx[i] - pad_x) / scale / src_w
            uy = (cy[i] - pad_y) / scale / src_h
            uw = bw[i] / scale / src_w
            uh = bh[i] / scale / src_h
            # clip
            ux = float(np.clip(ux, 0.0, 1.0))
            uy = float(np.clip(uy, 0.0, 1.0))
            uw = float(np.clip(uw, 0.0, 1.0))
            uh = float(np.clip(uh, 0.0, 1.0))
            label = COCO_CLASSES[int(cls_id[i])] if int(cls_id[i]) < len(COCO_CLASSES) else "obj"
            out.append(Detection(
                cls=label, conf=float(cls_conf[i]),
                x=ux, y=uy, w=uw, h=uh,
            ))
        return out

    def _infer_loop(self) -> None:
        if self._interpreter is None:
            self._init_interpreter()
        if self._interpreter is None:
            log.warning("yolo-qaihub: no interpreter, detections disabled")
            return

        last_ts = 0
        frame_count = 0
        window_start = monotonic()

        while not self._stop.is_set():
            frame_bgr, ts_ms = self._camera.get_latest_bgr()
            if frame_bgr is None or ts_ms == last_ts:
                self._stop.wait(0.005)
                continue
            last_ts = ts_ms

            try:
                result = self._infer_once(frame_bgr)
            except Exception as exc:
                log.warning("yolo-qaihub: invoke failed: %s", exc)
                self._stop.wait(0.05)
                continue
            if result is None:
                continue

            boxes_raw, scores_raw, scale, pad_x, pad_y = result
            src_h, src_w = frame_bgr.shape[:2]
            boxes = self._decode(boxes_raw, scores_raw, scale, pad_x, pad_y, src_w, src_h)
            with self._lock:
                self._latest = DetectionFrame(boxes=boxes, ts_ms=int(time() * 1000))

            frame_count += 1
            elapsed = monotonic() - window_start
            if elapsed >= 1.0:
                self._fps_actual = frame_count / elapsed
                frame_count = 0
                window_start = monotonic()
