# Fusion Belt

**Fusion Belt** is a wearable haptic navigation assistant: a belt that **sees** the world with a camera, **reasons** with on-device vision and voice AI, and **answers through your body** with directional vibration—not a screen, not a speaker. Say what you need (“find the exit,” “what do you see”), and the system fuses computer vision, time-of-flight distance, and IMU sensing into clear left / right / front / back haptic cues.

Built for **StarkHacks 2026** by team **sd-dreamliners**. The full build plan, protocols, and interface contracts live in [`docs/PLAN.md`](docs/PLAN.md).

## How it works

1. **Vision** — A camera feed runs **YOLO** on the Rubik Pi to detect objects and obstacles; a **ToF distance** sensor adds close-range geometry (including things vision might miss).
2. **Voice** — **Push-to-talk** on the M5 Stick captures audio; **local speech-to-text** and (when the hardware budget allows) a **local LLM** turn speech into navigation **intent**—e.g. seeking a target class—scoped to belt tasks, not a general chatbot.
3. **Fusion** — A **navigation fusion engine** combines detections, distance, voice intent, and fall signals from the IMU, then outputs **haptic commands** (direction, intensity, pattern) with safety rules (obstacles, falls, rate limits).
4. **Feel** — Four **vibration modules** around the belt give directional feedback; the **webapp** mirrors live camera, boxes, IMU, haptics, and transcripts for demos and debugging.

The design is **local-first** by default: YOLO and core reasoning stay on the Pi; cloud voice is optional for demos only.

## Hardware (summary)

| Role | Parts |
|------|--------|
| Brain & UI host | Rubik Pi 5 |
| Sensors & PTT | M5 StickC Plus (IMU, mic, button) |
| Haptics | 4× Modulino Vibro (front / back / left / right) |
| Extra sensing | Modulino Movement (IMU), Modulino Distance (ToF) |
| Vision | Raspberry Pi Camera (CSI) |

## Repo layout

```
pi/                  Pi orchestrator (camera, YOLO, voice, fusion, serial, webapp)
pi/bench/            Benchmarks (YOLO throughput exit gate)
tools/fake_m5.py     TCP-based M5 simulator for laptop dev
webapp/              Product UI (single page; MJPEG + WS)
bin/start-belt.sh    Boot script with warmup + memory watcher
docs/PLAN.md         Build plan & interface contracts (source of truth)
```

## Laptop dev (no Rubik needed)

Install minimal deps:

```bash
pip install pyserial fastapi 'uvicorn[standard]' numpy
```

Two terminals:

```bash
# terminal 1: simulated M5 (TCP server, prints haptic commands)
python -m tools.fake_m5
# type p<enter> for PTT cycle, f for fall, d for close obstacle

# terminal 2: pi orchestrator pointed at the simulator
BELT_SERIAL_URL=socket://localhost:5555 KEYWORD_INTENT_ONLY=1 \
  python -m pi.main
```

Open http://localhost:8000 in a browser. The MJPEG stream needs a real camera; without one you'll still see the UI shell, IMU trace from the simulator, distance updates, and (on `p`) the audio-mode round-trip in logs.

## On the Rubik Pi

System Python (no venv). Install deps:

```bash
sudo apt install -y python3-picamera2 ollama
pip install --break-system-packages -r requirements.txt
ollama pull gemma3:270m
```

Run the Phase 1 exit gate first:

```bash
python -m pi.bench.yolo_fps --frames 300 --source live
# pass: p95 < 333 ms (3 FPS floor)
# fail: flip to laptop YOLO offload immediately
```

Boot the belt:

```bash
BELT_SERIAL_URL=/dev/ttyUSB0 ./bin/start-belt.sh
```

## Hardware acceleration — what is on NPU vs CPU

What actually runs where on the Rubik Pi 5 (Qualcomm QCS6490) **today**:

| Stage | CPU / NPU | Notes |
|---|---|---|
| YOLO object detection | **LiteRT (XNNPACK on CPU)** by default; HTP NPU delegate also wired | Default backend is `qaihub`: `yolo26_det` from `qai_hub_models` traced with `include_postprocessing=False, split_output=True`, exported through ONNX → `onnx2tf` → TFLite, served by `ai_edge_litert`. **44 fps** at 320x320 on synthetic input on QCS6490 — leaves the live loop camera-bound, not compute-bound. The HTP delegate (`/usr/lib/libQnnTFLiteDelegate.so`) loads cleanly, but FP32 falls back entirely (QCS6490 HTP rejects FP16 ops); it is only useful for an int8-quantized model when you want to free CPU for STT/LLM. The original Ultralytics CPU path (`BELT_YOLO_BACKEND=cpu`) and the Edge Impulse path (`BELT_YOLO_BACKEND=ei`) are both still selectable. |
| MJPEG video stream | **CPU** (OpenCV JPEG encode) | Capped at the camera's native 14 fps. |
| Whisper STT (`tiny.en`) | **CPU** | ~1.9 s per turn. |
| Ollama / Gemma intent (`gemma3:270m`) | **CPU** | ~3.3 s per turn. Off the hot path — see "Voice budget". |
| Piper TTS (`amy-low`) | **CPU** | ~8.2 s per turn — **the dominator** in voice latency. |
| Narrator service (Gemma reactions to new YOLO boxes) | **CPU** | Async, off the hot path. ~2 s per reaction. |

Head-to-head benchmark on this board (synthetic 640x360 frame, then `bus.jpg`):

| Backend | Median latency | FPS | Boxes returned |
|---|---|---|---|
| `cpu` (Ultralytics YOLOv8n, imgsz=256, 4 threads) | 202.8 ms | 4.9 fps | correct |
| `ei` (QNN .eim, current uploaded model) | 591.0 ms | 1.7 fps | **0** (broken model export) |
| `qaihub` (yolo26n fp32, LiteRT XNNPACK, 4 threads) | **22.9 ms** | **43.5 fps** | correct (5/5 on `bus.jpg`) |
| `qaihub` (yolo26n fp32, LiteRT + HTP delegate) | 35.7 ms | 28.0 fps | correct, but graph falls back to CPU because QCS6490 HTP can't run FP16 |

Live loop FPS is gated by the camera (~14 fps from `qtiqmmfsrc`), so `yolo_fps` ≈ `camera_fps` ≈ 14 in `/healthz`. We're no longer compute-bound — every camera frame gets a fresh inference.

### qai-hub yolo26 — what we ship (`BELT_YOLO_BACKEND=qaihub`, default)

Pipeline:

1. `qai_hub_models.models.yolo26_det.model.Yolo26Detector(include_postprocessing=False, split_output=True)` — gives a PyTorch graph that's just backbone + detection head with raw `(boxes_xywh, class_scores_sigmoid)` outputs and no anchor decode / NMS in the graph. (We strip postproc because the QNN HTP delegate refuses several ops in the canonical YOLO postprocess subgraph and the whole thing falls back to CPU op-by-op, which is slower than just skipping the delegate.)
2. `torch.onnx.export(..., opset_version=13)` → `yolo26n_qaihub_320_split.onnx` (input `[1, 3, 320, 320]`, outputs `boxes [1, 4, 2100]`, `scores [1, 80, 2100]`).
3. `onnx2tf -i ... -oiqt -qt per-tensor` → emits float32, float16 and int8-quantized `.tflite` variants. We default to **float32** because on this board it's the fastest end-to-end (see table above) and it doesn't suffer from the int8 issue described next.
4. `pi/services/yolo_qaihub.py` loads the `.tflite` with `ai_edge_litert.interpreter.Interpreter`, optionally attaches `libQnnTFLiteDelegate.so` (set `BELT_QAIHUB_BACKEND=htp`), then in Python: letterbox BGR → 320x320 RGB tensor → invoke → dequantize (when needed) → argmax-class + threshold + `cv2.dnn.NMSBoxes` → emit normalized `Detection`s the rest of the belt already understands.

Quantization caveat we hit: `onnx2tf`'s default per-tensor int8 calibration picks scale `1/128` for the box output tensor (assuming a normalized range), which clips the actual box outputs (pixel coords 0-320) into garbage. The fix is either to feed real calibration data with `-cind`, or just ship FP32 — which is faster anyway, so that's what we do.

### Why the Edge Impulse NPU path is slower today (and how to fix it)

We **proved the NPU is firing**: during inference the runner process loads `libQnnHtp.so`, `libQnnHtpPrepare.so`, `libQnnHtpV68Stub.so`, and opens `/dev/fastrpc-cdsp-secure` (the Hexagon DSP IPC channel). EI's internal timing reports `classification: 558 ms` per call.

Two things are wrong with the model itself:

1. **8395 "labels"** (`class 1`...`class 8395`). That's the YOLOv8 grid output (`[1, 84, 8400]`) being misread as 8395 classification outputs. The raw `.tflite` was uploaded as a generic BYOM without telling Studio "this is YOLO, decode boxes from the output tensor".
2. **Slow even on NPU.** YOLOv8 has ops (Cast, ArgMax, certain Concat patterns) that the QNN HTP backend doesn't support natively. The delegate handles those by falling back to CPU op-by-op, with NPU↔CPU tensor transfers in between, which dominates wall-clock time.

**The fix:** deploy a model EI Studio actually knows is YOLO, instead of a raw BYOM upload. In Studio:
- New Object-Detection project → Impulse → add Image input + Image processing + **YOLOv5 transfer learning block** with COCO pre-trained weights (no training needed).
- Deploy with target `Linux (AARCH64 with Qualcomm QNN)`.
- Studio will emit a `.eim` whose post-processing is part of the impulse, not the raw model, so the output is decoded correctly and the grid head doesn't run on CPU.

### Flip on NPU (Edge Impulse + Qualcomm QNN)

The integration code is already shipped: `pi/services/yolo_ei.py` runs an Edge Impulse `.eim` runner and maps its `bounding_boxes` into the same `Detection` / `DetectionFrame` types the rest of the belt uses. `pi/main.py` picks the backend with `BELT_YOLO_BACKEND` (default `cpu`, set to `ei` for NPU). The launcher (`~/launch_belt.sh`) auto-falls back to CPU if no `.eim` is present, so flipping it on is one model file away.

**The one manual step (needs an Edge Impulse account):**

1. Go to https://studio.edgeimpulse.com and either create an object-detection project, or clone an existing public one (e.g. https://studio.edgeimpulse.com/public/624749/live — Jallson's YOLO-Pro Smart Parking Meter, already targeted at Rubik Pi).
2. **Deployment** page → pick **"Linux (AARCH64 with Qualcomm QNN)"** as the target. *This is the bit that matters — anything else (CPU/TFLite/ONNX) will run on CPU and you will get no speedup.*
3. Click **Build**. Download the `.eim` file.
4. Copy it to the Pi at the canonical path:
   ```bash
   scp downloaded-model.eim ubuntu@10.10.9.207:/home/ubuntu/sd-dreamliners/models/yolo.eim
   ```
   (or set `BELT_EI_MODEL=/path/to/your.eim` to override.)
5. Verify the runner alone:
   ```bash
   edge-impulse-linux-runner --model-file ~/sd-dreamliners/models/yolo.eim
   ```
   You should see "Inference @ Hexagon" or similar QNN backend logs.
6. Benchmark CPU vs NPU back-to-back:
   ```bash
   python3 -m pi.bench.yolo_backend cpu     # baseline ~200ms / 5 fps
   python3 -m pi.bench.yolo_backend ei      # NPU should be 5-50ms / 20-200 fps
   ```
   If `ei` is slower than `cpu`, the `.eim` was built with the wrong target.
7. Boot the belt with NPU:
   ```bash
   BELT_YOLO_BACKEND=ei ~/launch_belt.sh
   ```
   `tail -f /tmp/main.log` should show `yolo backend: edge-impulse (NPU via QNN .eim)`.

**Alternative (no `.eim` file, CLI auto-fetch):** `edge-impulse-linux` (interactive login → pick project) followed by `edge-impulse-linux-runner --download ~/sd-dreamliners/models/yolo.eim` will pull a built model from your project. Same result.

### Voice budget

The A4 voice-turn benchmark (`python -m pi.bench.voice_turn`, run on this Pi) measured **median 13.2 s** per round-trip vs the **3 s budget**. That fails the gate, so the launcher (`~/launch_belt.sh`) ships with `KEYWORD_INTENT_ONLY=1`. The ~3 s saved is just the LLM intent step; the remaining ~10 s comes from Whisper (1.9 s) + Piper TTS (8.2 s). If voice latency matters for the demo, the next lever is **TTS**, not the LLM.

## Environment flags

| Variable | Purpose |
|----------|---------|
| `BELT_SERIAL_URL` | pyserial URL: `/dev/ttyUSB0` or `socket://host:port` |
| `BELT_YOLO_BACKEND` | `qaihub` (default, qai-hub yolo26 via LiteRT), `cpu` (Ultralytics), or `ei` (Edge Impulse runner) |
| `BELT_QAIHUB_MODEL` | Path to the qai-hub `.tflite` (default `~/sd-dreamliners/models/qaihub/yolo26n_qaihub_split/yolo26n_qaihub_320_split_float32.tflite`) |
| `BELT_QAIHUB_BACKEND` | `cpu` (default, LiteRT XNNPACK) or `htp` (load QNN HTP delegate for NPU residency) |
| `BELT_QAIHUB_HTP_DELEGATE` | Path to the QNN TFLite delegate `.so` (default `/usr/lib/libQnnTFLiteDelegate.so`) |
| `BELT_EI_MODEL` | Path to the `.eim` file (default `~/sd-dreamliners/models/yolo.eim`) |
| `BELT_YOLO_IMGSZ` | YOLO inference input size (default 256, CPU/Ultralytics path only) |
| `BELT_YOLO_THREADS` | PyTorch thread cap for the CPU/Ultralytics YOLO path (default 4) |
| `KEYWORD_INTENT_ONLY=1` | Skip Ollama; route intents via keyword matcher (A4 fallback) |
| `OPENAI_VOICE=1` | Use OpenAI Realtime instead of local stack (last resort) |
| `BELT_WARMUP=0` | Skip warmup gate in `start-belt.sh` |
| `BELT_MEMWATCH=0` | Skip memory watcher in `start-belt.sh` |
| `BELT_LOG_DIR` | Where logs go (default `/tmp/belt-logs`) |
| `PIPER_VOICE` | Path to Piper voice model `.onnx` |
