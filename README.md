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
ollama pull llama3.2:3b
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

## Environment flags

| Variable | Purpose |
|----------|---------|
| `BELT_SERIAL_URL` | pyserial URL: `/dev/ttyUSB0` or `socket://host:port` |
| `KEYWORD_INTENT_ONLY=1` | Skip Ollama; route intents via keyword matcher (A4 fallback) |
| `OPENAI_VOICE=1` | Use OpenAI Realtime instead of local stack (last resort) |
| `BELT_WARMUP=0` | Skip warmup gate in `start-belt.sh` |
| `BELT_MEMWATCH=0` | Skip memory watcher in `start-belt.sh` |
| `BELT_LOG_DIR` | Where logs go (default `/tmp/belt-logs`) |
| `PIPER_VOICE` | Path to Piper voice model `.onnx` |
