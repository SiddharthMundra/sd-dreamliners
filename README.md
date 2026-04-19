# Fusion Belt

**An on-device, hardware-accelerated LLM + voice agent for a wearable haptic navigation belt.**

Feel the world. Speak your intent. See through silicon — all on-board.

---

## What it is

Fusion Belt is a wearable navigation aid that fuses a vision model, an
on-device LLM, and a voice agent into a single haptic loop — no cloud, no
phone tether. An M5StickC Plus (ESP32) reads the belt's IMU, ultrasonics,
and push-to-talk button; it tokenizes every sample into a compact framed
serial protocol and streams it to a Rubik Pi, which runs YOLO vision,
Whisper ASR, an Ollama-hosted LLM, and Piper TTS in parallel and pushes
haptic commands back down the same wire.

The ESP doing the parsing/tokenizing on-device is what lets the Pi's LLM
stack hit its latency budget: no JSON overhead on the hot path, no CPU
cycles wasted reshaping sensor blobs, and a single-byte frame header
(`R`, `I`, `D`, `B`, `M`, `MA`, `STOP`, `STATUS`) that the Pi's serial
bridge can dispatch in O(1).

## Architecture, in one paragraph

```
 ┌──────────────────────────┐       framed text @ 500k baud       ┌─────────────────────────────────┐
 │ M5StickC Plus (ESP32)    │  ─────────────────────────────────► │ Rubik Pi (on-device LLM stack)  │
 │                          │                                     │                                 │
 │ • IMU @ 20 Hz   → I,...  │                                     │ Serial bridge → fusion engine   │
 │ • Ultrasonic 5Hz→ D,a,b  │                                     │ → LLM narrator (Ollama)         │
 │ • BtnA PTT edges→ B,1/0  │                                     │ → voice pipeline (Whisper+Piper)│
 │ • On-boot       → R      │                                     │ → YOLO (CPU / QNN / HTP NPU)    │
 │                          │                                     │ → FastAPI webapp + MJPEG        │
 │ • Motors       ◄ M,i,p,ms│ ◄─────────────────────────────────  │                                 │
 │   (rate-limited pulses)  │        haptic commands back         │                                 │
 └──────────────────────────┘                                     └─────────────────────────────────┘
```

Full prose walkthrough: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
Phase plan and latency budgets: [`docs/PLAN.md`](docs/PLAN.md).

## Why this design

- **On-board LLM.** Intent routing runs on a Gemma-class model via Ollama
  on the Rubik Pi — no network, no leaked user speech. Falls back to a
  keyword matcher (`KEYWORD_INTENT_ONLY=1`) when the LLM is cold.
- **Hardware-accelerated vision.** Three backends share a common
  `YoloService` contract: `cpu` (Ultralytics), `ei` (Edge Impulse `.eim`
  on QNN NPU), and `qaihub` (QAI-Hub LiteRT on HTP NPU). Switch via
  `BELT_YOLO_BACKEND`.
- **ESP-side tokenization.** Raw sensor bursts are framed on the MCU, not
  on the Pi. This is the single biggest reason the haptic safety floor
  holds its budget under load: the Pi never has to parse a 1 kB blob
  just to learn "the left sonar sees something at 25 cm."
- **Two-loop safety.** A dumb, fast fusion loop (distance → haptic) is
  the floor. An LLM-driven narrator runs above it and is allowed to be
  slow. If the LLM stalls, you still feel obstacles.

## Quick start (laptop, no belt required)

```bash
pip install -r requirements.txt

# terminal 1 — simulated M5 (TCP, prints haptic pulses)
python -m tools.fake_m5
#   p<enter> = PTT press/release   f = fall   d = close obstacle

# terminal 2 — Pi orchestrator pointed at the simulator
BELT_SERIAL_URL=socket://localhost:5555 \
KEYWORD_INTENT_ONLY=1 \
  python -m pi.main
```

Open <http://localhost:8000> for the webapp (MJPEG stream + live IMU +
distance + voice turn log + haptic trace).

## Quick start (Rubik Pi, real belt)

```bash
sudo apt install -y python3-picamera2 ollama alsa-utils
pip install --break-system-packages -r requirements.txt
ollama pull llama3.2:3b

# Phase-1 exit gate — must pass before running the belt live
python -m pi.bench.yolo_fps --frames 300 --source live
#   pass: p95 < 333 ms (3 FPS floor)
#   fail: flip BELT_YOLO_BACKEND=ei or offload to a laptop

BELT_SERIAL_URL=/dev/ttyUSB0 ./bin/start-belt.sh
```

## Repository layout

```
pi/                         on-device orchestrator (Python 3.11, asyncio)
├── main.py                 wires every service; asyncio entrypoint
├── config.py               all BELT_* env var reads + defaults
├── models.py               dataclasses crossing service boundaries
├── services/
│   ├── serial_bridge.py    USB serial framing; PTT→mic; haptic out
│   ├── pi_mic_recorder.py  arecord wrapper for push-to-talk audio
│   ├── camera.py           picamera2 + v4l2 fallback
│   ├── fusion.py           distance → HapticCommand (safety floor)
│   ├── narrator.py         YOLO + distance → Ollama → steer
│   ├── voice.py            Whisper → intent → Piper TTS
│   ├── yolo.py             CPU Ultralytics backend
│   ├── yolo_ei.py          Edge Impulse / QNN NPU backend
│   ├── yolo_qaihub.py      QAI-Hub LiteRT / HTP NPU backend
│   └── webapp_server.py    FastAPI + MJPEG + WebSocket
└── bench/                  latency + FPS gates (do not refactor)

m5stack-starkhacks/         M5StickC Plus firmware (PlatformIO + Arduino)
├── platformio.ini
└── src/main.cpp            IMU, ultrasonic, BtnA, motors, wire protocol

webapp/                     single-page UI served by webapp_server
tools/                      dev helpers: fake_m5, preview_server, smoke tests
bin/                        start-belt.sh (warmup gate + mem watcher)
docs/                       PLAN.md, ARCHITECTURE.md, CLEANUP_SUMMARY.md
```

## Configuration

Every knob is a `BELT_*` environment variable read in `pi/config.py`.

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `BELT_SERIAL_URL` | str | `/dev/ttyUSB0` | pyserial URL (`/dev/ttyUSB0` or `socket://host:port`) |
| `BELT_SERIAL_BAUD` | int | `500000` | Must match M5 `Serial.begin()` |
| `BELT_MIC_DEVICE` | str | system default | ALSA `-D` arg for PTT capture |
| `BELT_YOLO_BACKEND` | str | `cpu` | `cpu` \| `ei` \| `qaihub` |
| `BELT_EI_MODEL` | path | — | `.eim` path for Edge Impulse backend |
| `BELT_YOLO_IMGSZ` | int | `320` | YOLO input size |
| `BELT_LOCAL_LLM_MODEL` | str | `gemma3:1b` | Ollama model tag |
| `BELT_OLLAMA_URL` | str | `http://localhost:11434` | Ollama endpoint |
| `BELT_OLLAMA_TIMEOUT_S` | float | `2.5` | LLM call timeout |
| `KEYWORD_INTENT_ONLY` | `1`/unset | unset | Skip LLM, use keyword intent |
| `BELT_M5_SERIAL_PCM` | `1`/unset | unset | Legacy: audio-over-serial fallback |
| `LEGACY_M5_AUDIO` | `1`/unset | unset | Alias of above |
| `BELT_WIRE_JSON` | `1`/unset | unset | JSON wire protocol (lab only) |
| `BELT_WEB_HOST` | str | `0.0.0.0` | FastAPI bind host |
| `BELT_WEB_PORT` | int | `8000` | FastAPI bind port |
| `BELT_LOG_DIR` | path | `/tmp/belt-logs` | Log file directory |
| `BELT_WARMUP` | `0`/`1` | `1` | Skip warmup gate in start-belt.sh |
| `BELT_MEMWATCH` | `0`/`1` | `1` | Skip memory watcher in start-belt.sh |
| `PIPER_VOICE` | path | — | Piper voice model `.onnx` |
| `OPENAI_VOICE` | `1`/unset | unset | Use OpenAI Realtime (last-resort fallback) |

Copy [`.env.example`](.env.example) to `.env` and fill in paths.

## Wire protocol (MCU → Pi)

Frozen contract. Every token is one character of header, comma-separated
fields, `\r\n` terminated.

| Frame | Direction | Meaning |
|---|---|---|
| `R` | M5 → Pi | boot ready |
| `I,ax,ay,az,gx,gy,gz` | M5 → Pi | IMU sample @ 20 Hz |
| `D,a_cm,b_cm` | M5 → Pi | ultrasonic pair @ 5 Hz (`-1` = no echo) |
| `B,1` / `B,0` | M5 → Pi | PTT button edge |
| `M,i,power,ms` | Pi → M5 | pulse motor `i` |
| `MA,power,ms` | Pi → M5 | pulse all motors |
| `STOP` | Pi → M5 | kill motors now |
| `STATUS` | Pi → M5 | request health reply |

## License

MIT. See `LICENSE` if present; the M5 firmware vendors `M5StickCPlus`
under its own terms.
