# Architecture

Fusion Belt is a three-stage pipeline stretched across two boards. The
M5StickC Plus (ESP32) owns the *sensor-time* domain: IMU, ultrasonics,
button edges, motor pulses. The Rubik Pi owns the *reasoning-time*
domain: vision, LLM intent, voice agent, safety-floor fusion, webapp.
The serial link between them is the single contract; everything else is
an implementation detail on either side.

```
         +-------- M5StickC Plus (ESP32) ---------+
         |  IMU @ 20 Hz   ---> I,ax,ay,az,gx,gy,gz|
         |  Ultrasonic    ---> D,a_cm,b_cm        |
         |    (2x HC-SR04 @ 5 Hz, TRIG pin 32)    |
         |  BtnA edges    ---> B,1  /  B,0        |
         |  Motor driver  <--- M,i,p,ms           |
         |                     MA,p,ms            |
         |                     STOP / STATUS      |
         +----------------+-----------------------+
                          |
                          |   500k baud, 8N1, \r\n framing
                          |   single-char header, comma fields
                          v
 +------------------ Rubik Pi (Linux, Python 3.11) ------------------+
 |                                                                   |
 |   serial_bridge.py ------> events queue -----> main.py orchestrator
 |        ^                                             |             |
 |        |                                             v             |
 |    PTT edges                                 fusion engine (20 Hz) |
 |    -> pi_mic_recorder (arecord 16 kHz mono)   safety-floor haptics |
 |    -> voice pipeline:                                              |
 |         Whisper (tiny.en, int8)                                    |
 |         intent router (Ollama gemma3 / keyword fallback)           |
 |         Piper TTS                                                  |
 |                                                                    |
 |   camera.py ---> YOLO backend (cpu | ei | qaihub) ---> detections  |
 |        |                                                |          |
 |        +----------> MJPEG + overlay --------+           |          |
 |                                             v           v          |
 |                                       webapp_server (FastAPI + WS) |
 |                                             |                      |
 |                                             +--> narrator.py       |
 |                                                  (LLM reasons over |
 |                                                   YOLO + distance, |
 |                                                   emits haptic +   |
 |                                                   spoken guidance) |
 |                                                                    |
 +--------------------------------------------------------------------+
```

## Why the ESP does the tokenizing

Every byte matters on the serial hot path. The ESP frames each sensor
reading into a compact line of ASCII:

```
I,0.012,-0.031,0.984,1.2,-0.5,0.1\r\n     ~36 bytes  @ 20 Hz = 720 B/s
D,25,180\r\n                              ~10 bytes  @  5 Hz = 50 B/s
B,1\r\n                                   ~5 bytes   on edges only
```

An equivalent JSON envelope would cost 80–120 bytes per sample and force
the Pi to hold a JSON parser in the hot loop. Instead, `serial_bridge.py`
dispatches on the first character of each line and hands off to a
per-frame parser — O(1) per event, no allocation beyond the line buffer.

The practical consequence is that the haptic safety floor keeps its
latency budget even when the LLM, Whisper, and MJPEG encoder are all
saturated: distance frames go `D,25,180` → fusion engine → `M,0,180,300`
in well under 50 ms, independent of whatever else the Pi is doing.

## Why the LLM runs on-device

The belt has to work in a basement, on an airplane, in a Faraday cage.
Every piece of reasoning runs locally:

- **ASR.** `faster-whisper` tiny.en, int8 quantized. ~400 ms for a 2 s
  utterance on the QCS6490 CPU.
- **Intent + narration.** Ollama hosts a Gemma-class model
  (`BELT_LOCAL_LLM_MODEL`, default `gemma3:270m`). The narrator shapes
  YOLO detections + distance into a structured prompt and asks the LLM
  to pick one of a fixed set of steering directives.
- **TTS.** Piper with `en_US-amy-low.onnx`. Streams straight to the
  default ALSA sink.
- **Vision.** Three hardware-accelerated backends share a single
  `YoloService` contract (`pi/services/yolo*.py`). Pick via
  `BELT_YOLO_BACKEND`:
  - `cpu` — Ultralytics, good enough for dev.
  - `ei` — Edge Impulse `.eim` with Qualcomm QNN delegate → Hexagon
    NPU residency on the Rubik Pi.
  - `qaihub` — QAI-Hub LiteRT with optional HTP delegate.

If any LLM call exceeds `BELT_OLLAMA_TIMEOUT_S`, the narrator falls back
silently and the safety-floor fusion continues. Voice with
`KEYWORD_INTENT_ONLY=1` bypasses the LLM entirely and routes intents by
keyword match — cheap, deterministic, demo-safe.

## Two-loop safety model

The belt has two concurrent control loops on the Pi side:

1. **Fast loop (20 Hz).** Distance-only → haptic. No LLM, no YOLO, no
   allocations in the hot path. This is the safety floor that protects
   the wearer if every smart subsystem dies.
2. **Slow loop (~5 Hz).** YOLO + distance + (optionally) voice intent →
   LLM → haptic + spoken guidance. This is the "smart" layer and is
   allowed to be slow, drop frames, or stall for a prompt completion.

Both loops emit `HapticCommand` objects through the same rate-limited
emit callback (`_on_haptic` in `pi/main.py`). The rate limiter
(`HAPTIC_RATE_LIMIT_MS`, 800 ms per motor) prevents the two loops from
fighting over the same motor.

## Wire protocol

All frames are `\r\n` terminated, comma-separated ASCII. One character
header, no trailing comma, no whitespace inside fields. The M5 firmware
is the source of truth: see `m5stack-starkhacks/src/main.cpp`.

### Uplink (M5 → Pi)

| Frame | Rate | Fields | Meaning |
|---|---|---|---|
| `R` | 1× on boot | — | firmware ready |
| `I` | 20 Hz | `ax,ay,az,gx,gy,gz` (g, dps) | IMU sample |
| `D` | 5 Hz | `a_cm,b_cm` (int, `-1` = no echo) | ultrasonic pair |
| `B` | on edge | `1` pressed / `0` released | PTT button |
| `OK …` | on demand | free-form | command ack |
| `ERR …` | on demand | free-form | command error |

### Downlink (Pi → M5)

| Frame | Fields | Meaning |
|---|---|---|
| `M,i,p,ms` | motor index, 0–255 power, duration ms | pulse single motor |
| `MA,p,ms` | 0–255 power, duration ms | pulse all motors |
| `STOP` | — | kill motors immediately |
| `STATUS` | — | request `OK ready` |
| `AUDIO_ON` / `AUDIO_OFF` | — | legacy: M5-mic PTT (see `BELT_M5_SERIAL_PCM`) |

### Invariants

- The protocol is frozen. Do not rename frames, reformat fields, or
  "improve" the parser without a corresponding firmware change and an
  explicit call-out in the commit message.
- `-1` means "no echo" for ultrasonic, not "error." Downstream code
  must tolerate it.
- Torn reads are the normal case. `serial_bridge.py` uses a rolling
  line buffer and only dispatches on `\n` — if you see code that looks
  redundant there, it is load-bearing.

## Data flow for one voice turn

1. Wearer presses PTT. M5 emits `B,1`.
2. Pi's `serial_bridge.py` wakes `pi_mic_recorder.py`, which spawns
   `arecord` into a rolling PCM buffer.
3. Wearer releases PTT. M5 emits `B,0`. Pi stops `arecord`, hands the
   PCM to the voice pipeline.
4. Whisper transcribes → `transcript: str`.
5. `voice.derive_intent(transcript, latest_yolo_frame)` either
   keyword-matches (fast path) or asks Ollama for a structured intent.
6. Narrator merges intent + distance + YOLO into a steering decision,
   emits a `HapticCommand` and, optionally, a spoken reply.
7. Piper synthesizes the reply; ALSA plays it. The webapp broadcasts
   the full turn for live observability.

## File-level responsibilities

- `pi/main.py` — orchestrator; one `Belt` class wiring services.
- `pi/config.py` — every `BELT_*` env var, every tunable constant.
- `pi/models.py` — dataclasses crossing service boundaries.
- `pi/services/serial_bridge.py` — serial I/O, line framer, PTT hook.
- `pi/services/pi_mic_recorder.py` — `arecord` subprocess wrapper.
- `pi/services/camera.py` — picamera2 / v4l2 source + MJPEG encoder.
- `pi/services/yolo*.py` — three interchangeable detector backends.
- `pi/services/fusion.py` — the dumb-fast safety-floor loop.
- `pi/services/narrator.py` — the LLM-driven slow loop.
- `pi/services/voice.py` — Whisper + intent router + Piper.
- `pi/services/webapp_server.py` — FastAPI, MJPEG, WebSocket broadcast.
- `m5stack-starkhacks/src/main.cpp` — everything the ESP does.

For the phase plan and exit gates, see [`PLAN.md`](PLAN.md).
