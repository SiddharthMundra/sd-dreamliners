"""Single source of truth for all Python-side tunables (CQ4)."""

import os

PROTOCOL_VERSION = 1

SERIAL_URL = os.environ.get("BELT_SERIAL_URL", "socket://localhost:5555")
SERIAL_BAUD = 921600
PING_INTERVAL_S = 1.0
PONG_TIMEOUT_S = 3.0
AUDIO_MODE_TIMEOUT_S = 2.0

CAMERA_W = 640
CAMERA_H = 360
CAMERA_FPS = 15

YOLO_WEIGHTS = "yolov8n.pt"
YOLO_IMGSZ = 320
YOLO_CONF = 0.35
YOLO_FRESH_MS = 500

DISTANCE_THRESHOLD_MM = 800
DISTANCE_FRESH_MS = 200
STALE_INPUT_MS = 1000
FUSION_FAST_HZ = 50
FUSION_SLOW_HZ = 5

HAPTIC_RATE_LIMIT_MS = 250
MAX_BUZZ_MS = 3000

STT_MODEL = "tiny.en"
STT_COMPUTE_TYPE = "int8"
STT_SAMPLE_RATE = 16000

LOCAL_LLM_MODEL = os.environ.get("BELT_LLM_MODEL", "gemma3:270m")
LOCAL_LLM_HOST = "http://localhost:11434"
VOICE_TURN_BUDGET_MS = 3000

USE_OPENAI_VOICE = os.environ.get("OPENAI_VOICE") == "1"
USE_KEYWORD_INTENT_ONLY = os.environ.get("KEYWORD_INTENT_ONLY") == "1"

PIPER_VOICE_PATH = os.environ.get(
    "PIPER_VOICE",
    os.path.expanduser("~/sd-dreamliners/voices/en_US-amy-low.onnx"),
)

WEBAPP_HOST = "0.0.0.0"
WEBAPP_PORT = 8000

LOG_DIR = os.environ.get("BELT_LOG_DIR", "/tmp/belt-logs")
MEMORY_WATCH_INTERVAL_S = 10.0
