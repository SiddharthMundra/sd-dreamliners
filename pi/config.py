"""Single source of truth for all Python-side tunables (CQ4)."""

import os

PROTOCOL_VERSION = 1

SERIAL_URL = os.environ.get("BELT_SERIAL_URL", "/dev/ttyUSB0")
SERIAL_BAUD = int(os.environ.get("BELT_SERIAL_BAUD", "500000"))
PING_INTERVAL_S = 1.0
PONG_TIMEOUT_S = 3.0
AUDIO_MODE_TIMEOUT_S = 5.0

CAMERA_W = 640
CAMERA_H = 360
CAMERA_FPS = 30

YOLO_BACKEND = os.environ.get("BELT_YOLO_BACKEND", "cpu")  # cpu | ei | qaihub
YOLO_WEIGHTS = "yolov8n.pt"
YOLO_IMGSZ = int(os.environ.get("BELT_YOLO_IMGSZ", "256"))
YOLO_CONF = float(os.environ.get("BELT_YOLO_CONF", "0.35"))
YOLO_FRESH_MS = 500
YOLO_CPU_THREADS = int(os.environ.get("BELT_YOLO_THREADS", "4"))

# Edge Impulse QNN path. The .eim must be built in EI Studio with deployment
# target "Linux (AARCH64 with Qualcomm QNN)" so it actually loads the
# Hexagon NPU at runtime. See README.md > "Hardware acceleration".
EI_MODEL_PATH = os.environ.get(
    "BELT_EI_MODEL", os.path.expanduser("~/sd-dreamliners/models/yolo.eim")
)

# Qualcomm AI Hub path: TFLite traced from qai_hub_models.yolo26_det
# (include_postprocessing=False, split_output=True). LiteRT runtime via
# ai_edge_litert. Optionally load the QNN HTP delegate for NPU residency.
# Defaults to fp32 because on QCS6490 it is the fastest (~44 fps end to end)
# and skips int8 quantization noise on raw box coords.
QAIHUB_MODEL_PATH = os.environ.get(
    "BELT_QAIHUB_MODEL",
    os.path.expanduser(
        "~/sd-dreamliners/models/qaihub/yolo26n_qaihub_split/"
        "yolo26n_qaihub_320_split_float32.tflite"
    ),
)
QAIHUB_BACKEND = os.environ.get("BELT_QAIHUB_BACKEND", "cpu")  # cpu | htp
QAIHUB_HTP_DELEGATE = os.environ.get(
    "BELT_QAIHUB_HTP_DELEGATE", "/usr/lib/libQnnTFLiteDelegate.so"
)

DISTANCE_THRESHOLD_MM = 1200  # 1.2 m safety floor
DISTANCE_FRESH_MS = 600
STALE_INPUT_MS = 1000
FUSION_FAST_HZ = 20
FUSION_SLOW_HZ = 5

# How often the safety-floor is allowed to re-fire the SAME motor while an
# obstacle is held. 250 ms was hammering the motors 4x/s; 800 ms is "steady
# drumbeat that a human can parse" without being fatiguing.
HAPTIC_RATE_LIMIT_MS = 800
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

MOTOR_LEFT_IDX = int(os.environ.get("BELT_MOTOR_LEFT_IDX", "0"))
MOTOR_FRONT_IDX = int(os.environ.get("BELT_MOTOR_FRONT_IDX", "1"))
MOTOR_RIGHT_IDX = int(os.environ.get("BELT_MOTOR_RIGHT_IDX", "2"))
MOTOR_BACK_IDX = int(os.environ.get("BELT_MOTOR_BACK_IDX", "3"))

US_PAIR_A_ROLE = os.environ.get("BELT_US_PAIR_A_ROLE", "left")
US_PAIR_A_ANGLE_DEG = float(os.environ.get("BELT_US_PAIR_A_ANGLE_DEG", "-45"))
US_PAIR_B_ROLE = os.environ.get("BELT_US_PAIR_B_ROLE", "right")
US_PAIR_B_ANGLE_DEG = float(os.environ.get("BELT_US_PAIR_B_ANGLE_DEG", "45"))

CAMERA_HFOV_DEG = float(os.environ.get("BELT_CAMERA_HFOV_DEG", "75"))

NAV_POLL_INTERVAL_S = float(os.environ.get("BELT_NAV_POLL_INTERVAL_S", "2.0"))
NAV_OBSTACLE_CM = float(os.environ.get("BELT_NAV_OBSTACLE_CM", "120"))
# gemma3:270m on the QCS6490 CPU takes 2-4 s end-to-end with num_predict=60.
# 2 s was almost always timing out in the real logs; 6 s lets it actually
# finish so "Gemma drives" is more than just a label.
NAV_LLM_TIMEOUT_S = float(os.environ.get("BELT_NAV_LLM_TIMEOUT_S", "6.0"))
NAV_COOLDOWN_MS = int(os.environ.get("BELT_NAV_COOLDOWN_MS", "3000"))

BELT_MIC_DEVICE = os.environ.get("BELT_MIC_DEVICE", "")
