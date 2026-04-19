#!/usr/bin/env bash
# Bring up the belt orchestrator with warmup gate + memory watcher.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG_DIR="${BELT_LOG_DIR:-/tmp/belt-logs}"
mkdir -p "$LOG_DIR"

echo "[start-belt] log dir: $LOG_DIR"
echo "[start-belt] serial: ${BELT_SERIAL_URL:-/dev/ttyUSB0}"

if [ "${BELT_WARMUP:-1}" = "1" ]; then
  echo "[start-belt] warming yolo + ollama ..."
  python -c "
from pi.services.yolo import YoloService
from pi.services.camera import CameraService
y = YoloService(CameraService())
y.warmup()
print('  yolo ok')
" || { echo "[start-belt] yolo warmup FAILED"; exit 2; }

  python -c "
from pi.services.voice import VoicePipeline
v = VoicePipeline()
v.warmup()
print('  voice ok')
" || { echo "[start-belt] voice warmup FAILED"; exit 3; }
fi

if [ "${BELT_MEMWATCH:-1}" = "1" ]; then
  (
    while true; do
      ts=$(date +%s)
      ps -o rss=,comm= -p $(pgrep -d, -f "ollama|uvicorn|python") 2>/dev/null \
        | awk -v ts=$ts '{ print ts, $0 }' >> "$LOG_DIR/memwatch.log"
      sleep 10
    done
  ) &
  MEM_PID=$!
  trap "kill $MEM_PID 2>/dev/null || true" EXIT
fi

echo "[start-belt] launching pi.main ..."
exec python -m pi.main
