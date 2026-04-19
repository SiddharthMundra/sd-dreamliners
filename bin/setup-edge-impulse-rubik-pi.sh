#!/usr/bin/env bash
# Install Edge Impulse Linux CLI on the Rubik Pi for QNN / hardware-accelerated inference.
#
# Run this ON the Pi (SSH or local terminal), not on your laptop:
#   chmod +x bin/setup-edge-impulse-rubik-pi.sh
#   ./bin/setup-edge-impulse-rubik-pi.sh
#
# After install: deploy your impulse from Studio as
#   "Linux (AARCH64 with Qualcomm QNN)"
# then run e.g.:
#   edge-impulse-linux-runner --model-file path/to/model.eim
#
# Ref: https://docs.edgeimpulse.com/hardware/boards/thundercomm-rubikpi3

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: missing command: $1" >&2
    exit 1
  }
}

echo "[edge-impulse] machine: $(uname -m)  os: $(grep -E '^NAME=' /etc/os-release 2>/dev/null || echo unknown)"

is_qualcomm_linux() {
  if command -v rubikpi_config >/dev/null 2>&1; then
    return 0
  fi
  if [[ -r /etc/os-release ]] && grep -qiE 'qualcomm|qcom|qti' /etc/os-release; then
    return 0
  fi
  return 1
}

install_npm_tools() {
  echo "[edge-impulse] installing npm packages (edge-impulse-linux + edge-impulse-cli bundle) ..."
  # Native addons + global install: use unsafe-perm on embedded Linux.
  sudo npm install -g edge-impulse-linux edge-impulse-cli --unsafe-perm
}

if is_qualcomm_linux; then
  echo "[edge-impulse] detected Qualcomm Linux — using Edge Impulse QC installer"
  need_cmd wget
  TMP="$(mktemp)"
  trap 'rm -f "$TMP"' EXIT
  wget -qO "$TMP" https://cdn.edgeimpulse.com/firmware/linux/setup-edge-impulse-qc-linux.sh
  echo "[edge-impulse] running setup-edge-impulse-qc-linux.sh (follow any prompts) ..."
  sudo bash "$TMP"
  echo "[edge-impulse] QC script done; ensuring latest CLI npm bundle ..."
  need_cmd npm
  install_npm_tools
else
  echo "[edge-impulse] Ubuntu/Debian path — installing Node + GStreamer + build deps"
  need_cmd curl
  need_cmd sudo

  curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
  sudo apt-get update
  sudo apt-get install -y \
    gcc g++ make build-essential \
    nodejs \
    sox \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-base-apps

  install_npm_tools
fi

echo ""
echo "[edge-impulse] verifying binaries ..."
for c in edge-impulse-linux edge-impulse-linux-runner edge-impulse-uploader; do
  if command -v "$c" >/dev/null 2>&1; then
    echo "  ok: $(command -v "$c")"
  else
    echo "  MISSING: $c — check PATH (e.g. export PATH=\"\$HOME/.npm-global/bin:\$PATH\")" >&2
  fi
done

echo ""
echo "[edge-impulse] done."
echo "  Next: Studio → Deployment → Linux (AARCH64 with Qualcomm QNN) → download .eim"
echo "  Then:  edge-impulse-linux-runner --model-file ./your-model.eim"
