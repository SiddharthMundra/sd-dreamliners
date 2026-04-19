"""Capture PTT audio from the Rubik Pi (or any Linux host) default mic.

Uses ``arecord`` (alsa-utils) for minimal native deps on ARM. Produces mono
16-bit little-endian PCM at ``STT_SAMPLE_RATE`` (16 kHz) to match Whisper.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from typing import Optional

from pi.config import BELT_MIC_DEVICE, STT_SAMPLE_RATE

log = logging.getLogger(__name__)


class PiMicRecorder:
    """Start/stop background recording; ``stop()`` returns raw PCM16 bytes."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        return shutil.which("arecord") is not None

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    def start(self) -> bool:
        """Spawn ``arecord``. Returns False if already recording or binary missing."""
        if not self.available():
            log.error("arecord not found; install alsa-utils (sudo apt install alsa-utils)")
            return False
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                log.warning("PiMicRecorder: already recording")
                return False
            cmd = [
                "arecord",
                "-q",
                "-f", "S16_LE",
                "-c", "1",
                "-r", str(STT_SAMPLE_RATE),
                "-t", "raw",
            ]
            if BELT_MIC_DEVICE:
                cmd.extend(["-D", BELT_MIC_DEVICE])
            cmd.append("-")  # stdout
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except OSError as e:
                log.error("failed to start arecord: %s", e)
                self._proc = None
                return False
            log.info("PiMicRecorder: started (%s)", " ".join(cmd))
            return True

    def stop(self) -> bytes:
        """Terminate recording and return all PCM16 bytes captured."""
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return b""
        try:
            out, err = proc.communicate(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            log.warning("PiMicRecorder: arecord killed after timeout")
        if err:
            log.debug("arecord stderr: %s", err[:500] if err else b"")
        raw = bytes(out or b"")
        log.info("PiMicRecorder: stopped, %d bytes", len(raw))
        return raw
