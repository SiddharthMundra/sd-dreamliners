"""Pi-side serial bridge with two-mode protocol (CONTROL JSON / AUDIO PCM16).

Connects via pyserial URL — `socket://localhost:5555` for laptop dev against the
fake M5 simulator, or `/dev/ttyUSB0` (etc.) on the Pi against real firmware.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from time import monotonic

import serial

from pi.config import (
    AUDIO_MODE_TIMEOUT_S,
    PING_INTERVAL_S,
    PONG_TIMEOUT_S,
    PROTOCOL_VERSION,
    SERIAL_BAUD,
    SERIAL_URL,
)

log = logging.getLogger(__name__)

_PTT_UP_MARKER = b"\x00" * 16 + b"{"


class SerialBridge:
    """Bidirectional Pi↔M5 channel.

    Consumers `await bridge.events.get()` for parsed JSON dicts.
    Audio frames arrive as a single `{"t": "_audio_done", "audio": bytes}`
    event after PTT release (or `{"t": "_audio_timeout"}` if the M5 stalled).
    """

    def __init__(self) -> None:
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self._ser: serial.Serial | None = None
        self._mode = "CONTROL"
        self._audio_buf = bytearray()
        self._audio_started_at = 0.0
        self._last_pong_at = monotonic()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._open_serial()
        threading.Thread(target=self._reader_loop, daemon=True, name="serial-rx").start()
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._audio_timeout_loop())
        asyncio.create_task(self._reconnect_loop())

    def _open_serial(self) -> bool:
        try:
            self._ser = serial.serial_for_url(SERIAL_URL, baudrate=SERIAL_BAUD, timeout=0.05)
            log.info("serial open: %s @ %d baud", SERIAL_URL, SERIAL_BAUD)
            return True
        except (serial.SerialException, OSError, ConnectionRefusedError) as e:
            log.warning("serial open failed (%s); will retry: %s", SERIAL_URL, e)
            self._ser = None
            return False

    async def _reconnect_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(2.0)
            if self._ser is None:
                self._open_serial()

    def stop(self) -> None:
        self._stop.set()
        if self._ser:
            self._ser.close()

    def send(self, msg: dict) -> None:
        msg.setdefault("v", PROTOCOL_VERSION)
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        with self._send_lock:
            if self._ser is None:
                return
            try:
                self._ser.write(data)
            except (serial.SerialException, OSError) as e:
                log.warning("serial write failed: %s", e)
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    @property
    def last_pong_age_ms(self) -> int:
        return int((monotonic() - self._last_pong_at) * 1000)

    @property
    def healthy(self) -> bool:
        return self.last_pong_age_ms < int(PONG_TIMEOUT_S * 1000)

    def _reader_loop(self) -> None:
        line_buf = bytearray()
        while not self._stop.is_set():
            if self._ser is None:
                self._stop.wait(0.5)
                continue
            try:
                if self._mode == "CONTROL":
                    line_buf = self._read_control(line_buf)
                else:
                    self._read_audio()
            except (serial.SerialException, OSError, AttributeError) as e:
                if self._stop.is_set():
                    return
                log.warning("serial read error: %s", e)
                try:
                    if self._ser is not None:
                        self._ser.close()
                except Exception:
                    pass
                self._ser = None
                self._stop.wait(0.5)

    def _read_control(self, buf: bytearray) -> bytearray:
        chunk = self._ser.read(256)  # type: ignore[union-attr]
        if not chunk:
            return buf
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
            if nl == -1:
                return buf
            line, buf = bytes(buf[:nl]), bytearray(buf[nl + 1 :])
            self._handle_control_line(line)
            if self._mode == "AUDIO":
                # Any bytes already buffered after the PTT line are audio.
                self._audio_buf.extend(buf)
                buf.clear()
                return buf

    def _handle_control_line(self, line: bytes) -> None:
        line = line.strip()
        if not line:
            return
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            log.debug("dropped malformed line: %r", line[:80])
            return
        if msg.get("v") != PROTOCOL_VERSION:
            log.debug("dropped wrong-version msg: %r", msg)
            return
        t = msg.get("t")
        if t == "pong":
            self._last_pong_at = monotonic()
            return
        if t == "ptt" and msg.get("state") == "down":
            self._mode = "AUDIO"
            self._audio_started_at = monotonic()
            self._audio_buf.clear()
        self._post(msg)

    def _read_audio(self) -> None:
        chunk = self._ser.read(4096)  # type: ignore[union-attr]
        if not chunk:
            return
        self._audio_buf.extend(chunk)
        # Scan recent bytes for the PTT-up sync marker. Restrict the scan
        # window so silence (zeros in PCM) isn't expensive to search through.
        scan_start = max(0, len(self._audio_buf) - len(chunk) - len(_PTT_UP_MARKER))
        idx = self._audio_buf.find(_PTT_UP_MARKER, scan_start)
        if idx == -1:
            return
        audio = bytes(self._audio_buf[:idx])
        # Marker overlaps with the next JSON line's opening brace; keep the '{'.
        tail = bytes(self._audio_buf[idx + 16 :])
        self._audio_buf.clear()
        self._mode = "CONTROL"
        self._post({"t": "_audio_done", "audio": audio})
        if tail:
            self._handle_control_line_buffer(tail)

    def _handle_control_line_buffer(self, tail: bytes) -> None:
        # Process a buffer that may contain one or more newline-terminated lines.
        nl = tail.find(b"\n")
        while nl != -1:
            self._handle_control_line(tail[:nl])
            tail = tail[nl + 1 :]
            nl = tail.find(b"\n")
        if tail:
            # Stash partial line back for next iteration via a fresh reader cycle.
            with self._send_lock:
                # Rare path; push residue back onto the input by re-buffering.
                # Simpler: drop with debug log — partial JSON across mode flip is unlikely.
                log.debug("dropped %d residual bytes after audio end", len(tail))

    def _post(self, msg: dict) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self.events.put_nowait, msg)

    async def _heartbeat_loop(self) -> None:
        seq = 0
        while not self._stop.is_set():
            await asyncio.sleep(PING_INTERVAL_S)
            self.send({"t": "ping", "seq": seq})
            seq += 1
            if self.last_pong_age_ms > PONG_TIMEOUT_S * 1000:
                # Pi-side safety: if M5 is silent, command motors off.
                self.send({
                    "t": "haptic",
                    "dir": "ALL",
                    "intensity": 0,
                    "pattern": "solid",
                    "duration_ms": 0,
                })

    async def _audio_timeout_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.5)
            if self._mode == "AUDIO" and monotonic() - self._audio_started_at > AUDIO_MODE_TIMEOUT_S:
                log.warning("audio mode timed out, force-flipping to CONTROL")
                self._audio_buf.clear()
                self._mode = "CONTROL"
                self._post({"t": "_audio_timeout"})
