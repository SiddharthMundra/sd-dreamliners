"""Pi-side serial bridge speaking the real M5 firmware protocol.

Wire format (matches ``m5firmwarestarkhacks/src/main.cpp``, plain text + binary
audio, 500000 baud):

  M5 -> Pi:
    R                                 ready banner
    D,<echoA_cm>,<echoB_cm>           ultrasonic pair distances (-1 = no echo)
    I,<ax>,<ay>,<az>,<gx>,<gy>,<gz>   IMU @ ~20 Hz (g, dps)
    F                                 fall confirmed
    B,<1|0>                           button A state changed
    OK <what>                         command acknowledged
    ERR <what>                        command rejected
    AUDIO_CHUNK <N>\\n<N bytes>\\n    binary audio packet (16 kHz s16le)
    AUDIO_END                         audio stream stopped

  Pi -> M5:
    M,<i>,<power>,<ms>                pulse motor i (0..3), power 0..50, ms
    MA,<power>,<ms>                   pulse all 4 motors
    STOP                              stop all motors
    AUDIO_ON / AUDIO_OFF              start/stop audio capture
    STATUS                            request OK banner

Connect via pyserial URL — ``/dev/ttyUSB0`` for real hardware,
``socket://localhost:5555`` for the laptop simulator (``tools.fake_m5``).

Consumers ``await bridge.events.get()`` for parsed dict events with the
shapes ``pi/main.py``'s ``_serial_event_loop`` understands:

    {"t": "hello"}
    {"t": "imu", "ax": .., "ay": .., "az": .., "gx": .., "gy": .., "gz": ..}
    {"t": "distance", "front_cm": int, "back_cm": int, "mm": int}
    {"t": "fall", "severity": "hard", "az_peak": 0.0}
    {"t": "button", "down": bool}
    {"t": "ack", "what": str}
    {"t": "err", "what": str}
    {"t": "_audio_done", "audio": bytes}
    {"t": "_audio_timeout"}
"""

from __future__ import annotations

import asyncio
import logging
import threading
from time import monotonic
from typing import Any, Optional

import serial

from pi.config import (
    AUDIO_MODE_TIMEOUT_S,
    MOTOR_BACK_IDX,
    MOTOR_FRONT_IDX,
    MOTOR_LEFT_IDX,
    MOTOR_RIGHT_IDX,
    PONG_TIMEOUT_S,
    SERIAL_BAUD,
    SERIAL_URL,
)

log = logging.getLogger(__name__)

# Map our logical Direction → motor index on the I2C chain (configurable in pi/config.py).
_DIR_TO_MOTOR_IDX: dict[str, int] = {
    "L": MOTOR_LEFT_IDX,
    "F": MOTOR_FRONT_IDX,
    "R": MOTOR_RIGHT_IDX,
    "B": MOTOR_BACK_IDX,
}

# Firmware accepts power 0..50; our HapticCommand carries 0..255.
_FW_POWER_MAX = 50
_PI_POWER_MAX = 255


def _scale_power(intensity: int) -> int:
    if intensity <= 0:
        return 0
    return max(1, min(_FW_POWER_MAX, round(intensity * _FW_POWER_MAX / _PI_POWER_MAX)))


class SerialBridge:
    """Bidirectional Pi↔M5 channel speaking the M5 firmware text protocol."""

    def __init__(self, ptt_recorder: Optional[Any] = None) -> None:
        self.events: asyncio.Queue[dict] = asyncio.Queue()
        self._ser: Optional[serial.Serial] = None
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._last_rx_at = monotonic()
        # Rubik Pi mic on BtnA PTT (see PiMicRecorder). If None, use M5 AUDIO_CHUNK path.
        self._ptt_recorder = ptt_recorder
        self._pi_ptt_recording = False
        self._pi_ptt_started_at = 0.0

        # Audio capture state machine.
        self._audio_active = False
        self._audio_buf = bytearray()
        self._audio_started_at = 0.0
        # When inside an AUDIO_CHUNK header, this is the number of binary bytes
        # we still owe the parser. 0 means "back to text-line mode".
        self._audio_chunk_remaining = 0

    # ----- lifecycle -----

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._open_serial()
        threading.Thread(target=self._reader_loop, daemon=True, name="serial-rx").start()
        asyncio.create_task(self._audio_timeout_loop())
        asyncio.create_task(self._reconnect_loop())

    def stop(self) -> None:
        self._stop.set()
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass

    def _open_serial(self) -> bool:
        try:
            self._ser = serial.serial_for_url(SERIAL_URL, baudrate=SERIAL_BAUD, timeout=0.05)
            log.info("serial open: %s @ %d baud", SERIAL_URL, SERIAL_BAUD)
            self._last_rx_at = monotonic()
            return True
        except (serial.SerialException, OSError, ConnectionRefusedError) as e:
            log.warning("serial open failed (%s): %s", SERIAL_URL, e)
            self._ser = None
            return False

    async def _reconnect_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(2.0)
            if self._ser is None:
                self._open_serial()

    # ----- health -----

    @property
    def last_pong_age_ms(self) -> int:
        """Time since last byte received (firmware has no ping/pong, but it
        streams IMU @ 20 Hz, so silence > 1 s = something is wrong)."""
        return int((monotonic() - self._last_rx_at) * 1000)

    @property
    def healthy(self) -> bool:
        return self._ser is not None and self.last_pong_age_ms < int(PONG_TIMEOUT_S * 1000)

    # ----- send commands -----

    def send(self, msg) -> None:
        """Accepts:
          * a HapticCommand dataclass (uses its ``to_wire_lines()`` directly),
          * a dict shaped like the legacy JSON wire format (translated to
            firmware syntax for back-compat with old call sites),
          * a string (written as a single firmware line)."""
        # HapticCommand or anything else that knows how to encode itself.
        to_lines = getattr(msg, "to_wire_lines", None)
        if callable(to_lines):
            for line in to_lines():
                self._write_bytes(line if isinstance(line, (bytes, bytearray)) else (str(line) + "\n").encode())
            return
        if isinstance(msg, dict):
            line = self._dict_to_firmware_line(msg)
            if line is not None:
                self._write_line(line)
            return
        if isinstance(msg, str):
            self._write_line(msg)
            return

    def send_raw(self, line: str) -> None:
        """Escape hatch for code that wants to write a raw firmware line."""
        self._write_line(line)

    def send_line(self, line: bytes | str) -> None:
        """Write one newline-terminated command (used by tests and tooling)."""
        if isinstance(line, str):
            self._write_line(line)
        else:
            self._write_bytes(line if line.endswith(b"\n") else line + b"\n")

    @staticmethod
    def _dict_to_firmware_line(msg: dict) -> Optional[str]:
        t = msg.get("t")
        if t == "haptic":
            direction = str(msg.get("dir", "ALL")).upper()
            power = _scale_power(int(msg.get("intensity", 0)))
            duration_ms = max(0, int(msg.get("duration_ms", 0)))
            if direction == "ALL":
                return f"MA,{power},{duration_ms}"
            if power == 0 or duration_ms == 0:
                return "STOP"
            idx = _DIR_TO_MOTOR_IDX.get(direction)
            if idx is None:
                return None
            return f"M,{idx},{power},{duration_ms}"
        if t == "audio_on":
            return "AUDIO_ON"
        if t == "audio_off":
            return "AUDIO_OFF"
        if t == "status":
            return "STATUS"
        if t == "stop":
            return "STOP"
        # ping / unknown control msgs are dropped silently.
        return None

    def _write_line(self, line: str) -> None:
        self._write_bytes((line.rstrip("\n") + "\n").encode())

    def _write_bytes(self, data: bytes) -> None:
        with self._send_lock:
            if self._ser is None:
                log.debug("serial write skipped (port not open)")
                return
            try:
                self._ser.write(data)
            except (serial.SerialException, OSError) as e:
                log.warning("serial write failed: %s", e)
                self._drop_serial()

    def _drop_serial(self) -> None:
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    # ----- read loop -----

    def _reader_loop(self) -> None:
        line_buf = bytearray()
        while not self._stop.is_set():
            if self._ser is None:
                self._stop.wait(0.5)
                continue
            try:
                if self._audio_chunk_remaining > 0:
                    line_buf = self._read_audio_chunk(line_buf)
                else:
                    line_buf = self._read_text(line_buf)
            except (serial.SerialException, OSError, AttributeError) as e:
                if self._stop.is_set():
                    return
                log.warning("serial read error: %s", e)
                self._drop_serial()
                self._stop.wait(0.5)

    def _read_text(self, buf: bytearray) -> bytearray:
        chunk = self._ser.read(512)  # type: ignore[union-attr]
        if not chunk:
            return buf
        self._last_rx_at = monotonic()
        buf.extend(chunk)
        while True:
            nl = buf.find(b"\n")
            if nl == -1:
                return buf
            line = bytes(buf[:nl]).rstrip(b"\r")
            buf = bytearray(buf[nl + 1 :])
            self._handle_text_line(line)
            if self._audio_chunk_remaining > 0:
                # Bytes already buffered after the AUDIO_CHUNK header are payload.
                take = min(self._audio_chunk_remaining, len(buf))
                if take:
                    self._audio_buf.extend(buf[:take])
                    buf = bytearray(buf[take:])
                    self._audio_chunk_remaining -= take
                if self._audio_chunk_remaining == 0:
                    # Eat the trailing newline the firmware writes after the bytes.
                    if buf.startswith(b"\n"):
                        buf = bytearray(buf[1:])
                    elif buf.startswith(b"\r\n"):
                        buf = bytearray(buf[2:])
                # Either way, we're back in text mode.
                if self._audio_chunk_remaining > 0:
                    return buf

    def _read_audio_chunk(self, buf: bytearray) -> bytearray:
        # We're mid-chunk; pull binary bytes off the wire until satisfied.
        chunk = self._ser.read(min(4096, self._audio_chunk_remaining))  # type: ignore[union-attr]
        if not chunk:
            return buf
        self._last_rx_at = monotonic()
        self._audio_buf.extend(chunk)
        self._audio_chunk_remaining -= len(chunk)
        if self._audio_chunk_remaining == 0:
            # Drop the trailing newline. We may not have it yet; the next text
            # read will eat any leading \n / \r\n.
            tail = self._ser.read(1)  # type: ignore[union-attr]
            if tail in (b"\r",):
                self._ser.read(1)  # type: ignore[union-attr]
        return buf

    # ----- line dispatch -----

    def _handle_text_line(self, line: bytes) -> None:
        if not line:
            return
        try:
            text = line.decode("ascii", errors="ignore").strip()
        except Exception:
            return
        if not text:
            return

        if text == "R":
            self._post({"t": "hello", "fw": "m5-stickc+", "caps": ["imu", "us", "haptic", "mic"]})
            return
        if text == "F":
            self._post({"t": "fall", "severity": "hard", "az_peak": 0.0})
            return
        if text == "AUDIO_END":
            log.info("AUDIO_END received, buf=%d bytes", len(self._audio_buf))
            self._finalize_audio()
            return
        if text.startswith("AUDIO_CHUNK"):
            log.info("audio chunk header: %s (active=%s)", text, self._audio_active)
            self._begin_audio_chunk(text)
            return
        if text.startswith("OK "):
            ack = text[3:].strip()
            if ack == "audio_on":
                self._begin_audio_capture()
            self._post({"t": "ack", "what": ack})
            return
        if text.startswith("ERR"):
            self._post({"t": "err", "what": text[3:].strip()})
            return
        if text.startswith("I,"):
            self._handle_imu(text[2:])
            return
        if text.startswith("D,"):
            self._handle_distance(text[2:])
            return
        if text.startswith("B,"):
            self._handle_button(text[2:])
            return
        log.debug("unknown m5 line: %r", text[:80])

    def _handle_imu(self, body: str) -> None:
        parts = body.split(",")
        if len(parts) < 3:
            return
        try:
            ax, ay, az = float(parts[0]), float(parts[1]), float(parts[2])
            gx = float(parts[3]) if len(parts) > 3 else 0.0
            gy = float(parts[4]) if len(parts) > 4 else 0.0
            gz = float(parts[5]) if len(parts) > 5 else 0.0
        except ValueError:
            return
        self._post({
            "t": "imu", "ax": ax, "ay": ay, "az": az,
            "gx": gx, "gy": gy, "gz": gz, "source": "m5",
        })

    def _handle_distance(self, body: str) -> None:
        parts = body.split(",")
        if len(parts) < 2:
            return
        try:
            a_cm = int(parts[0])
            b_cm = int(parts[1])
        except ValueError:
            return
        valid = [v for v in (a_cm, b_cm) if v >= 0]
        closest_cm = min(valid) if valid else -1
        msg = {
            "t": "distance",
            "left_cm": a_cm,
            "right_cm": b_cm,
            "mm": -1 if closest_cm < 0 else closest_cm * 10,
        }
        self._post(msg)

    def _handle_button(self, body: str) -> None:
        try:
            down = bool(int(body.strip()))
        except ValueError:
            return

        rec = self._ptt_recorder
        use_pi = False
        if rec is not None:
            av = getattr(type(rec), "available", None)
            use_pi = bool(av()) if callable(av) else True

        if down:
            if use_pi and rec is not None:
                if rec.start():
                    self._pi_ptt_recording = True
                    self._pi_ptt_started_at = monotonic()
                else:
                    self._write_line("AUDIO_ON")
                    self._begin_audio_capture()
            else:
                self._write_line("AUDIO_ON")
                self._begin_audio_capture()
        else:
            if self._pi_ptt_recording and rec is not None:
                self._pi_ptt_recording = False
                pcm = rec.stop()
                self._post({"t": "_audio_done", "audio": pcm})
            elif self._audio_active:
                self._write_line("AUDIO_OFF")

        self._post({"t": "button", "down": down})

    # ----- audio capture state machine -----

    def _begin_audio_capture(self) -> None:
        if self._audio_active:
            return
        self._audio_active = True
        self._audio_buf.clear()
        self._audio_started_at = monotonic()

    def _begin_audio_chunk(self, header: str) -> None:
        # AUDIO_CHUNK <N>
        try:
            n = int(header.split()[1])
        except (IndexError, ValueError):
            return
        if n <= 0 or n > 1 << 20:
            return
        if not self._audio_active:
            self._begin_audio_capture()
        self._audio_chunk_remaining = n

    def _finalize_audio(self) -> None:
        if not self._audio_active:
            return
        audio = bytes(self._audio_buf)
        self._audio_buf.clear()
        self._audio_active = False
        self._audio_chunk_remaining = 0
        if audio:
            self._post({"t": "_audio_done", "audio": audio})

    # ----- helpers -----

    def _post(self, msg: dict) -> None:
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self.events.put_nowait, msg)

    async def _audio_timeout_loop(self) -> None:
        while not self._stop.is_set():
            await asyncio.sleep(0.5)
            if self._pi_ptt_recording and self._ptt_recorder is not None:
                if monotonic() - self._pi_ptt_started_at > AUDIO_MODE_TIMEOUT_S:
                    log.warning("Pi PTT timed out, stopping mic capture")
                    self._pi_ptt_recording = False
                    pcm = self._ptt_recorder.stop()
                    if pcm:
                        self._post({"t": "_audio_done", "audio": pcm})
                    else:
                        self._post({"t": "_audio_timeout"})
                continue
            if self._audio_active and monotonic() - self._audio_started_at > AUDIO_MODE_TIMEOUT_S:
                log.warning("audio mode timed out, force-flushing")
                self._audio_active = False
                self._audio_chunk_remaining = 0
                self._audio_buf.clear()
                self._post({"t": "_audio_timeout"})
