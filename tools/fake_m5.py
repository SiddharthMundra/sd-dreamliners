"""Fake M5 firmware over TCP at localhost:5555.

Speaks the *real* M5 text protocol (``m5firmwarestarkhacks/src/main.cpp``),
so the Pi sees the exact same lines it would from hardware. Lets you develop
``pi/*`` on a laptop with no hardware by pointing at
``BELT_SERIAL_URL=socket://localhost:5555``.

Interactive commands (typed in the terminal running this script, or echoed
into TCP port 5556):
    p   simulate button PTT (Pi records Rubik mic; no serial PCM unless LEGACY)
    f   simulate a hard fall
    d   send one very-close obstacle distance
    l   send obstacle on the left pair (A close, B far)
    r   send obstacle on the right pair (B close, A far)
    q   quit
"""

from __future__ import annotations

import asyncio
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from time import time

HOST = os.environ.get("FAKE_M5_HOST", "localhost")
PORT = int(os.environ.get("FAKE_M5_PORT", "5555"))
CTRL_PORT = int(os.environ.get("FAKE_M5_CTRL_PORT", "5556"))
SAMPLE_RATE = 16000

_PIPER_BIN = shutil.which("piper") or os.path.expanduser("~/.local/bin/piper")
_PIPER_VOICE = os.path.expanduser("~/sd-dreamliners/voices/en_US-amy-low.onnx")
_PIPER_AVAILABLE = os.path.exists(_PIPER_BIN) and os.path.exists(_PIPER_VOICE)


def _sine_pcm16(seconds: float, freq: int = 200) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    out = bytearray()
    for i in range(n):
        v = int(3000 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
        out += int.to_bytes(v & 0xFFFF, 2, "little", signed=False)
    return bytes(out)


def _piper_pcm16(phrase: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav:
        subprocess.run(
            [_PIPER_BIN, "--model", _PIPER_VOICE, "--output_file", wav.name],
            input=phrase.encode(), check=True, stderr=subprocess.DEVNULL,
        )
        with wave.open(wav.name, "rb") as w:
            sr = w.getframerate()
            frames = w.readframes(w.getnframes())
    if sr == SAMPLE_RATE:
        return frames
    arr = bytearray()
    ratio = SAMPLE_RATE / sr
    src = memoryview(frames).cast("h")
    new_len = int(len(src) * ratio)
    for i in range(new_len):
        s = src[min(int(i / ratio), len(src) - 1)]
        arr += int.to_bytes(s & 0xFFFF, 2, "little", signed=False)
    return bytes(arr)


class FakeM5:
    def __init__(self) -> None:
        self.writer: asyncio.StreamWriter | None = None
        self.connected = asyncio.Event()
        # simulated ultrasonic pair state (cm, -1 = no echo)
        self._a_cm = 200
        self._b_cm = 220
        self._transient_until_ms = 0

    async def serve(self) -> None:
        server = await asyncio.start_server(self._on_client, HOST, PORT)
        print(f"[fake-m5] listening on {HOST}:{PORT}")
        async with server:
            await server.serve_forever()

    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        print(f"[fake-m5] client connected: {peer}")
        self.writer = writer
        self.connected.set()
        await self._write_line("R")
        try:
            await asyncio.gather(
                self._read_loop(reader),
                self._sensor_loop(),
            )
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            print("[fake-m5] client disconnected")
            self.writer = None
            self.connected.clear()

    async def _read_loop(self, reader: asyncio.StreamReader) -> None:
        while True:
            line = await reader.readline()
            if not line:
                break
            text = line.decode("ascii", errors="replace").strip()
            if not text:
                continue
            await self._handle_incoming(text)

    async def _handle_incoming(self, text: str) -> None:
        if text.startswith("M,"):
            parts = text.split(",")
            if len(parts) == 4:
                i, p, ms = parts[1], parts[2], parts[3]
                print(f"[fake-m5] HAPTIC motor={i} power={p} {ms}ms")
                await self._write_line("OK motor")
                return
        if text.startswith("MA,"):
            parts = text.split(",")
            if len(parts) == 3:
                p, ms = parts[1], parts[2]
                print(f"[fake-m5] HAPTIC ALL power={p} {ms}ms")
                await self._write_line("OK motor_all")
                return
        if text == "STOP":
            print("[fake-m5] STOP")
            await self._write_line("OK stop")
            return
        if text == "STATUS":
            await self._write_line("OK ready")
            return
        if text == "AUDIO_ON":
            await self._write_line("OK audio_on")
            return
        if text == "AUDIO_OFF":
            await self._write_line("AUDIO_END")
            return
        print(f"[fake-m5] unknown: {text!r}")
        await self._write_line(f"ERR unknown:{text}")

    async def _sensor_loop(self) -> None:
        # Background telemetry: IMU @ 20Hz, distance @ 5Hz.
        tick = 0
        while self.writer is not None:
            await asyncio.sleep(0.05)
            tick += 1
            ax = 0.05 * math.sin(tick / 30)
            ay = 0.02
            az = 0.98
            gx = gy = gz = 0.0
            if tick % 1 == 0:
                await self._write_line(f"I,{ax:.3f},{ay:.3f},{az:.3f},{gx:.2f},{gy:.2f},{gz:.2f}")
            if tick % 4 == 0:
                now_ms = int(time() * 1000)
                if now_ms > self._transient_until_ms:
                    # Drift around open space with a little noise.
                    self._a_cm = max(30, int(200 + 40 * math.sin(tick / 30) + random.randint(-8, 8)))
                    self._b_cm = max(30, int(220 + 40 * math.cos(tick / 30) + random.randint(-8, 8)))
                await self._write_line(f"D,{self._a_cm},{self._b_cm}")

    async def _write_line(self, text: str) -> None:
        if self.writer is None:
            return
        self.writer.write((text + "\n").encode())
        try:
            await self.writer.drain()
        except ConnectionResetError:
            pass

    async def _write_raw(self, data: bytes) -> None:
        if self.writer is None:
            return
        self.writer.write(data)
        try:
            await self.writer.drain()
        except ConnectionResetError:
            pass

    # ---------- interactive triggers ----------

    async def trigger_ptt(self, phrase: str = "find a bottle") -> None:
        if self.writer is None:
            print("[fake-m5] no client connected")
            return
        use_pcm = (
            os.environ.get("LEGACY_M5_AUDIO", "0") == "1"
            and os.environ.get("BELT_M5_SERIAL_PCM", "0") == "1"
        )
        if use_pcm:
            loop = asyncio.get_running_loop()
            if _PIPER_AVAILABLE:
                pcm = await loop.run_in_executor(None, _piper_pcm16, phrase)
            else:
                pcm = _sine_pcm16(0.8)
            print(f"[fake-m5] LEGACY serial PCM PTT ({phrase!r})")
            await self._write_line("B,1")
            chunk_bytes = 512 * 2
            for off in range(0, len(pcm), chunk_bytes):
                blob = pcm[off:off + chunk_bytes]
                await self._write_line(f"AUDIO_CHUNK {len(blob)}")
                await self._write_raw(blob + b"\n")
                await asyncio.sleep(0.02)
            await self._write_line("AUDIO_END")
            await self._write_line("B,0")
            return
        print("[fake-m5] PTT B,1 / B,0 (Pi mic path; phrase unused)")
        await self._write_line("B,1")
        await asyncio.sleep(0.12)
        await self._write_line("B,0")

    async def trigger_fall(self) -> None:
        await self._write_line("F")

    async def trigger_distance(self, a_cm: int, b_cm: int, hold_ms: int = 2000) -> None:
        self._a_cm = a_cm
        self._b_cm = b_cm
        self._transient_until_ms = int(time() * 1000) + hold_ms
        await self._write_line(f"D,{a_cm},{b_cm}")


async def _handle_command(m5: FakeM5, c: str) -> str:
    c = c.strip().lower()
    if c == "p":
        await m5.trigger_ptt()
        return "OK ptt\n"
    if c == "f":
        await m5.trigger_fall()
        return "OK fall\n"
    if c == "d":
        cm = random.randint(15, 60)
        await m5.trigger_distance(cm, cm + random.randint(-5, 5))
        return f"OK distance {cm}cm\n"
    if c == "l":
        await m5.trigger_distance(25, 220)
        return "OK obstacle left (A=25cm)\n"
    if c == "r":
        await m5.trigger_distance(230, 28)
        return "OK obstacle right (B=28cm)\n"
    if c == "q":
        return "BYE\n"
    return f"ERR unknown command: {c!r}\n"


async def _stdin_loop(m5: FakeM5) -> None:
    print("[fake-m5] commands: p=PTT, f=fall, d=close, l=left-close, r=right-close, q=quit")
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            await asyncio.sleep(0.1)
            continue
        reply = await _handle_command(m5, line)
        print(f"[fake-m5] {reply.strip()}")
        if line.strip().lower() == "q":
            return


async def _ctrl_server(m5: FakeM5) -> None:
    async def on_ctrl(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        line = await reader.readline()
        if line:
            reply = await _handle_command(m5, line.decode(errors="ignore"))
            writer.write(reply.encode())
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(on_ctrl, HOST, CTRL_PORT)
    print(f"[fake-m5] control listening on {HOST}:{CTRL_PORT} (echo p/f/d/l/r into TCP to trigger)")
    async with server:
        await server.serve_forever()


async def main() -> None:
    m5 = FakeM5()
    use_stdin = sys.stdin.isatty()
    tasks = [m5.serve(), _ctrl_server(m5)]
    if use_stdin:
        tasks.append(_stdin_loop(m5))
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
