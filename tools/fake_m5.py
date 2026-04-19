"""Fake M5 firmware over TCP at localhost:5555.

Speaks the same wire protocol the real M5 will. Lets you develop pi/* on a
laptop with no hardware. Connect by setting `BELT_SERIAL_URL=socket://localhost:5555`.

Interactive commands (typed in the terminal running this script):
    p   simulate PTT cycle (0.8s of fake audio, then PTT up)
    f   simulate hard fall
    d   send a single distance reading (random close obstacle)
    q   quit
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
import wave
from time import time

HOST = "localhost"
PORT = 5555
CTRL_PORT = 5556
PROTOCOL_VERSION = 1
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
    """Synth `phrase` with Piper and return mono 16kHz PCM16."""
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


def _now_ms() -> int:
    return int(time() * 1000)


def _line(msg: dict) -> bytes:
    msg.setdefault("v", PROTOCOL_VERSION)
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode()


class FakeM5:
    def __init__(self) -> None:
        self.writer: asyncio.StreamWriter | None = None
        self.connected = asyncio.Event()

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
        await self._send({"t": "hello", "fw": "sim-1.0", "caps": ["imu", "mic", "haptic", "display"]})
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
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = msg.get("t")
            if t == "ping":
                await self._send({"t": "pong", "seq": msg.get("seq", 0)})
            elif t == "haptic":
                d, i, p, ms = msg.get("dir"), msg.get("intensity"), msg.get("pattern"), msg.get("duration_ms")
                print(f"[fake-m5] HAPTIC {d} intensity={i} pattern={p} {ms}ms")
            elif t == "display":
                print(f"[fake-m5] DISPLAY {msg.get('line1', '')!r} / {msg.get('line2', '')!r}")

    async def _sensor_loop(self) -> None:
        # Constant background telemetry: IMU at 50Hz, distance at 50Hz.
        tick = 0
        while self.writer is not None:
            await asyncio.sleep(0.02)
            tick += 1
            await self._send({
                "t": "imu",
                "ax": 0.05 * math.sin(tick / 30),
                "ay": 9.79,
                "az": 0.05 * math.cos(tick / 30),
                "gx": 0.0, "gy": 0.0, "gz": 0.0,
                "ts": _now_ms(),
            })
            if tick % 10 == 0:
                # Distance varies slowly; occasionally drops below threshold.
                base = 1500 + 800 * math.sin(tick / 200)
                noise = random.randint(-50, 50)
                await self._send({"t": "distance", "mm": int(base + noise)})

    async def _send(self, msg: dict) -> None:
        if self.writer is None:
            return
        self.writer.write(_line(msg))
        try:
            await self.writer.drain()
        except ConnectionResetError:
            pass

    async def trigger_ptt(self, phrase: str = "find a bottle") -> None:
        """Send a PTT cycle. Uses real Piper-synthesized speech when available
        (so Whisper actually transcribes something); falls back to a 0.8s sine.
        Synthesis is offloaded to a thread so the event loop stays responsive."""
        if self.writer is None:
            print("[fake-m5] no client connected")
            return
        loop = asyncio.get_running_loop()
        if _PIPER_AVAILABLE:
            pcm = await loop.run_in_executor(None, _piper_pcm16, phrase)
        else:
            pcm = _sine_pcm16(0.8)
        print(f"[fake-m5] PTT down -> {len(pcm) / 32000:.2f}s audio ({phrase!r}) -> PTT up")
        await self._send({"t": "ptt", "state": "down"})
        chunk = 640  # 20ms at 16kHz
        for off in range(0, len(pcm), chunk):
            self.writer.write(pcm[off:off + chunk])
            await self.writer.drain()
            await asyncio.sleep(0.02)
        self.writer.write(b"\x00" * 16)
        self.writer.write(_line({"t": "ptt", "state": "up"}))
        await self.writer.drain()

    async def trigger_fall(self) -> None:
        await self._send({"t": "fall", "severity": "hard", "az_peak": -28.4})

    async def trigger_distance(self, mm: int) -> None:
        await self._send({"t": "distance", "mm": mm})


async def _handle_command(m5: FakeM5, c: str) -> str:
    c = c.strip().lower()
    if c == "p":
        await m5.trigger_ptt()
        return "OK ptt\n"
    if c == "f":
        await m5.trigger_fall()
        return "OK fall\n"
    if c == "d":
        mm = random.randint(200, 600)
        await m5.trigger_distance(mm)
        return f"OK distance {mm}\n"
    if c == "q":
        return "BYE\n"
    return f"ERR unknown command: {c!r}\n"


async def _stdin_loop(m5: FakeM5) -> None:
    print("[fake-m5] commands: p=PTT cycle, f=fall, d=close obstacle, q=quit")
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
    print(f"[fake-m5] control listening on {HOST}:{CTRL_PORT} (echo p/f/d into TCP to trigger)")
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
