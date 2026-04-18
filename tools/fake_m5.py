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
import random
import sys
from time import time

HOST = "localhost"
PORT = 5555
PROTOCOL_VERSION = 1
SAMPLE_RATE = 16000


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

    async def trigger_ptt(self, duration_s: float = 0.8) -> None:
        if self.writer is None:
            print("[fake-m5] no client connected")
            return
        print(f"[fake-m5] PTT down -> {duration_s}s audio -> PTT up")
        await self._send({"t": "ptt", "state": "down"})
        # Stream raw PCM16 silence with a small sine for non-zero content.
        n_samples = int(SAMPLE_RATE * duration_s)
        chunk_n = 320  # 20ms chunks
        for start in range(0, n_samples, chunk_n):
            end = min(start + chunk_n, n_samples)
            samples = bytearray()
            for i in range(start, end):
                # 200 Hz tone, low amplitude (~10% full scale).
                v = int(3000 * math.sin(2 * math.pi * 200 * i / SAMPLE_RATE))
                samples += int.to_bytes(v & 0xFFFF, 2, "little", signed=False)
            self.writer.write(bytes(samples))
            await self.writer.drain()
            await asyncio.sleep(0.02)
        # Sync preamble: 16 zero bytes, then the JSON line.
        self.writer.write(b"\x00" * 16)
        self.writer.write(_line({"t": "ptt", "state": "up"}))
        await self.writer.drain()

    async def trigger_fall(self) -> None:
        await self._send({"t": "fall", "severity": "hard", "az_peak": -28.4})

    async def trigger_distance(self, mm: int) -> None:
        await self._send({"t": "distance", "mm": mm})


async def _stdin_loop(m5: FakeM5) -> None:
    print("[fake-m5] commands: p=PTT cycle, f=fall, d=close obstacle, q=quit")
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            await asyncio.sleep(0.1)
            continue
        c = line.strip().lower()
        if c == "p":
            await m5.trigger_ptt()
        elif c == "f":
            await m5.trigger_fall()
        elif c == "d":
            await m5.trigger_distance(random.randint(200, 600))
        elif c == "q":
            print("[fake-m5] bye")
            return


async def main() -> None:
    m5 = FakeM5()
    await asyncio.gather(m5.serve(), _stdin_loop(m5))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
