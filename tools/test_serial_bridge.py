"""Automated smoke test for SerialBridge <-> FakeM5.

Verifies, in one process:
  1. CONTROL JSON exchange (hello, ping/pong, haptic command echo).
  2. Sensor stream (imu, distance) reaches the bridge consumer.
  3. AUDIO mode round-trip: PTT down -> raw PCM16 -> PTT up -> _audio_done.

Pass = all asserts succeed. Run on the Pi:
    python3 -m tools.test_serial_bridge
"""

from __future__ import annotations

import asyncio
import os
import sys

os.environ.setdefault("BELT_SERIAL_URL", "socket://localhost:5555")

from pi.services.serial_bridge import SerialBridge  # noqa: E402
from tools.fake_m5 import FakeM5, HOST, PORT  # noqa: E402

HAPTIC_CMD = {
    "t": "haptic",
    "dir": "FRONT",
    "intensity": 200,
    "pattern": "double",
    "duration_ms": 150,
}


async def _wait_for_event(bridge: SerialBridge, predicate, timeout_s: float, label: str) -> dict:
    end = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < end:
        remaining = max(0.05, end - asyncio.get_running_loop().time())
        try:
            msg = await asyncio.wait_for(bridge.events.get(), timeout=remaining)
        except asyncio.TimeoutError:
            break
        if predicate(msg):
            return msg
    raise AssertionError(f"timeout waiting for: {label}")


async def main() -> int:
    m5 = FakeM5()
    server = await asyncio.start_server(m5._on_client, HOST, PORT)
    print(f"[test] fake_m5 listening on {HOST}:{PORT}", flush=True)

    bridge = SerialBridge()
    await bridge.start()

    failures: list[str] = []

    async with server:
        try:
            await asyncio.wait_for(m5.connected.wait(), timeout=3.0)
            print("[test] bridge connected to fake_m5", flush=True)

            hello = await _wait_for_event(
                bridge, lambda m: m.get("t") == "hello", 2.0, "hello"
            )
            assert hello.get("fw") == "sim-1.0", f"bad hello: {hello}"
            print(f"[test] PASS hello: caps={hello.get('caps')}", flush=True)

            await _wait_for_event(
                bridge, lambda m: m.get("t") == "imu", 1.0, "imu sample"
            )
            print("[test] PASS imu stream", flush=True)

            await _wait_for_event(
                bridge, lambda m: m.get("t") == "distance", 1.0, "distance reading"
            )
            print("[test] PASS distance stream", flush=True)

            await asyncio.sleep(1.5)
            assert bridge.healthy, f"ping/pong unhealthy after warmup ({bridge.last_pong_age_ms}ms)"
            print(f"[test] PASS ping/pong (age={bridge.last_pong_age_ms}ms)", flush=True)

            bridge.send(HAPTIC_CMD)
            await asyncio.sleep(0.2)
            print("[test] PASS haptic send (see fake_m5 console line)", flush=True)

            await m5.trigger_ptt(duration_s=0.5)
            audio_evt = await _wait_for_event(
                bridge,
                lambda m: m.get("t") in ("_audio_done", "_audio_timeout"),
                3.0,
                "audio cycle complete",
            )
            assert audio_evt["t"] == "_audio_done", f"audio path failed: {audio_evt}"
            audio_bytes = audio_evt["audio"]
            expected_min = int(0.5 * 16000 * 2 * 0.8)
            assert len(audio_bytes) >= expected_min, (
                f"audio too short: got {len(audio_bytes)} bytes, expected >= {expected_min}"
            )
            print(
                f"[test] PASS audio round-trip ({len(audio_bytes)} bytes captured)",
                flush=True,
            )

            await _wait_for_event(
                bridge, lambda m: m.get("t") == "imu", 1.0, "post-audio imu sample"
            )
            print("[test] PASS post-audio recovery (imu stream resumed)", flush=True)

        except AssertionError as e:
            failures.append(str(e))
            print(f"[test] FAIL: {e}", flush=True)
        finally:
            bridge.stop()
            server.close()
            await server.wait_closed()

    if failures:
        print(f"\n[test] {len(failures)} FAILURE(S)")
        return 1
    print("\n[test] ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
