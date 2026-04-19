"""Automated smoke test for SerialBridge <-> FakeM5 (text wire protocol).

Verifies:
  1. Ready banner ``R`` -> internal ``hello`` event.
  2. IMU ``I,...`` and distance ``D,...`` reach the consumer queue.
  3. STATUS heartbeat -> ``OK ready`` keeps the bridge healthy.
  4. PTT ``B,1`` / ``B,0`` -> Pi mic path delivers ``_audio_done`` (here: injected synth PCM).

Run:
    python3 -m tools.test_serial_bridge
"""

from __future__ import annotations

import asyncio
import os
import sys

os.environ.setdefault("FAKE_M5_PORT", "15678")
os.environ.setdefault("FAKE_M5_CTRL_PORT", "15679")
os.environ.setdefault(
    "BELT_SERIAL_URL",
    f"socket://127.0.0.1:{os.environ['FAKE_M5_PORT']}",
)

from pi.services.serial_bridge import SerialBridge  # noqa: E402
from tools.fake_m5 import FakeM5, HOST, PORT, _sine_pcm16  # noqa: E402


class _SynthMic:
    """Avoid ``arecord`` in CI: return deterministic PCM on PTT release."""

    def __init__(self) -> None:
        self._on = False

    def start(self) -> bool:
        self._on = True
        return True

    def stop(self) -> bytes:
        self._on = False
        return _sine_pcm16(0.5)

    @property
    def is_recording(self) -> bool:
        return self._on

    @staticmethod
    def available() -> bool:
        return True


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

    bridge = SerialBridge(ptt_recorder=_SynthMic())
    await bridge.start()

    failures: list[str] = []

    async with server:
        try:
            await asyncio.wait_for(m5.connected.wait(), timeout=3.0)
            print("[test] bridge connected to fake_m5", flush=True)

            hello = await _wait_for_event(
                bridge, lambda m: m.get("t") == "hello", 2.0, "hello"
            )
            assert hello.get("fw") == "m5-stickc+", f"bad hello: {hello}"
            print(f"[test] PASS hello: {hello}", flush=True)

            await _wait_for_event(
                bridge, lambda m: m.get("t") == "imu", 1.0, "imu sample"
            )
            print("[test] PASS imu stream", flush=True)

            await _wait_for_event(
                bridge, lambda m: m.get("t") == "distance", 1.0, "distance reading"
            )
            print("[test] PASS distance stream", flush=True)

            await asyncio.sleep(1.5)
            assert bridge.healthy, f"STATUS liveness unhealthy ({bridge.last_pong_age_ms}ms)"
            print(f"[test] PASS STATUS heartbeat (age={bridge.last_pong_age_ms}ms)", flush=True)

            bridge.send_line(b"M,0,25,150\n")
            await asyncio.sleep(0.2)
            print("[test] PASS haptic send (see fake_m5 console line)", flush=True)

            await m5.trigger_ptt()
            audio_evt = await _wait_for_event(
                bridge,
                lambda m: m.get("t") in ("_audio_done", "_audio_timeout"),
                3.0,
                "audio cycle complete",
            )
            assert audio_evt["t"] == "_audio_done", f"audio path failed: {audio_evt}"
            audio_bytes = audio_evt["audio"]
            expected_min = int(0.5 * 16000 * 2 * 0.95)
            assert len(audio_bytes) >= expected_min, (
                f"audio too short: got {len(audio_bytes)} bytes, expected >= {expected_min}"
            )
            print(
                f"[test] PASS Pi-mic PTT path ({len(audio_bytes)} bytes synth)",
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
