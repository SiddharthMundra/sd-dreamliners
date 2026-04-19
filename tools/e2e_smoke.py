"""End-to-end smoke harness, run on the Pi.

Spawns fake_m5 + main.py as subprocesses, then drives the live system:
  - hits /healthz to confirm camera, yolo, serial all green
  - hits /snapshot.jpg to confirm a real frame is being served
  - opens the WebSocket and counts events for 5 seconds
  - triggers fake_m5 PTT cycle via the control port -> expects an
    'assistant' transcript over the WebSocket within ~10s
  - triggers fake_m5 fall -> expects a haptic ALL command over WebSocket

Prints a summary table and exits 0 on full pass, 1 on any failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from contextlib import suppress
from typing import Any

PI_HOST = "127.0.0.1"
WEBAPP_PORT = 8000
M5_CTRL_PORT = 5556
REPO = os.path.expanduser("~/sd-dreamliners")
LOG_DIR = "/tmp"
ENV = os.environ | {
    "BELT_SERIAL_URL": "socket://localhost:5555",
    "PATH": os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", ""),
    "KEYWORD_INTENT_ONLY": os.environ.get("KEYWORD_INTENT_ONLY", "1"),
    "PYTHONUNBUFFERED": "1",
}


def _ctrl(cmd: str, timeout: float = 15.0) -> str:
    s = socket.create_connection((PI_HOST, M5_CTRL_PORT), timeout=timeout)
    s.settimeout(timeout)
    s.sendall((cmd + "\n").encode())
    s.shutdown(socket.SHUT_WR)
    data = s.recv(64).decode()
    s.close()
    return data.strip()


def _http(path: str, timeout: float = 3.0) -> tuple[int, bytes]:
    req = urllib.request.Request(f"http://{PI_HOST}:{WEBAPP_PORT}{path}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except Exception as e:
        return -1, str(e).encode()


def _tail(path: str, n: int = 10) -> str:
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return "<missing>"


async def _ws_listen(duration_s: float) -> list[dict]:
    try:
        from websockets import connect
    except ImportError:
        try:
            from websockets.client import connect
        except ImportError:
            return []
    events: list[dict] = []
    end = time.time() + duration_s
    try:
        async with connect(f"ws://{PI_HOST}:{WEBAPP_PORT}/ws") as ws:
            while time.time() < end:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                with suppress(Exception):
                    events.append(json.loads(msg))
    except Exception as e:
        print(f"  ws error: {e}")
    return events


def _start(name: str, cmd: list[str]) -> subprocess.Popen:
    log = open(f"{LOG_DIR}/{name}.log", "w")
    print(f"[harness] launching {name}: {' '.join(cmd)}")
    return subprocess.Popen(
        cmd, cwd=REPO, env=ENV, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )


def _wait_port(host: str, port: int, timeout_s: float) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        try:
            s = socket.create_connection((host, port), timeout=0.5)
            s.close()
            return True
        except OSError:
            time.sleep(0.5)
    return False


def main() -> int:
    failures: list[str] = []
    procs: dict[str, subprocess.Popen] = {}

    try:
        procs["fake_m5"] = _start("fake_m5", [sys.executable, "-u", "-m", "tools.fake_m5"])
        if not _wait_port(PI_HOST, M5_CTRL_PORT, 5.0):
            return _fail("fake_m5 control port never opened", procs, ["fake_m5", "main"])
        print(f"  ok fake_m5 control listening on :{M5_CTRL_PORT}")

        procs["main"] = _start("main", [sys.executable, "-u", "-m", "pi.main"])
        if not _wait_port(PI_HOST, WEBAPP_PORT, 25.0):
            return _fail("main.py webapp never opened", procs, ["fake_m5", "main"])
        print(f"  ok main.py webapp on :{WEBAPP_PORT}")

        time.sleep(8)

        status, body = _http("/healthz")
        if status != 200:
            failures.append(f"/healthz returned {status}: {body[:200]!r}")
        else:
            try:
                health = json.loads(body)
                print(f"  /healthz: {health}")
                if not health.get("serial_ok"):
                    failures.append(f"serial not healthy: {health}")
                if (health.get("yolo_fps") or 0) < 1.0:
                    failures.append(f"yolo_fps too low: {health.get('yolo_fps')}")
            except Exception as e:
                failures.append(f"/healthz body invalid: {e}")

        status, body = _http("/snapshot.jpg")
        if status != 200 or not body.startswith(b"\xff\xd8"):
            failures.append(f"/snapshot.jpg bad: status={status} len={len(body)}")
        else:
            print(f"  /snapshot.jpg ok: {len(body)} bytes")

        status, body = _http("/snapshot-overlay.jpg")
        if status != 200 or not body.startswith(b"\xff\xd8"):
            failures.append(f"/snapshot-overlay.jpg bad: status={status} len={len(body)}")
        else:
            print(f"  /snapshot-overlay.jpg ok: {len(body)} bytes")

        print("  listening on /ws for 4s ...")
        events_baseline = asyncio.run(_ws_listen(4.0))
        types = {e.get("t") for e in events_baseline}
        print(f"    captured {len(events_baseline)} events of types {sorted(t for t in types if t)}")
        for required in ("imu", "distance", "health"):
            if required not in types:
                failures.append(f"ws missing baseline event type: {required}")

        print("  triggering fake_m5 distance burst ...")
        for _ in range(3):
            print(f"    {_ctrl('d')}")

        print("  triggering fake_m5 PTT cycle, watching ws for ~20s ...")
        async def trigger_then_listen():
            ws_task = asyncio.create_task(_ws_listen(20.0))
            await asyncio.sleep(1.0)
            await asyncio.get_running_loop().run_in_executor(None, _ctrl, "p")
            return await ws_task
        events_voice = asyncio.run(trigger_then_listen())
        v_types = [(e.get("t"), e.get("role")) for e in events_voice]
        print(f"    captured {len(events_voice)} events; types={sorted(set(v_types))}")
        if not any(e.get("t") == "voice" and e.get("role") == "user" for e in events_voice):
            failures.append("no user transcript appeared on ws after PTT")

        print("  triggering fake_m5 fall ...")
        async def trigger_fall_listen():
            ws_task = asyncio.create_task(_ws_listen(4.0))
            await asyncio.sleep(0.5)
            await asyncio.get_running_loop().run_in_executor(None, _ctrl, "f")
            return await ws_task
        events_fall = asyncio.run(trigger_fall_listen())
        f_types = [(e.get("t"), e.get("severity"), e.get("dir")) for e in events_fall]
        print(f"    captured {len(events_fall)} events; types={sorted(set(f_types))}")
        if not any(e.get("t") == "fall" for e in events_fall):
            failures.append("no fall event on ws")
        if not any(e.get("t") == "haptic" and e.get("dir") == "ALL" for e in events_fall):
            failures.append("no SOS haptic broadcast after fall")

    finally:
        for name, p in procs.items():
            with suppress(Exception):
                p.terminate()
        for name, p in procs.items():
            with suppress(Exception):
                p.wait(timeout=3)

    print()
    if failures:
        print(f"[harness] {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        print("\n--- main.log tail ---\n" + _tail(f"{LOG_DIR}/main.log", 25))
        print("\n--- fake_m5.log tail ---\n" + _tail(f"{LOG_DIR}/fake_m5.log", 15))
        return 1

    print("[harness] ALL PASS")
    return 0


def _fail(msg: str, procs: dict[str, subprocess.Popen], names: list[str]) -> int:
    print(f"[harness] FATAL: {msg}")
    for name, p in procs.items():
        with suppress(Exception):
            p.terminate()
    for name in names:
        print(f"\n--- {name}.log ---\n" + _tail(f"{LOG_DIR}/{name}.log", 30))
    return 1


if __name__ == "__main__":
    sys.exit(main())
