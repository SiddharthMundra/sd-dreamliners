"""Standalone preview server — no Pi hardware required.

Serves the webapp at http://localhost:8000 and pumps fake sensor data over
WebSocket so the UI can be fully exercised on any machine.

Usage:
    python tools/preview_server.py
"""

from __future__ import annotations

import asyncio
import math
import random
import struct
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

WEBAPP_DIR = Path(__file__).resolve().parents[1] / "webapp"

app = FastAPI(title="Belt preview")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR)), name="static")

# ── connected WebSocket clients ───────────────────────────────────────────────
_clients: set[WebSocket] = set()


async def _broadcast(msg: dict) -> None:
    dead = []
    for ws in list(_clients):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _clients.discard(ws)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(WEBAPP_DIR / "index.html"))


@app.get("/mjpeg")
async def mjpeg() -> StreamingResponse:
    return StreamingResponse(
        _fake_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/mjpeg-overlay")
async def mjpeg_overlay() -> StreamingResponse:
    # Preview already bakes a fake YOLO box into _fake_mjpeg, so we alias.
    return StreamingResponse(
        _fake_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(websocket)


# ── fake MJPEG stream (animated gradient JPEG) ────────────────────────────────

def _make_jpeg(tick: int) -> bytes:
    """Return a tiny synthetic JPEG — dark gradient with a moving highlight."""
    import io
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Fallback: return a minimal valid JPEG (1-pixel black)
        return (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
            b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
            b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e\xc0"
            b"\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00"
            b"\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00"
            b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10"
            b"\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}"
            b"\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91"
            b"\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a"
            b"%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87"
            b"\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5"
            b"\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3"
            b"\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda"
            b"\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6"
            b"\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xf5\x0f\xff\xd9"
        )

    size = 320
    img = Image.new("RGB", (size, size), (8, 8, 18))
    draw = ImageDraw.Draw(img)

    # Animated scanning beam
    beam_y = int((math.sin(tick / 20) * 0.5 + 0.5) * size)
    draw.rectangle([0, beam_y - 2, size, beam_y + 2], fill=(20, 60, 120))

    # Random noise blobs to simulate scene
    rng = random.Random(tick // 10)
    for _ in range(4):
        x, y = rng.randint(30, 290), rng.randint(30, 290)
        r = rng.randint(12, 40)
        gray = rng.randint(20, 50)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(gray, gray, gray + 8))

    # Fake bounding box
    bx1, by1 = 80, 90
    bx2, by2 = 200, 230
    draw.rectangle([bx1, by1, bx2, by2], outline=(59, 130, 246), width=2)
    draw.rectangle([bx1, by1 - 16, bx1 + 70, by1], fill=(59, 130, 246))
    draw.text((bx1 + 4, by1 - 14), "person 91%", fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return buf.getvalue()


async def _fake_mjpeg():
    tick = 0
    while True:
        jpeg = await asyncio.get_running_loop().run_in_executor(None, _make_jpeg, tick)
        yield b"--frame\r\nContent-Type: image/jpeg\r\n"
        yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
        yield jpeg + b"\r\n"
        tick += 1
        await asyncio.sleep(0.12)  # ~8 fps


# ── fake sensor data pumped over WebSocket ────────────────────────────────────

async def _fake_data_loop() -> None:
    tick = 0
    last_haptic = 0.0
    last_voice  = 0.0
    last_fall   = 0.0

    voice_script = [
        ("user",      "What do you see?"),
        ("assistant", "There's a person directly ahead, about 1.2 meters away."),
        ("user",      "Find the exit."),
        ("assistant", "Scanning for exit signs. Guiding you left."),
        ("user",      "How far is the obstacle?"),
        ("assistant", "Obstacle detected at 420 mm — haptic warning active."),
    ]
    voice_idx = 0

    while True:
        now = time.time()
        tick += 1

        # ── IMU (50 Hz, but we broadcast at 10 Hz to keep WS lightweight) ──
        ax = 0.08 * math.sin(tick / 25)
        ay = 9.79 + 0.04 * math.cos(tick / 18)
        az = 0.06 * math.sin(tick / 40 + 1.2)
        await _broadcast({"t": "imu", "source": "m5",
                          "ax": round(ax, 3), "ay": round(ay, 3), "az": round(az, 3)})

        # ── Distance (oscillates, occasionally close) ─────────────────────
        dist_mm = int(1200 + 900 * math.sin(tick / 80) + random.randint(-30, 30))
        dist_mm = max(100, min(2000, dist_mm))
        await _broadcast({"t": "distance", "mm": dist_mm})

        # ── Haptic — fires every ~3 s on a random direction ───────────────
        if now - last_haptic > 3.0 + random.uniform(-0.5, 0.5):
            last_haptic = now
            direction   = random.choice(["F", "L", "R", "B"])
            pattern     = random.choice(["pulse", "solid", "ramp"])
            intensity   = random.randint(120, 220)
            await _broadcast({
                "t": "haptic", "dir": direction, "intensity": intensity,
                "pattern": pattern, "duration_ms": 500,
            })

        # ── YOLO detections synced with fake MJPEG boxes ──────────────────
        if tick % 5 == 0:
            await _broadcast({
                "t": "detections",
                "boxes": [
                    {"cls": "person", "conf": 0.91, "x": 0.44, "y": 0.5, "w": 0.38, "h": 0.44},
                ],
            })

        # ── Voice transcript — cycles through script every ~12 s ──────────
        if now - last_voice > 5.5:
            last_voice = now
            role, text = voice_script[voice_idx % len(voice_script)]
            await _broadcast({"t": "voice", "role": role, "text": text})
            voice_idx += 1

        # ── Health ────────────────────────────────────────────────────────
        if tick % 10 == 0:
            await _broadcast({
                "t": "health",
                "serial_ok":    True,
                "yolo_fps":     round(4.8 + random.uniform(-0.4, 0.4), 1),
                "warmup_ok":    True,
                "ollama_alive": True,
                "last_pong_age_ms": random.randint(10, 60),
            })

        # ── Simulated fall every ~30 s ─────────────────────────────────────
        if now - last_fall > 30.0:
            last_fall = now
            await _broadcast({"t": "fall", "severity": "hard", "az_peak": -27.8})

        await asyncio.sleep(0.1)  # 10 Hz main loop


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_fake_data_loop())


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tools.preview_server:app", host="0.0.0.0", port=8000, reload=False)
