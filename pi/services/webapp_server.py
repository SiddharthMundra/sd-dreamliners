"""FastAPI webapp server.

Camera streaming surface (any HTTP client / browser / VLC):
  GET /mjpeg          MJPEG stream of raw camera frames.
  GET /mjpeg-overlay  MJPEG stream with YOLO boxes baked in.
  GET /snapshot.jpg   Single-frame JPEG (raw).
  GET /snapshot-overlay.jpg  Single-frame JPEG with overlay.

Other surfaces:
  GET /              Product UI shell.
  GET /static/*      Static assets (JS/CSS).
  WS  /ws            Structured event broadcast.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pi.services.camera import CameraService, draw_overlay, encode_jpeg

log = logging.getLogger(__name__)

WEBAPP_DIR = Path(__file__).resolve().parents[2] / "webapp"


class WebappServer:
    def __init__(
        self,
        camera: CameraService,
        detections_supplier: Callable[[], list] | None = None,
    ) -> None:
        self.app = FastAPI(title="Belt")
        self._camera = camera
        self._detections_supplier = detections_supplier or (lambda: [])
        self._clients: set[WebSocket] = set()
        self._clients_lock = asyncio.Lock()
        self._register_routes()

    def _register_routes(self) -> None:
        app = self.app
        app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR)), name="static")

        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(str(WEBAPP_DIR / "index.html"))

        @app.get("/mjpeg")
        async def mjpeg() -> StreamingResponse:
            return StreamingResponse(
                self._mjpeg_stream(overlay=False),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @app.get("/mjpeg-overlay")
        async def mjpeg_overlay() -> StreamingResponse:
            return StreamingResponse(
                self._mjpeg_stream(overlay=True),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @app.get("/snapshot.jpg")
        async def snapshot() -> Response:
            return self._snapshot_response(overlay=False)

        @app.get("/snapshot-overlay.jpg")
        async def snapshot_overlay() -> Response:
            return self._snapshot_response(overlay=True)

        @app.get("/healthz")
        async def healthz() -> dict:
            frame, ts = self._camera.get_latest_bgr()
            return {
                "camera_backend": self._camera.backend,
                "camera_fps": round(self._camera.fps, 2),
                "camera_frame_age_ms": self._camera.frame_age_ms,
                "camera_has_frame": frame is not None,
            }

        @app.websocket("/ws")
        async def ws(websocket: WebSocket) -> None:
            await websocket.accept()
            async with self._clients_lock:
                self._clients.add(websocket)
            try:
                while True:
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                async with self._clients_lock:
                    self._clients.discard(websocket)

    async def broadcast(self, msg: dict | object) -> None:
        payload = _to_dict(msg)
        async with self._clients_lock:
            dead: list[WebSocket] = []
            for ws in self._clients:
                try:
                    await ws.send_json(payload)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    def _snapshot_response(self, overlay: bool) -> Response:
        frame, _ = self._camera.get_latest_bgr()
        if frame is None:
            return Response(status_code=503, content=b"camera not ready", media_type="text/plain")
        if overlay:
            frame = draw_overlay(frame, self._detections_supplier())
        jpeg = encode_jpeg(frame)
        return Response(
            content=jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store"},
        )

    async def _mjpeg_stream(self, overlay: bool):
        boundary = b"--frame\r\n"
        last_ts = 0
        loop = asyncio.get_running_loop()
        while True:
            frame, ts_ms = self._camera.get_latest_bgr()
            if frame is None or ts_ms == last_ts:
                await asyncio.sleep(0.03)
                continue
            last_ts = ts_ms
            if overlay:
                detections = self._detections_supplier()
                jpeg = await loop.run_in_executor(None, lambda: encode_jpeg(draw_overlay(frame, detections)))
            else:
                jpeg = await loop.run_in_executor(None, encode_jpeg, frame)
            if not jpeg:
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
            yield jpeg + b"\r\n"
            await asyncio.sleep(0.03)


def _to_dict(msg: dict | object) -> dict:
    if isinstance(msg, dict):
        return msg
    if is_dataclass(msg):
        return asdict(msg)
    raise TypeError(f"cannot serialize {type(msg)}")
