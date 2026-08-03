import json
import logging
import time

from fastapi import WebSocket

from app.schemas.websocket import ErrorEvent

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self) -> None:
        self.active: dict[str, WebSocket] = {}

    @property
    def connection_count(self) -> int:
        return len(self.active)

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        self.active[client_id] = websocket
        logger.info("Client connected: %s (total: %d)", client_id, self.connection_count)

    def disconnect(self, client_id: str) -> None:
        self.active.pop(client_id, None)
        logger.info("Client disconnected: %s (total: %d)", client_id, self.connection_count)

    async def send_json(self, websocket: WebSocket, payload: dict[str, object]) -> None:
        await websocket.send_text(json.dumps(payload))

    async def send_start(self, websocket: WebSocket, request_id: str) -> None:
        await self.send_json(
            websocket,
            {"event": "start", "request_id": request_id, "ts": time.time()},
        )

    async def send_chunk(self, websocket: WebSocket, request_id: str, text: str) -> None:
        await self.send_json(
            websocket,
            {"event": "chunk", "request_id": request_id, "text": text},
        )

    async def send_done(self, websocket: WebSocket, request_id: str) -> None:
        await self.send_json(
            websocket,
            {"event": "done", "request_id": request_id, "ts": time.time()},
        )

    async def send_error(self, websocket: WebSocket, payload: ErrorEvent) -> None:
        await self.send_json(
            websocket,
            {
                "event": "error",
                "request_id": payload.request_id,
                "message": payload.message,
            },
        )


manager = ConnectionManager()
