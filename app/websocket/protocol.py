import base64
import json

from fastapi import WebSocket

from app.schemas.websocket import ErrorEvent, ImageRequest
from app.services.rate_limit import rate_limiter
from app.websocket.manager import manager


def _decode_data_uri(data_uri: str) -> bytes:
    """Extract raw bytes from a data URI like 'data:image/jpeg;base64,...'."""
    if data_uri.startswith("data:"):
        _, encoded = data_uri.split(",", 1)
    else:
        encoded = data_uri
    return base64.b64decode(encoded)


async def receive_client_message(
    websocket: WebSocket,
    client_id: str,
) -> ImageRequest | None:
    message = await websocket.receive()

    if not message.get("text"):
        return None

    try:
        payload = json.loads(message["text"])
    except json.JSONDecodeError:
        await manager.send_error(
            websocket,
            ErrorEvent(message="Invalid JSON in text frame"),
        )
        return None

    image_data = payload.get("image")
    if not image_data:
        await manager.send_error(
            websocket,
            ErrorEvent(message="Missing 'image' field (base64 data URI expected)"),
        )
        return None

    try:
        image_bytes = _decode_data_uri(image_data)
    except Exception:
        await manager.send_error(
            websocket,
            ErrorEvent(message="Invalid base64 image data"),
        )
        return None

    prompt = payload.get("prompt", "What is in this image?")
    request_id = payload.get("request_id", f"req-{id(message)}")

    if not rate_limiter.allow(client_id):
        await manager.send_error(
            websocket,
            ErrorEvent(
                request_id=request_id,
                message="Rate limit exceeded; slow down requests",
            ),
        )
        return None

    return ImageRequest(
        client_id=client_id,
        request_id=request_id,
        prompt=prompt,
        image_bytes=image_bytes,
    )
