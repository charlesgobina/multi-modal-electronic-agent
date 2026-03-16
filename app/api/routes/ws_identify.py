import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
import openai

from app.agents.vision_agent import run_vision_agent
from app.schemas.agent import AgentResult
from app.schemas.websocket import ErrorEvent
from app.websocket.manager import manager
from app.websocket.protocol import receive_client_message

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/identify/{client_id}")
async def ws_identify(websocket: WebSocket, client_id: str) -> None:
    await manager.connect(websocket, client_id)

    try:
        while True:
            message = await receive_client_message(websocket, client_id)
            if message is None:
                continue

            await manager.send_start(
                websocket=websocket,
                request_id=message.request_id,
            )

            try:
                result: AgentResult = await run_vision_agent(message)
                for text_chunk in _chunk_text(result.response_text):
                    await manager.send_chunk(
                        websocket=websocket,
                        request_id=result.request_id,
                        text=text_chunk,
                    )

                await manager.send_done(websocket=websocket, request_id=result.request_id)
                logger.info(
                    "Completed identification for %s / %s in mode %s",
                    client_id,
                    result.request_id,
                    result.task_type,
                )
            except ValueError as exc:
                await manager.send_error(
                    websocket=websocket,
                    payload=ErrorEvent(request_id=message.request_id, message=str(exc)),
                )
            except openai.APIError as exc:
                logger.error("OpenAI error: %s", exc)
                await manager.send_error(
                    websocket=websocket,
                    payload=ErrorEvent(
                        request_id=message.request_id,
                        message=f"Model error: {exc.message}",
                    ),
                )

    except (WebSocketDisconnect, RuntimeError):
        manager.disconnect(client_id)
    except Exception:
        logger.exception("Unexpected error for client %s", client_id)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        finally:
            manager.disconnect(client_id)


def _chunk_text(text: str, chunk_size: int = 180) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [stripped[index : index + chunk_size] for index in range(0, len(stripped), chunk_size)]
