from fastapi import APIRouter

from app.websocket.manager import manager

router = APIRouter()


@router.get("/health")
async def healthcheck() -> dict[str, int | str]:
    return {"status": "ok", "active_connections": manager.connection_count}
