from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    client_id: str
    request_id: str
    prompt: str
    image_bytes: bytes
    image_b64: str
    media_type: str
    task_type: str
    memory_context: str
    response_text: str
    structured_data: dict[str, Any] | None
