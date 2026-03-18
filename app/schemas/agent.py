from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentResult:
    request_id: str
    task_type: str
    response_text: str
    structured_data: dict[str, Any] | None = field(default=None)
