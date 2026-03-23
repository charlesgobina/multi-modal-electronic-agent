from dataclasses import dataclass


@dataclass(frozen=True)
class AgentResult:
    request_id: str
    task_type: str
    response_text: str
