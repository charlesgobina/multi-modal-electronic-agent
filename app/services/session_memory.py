from collections import defaultdict
from collections import deque
from dataclasses import dataclass

from app.core.config import get_settings


@dataclass(frozen=True)
class SessionTurn:
    prompt: str
    task_type: str
    response_text: str


class SessionMemoryStore:
    def __init__(self) -> None:
        self._sessions: dict[str, deque[SessionTurn]] = defaultdict(
            lambda: deque(maxlen=get_settings().session_memory_turns)
        )

    def recent_summary(self, client_id: str) -> str:
        turns = list(self._sessions[client_id])
        if not turns:
            return "No prior session context."
        return "\n".join(
            f"- Prompt: {turn.prompt}\n  Task: {turn.task_type}\n  Response: {turn.response_text}"
            for turn in turns
        )

    def append(self, client_id: str, prompt: str, task_type: str, response_text: str) -> None:
        self._sessions[client_id].append(
            SessionTurn(prompt=prompt, task_type=task_type, response_text=response_text)
        )


session_memory = SessionMemoryStore()
