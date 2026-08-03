from dataclasses import dataclass, field
import time


@dataclass
class ClientMetadata:
    prompt: str = "What is in this image?"
    request_id: str = field(default_factory=lambda: f"req-{time.monotonic():.4f}")


@dataclass(frozen=True)
class ImageRequest:
    client_id: str
    request_id: str
    prompt: str
    image_bytes: bytes


@dataclass(frozen=True)
class ErrorEvent:
    request_id: str | None = None
    message: str = "Unknown error"
