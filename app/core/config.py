from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


_load_env_file()


@dataclass(frozen=True)
class Settings:
    app_name: str = "multi-modal"
    model_provider: str = "openai"
    openai_api_key: str | None = None
    gpt_model: str = "gpt-4o"
    groq_api_key: str | None = None
    groq_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_intent_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com"
    max_image_bytes: int = 10 * 1024 * 1024
    max_image_dim: int = 2048
    rate_limit_rps: float = 2.0
    session_memory_turns: int = 6
    system_prompt: str = (
        "You are an expert electrical engineering teaching assistant with deep knowledge of "
        "circuit design, electronic components, and lab equipment. Students send you photos "
        "of their circuits, breadboards, and components for help.\n\n"
        "Your analysis style:\n"
        "- Answer the student's specific question first, then provide supporting detail.\n"
        "- Be precise: name exact component types, values (if readable), pin functions, and standards.\n"
        "- When analyzing connections, trace the actual signal/power path and explain WHY "
        "something is correct or incorrect, not just WHAT you see.\n"
        "- Clearly distinguish what you can confirm visually from what you are inferring. "
        "Use phrases like 'I can see that…' vs 'This appears to be…' to signal confidence level.\n"
        "- Keep language accessible to students while remaining technically accurate.\n"
        "- Use short paragraphs and bullet points for readability."
    )
    cors_origins: tuple[str, ...] = ("*",)

    @property
    def active_model(self) -> str:
        if self.model_provider == "groq":
            return self.groq_model
        return self.gpt_model

    @property
    def active_api_key(self) -> str | None:
        if self.model_provider == "groq":
            return self.groq_api_key
        return self.openai_api_key

    @property
    def active_base_url(self) -> str | None:
        if self.model_provider == "groq":
            return self.groq_base_url
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    origins = tuple(
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "*").split(",")
        if origin.strip()
    )
    return Settings(
        app_name=os.getenv("APP_NAME", "multi-modal"),
        model_provider=os.getenv("MODEL_PROVIDER", "openai").strip().lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gpt_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct"),
        groq_intent_model=os.getenv("GROQ_INTENT_MODEL", "openai/gpt-oss-20b"),
        groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com"),
        max_image_bytes=int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024))),
        max_image_dim=int(os.getenv("MAX_IMAGE_DIM", "2048")),
        rate_limit_rps=float(os.getenv("RATE_LIMIT_RPS", "2")),
        session_memory_turns=int(os.getenv("SESSION_MEMORY_TURNS", "6")),
        system_prompt=os.getenv("SYSTEM_PROMPT", Settings.system_prompt),
        cors_origins=origins or ("*",),
    )
