from functools import lru_cache
from typing import Any

from app.core.config import get_settings


def get_llm(provider: str | None = None) -> Any:
    settings = get_settings()
    provider_norm = (provider or settings.model_provider).lower().strip()

    if provider_norm == "groq":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured")
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            max_retries=0,
        )

    if provider_norm == "groq-intent":
        from langchain_groq import ChatGroq

        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured")
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_intent_model,
            temperature=0,
            max_retries=0,
        )

    if provider_norm == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.gpt_model,
        )

    raise ValueError(f"Unknown LLM provider: {provider_norm}")


@lru_cache(maxsize=1)
def get_model_client():
    """Backward-compatible helper — returns the default vision LLM."""
    return get_llm()
