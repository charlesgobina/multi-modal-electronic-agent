import io

from app.core.config import get_settings


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    from openai import AsyncOpenAI

    settings = get_settings()

    if settings.model_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured")
        client = AsyncOpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = "whisper-large-v3"
    else:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured")
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        model = "whisper-1"

    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename

    transcript = await client.audio.transcriptions.create(
        model=model,
        file=audio_file,
    )
    return transcript.text
