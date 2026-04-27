import logging

from fastapi import APIRouter, File, Form, UploadFile

from app.agents.vision_agent import run_vision_agent
from app.schemas.agent import AgentResult
from app.schemas.voice import VoiceIdentifyResponse
from app.schemas.websocket import ImageRequest
from app.services.transcription import transcribe_audio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Voice Analysis"])


@router.post(
    "/voice-identify",
    summary="Analyze an image using a spoken question",
    description=(
        "Upload an audio recording of your question alongside a circuit or component photo. "
        "The endpoint transcribes your speech with Whisper, then passes the transcript as the "
        "prompt to the vision model — same analysis pipeline as `/api/identify`, just hands-free.\n\n"
        "**Supported audio formats:** WebM, MP3, WAV, M4A (max ~25 MB)\n\n"
        "**Supported image formats:** JPEG, PNG (max 10 MB)"
    ),
    response_description="Transcript of the spoken question plus the model's image analysis",
)
async def voice_identify(
    audio: UploadFile = File(..., description="Audio recording of your question"),
    image: UploadFile = File(..., description="Photo to analyze (JPEG or PNG, max 10 MB)"),
    client_id: str = Form(
        "rest-client",
        description="Session identifier — use the same value across requests to maintain conversation memory",
    ),
    request_id: str = Form(
        "req-voice",
        description="Client-generated identifier to correlate responses with requests",
    ),
) -> VoiceIdentifyResponse:
    audio_bytes, image_bytes = await audio.read(), await image.read()

    transcript = await transcribe_audio(audio_bytes, audio.filename or "audio.webm")
    logger.info("Transcribed audio for request_id=%s: %r", request_id, transcript)

    request = ImageRequest(
        client_id=client_id,
        request_id=request_id,
        prompt=transcript,
        image_bytes=image_bytes,
    )
    result: AgentResult = await run_vision_agent(request)

    return VoiceIdentifyResponse(
        request_id=result.request_id,
        transcript=transcript,
        task_type=result.task_type,
        analysis=result.response_text,
    )
