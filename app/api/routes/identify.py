import logging
from typing import Any

from fastapi import APIRouter, File, Form, UploadFile
from pydantic import BaseModel, Field

from app.agents.vision_agent import run_vision_agent
from app.schemas.agent import AgentResult
from app.schemas.websocket import ImageRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Vision Analysis"])


class IdentifyResponse(BaseModel):
    request_id: str = Field(description="The unique identifier for this request")
    task_type: str = Field(
        description=(
            "The inferred analysis mode. One of: "
            "describe_scene, identify_component, read_text, analyze_circuit"
        ),
    )
    analysis: str = Field(description="The model's detailed analysis of the image")
    structured_data: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Structured component data returned when task_type is 'identify_component'. "
            "Contains a 'components' list and a 'summary'. Null for other task types."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "req-001",
                    "task_type": "analyze_circuit",
                    "analysis": "I can see an LED connected to a 220Ω resistor on a breadboard...",
                    "structured_data": None,
                }
            ]
        }
    }


@router.post(
    "/identify",
    summary="Analyze an image of electrical components or circuits",
    description=(
        "Upload a photo of a circuit, breadboard, or electrical component along with a question. "
        "The system automatically classifies your intent (identify components, analyze wiring, "
        "read labels, or describe the scene) and returns a detailed analysis.\n\n"
        "**Supported image formats:** JPEG, PNG (max 10 MB)\n\n"
        "**Example prompts:**\n"
        "- *\"What components are on this breadboard?\"* → identifies each component\n"
        "- *\"Is this LED circuit wired correctly?\"* → traces connections and flags issues\n"
        "- *\"Read the markings on this IC chip\"* → transcribes and explains visible text\n"
        "- *\"What am I looking at?\"* → general scene description"
    ),
    response_description="The analysis result including inferred task type and detailed response",
)
async def identify(
    image: UploadFile = File(
        ...,
        description="Photo to analyze (JPEG or PNG, max 10 MB)",
    ),
    prompt: str = Form(
        "What is in this image?",
        description="Your question about the image",
        examples=["Is this circuit wired correctly?", "What component is this?"],
    ),
    client_id: str = Form(
        "rest-client",
        description="Session identifier — use the same value across requests to maintain conversation memory",
    ),
    request_id: str = Form(
        "req-rest",
        description=(
            "Client-generated identifier to correlate responses with requests. "
            "Primarily useful over WebSocket where multiple requests share one connection. "
            "For REST, this is optional — the HTTP response is already tied to the request."
        ),
    ),
) -> IdentifyResponse:
    image_bytes = await image.read()

    request = ImageRequest(
        client_id=client_id,
        request_id=request_id,
        prompt=prompt,
        image_bytes=image_bytes,
    )

    result: AgentResult = await run_vision_agent(request)

    return IdentifyResponse(
        request_id=result.request_id,
        task_type=result.task_type,
        analysis=result.response_text,
        structured_data=result.structured_data,
    )
