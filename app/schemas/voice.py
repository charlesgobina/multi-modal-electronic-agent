from pydantic import BaseModel, Field


class VoiceIdentifyResponse(BaseModel):
    request_id: str = Field(description="The unique identifier for this request")
    transcript: str = Field(description="The transcribed text from the audio input")
    task_type: str = Field(description="The inferred analysis mode")
    analysis: str = Field(description="The model's detailed analysis of the image")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "req-voice-001",
                    "transcript": "Is this LED wired correctly?",
                    "task_type": "analyze_circuit",
                    "analysis": "The LED is wired with correct polarity — anode to 3.3 V through a 220 Ω resistor...",
                }
            ]
        }
    }
