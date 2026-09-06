from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    text: str = Field(..., description="Recognized text from the audio.")
    model: str = Field(..., description="Model that produced the transcription.")


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    ready: bool
