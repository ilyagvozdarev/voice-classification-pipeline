from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify.")


class ClassifyResponse(BaseModel):
    label: str = Field(..., description="Predicted label, e.g. POSITIVE / NEGATIVE.")
    score: float = Field(..., ge=0.0, le=1.0, description="Confidence for the label.")
    model: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    ready: bool
