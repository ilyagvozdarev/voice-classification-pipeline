from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    """Request body for multi-label classification."""

    text: str = Field(..., min_length=1, description="Text to classify.")


class LabelScore(BaseModel):
    """A single predicted label with its probability."""

    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    """Multi-label result: every label whose score passes the threshold."""

    labels: list[LabelScore] = Field(
        ..., description="Labels above threshold, sorted by score descending."
    )
    threshold: float = Field(..., description="Threshold used for this prediction.")
    model: str


class HealthResponse(BaseModel):
    """Liveness/readiness payload for the BERT service."""

    status: str = "ok"
    model: str
    ready: bool
