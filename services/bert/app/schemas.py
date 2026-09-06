from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to classify.")


class LabelScore(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class ClassifyResponse(BaseModel):
    labels: list[LabelScore] = Field(
        ..., description="Labels above threshold, sorted by score descending."
    )
    threshold: float = Field(..., description="Threshold used for this prediction.")
    model: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    ready: bool
