from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User message to respond to.")
    max_new_tokens: int | None = Field(
        default=None, ge=1, le=512, description="Override for generation length."
    )


class GenerateResponse(BaseModel):
    answer: str = Field(..., description="Generated answer.")
    model: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    ready: bool
