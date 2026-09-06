from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Fully-built prompt.")
    max_new_tokens: int | None = Field(
        default=None, ge=1, le=2048, description="Override for generation length."
    )


class GenerateResponse(BaseModel):
    text: str = Field(..., description="Generated text.")
    model: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    backend: str
    ready: bool
