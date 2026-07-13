from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for text generation.

    The caller (master) sends a fully-built prompt — the LLM service itself is
    prompt-agnostic, it just completes whatever it receives. This keeps the
    service reusable for both the "restore" and "classify" steps.
    """

    prompt: str = Field(..., min_length=1, description="Fully-built prompt.")
    max_new_tokens: int | None = Field(
        default=None, ge=1, le=2048, description="Override for generation length."
    )


class GenerateResponse(BaseModel):
    """Generated completion for a single prompt."""

    text: str = Field(..., description="Generated text.")
    model: str


class HealthResponse(BaseModel):
    """Liveness/readiness payload for the LLM service."""

    status: str = "ok"
    model: str
    backend: str
    ready: bool
