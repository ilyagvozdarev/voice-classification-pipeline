from pydantic import BaseModel, Field


class LabelScore(BaseModel):
    """A single predicted label with its probability (from BERT)."""

    label: str
    score: float


class PipelineResponse(BaseModel):
    """Aggregated result the master returns to the gradio front-end.

    The pipeline is two-stage: the raw transcript is first *restored* by the
    LLM, then the restored text is classified two ways — by BERT (multi-label)
    and by the LLM (its own classification) — and both are returned.
    """

    transcript: str = Field(..., description="Raw text recognized from audio (ASR).")
    restored_text: str = Field(
        ..., description="LLM-restored transcript (diarized, punctuated, typed)."
    )
    bert_labels: list[LabelScore] = Field(
        ..., description="Multi-label classification of the restored text (BERT)."
    )
    llm_classification: str = Field(
        ..., description="LLM's own classification of the restored text."
    )


class HealthResponse(BaseModel):
    """Liveness payload plus the configured downstream URLs (for debugging)."""

    status: str = "ok"
    downstream: dict[str, str]
