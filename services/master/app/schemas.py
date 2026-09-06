from pydantic import BaseModel, Field


class LabelScore(BaseModel):
    label: str
    score: float


class PipelineResponse(BaseModel):
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
    status: str = "ok"
    downstream: dict[str, str]
