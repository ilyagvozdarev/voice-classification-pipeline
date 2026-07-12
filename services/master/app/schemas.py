from pydantic import BaseModel, Field


class Sentiment(BaseModel):
    label: str
    score: float


class PipelineResponse(BaseModel):
    """Aggregated result the master returns to the gradio front-end."""

    transcript: str = Field(..., description="Text recognized from the audio (ASR).")
    sentiment: Sentiment = Field(..., description="BERT classification of the transcript.")
    answer: str = Field(..., description="LLM answer generated from the transcript.")


class HealthResponse(BaseModel):
    status: str = "ok"
    downstream: dict[str, str]
