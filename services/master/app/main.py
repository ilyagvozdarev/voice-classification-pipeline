import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from .clients import DownstreamError, clients
from .config import settings
from .prompts import build_classification_prompt, build_restoration_prompt
from .schemas import HealthResponse, LabelScore, PipelineResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("master")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await clients.start()
    try:
        yield
    finally:
        await clients.close()


app = FastAPI(title="Master Orchestrator", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        downstream={
            "asr": settings.asr_url,
            "bert": settings.bert_url,
            "llm": settings.llm_url,
        }
    )


@app.post("/process", response_model=PipelineResponse)
async def process(audio: UploadFile = File(...)) -> PipelineResponse:
    """
    pipeline: audio -> ASR -> LLM(restore) -> {BERT(multi-label) ‖ LLM(classify)}.
    The restore step is sequential (everything below needs the restored text);
    the two classifications then run concurrently.

    Parameters
    ----------
    audio : UploadFile
        Uploaded audio (multipart file field ``audio``).

    Returns
    -------
    PipelineResponse
        Raw transcript, restored text, BERT multi-label result and the LLM classification.

    Raises
    ------
    HTTPException
        400 on empty upload, 422 on an empty transcript, 502 if any downstream
        service fails.
    """
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    # Step 1: transcribe. Everything downstream depends on the text.
    try:
        transcript = await clients.transcribe(
            audio_bytes, audio.filename or "audio.wav"
        )
    except DownstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if not transcript:
        raise HTTPException(status_code=422, detail="ASR produced empty transcript.")

    # Step 2: LLM restores the raw transcript (diarization, punctuation, types).
    # Sequential — both classifiers below work on the restored text.
    try:
        restored_text = await clients.generate(build_restoration_prompt(transcript))
    except DownstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Step 3: classify the restored text two ways (BERT, LLM) concurrently:
    try:
        bert_labels_raw, llm_classification = await asyncio.gather(
            clients.classify_bert(restored_text),
            clients.generate(build_classification_prompt(restored_text)),
        )
    except DownstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return PipelineResponse(
        transcript=transcript,
        restored_text=restored_text,
        bert_labels=[LabelScore(**item) for item in bert_labels_raw],
        llm_classification=llm_classification,
    )
