import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .config import settings
from .model import asr_model
from .schemas import HealthResponse, TranscriptionResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("asr")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Blocking model load; run off the event loop so startup doesn't stall it.
    await run_in_threadpool(asr_model.load)
    yield


app = FastAPI(title="ASR Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(model=settings.asr_model_name, ready=asr_model.ready)


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(audio: UploadFile = File(...)) -> TranscriptionResponse:
    if not asr_model.ready:
        raise HTTPException(status_code=503, detail="Model not ready.")

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio upload.")

    try:
        text = await run_in_threadpool(asr_model.transcribe, audio_bytes)
    except Exception as exc:
        logger.exception("Transcription failed")
        raise HTTPException(
            status_code=422, detail=f"Could not process audio: {exc}"
        ) from exc

    return TranscriptionResponse(text=text, model=settings.asr_model_name)


# NOTE:
# - transcribe в run_in_threadpool
