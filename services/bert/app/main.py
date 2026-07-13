import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from .config import settings
from .model import bert_model
from .schemas import ClassifyRequest, ClassifyResponse, HealthResponse, LabelScore

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("bert")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(bert_model.load)
    yield


app = FastAPI(title="BERT Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(model=settings.bert_model_name, ready=bert_model.ready)


@app.post("/classify", response_model=ClassifyResponse)
async def classify(req: ClassifyRequest) -> ClassifyResponse:
    if not bert_model.ready:
        raise HTTPException(status_code=503, detail="Model not ready.")

    try:
        pairs = await run_in_threadpool(bert_model.classify, req.text)
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(
            status_code=422, detail=f"Classification error: {exc}"
        ) from exc

    labels = [LabelScore(label=label, score=score) for label, score in pairs]
    return ClassifyResponse(
        labels=labels, threshold=settings.bert_threshold, model=settings.bert_model_name
    )
