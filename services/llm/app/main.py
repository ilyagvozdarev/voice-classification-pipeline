import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from .backends import get_backend
from .batcher import BatchProcessor
from .config import settings
from .schemas import GenerateRequest, GenerateResponse, HealthResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("llm")

backend = get_backend()
batcher = BatchProcessor(
    backend,
    max_batch_size=settings.llm_max_batch_size,
    batch_timeout_ms=settings.llm_batch_timeout_ms,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model off the event loop, then start the batching worker.
    await run_in_threadpool(backend.load)
    await batcher.start()
    try:
        yield
    finally:
        await batcher.stop()


app = FastAPI(title="LLM Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        model=settings.llm_model_name,
        backend=settings.llm_backend,
        ready=batcher is not None,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    max_new_tokens = req.max_new_tokens or settings.llm_max_new_tokens
    try:
        text = await batcher.submit(req.prompt, max_new_tokens)
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=422, detail=f"Generation error: {exc}"
        ) from exc

    return GenerateResponse(text=text, model=settings.llm_model_name)
