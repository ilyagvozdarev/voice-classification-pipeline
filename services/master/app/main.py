import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from .clients import DownstreamError, clients
from .config import settings
from .schemas import HealthResponse, PipelineResponse, Sentiment

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
    """Orchestrate the full pipeline for one audio clip.

    Flow: audio -> ASR (text) -> {BERT, LLM} in parallel -> aggregate.
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

    # Step 2: fan out to BERT and LLM concurrently; they are independent.
    try:
        sentiment_raw, answer = await asyncio.gather(
            clients.classify(transcript),
            clients.generate(transcript),
        )
    except DownstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Step 3: aggregate and return.
    return PipelineResponse(
        transcript=transcript,
        sentiment=Sentiment(
            label=sentiment_raw["label"], score=sentiment_raw["score"]
        ),
        answer=answer,
    )


# Некоторые вещи нужно сделать один раз при старте приложения и один раз при остановке, а не на каждый запрос:
#
# - В master: создать aiohttp-сессию (пул соединений) при старте и закрыть её при выключении.
# - В ML-сервисах: загрузить модель в память при старте (см. asr/app/main.py).
#
# Создавать сессию/грузить модель на каждый HTTP-запрос — расточительно. Нужен механизм «сделай ДО того, как начнём принимать запросы»
# и «прибери ПОСЛЕ того, как закончили». В FastAPI это называется lifespan («жизненный цикл» приложения).
#
# lifespan — это параметр конструктора FastAPI, куда ты передаёшь функцию, описывающую «что сделать на старте и на остановке»:
#
#     app = FastAPI(title="Master Orchestrator", lifespan=lifespan)
#
# FastAPI при запуске вызовет эту функцию, выполнит её startup-часть, потом начнёт обслуживать запросы, а когда приложение
# гасят (Ctrl+C, docker stop) — выполнит её shutdown-часть.
#
# Вопрос только в том, как в одной функции описать «до» и «после». Для этого и нужен @asynccontextmanager с yield.
#
# asynccontextmanager из contextlib — это асинхронная версия того же самого. Нужна, потому что наши setup/teardown
# содержат await (await clients.start()), а обычный @contextmanager с await внутри работать не умеет.
#
# Пошагово, что делает FastAPI:
# 1. Старт контейнера → FastAPI входит в lifespan, выполняет await clients.start() — создаётся aiohttp-сессия.
# 2. Доходит до yield — «зависает» на нём. В этот момент приложение начинает принимать HTTP-запросы. Всё время,
#    пока сервис живёт и обрабатывает /process, выполнение стоит на yield.
# 3. Остановка (docker stop, Ctrl+C) → FastAPI «возобновляет» функцию после yield, выполняет await clients.close() — сессия
#    аккуратно закрывается (закрываются TCP-соединения).
#
#
# lifespan обязан принимать один аргумент — сам объект приложения (FastAPI его передаёт при вызове). Часто он не используется
# (как у нас — мы пишем app, но не трогаем его). Но иногда через него удобно сложить общие ресурсы, например app.state.session = ...,
# чтобы потом доставать их в эндпоинтах. У нас вместо этого глобальный синглтон clients, поэтому аргумент просто присутствует «по контракту».


# почему aiohttp.ClientSession.post а не requests.post:
# конкурентность через asyncio.gather работает только с асинхронным I/O — то есть с aiohttp. С requests он бы не заработал.
# requests — блокирующая (синхронная) библиотека. Когда ты вызываешь requests.post(...), поток замирает и ждёт ответа, ничего больше не делая.
# То есть пока requests.post висит в async-эндпоинте, он замораживает весь event loop

# Короче говоря, aiohttp же — асинхронный клиент: await session.post(...) на время ожидания сети уступает цикл, поэтому и gather
# реально параллелит, и сервер остаётся отзывчивым.


# healthcheck:
# простой liveness, сейчас никем автоматически не дёргается — он существует «на всякий случай» и для ручной проверки,
# так как никто не зависит от master через condition: service_healthy
