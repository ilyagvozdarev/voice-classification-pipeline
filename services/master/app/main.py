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
    """Orchestrate the full pipeline for one audio clip.

    Flow: audio -> ASR -> LLM(restore) -> {BERT(multi-label) ‖ LLM(classify)}.
    The restore step is sequential (everything below needs the restored text);
    the two classifications then run concurrently.

    Parameters
    ----------
    audio : UploadFile
        Uploaded audio clip (multipart file field ``audio``).

    Returns
    -------
    PipelineResponse
        Raw transcript, restored text, BERT multi-label result and the LLM's
        own classification.

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

    # Step 3: classify the restored text two ways, concurrently:
    #   - BERT  -> multi-label prediction
    #   - LLM   -> its own classification (different prompt)
    try:
        bert_labels_raw, llm_classification = await asyncio.gather(
            clients.classify_bert(restored_text),
            clients.generate(build_classification_prompt(restored_text)),
        )
    except DownstreamError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Step 4: aggregate and return.
    return PipelineResponse(
        transcript=transcript,
        restored_text=restored_text,
        bert_labels=[LabelScore(**item) for item in bert_labels_raw],
        llm_classification=llm_classification,
    )


# NOTE:
# lifespan:
# это параметр конструктора FastAPI, куда ты передаёшь функцию, описывающую «что сделать на старте и на остановке» («жизненный цикл» приложения).
# Нужен так как некоторые вещи нужно сделать один раз при старте приложения и один раз при остановке, а не на каждый запрос:
#   app = FastAPI(title="Master Orchestrator", lifespan=lifespan)
# FastAPI при запуске вызовет эту функцию, выполнит её startup-часть, потом начнёт обслуживать запросы, а когда приложение гасят (Ctrl+C, docker stop) 
# — выполнит её shutdown-часть.
# Вопрос только в том, как в одной функции описать «до» и «после». Для этого и нужен @asynccontextmanager с yield:
# asynccontextmanager из contextlib — это асинхронная версия contextmanager. Нужна, потому что наши setup/teardown
# содержат await (await clients.start()), а обычный @contextmanager с await внутри работать не умеет.
#
# lifespan обязан принимать один аргумент — сам объект приложения (FastAPI его передаёт при вызове). Часто он не используется
# (как у нас — мы пишем app, но не трогаем его). Но иногда через него удобно сложить общие ресурсы, например app.state.session = ...,
# чтобы потом доставать их в эндпоинтах. У нас вместо этого глобальный синглтон clients, поэтому аргумент просто присутствует «по контракту».
#
#
# 1. healthcheck:
#    простой liveness, сейчас никем автоматически не дёргается — он существует «на всякий случай» и для ручной проверки,
#    так как никто не зависит от master через condition: service_healthy
#
# 2. FastAPI lifespan:
#    создать aiohttp-сессию (пул соединений) при старте и закрыть её при выключении:
#       1. Старт контейнера → FastAPI входит в lifespan, выполняет await clients.start() — создаётся aiohttp-сессия.
#       2. Доходит до yield — «зависает» на нём. В этот момент приложение начинает принимать HTTP-запросы. Всё время,
#          пока сервис живёт и обрабатывает /process, выполнение стоит на yield.
#       3. Остановка (docker stop, Ctrl+C) → FastAPI «возобновляет» функцию после yield, выполняет await clients.close() — сессия
#          аккуратно закрывается (закрываются TCP-соединения).
# 
# 3. @app.post("/process"):
#    audio -> aiohttp.ClientSession aiohttp.FormData -> {settings.asr_url}/transcribe
#
#    почему aiohttp.ClientSession.post а не requests.post:
#       конкурентность через asyncio.gather работает только с асинхронным I/O — то есть с aiohttp. С requests он бы не заработал.
#       requests — блокирующая (синхронная) библиотека. Когда ты вызываешь requests.post(...), поток замирает и ждёт ответа, ничего больше не делая.
#       То есть пока requests.post висит в async-эндпоинте, он замораживает весь event loop
#       Короче говоря, aiohttp же — асинхронный клиент: await session.post(...) на время ожидания сети уступает цикл, поэтому и gather
#       реально параллелит, и сервер остаётся отзывчивым.
#
#    полученный текст -> "{settings.llm_url}/generate" (получаем restored текст)
#                   
#    предобработанный текст -> asyncio.gather
#                                   -> "{settings.bert_url}/classify"
#                                   -> "{settings.llm_url}/generate"
#
#    ответы возвращаем клиенту (restored_text, bert_labels, llm_classification)
#


