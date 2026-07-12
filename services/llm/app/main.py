import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool

from .config import settings
from .model import llm_model
from .schemas import GenerateRequest, GenerateResponse, HealthResponse

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("llm")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await run_in_threadpool(llm_model.load)
    yield


app = FastAPI(title="LLM Service", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(model=settings.llm_model_name, ready=llm_model.ready)


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    if not llm_model.ready:
        raise HTTPException(status_code=503, detail="Model not ready.")

    try:
        answer = await run_in_threadpool(
            llm_model.generate, req.text, req.max_new_tokens
        )
    except Exception as exc:
        logger.exception("Generation failed")
        raise HTTPException(
            status_code=422, detail=f"Generation error: {exc}"
        ) from exc

    return GenerateResponse(answer=answer, model=settings.llm_model_name)


# - await run_in_threadpool(llm_model.load):
#   Справедливый вопрос — во время lifespan-startup сервер ещё не принимает HTTP. Тем не менее уводить блокировку в поток
#   правильно по нескольким причинам:
#   - event loop должен оставаться отзывчивым даже на старте — он обрабатывает системные сигналы (например SIGTERM, чтобы можно
#     было корректно прервать долгий запуск), внутренние таймеры uvicorn и т.п. Заблокированный намертво цикл на минуту хуже реагирует на «останови контейнер».
#   - единый корректный паттерн. «Блокирующий код вызываем через run_in_threadpool» — это правило одинаково и на старте, и в рантайме.
#     Не нужно держать в голове исключения.
#   - согласованность с рантаймом. В этом же сервисе мы точно так же уводим в поток сам инференс (llm/app/main.py):
#     answer = await run_in_threadpool(llm_model.generate, req.text, req.max_new_tokens)
#     Там это уже критично (запросы идут, цикл нельзя морозить), и load следует тому же принципу.
#
# -
#   комментарий для линтера, который говорит «не ругайся на эту строку по правилу BLE001
#   BLE001 - это код конкретного правила. BLE — от плагина flake8-blind-except (Blind Except), правило BLE001 называется
#   примерно «Do not catch blind exception: Exception». Линтер предупреждает, когда ты ловишь слишком широкий except Exception:
#   — потому что это может «проглотить» неожиданные ошибки и усложнить отладку.
#   Здесь широкий except Exception — намеренный. Инференс модели может упасть кучей разных способов (ошибка torch, нехватка памяти,
#   кривой вход, баг в transformers), и мы не хотим их перечислять поштучно. Задача — любой сбой превратить в аккуратный HTTP-422
#   для клиента, а не в необработанный 500-краш.
