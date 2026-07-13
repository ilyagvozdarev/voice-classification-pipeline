"""Request batching for the LLM service.

Instead of running the model once per HTTP request, incoming prompts are put on
an ``asyncio.Queue`` and a single background worker drains it, forming a batch
and running one inference call for many prompts at once. Each request awaits
its own ``Future``, which the worker resolves once the batch is done.

This is the classic "accumulate a batch, then flush" pattern and a direct use
of the event-loop mechanics: the worker suspends on ``queue.get`` / ``wait_for``
(yielding the loop), and callers suspend on their per-request Future.
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from fastapi.concurrency import run_in_threadpool

from .backends import Backend

logger = logging.getLogger("llm.batcher")


@dataclass
class _Job:
    prompt: str
    max_new_tokens: int
    future: asyncio.Future


class BatchProcessor:
    """Owns the queue and the background worker that flushes batches."""

    def __init__(
        self, backend: Backend, max_batch_size: int, batch_timeout_ms: int
    ) -> None:
        self._backend = backend
        self._max_batch_size = max_batch_size
        self._batch_timeout = batch_timeout_ms / 1000.0
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        """Spawn the background worker (call inside the running event loop)."""
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Cancel the worker and wait for it to unwind."""
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def submit(self, prompt: str, max_new_tokens: int) -> str:
        """Enqueue one prompt and await its generated text.

        Parameters
        ----------
        prompt : str
            Fully-built prompt to complete.
        max_new_tokens : int
            Generation length limit for this request.

        Returns
        -------
        str
            The generated completion for this prompt.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(_Job(prompt, max_new_tokens, future))
        return await future

    async def _run(self) -> None:
        """Worker loop: collect a batch, then process it, forever."""
        while True:
            batch = await self._collect_batch()
            await self._process(batch)

    async def _collect_batch(self) -> list[_Job]:
        # Block until at least one job arrives, then keep pulling more until
        # the batch is full or the time budget (since the first job) is spent.
        first = await self._queue.get()
        batch = [first]
        deadline = asyncio.get_running_loop().time() + self._batch_timeout

        while len(batch) < self._max_batch_size:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                job = await asyncio.wait_for(self._queue.get(), remaining)
            except TimeoutError:
                break
            batch.append(job)
        return batch

    async def _process(self, batch: list[_Job]) -> None:
        prompts = [job.prompt for job in batch]
        # One shared length budget for the whole batch (simplification).
        max_new_tokens = max(job.max_new_tokens for job in batch)
        logger.info("Processing batch of %d prompt(s)", len(batch))

        try:
            # Inference is blocking (CPU/GPU) — offload off the event loop.
            texts = await run_in_threadpool(
                self._backend.generate_batch, prompts, max_new_tokens
            )
        except Exception as exc:
            logger.exception("Batch generation failed")
            for job in batch:
                if not job.future.done():
                    job.future.set_exception(exc)
            return

        for job, text in zip(batch, texts, strict=True):
            if not job.future.done():
                job.future.set_result(text)



# NOTE:
# [Producer] - [Consumer] работают параллельно через asyncio.Queue
#
# 1. event loop уже запущен, его создает uvicorn: когда контейнер стартует командой вида uvicorn app.main:app --host 0.0.0.0 --port 8000, 
#    происходит примерно следующее (упрощённо):
#       asyncio.run(server.serve())
#
# 2. FastAPI lifespan:
#    запуск задачи на накопление батча и его обработки (batcher.start() -> asyncio.create_task(self._run())).
#    обработка батча происходит в отдельном потоке (run_in_threadpool), поэтому она перекрывается с накоплением батча 
#    (поэтому нет смысла накопление батча и его обработку помещать в разные таски)
#
# 3. [Producer]
#    @app.post("/generate"):
#    каждый запрос генерации кладем в очередь asyncio.Queue[_Job] в виде работы - тройки _Job(prompt, max_new_tokens, future) (batcher.submit)
#
# 4. [Consumer]
#    задача на накопление батча (_collect_batch):
#    берем из очереди _max_batch_size работ или меньше если прошло _batch_timeout секунд (у первого запроса в батче таймаута нет)
#
# 5. корутина обработки батча (_process):
#    запускается в отдельном потоке через fastapi.concurrency.run_in_threadpool
#
#    почему не например asyncio.to_thread?:
#       run_in_threadpool из starlette построен поверх anyio, и отсюда реальное отличие от asyncio.to_thread:
#       Единый пул с общим ограничителем (capacity limiter):
#       Через этот же пул FastAPI гоняет синхронные роут-обработчики, синхронные зависимости и такие явные run_in_threadpool
#       Значит, вся блокирующая работа приложения делит общий бюджет потоков, поток-пул не раздувается бесконтрольно
#       А asyncio.to_thread использует отдельный дефолтный executor цикла со своим лимитом не связанный с anyio-пулом FastAPI.
#       
#       Однако в нашем коде разница практически не проявится - у нас один воркер (_run в единственном экземпляре), поэтому в любой момент 
#       времени крутится максимум один generate_batch. Размер пула и общий лимитер тут почти не важны — конкуренции за потоки нет.
#       Так что оба варианта дали бы одинаковое поведение:
#           texts = await asyncio.to_thread(self._backend.generate_batch, prompts, max_new_tokens)
#       Выбор run_in_threadpool здесь — вопрос консистентности со стеком (мы и так внутри FastAPI/Starlette/anyio), а не производительности.
#
#    после получения батча ответов всем future работ задаем future.set_result(text) (делаем done), text - ответ
#
# 6. после того как future done соответствующий batcher.submit просыпается и отправляет ответ в master-сервис

