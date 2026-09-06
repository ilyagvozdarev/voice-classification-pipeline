"""
Request batching for the LLM service.

Incoming prompts are put on an ``asyncio.Queue`` and a single background worker consumes
it (suspends on ``queue.get`` / ``wait_for``), forming a batch and running one inference
call for many prompts at once.
Each request awaits its own ``Future``, which the worker resolves once the batch is done.
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
    def __init__(
        self, backend: Backend, max_batch_size: int, batch_timeout_ms: int
    ) -> None:
        self._backend = backend
        self._max_batch_size = max_batch_size
        self._batch_timeout = batch_timeout_ms / 1000.0
        self._queue: asyncio.Queue[_Job] = asyncio.Queue()
        self._worker: asyncio.Task | None = None

    async def start(self) -> None:
        self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

    async def submit(self, prompt: str, max_new_tokens: int) -> str:
        """
        Enqueue one prompt and await its generated text.

        Parameters
        ----------
        prompt : str
            prompt to complete.
        max_new_tokens : int
            generation length limit.

        Returns
        -------
        str
            The generated completion.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(_Job(prompt, max_new_tokens, future))
        return await future

    async def _run(self) -> None:
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
