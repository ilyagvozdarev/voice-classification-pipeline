import logging

import aiohttp

from .config import settings

logger = logging.getLogger("master.clients")


class DownstreamError(RuntimeError):
    """Raised when a downstream service returns a non-2xx response."""


class ServiceClients:
    """Owns a single shared aiohttp session for all downstream calls.

    One session (connection pool) is created on startup and reused, which is
    the recommended aiohttp pattern — creating a session per request is costly.
    """

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=settings.request_timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("Client session is not initialized.")
        return self._session

    async def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        form = aiohttp.FormData()
        form.add_field(
            "audio", audio_bytes, filename=filename, content_type="application/octet-stream"
        )
        async with self.session.post(f"{settings.asr_url}/transcribe", data=form) as resp:
            await _raise_for_status(resp, "asr")
            data = await resp.json()
        return data["text"]

    async def classify(self, text: str) -> dict:
        payload = {"text": text}
        async with self.session.post(f"{settings.bert_url}/classify", json=payload) as resp:
            await _raise_for_status(resp, "bert")
            return await resp.json()

    async def generate(self, text: str) -> str:
        payload = {"text": text}
        async with self.session.post(f"{settings.llm_url}/generate", json=payload) as resp:
            await _raise_for_status(resp, "llm")
            data = await resp.json()
        return data["answer"]


async def _raise_for_status(resp: aiohttp.ClientResponse, service: str) -> None:
    if resp.status >= 400:
        body = await resp.text()
        raise DownstreamError(f"{service} returned {resp.status}: {body[:300]}")


clients = ServiceClients()



# Notes:
# - метод start(self) не делает await, технически её можно было бы объявить обычной def start(self) а не делать корутиной.
#   Async её сделали по двум причинам:
#   1. Симметрия и единый контракт. Оба метода — управление жизненным циклом сессии, и оба вызываются одинаково в lifespan:
#      await clients.start()
#      ...
#      await clients.close()
#
#   2. Задел на будущее (future-proofing). Если завтра в инициализацию добавится что-то по-настоящему асинхронное (например «пингануть
#      downstream-сервисы при старте», прогреть соединения, сходить за токеном), то start уже async — не придётся менять сигнатуру и все места вызова.
#
# - Ключевой нюанс, который сбивает при переходе с requests на aiohttp: у aiohttp нет отдельного параметра files=.
#   Есть только data=, и FormData — универсальный строитель тела формат которого выбирается по содержимому:
#   - добавил поле с filename / бинарными данными → multipart (файловая часть);
#   - просто словарь строк → urlencoded (как data= в requests).
#
# - async with self.session.post(f"{settings.asr_url}/transcribe", data=form) as resp:
#      await _raise_for_status(resp, "asr")
#      data = await resp.json()
#   Реальная отправка происходит в __aenter__
#   __aexit__(...) освобождает ответ: дочитывает/сбрасывает оставшееся тело и возвращает соединение в пул (keep-alive), чтобы его
#   можно было переиспользовать для следующего запроса. Это тоже может требовать I/O — поэтому __aexit__ асинхронный.

#   В момент входа в блок async with ответ уже пришел, но не весь а только его начало без тела.
#   то есть __aenter__() завершается, когда сервер прислал строку статуса и заголовки ответа
#   Тело подтягивается позже — как раз на await resp.json().
