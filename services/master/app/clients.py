import logging

import aiohttp

from .config import settings

logger = logging.getLogger("master.clients")


class DownstreamError(RuntimeError):
    """Raised when a downstream service returns a non-2xx response."""


class ServiceClients:
    """Owns a single shared aiohttp session (connection pool) for all downstream calls."""

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
        """Send audio to the ASR service and return the raw transcript."""
        form = aiohttp.FormData()
        form.add_field(
            "audio",
            audio_bytes,
            filename=filename,
            content_type="application/octet-stream",
        )
        async with self.session.post(
            f"{settings.asr_url}/transcribe", data=form
        ) as resp:
            await _raise_for_status(resp, "asr")
            data = await resp.json()
        return data["text"]

    async def classify_bert(self, text: str) -> list[dict]:
        """
        Multi-label classification via the BERT service.

        Returns
        -------
        list of dict
            Each item is ``{"label": str, "score": float}`` above threshold.
        """
        payload = {"text": text}
        async with self.session.post(
            f"{settings.bert_url}/classify", json=payload
        ) as resp:
            await _raise_for_status(resp, "bert")
            data = await resp.json()
        return data["labels"]

    async def generate(self, prompt: str) -> str:
        """Send a prompt to the LLM service and return its completion."""
        payload = {"prompt": prompt}
        async with self.session.post(
            f"{settings.llm_url}/generate", json=payload
        ) as resp:
            await _raise_for_status(resp, "llm")
            data = await resp.json()
        return data["text"]


async def _raise_for_status(resp: aiohttp.ClientResponse, service: str) -> None:
    if resp.status >= 400:
        body = await resp.text()
        raise DownstreamError(f"{service} returned {resp.status}: {body[:300]}")


clients = ServiceClients()
