"""Pluggable inference backends for the LLM service.

Both backends expose the same tiny surface — ``load()`` and
``generate_batch(prompts)`` — so the batching layer (see ``batcher.py``) does
not care which one is in use. The active backend is chosen by
``settings.llm_backend`` and swapped via a compose overlay, not code changes.
"""

import logging
from typing import Protocol

from .config import settings

logger = logging.getLogger("llm.backend")


class Backend(Protocol):
    """Minimal contract every inference backend must satisfy."""

    def load(self) -> None:
        """Load weights / start the engine (blocking)."""
        ...

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Generate one completion per prompt, preserving order."""
        ...


class TransformersBackend:
    """CPU-friendly backend using a HF text2text pipeline (default flan-t5).

    The HF pipeline natively accepts a list of prompts and runs them as a
    batch, which is exactly what our batcher hands it.
    """

    def __init__(self) -> None:
        self._pipe = None

    def load(self) -> None:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        logger.info(
            "Loading transformers backend: %s (device=%s)",
            settings.llm_model_name,
            "cuda:0" if device == 0 else "cpu",
        )
        self._pipe = pipeline(
            "text2text-generation", model=settings.llm_model_name, device=device
        )
        logger.info("Transformers backend ready.")

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        if self._pipe is None:
            raise RuntimeError("Backend is not loaded yet.")
        outputs = self._pipe(
            prompts, max_new_tokens=max_new_tokens, do_sample=False
        )
        return [o["generated_text"].strip() for o in outputs]


class VLLMBackend:
    """GPU backend using vLLM. Best suited to causal/instruct models.

    vLLM already does continuous batching internally, but our application-level
    batcher still groups requests so a single ``generate`` call covers many
    prompts at once.
    """

    def __init__(self) -> None:
        self._llm = None
        self._sampling_params_cls = None

    def load(self) -> None:
        from vllm import LLM, SamplingParams

        logger.info("Loading vLLM backend: %s", settings.llm_model_name)
        self._llm = LLM(model=settings.llm_model_name)
        self._sampling_params_cls = SamplingParams
        logger.info("vLLM backend ready.")

    def generate_batch(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        if self._llm is None:
            raise RuntimeError("Backend is not loaded yet.")
        params = self._sampling_params_cls(
            max_tokens=max_new_tokens, temperature=0.0
        )
        results = self._llm.generate(prompts, params)
        return [r.outputs[0].text.strip() for r in results]


def get_backend() -> Backend:
    """Instantiate the backend selected by ``settings.llm_backend``."""
    if settings.llm_backend == "vllm":
        return VLLMBackend()
    if settings.llm_backend == "transformers":
        return TransformersBackend()
    raise ValueError(f"Unknown LLM_BACKEND: {settings.llm_backend!r}")
