import logging

from .config import settings

logger = logging.getLogger("llm.model")

PROMPT_TEMPLATE = "Respond helpfully and concisely to the user message:\n{text}"


class LLMModel:
    """HF text2text-generation pipeline wrapper (default: flan-t5-small)."""

    def __init__(self) -> None:
        self._pipe = None

    def load(self) -> None:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        logger.info(
            "Loading LLM model: %s (device=%s)",
            settings.llm_model_name,
            "cuda:0" if device == 0 else "cpu",
        )
        self._pipe = pipeline(
            "text2text-generation", model=settings.llm_model_name, device=device
        )
        logger.info("LLM model loaded.")

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    def generate(self, text: str, max_new_tokens: int | None = None) -> str:
        if self._pipe is None:
            raise RuntimeError("LLM model is not loaded yet.")

        prompt = PROMPT_TEMPLATE.format(text=text)
        result = self._pipe(
            prompt,
            max_new_tokens=max_new_tokens or settings.llm_max_new_tokens,
            do_sample=False,
        )[0]
        return result["generated_text"].strip()


llm_model = LLMModel()
