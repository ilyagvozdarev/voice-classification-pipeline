import logging

from .config import settings

logger = logging.getLogger("bert.model")


class BertClassifier:
    """HF text-classification pipeline wrapper (default: SST-2 sentiment)."""

    def __init__(self) -> None:
        self._pipe = None

    def load(self) -> None:
        import torch
        from transformers import pipeline

        device = 0 if torch.cuda.is_available() else -1
        logger.info(
            "Loading BERT model: %s (device=%s)",
            settings.bert_model_name,
            "cuda:0" if device == 0 else "cpu",
        )
        self._pipe = pipeline("text-classification", model=settings.bert_model_name, device=device)
        logger.info("BERT model loaded.")

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    def classify(self, text: str) -> tuple[str, float]:
        if self._pipe is None:
            raise RuntimeError("BERT model is not loaded yet.")

        result = self._pipe(text, truncation=True)[0]
        return result["label"], float(result["score"])


bert_model = BertClassifier()
