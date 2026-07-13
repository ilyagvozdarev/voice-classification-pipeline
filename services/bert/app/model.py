import logging

from .config import settings

logger = logging.getLogger("bert.model")


class BertClassifier:
    """Multi-label text-classification pipeline wrapper (default: go_emotions).

    Unlike single-label sentiment, a multi-label model applies a sigmoid per
    label, so several labels can be "on" at once. We return every label whose
    probability clears the configured threshold.
    """

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
        # top_k=None makes the pipeline return scores for ALL labels rather
        # than just the top one — required for multi-label output.
        self._pipe = pipeline(
            "text-classification",
            model=settings.bert_model_name,
            top_k=None,
            device=device,
        )
        logger.info("BERT model loaded.")

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    def classify(self, text: str) -> list[tuple[str, float]]:
        """Return (label, score) pairs above the threshold, sorted descending.

        Parameters
        ----------
        text : str
            The (restored) text to classify.

        Returns
        -------
        list of tuple of (str, float)
            Labels whose probability is >= ``bert_threshold``. May be empty if
            no label is confident enough.

        Raises
        ------
        RuntimeError
            If the model has not been loaded yet.
        """
        if self._pipe is None:
            raise RuntimeError("BERT model is not loaded yet.")

        # With top_k=None the pipeline returns a list of {label, score} dicts.
        scores = self._pipe(text, truncation=True)[0]
        selected = [
            (item["label"], float(item["score"]))
            for item in scores
            if item["score"] >= settings.bert_threshold
        ]
        selected.sort(key=lambda pair: pair[1], reverse=True)
        return selected


bert_model = BertClassifier()
