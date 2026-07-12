import io
import logging

import numpy as np
import soundfile as sf

from .config import settings

logger = logging.getLogger("asr.model")


class ASRModel:
    """Thin wrapper around a HF automatic-speech-recognition pipeline.

    Loading is deferred to `load()` so the FastAPI lifespan controls when the
    (potentially slow) weight download / init happens.
    """

    def __init__(self) -> None:
        self._pipe = None

    def load(self) -> None:
        # Imported lazily so the module can be imported without torch present.
        import torch
        from transformers import pipeline

        # Use GPU automatically when a CUDA build of torch sees a device;
        # the CPU image reports False here and falls back to -1 (CPU).
        device = 0 if torch.cuda.is_available() else -1
        logger.info(
            "Loading ASR model: %s (device=%s)",
            settings.asr_model_name,
            "cuda:0" if device == 0 else "cpu",
        )
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=settings.asr_model_name,
            device=device,
        )
        logger.info("ASR model loaded.")

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    def transcribe(self, audio_bytes: bytes) -> str:
        """Decode raw audio bytes and return the recognized text.

        The bytes may be in any format libsndfile can read (WAV, FLAC, OGG,
        ...). Stereo input is downmixed to mono and converted to float32
        before being passed to the ASR pipeline. Runs synchronously and is
        CPU-bound, so callers on an event loop should offload it to a thread.

        Parameters
        ----------
        audio_bytes : bytes
            Encoded audio file content (not a decoded waveform). Must be a
            non-empty buffer in a libsndfile-supported container.

        Returns
        -------
        str
            The transcribed text, stripped of surrounding whitespace. May be
            an empty string if the model recognized no speech.

        Raises
        ------
        RuntimeError
            If the model has not been loaded yet (``load`` was not called).

        Examples
        --------
        >>> model = ASRModel()
        >>> model.load()
        >>> with open("hello.wav", "rb") as fh:
        ...     model.transcribe(fh.read())
        'hello world'
        """
        if self._pipe is None:
            raise RuntimeError("ASR model is not loaded yet.")

        data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:  # stereo -> mono
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        result = self._pipe({"array": data, "sampling_rate": sample_rate})
        return (result.get("text") or "").strip()


asr_model = ASRModel()
