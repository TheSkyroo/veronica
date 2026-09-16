import asyncio

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    """faster-whisper wrapper. Input: int16 mono 16 kHz."""

    _model_cls = WhisperModel  # swapped in tests

    def __init__(self, model_name: str, language: str | None = "en") -> None:
        self.model_name = model_name
        self.language = language          # None = let whisper detect
        self._model = self._model_cls(model_name, device="cpu", compute_type="int8")

    def set_language(self, language: str | None) -> None:
        self.language = language

    def transcribe_detailed(self, pcm16: np.ndarray) -> tuple[str, str]:
        audio = pcm16.astype(np.float32) / 32768.0
        segments, info = self._model.transcribe(
            audio, beam_size=1, language=self.language, vad_filter=False
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        detected = self.language or getattr(info, "language", None) or "en"
        return text, detected

    def transcribe(self, pcm16: np.ndarray) -> str:
        return self.transcribe_detailed(pcm16)[0]

    async def atranscribe(self, pcm16: np.ndarray) -> str:
        return await asyncio.to_thread(self.transcribe, pcm16)

    async def atranscribe_detailed(self, pcm16: np.ndarray) -> tuple[str, str]:
        return await asyncio.to_thread(self.transcribe_detailed, pcm16)
