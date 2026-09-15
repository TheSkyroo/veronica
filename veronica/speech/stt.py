import asyncio

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    """faster-whisper wrapper. Input: int16 mono 16 kHz."""

    _model_cls = WhisperModel  # swapped in tests

    def __init__(self, model_name: str) -> None:
        self._model = self._model_cls(model_name, device="cpu", compute_type="int8")

    def transcribe(self, pcm16: np.ndarray) -> str:
        audio = pcm16.astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio, beam_size=1, language="en", vad_filter=False
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def atranscribe(self, pcm16: np.ndarray) -> str:
        return await asyncio.to_thread(self.transcribe, pcm16)
