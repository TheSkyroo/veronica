import asyncio
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

from veronica.brain.sentences import (
    has_devanagari,  # also re-exported for compatibility
)


class Synthesizer:
    """Kokoro ONNX text-to-speech. Output: float32 mono at 24 kHz."""

    _kokoro_cls = Kokoro  # swapped in tests

    def __init__(
        self,
        voice: str,
        models_dir: Path,
        speed: float = 1.0,
        hindi_voice: str = "hf_beta",
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.hindi_voice = hindi_voice
        self._engine = self._kokoro_cls(
            str(models_dir / "kokoro-v1.0.onnx"),
            str(models_dir / "voices-v1.0.bin"),
        )

    def synth(self, text: str, lang: str | None = None) -> tuple[np.ndarray, int]:
        # Script wins over the turn's language: an English voice reading
        # Devanagari produces noise, so a reply that came back in Hindi is
        # spoken in the Hindi voice even when the question was English.
        hindi = lang == "hi" or has_devanagari(text)
        voice, kl = (self.hindi_voice, "hi") if hindi else (self.voice, "en-us")
        samples, sr = self._engine.create(text, voice=voice, speed=self.speed, lang=kl)
        return np.asarray(samples, dtype=np.float32), sr

    async def asynth(self, text: str, lang: str | None = None) -> tuple[np.ndarray, int]:
        return await asyncio.to_thread(self.synth, text, lang)
