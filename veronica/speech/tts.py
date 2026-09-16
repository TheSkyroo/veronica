import asyncio
import re
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))


class Synthesizer:
    """Kokoro ONNX text-to-speech. Output: float32 mono at 24 kHz."""

    _kokoro_cls = Kokoro  # swapped in tests

    def __init__(
        self,
        voice: str,
        models_dir: Path,
        speed: float = 1.0,
        hindi_voice: str = "hf_alpha",
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.hindi_voice = hindi_voice
        self._engine = self._kokoro_cls(
            str(models_dir / "kokoro-v1.0.onnx"),
            str(models_dir / "voices-v1.0.bin"),
        )

    def synth(self, text: str, lang: str | None = None) -> tuple[np.ndarray, int]:
        hindi = lang == "hi" or (lang is None and has_devanagari(text))
        voice, kl = (self.hindi_voice, "hi") if hindi else (self.voice, "en-us")
        samples, sr = self._engine.create(text, voice=voice, speed=self.speed, lang=kl)
        return np.asarray(samples, dtype=np.float32), sr

    async def asynth(self, text: str, lang: str | None = None) -> tuple[np.ndarray, int]:
        return await asyncio.to_thread(self.synth, text, lang)
