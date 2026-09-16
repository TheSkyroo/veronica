import asyncio
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro


class Synthesizer:
    """Kokoro ONNX text-to-speech. Output: float32 mono at 24 kHz."""

    _kokoro_cls = Kokoro  # swapped in tests

    def __init__(self, voice: str, models_dir: Path, speed: float = 1.0) -> None:
        self.voice = voice
        self.speed = speed
        self._engine = self._kokoro_cls(
            str(models_dir / "kokoro-v1.0.onnx"),
            str(models_dir / "voices-v1.0.bin"),
        )

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        samples, sr = self._engine.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        return np.asarray(samples, dtype=np.float32), sr

    async def asynth(self, text: str) -> tuple[np.ndarray, int]:
        return await asyncio.to_thread(self.synth, text)
