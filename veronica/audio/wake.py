import asyncio
from collections.abc import Callable, Iterator

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from veronica.config import Settings

CHUNK = 1280  # 80 ms @ 16 kHz, openwakeword's native chunk


class WakeWord:
    """Blocks until the configured wake word scores above threshold."""

    _model_cls = Model  # swapped in tests

    def __init__(self, settings: Settings, frames: Callable[[], Iterator[bytes]] | None = None) -> None:
        self.s = settings
        self._frames = frames or self._mic_frames
        self._model = self._model_cls(wakeword_models=[settings.wake_model], inference_framework="onnx")

    def _mic_frames(self) -> Iterator[bytes]:
        with sd.RawInputStream(samplerate=self.s.sample_rate, channels=1, dtype="int16", blocksize=CHUNK) as stream:
            while True:
                data, _ = stream.read(CHUNK)
                yield bytes(data)

    async def wait(self) -> None:
        await asyncio.to_thread(self._wait)

    def _wait(self) -> None:
        self._model.reset()
        for frame in self._frames():
            chunk = np.frombuffer(frame, dtype=np.int16)
            scores = self._model.predict(chunk)
            if scores[self.s.wake_model] >= self.s.wake_threshold:
                return
