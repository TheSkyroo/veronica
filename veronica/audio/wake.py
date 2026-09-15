import asyncio
import logging
from collections.abc import Callable, Iterator

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from veronica.config import Settings

CHUNK = 1280  # 80 ms @ 16 kHz, openwakeword's native chunk

# Fallback set of openwakeword's bundled pretrained model names, used only if
# openwakeword.get_pretrained_model_paths() is unavailable on the installed version.
_PRETRAINED_FALLBACK = frozenset({"alexa", "hey_mycroft", "hey_jarvis", "hey_rhasspy", "timer", "weather"})

log = logging.getLogger("veronica.audio")


def _pretrained_names() -> frozenset[str]:
    try:
        import openwakeword

        # openwakeword 0.6.0 exposes a MODELS dict keyed by pretrained model name.
        models = getattr(openwakeword, "MODELS", None)
        if models:
            return frozenset(models.keys())
    except Exception:
        pass
    return _PRETRAINED_FALLBACK


class WakeWord:
    """Blocks until the configured wake word scores above threshold."""

    _model_cls = Model  # swapped in tests

    def __init__(self, settings: Settings, frames: Callable[[], Iterator[bytes]] | None = None) -> None:
        self.s = settings
        self._frames = frames or self._mic_frames

        custom = settings.models_dir / f"{settings.wake_model}.onnx"
        if custom.exists():
            model_ref = str(custom)
            self._key = settings.wake_model
        elif settings.wake_model in _pretrained_names():
            model_ref = settings.wake_model
            self._key = settings.wake_model
        else:
            log.warning("wake model '%s' not found at %s; falling back to 'hey_jarvis'", settings.wake_model, custom)
            model_ref = "hey_jarvis"
            self._key = "hey_jarvis"

        self._model = self._model_cls(wakeword_models=[model_ref], inference_framework="onnx")

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
            if scores[self._key] >= self.s.wake_threshold:
                return
