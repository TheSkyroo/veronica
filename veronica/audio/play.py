import asyncio

import numpy as np
import sounddevice as sd


class Player:
    """Plays float32 mono chunks sequentially; stop() cancels playback."""

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.is_playing = False
        self._stopped = False

    def reset(self) -> None:
        """Clear a previous stop() so new playback is accepted."""
        self._stopped = False

    async def play(self, samples: np.ndarray) -> None:
        if self._stopped:
            return
        self.is_playing = True
        try:
            await asyncio.to_thread(self._blocking_play, samples)
        finally:
            self.is_playing = False

    def _blocking_play(self, samples: np.ndarray) -> None:
        sd.play(samples, samplerate=self.sample_rate)
        sd.wait()

    def stop(self) -> None:
        self._stopped = True
        sd.stop()
