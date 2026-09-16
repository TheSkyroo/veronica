"""Print the live wake-word score for every mic frame that exceeds 0.1.

Say the wake phrase while this is running and watch the printed scores; set
VERONICA_WAKE_THRESHOLD in .env just below the scores you see. Ctrl-C to stop.
"""
import numpy as np

from veronica.audio.wake import WakeWord
from veronica.config import settings


def main() -> None:
    w = WakeWord(settings)
    model = w._model
    key = w._key
    print(f"listening for '{key}' (threshold={settings.wake_threshold}); Ctrl-C to stop")
    try:
        for frame in w._mic_frames():
            chunk = np.frombuffer(frame, dtype=np.int16)
            score = model.predict(chunk)[key]
            if score > 0.1:
                marker = "*" if score >= settings.wake_threshold else ""
                print(f"{score:.2f}{marker}")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
