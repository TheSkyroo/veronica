"""Live wake-word debug: openwakeword scores, or whisper window transcripts.

For the openwakeword engine, prints the live wake-word score for every mic frame
that exceeds 0.1. For the whisper engine, prints each analysis window's
transcript. Say the wake phrase while this is running. Ctrl-C to stop.
"""
import numpy as np

from veronica.audio.wake import make_wake
from veronica.config import settings


def _run_openwakeword(w) -> None:
    model = w._model
    key = w._key
    print(f"listening for '{key}' (threshold={settings.wake_threshold}); Ctrl-C to stop")
    for frame in w._mic_frames():
        chunk = np.frombuffer(frame, dtype=np.int16)
        score = model.predict(chunk)[key]
        if score > 0.1:
            marker = "*" if score >= settings.wake_threshold else ""
            print(f"{score:.2f}{marker}")


def _run_whisper(w) -> None:
    from veronica.ui.events import rms

    print(f"listening for phrases {settings.wake_phrases}; Ctrl-C to stop")
    buf = np.zeros(0, dtype=np.int16)
    since_hop = 0
    for frame in w._mic_frames():
        chunk = np.frombuffer(frame, dtype=np.int16)
        buf = np.concatenate([buf, chunk])[-w._window_samples:]
        since_hop += chunk.size
        if since_hop < w._hop_samples:
            continue
        since_hop = 0
        if rms(buf) < settings.wake_min_rms:
            continue
        text = w._transcribe(buf)
        print(repr(text))


def main() -> None:
    print(f"engine: {settings.wake_engine}")
    w = make_wake(settings)
    try:
        if settings.wake_engine == "whisper":
            _run_whisper(w)
        else:
            _run_openwakeword(w)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
