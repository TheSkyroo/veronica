"""The real models on the committed fixtures (tests/fixtures/voice: Kokoro
voices, reverberated; "me" is af_sarah, "other" am_adam). No microphone:
the recorder is fed the fixture audio through its `frames` hook. Fetches
the models into ~/.veronica/models if they aren't there yet."""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from veronica.audio import denoise, models
from veronica.audio.record import Recorder
from veronica.audio.speaker import SpeakerGate
from veronica.config import Settings

pytestmark = pytest.mark.live
FIX = Path(__file__).parent / "fixtures" / "voice"


def load(name: str) -> np.ndarray:
    return sf.read(FIX / f"{name}.flac", dtype="int16")[0]


def db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-9)


def suppressed(x: np.ndarray) -> np.ndarray:
    path = models.ensure(models.GTCRN, Settings().models_dir)
    d = denoise.Denoiser(denoise._session(str(path)))
    y = np.concatenate([d.process(x[i:i + 480]) for i in range(0, x.size, 480)])
    return y[denoise.LAG:]


def test_suppression_removes_noise_and_keeps_the_voice():
    noise = load("noise_pink")
    assert db(suppressed(noise)) < db(noise) - 15
    speech = load("me_command")
    out = suppressed(speech)
    assert np.corrcoef(out, speech[: out.size])[0, 1] > 0.95


def test_steady_room_noise_alone_does_not_open_a_capture(tmp_path):
    # (Babble is other people's speech: suppression keeps voices, so that
    # one is the speaker check's job, below.)
    models.ensure(models.GTCRN, Settings().models_dir)
    noise = np.concatenate([load("noise_pink")] * 2)

    def frames():
        for i in range(0, noise.size - 479, 480):
            yield noise[i:i + 480].tobytes()
        while True:
            yield np.zeros(480, np.int16).tobytes()

    r = Recorder(Settings(), frames=frames)
    import asyncio

    assert asyncio.run(r.capture(max_s=3)) is None


def test_speaker_check_on_the_fixture_voices(tmp_path):
    s = Settings(home=tmp_path)
    gate = SpeakerGate(s)
    gate._model = gate._model_cls(models.ensure(models.CAMPPLUS, Settings().models_dir))
    lead, tail = np.zeros(4800, np.int16), np.zeros(19200, np.int16)
    profile, agreement = gate.enrol([np.concatenate([lead, load(f"me_enrol_{i}"), tail]) for i in (1, 2, 3)])
    assert profile is not None and agreement > 0.5
    assert gate.check(np.concatenate([lead, load("me_command"), tail]), "request")[0]
    assert gate.check(np.concatenate([lead, load("me_yes"), tail]), "confirm")[0]
    assert not gate.check(np.concatenate([lead, load("other_command"), tail]), "request")[0]
    assert not gate.check(load("noise_pink"), "confirm")[0]
