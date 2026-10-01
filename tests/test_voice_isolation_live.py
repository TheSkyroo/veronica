"""The real models on the committed fixtures (tests/fixtures/voice: Kokoro
voices, reverberated; "me" is af_sarah, "other" am_adam). No microphone:
the recorder is fed the fixture audio through its `frames` hook. The models
are downloaded (and checksummed) into a temporary directory, never into
~/.veronica."""
import asyncio
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


@pytest.fixture(scope="module")
def home(tmp_path_factory):
    """A throwaway VERONICA home holding both models for the module."""
    h = tmp_path_factory.mktemp("veronica")
    for m in (models.GTCRN, models.CAMPPLUS):
        models.ensure(m, h / "models")
    return h


def load(name: str) -> np.ndarray:
    return sf.read(FIX / f"{name}.flac", dtype="int16")[0]


def db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-9)


def suppressed(x: np.ndarray, home: Path) -> np.ndarray:
    path = home / "models" / models.GTCRN.name
    d = denoise.Denoiser(denoise._session(str(path)))
    y = np.concatenate([d.process(x[i:i + 480]) for i in range(0, x.size, 480)])
    return y[denoise.LAG:]


def test_suppression_removes_noise_and_keeps_the_voice(home):
    noise = load("noise_pink")
    assert db(suppressed(noise, home)) < db(noise) - 15
    speech = load("me_command")
    out = suppressed(speech, home)
    assert np.corrcoef(out, speech[: out.size])[0, 1] > 0.95


def test_steady_room_noise_alone_does_not_open_a_capture(home):
    # (Babble is other people's speech: suppression keeps voices, so that
    # one is the speaker check's job, below.)
    noise = np.concatenate([load("noise_pink")] * 2)

    def frames():
        for i in range(0, noise.size - 479, 480):
            yield noise[i:i + 480].tobytes()
        while True:
            yield np.zeros(480, np.int16).tobytes()

    r = Recorder(Settings(home=home), frames=frames)
    assert asyncio.run(r.capture(max_s=3)) is None


@pytest.mark.parametrize("name", ["me_yes", "me_command"])
def test_the_capture_starts_where_the_voice_does(home, name):
    """The suppressor lags the mic by 32 ms: the recorder must still hand
    over the raw audio from the voice's first frame, not 30 ms into it."""
    speech = load(name)
    lead = np.zeros(9600, np.int16)
    audio = np.concatenate([lead, speech, np.zeros(32000, np.int16)])

    def frames():
        for i in range(0, audio.size - 479, 480):
            yield audio[i:i + 480].tobytes()
        while True:
            yield np.zeros(480, np.int16).tobytes()

    pcm = asyncio.run(Recorder(Settings(home=home), frames=frames).capture(max_s=3))
    assert pcm is not None
    onset = int(np.flatnonzero(np.abs(audio) > 0.05 * np.abs(speech).max())[0])
    start = next(i for i in range(0, audio.size - pcm.size + 1, 480) if np.array_equal(audio[i:i + 480], pcm[:480]))
    assert start <= onset


def test_speaker_check_on_the_fixture_voices(home):
    gate = SpeakerGate(Settings(home=home))
    assert gate.prepare()
    lead, tail = np.zeros(4800, np.int16), np.zeros(19200, np.int16)
    profile, agreement = gate.enrol([np.concatenate([lead, load(f"me_enrol_{i}"), tail]) for i in (1, 2, 3)])
    assert profile is not None and agreement > 0.5
    assert gate.check(np.concatenate([lead, load("me_command"), tail]), "request")[0]
    assert gate.check(np.concatenate([lead, load("me_yes"), tail]), "confirm")[0]
    assert not gate.check(np.concatenate([lead, load("other_command"), tail]), "request")[0]
    assert not gate.check(load("noise_pink"), "confirm")[0]
