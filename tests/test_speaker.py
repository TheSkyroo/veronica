import json
import logging
import stat

import numpy as np
import pytest

from veronica.audio import models, speaker
from veronica.audio.speaker import SpeakerGate, SpeakerModel, VoiceProfile
from veronica.config import Settings

DIM = 4


def voice(k: int, n: int = 32000) -> np.ndarray:
    """A fake utterance of 'voice' k, loud enough to count as voiced:
    FakeModel reads the voice off the first sample."""
    return np.full(n, k * 1000, dtype=np.int16)


class FakeModel:
    loads = 0

    def __init__(self, path):
        FakeModel.loads += 1
        self.path = path

    def embed(self, pcm):
        k = int(pcm[0]) // 1000
        v = np.zeros(DIM, dtype=np.float32)
        if k >= 10:          # "a noisy take of voice k-10": mostly k, some of the next voice
            v[(k - 10) % DIM], v[(k - 9) % DIM] = 0.8, 0.6
        else:
            v[k % DIM] = 1.0
        return v


@pytest.fixture
def gate(tmp_home, monkeypatch):
    FakeModel.loads = 0
    monkeypatch.setattr(SpeakerGate, "_model_cls", FakeModel)
    monkeypatch.setattr(models, "ensure", lambda m, d, **k: d / m.name)
    return SpeakerGate(Settings(speaker_threshold=0.5))


def test_no_profile_accepts_without_loading_the_model(gate):
    assert gate.check(voice(1), "request") == (True, None)
    assert not gate.active
    assert FakeModel.loads == 0


def test_enrol_saves_a_private_profile_that_a_new_gate_loads(gate):
    profile, agreement = gate.enrol([voice(1), voice(1), voice(1)])
    assert profile is not None and agreement == pytest.approx(1.0)
    mode = stat.S_IMODE(gate.path.stat().st_mode)
    assert mode == 0o600
    data = json.loads(gate.path.read_text())
    assert data["model"] == models.CAMPPLUS.name and data["clips"] == 3 and len(data["embedding"]) == DIM
    again = SpeakerGate(gate.s)
    assert again.active and np.allclose(again.profile.embedding, profile.embedding)


def test_check_accepts_the_enrolled_voice_and_ignores_others(gate, caplog):
    gate.enrol([voice(1), voice(1), voice(1)])
    caplog.set_level(logging.INFO, logger="veronica.audio")
    ok, score = gate.check(voice(1), "request")
    assert ok and score == pytest.approx(1.0)
    ok, score = gate.check(voice(2), "confirm")
    assert not ok and score == pytest.approx(0.0)
    assert "speaker confirm: score=0.000 threshold=0.50 -> ignored (2.0 s voiced of 2.0 s" in caplog.text
    assert [r["accepted"] for r in gate.recent] == [False, True]      # newest first
    assert gate.status()["recent"][0]["where"] == "confirm"


def test_threshold_and_switch_are_read_live(gate):
    gate.enrol([voice(1)] * 3)
    noisy = voice(11)                       # scores 0.8 against voice 1
    assert gate.check(noisy, "request")[0]
    gate.s.speaker_threshold = 0.9
    assert not gate.check(noisy, "request")[0]
    gate.s.speaker_verification = False
    assert gate.check(voice(2), "request") == (True, None)


def test_wake_check_only_when_asked_for(gate):
    gate.enrol([voice(1)] * 3)
    assert gate.check_wake(voice(2)) is True           # speaker_verification_wake off
    gate.s.speaker_verification_wake = True
    assert gate.check_wake(voice(2)) is False
    assert gate.check_wake(voice(1)) is True


def test_enrolment_that_disagrees_saves_nothing(gate):
    profile, agreement = gate.enrol([voice(1), voice(1), voice(2)])
    assert profile is None and agreement < speaker.ENROL_MIN_AGREEMENT
    assert not gate.path.exists() and gate.profile is None


def test_forget_deletes_the_profile(gate):
    gate.enrol([voice(1)] * 3)
    assert gate.forget() is True
    assert not gate.path.exists() and not gate.active
    assert gate.forget() is False


@pytest.mark.parametrize("content", [
    "not json",
    json.dumps({"version": 99, "model": models.CAMPPLUS.name, "embedding": [1, 0]}),
    json.dumps({"version": 1, "model": "some-other-model.onnx", "embedding": [1, 0]}),
    json.dumps({"version": 1, "model": models.CAMPPLUS.name, "embedding": []}),
])
def test_a_bad_profile_is_ignored(tmp_home, content, caplog):
    s = Settings()
    s.home.mkdir(parents=True, exist_ok=True)
    speaker.profile_path(s).write_text(content)
    assert VoiceProfile.load(speaker.profile_path(s)) is None
    assert "ignoring voice profile" in caplog.text


def test_check_never_loads_the_model_itself(gate, monkeypatch, caplog):
    """A turn's check must never wait on a download or a load: until the
    model is in, it accepts (logged once) and loads it in the background."""
    gate.enrol([voice(1)] * 3)
    gate._model = None
    kicked = []
    monkeypatch.setattr(gate, "_load_in_background", lambda: kicked.append(1))
    monkeypatch.setattr(models, "ensure", lambda *a, **k: pytest.fail("check fetched the model"))
    assert gate.check(voice(2), "confirm") == (True, None)
    assert gate.check(voice(2), "request") == (True, None)
    assert len(kicked) == 2
    assert caplog.text.count("model not loaded yet; accepting") == 1
    assert gate.check_wake(voice(2)) is True


def test_a_failed_load_is_visible_and_retried_only_now_and_then(gate, monkeypatch):
    gate.enrol([voice(1)] * 3)
    gate._model = None
    calls = []

    def boom(m, d, **k):
        calls.append(1)
        raise OSError("offline")

    monkeypatch.setattr(models, "ensure", boom)
    assert gate.prepare() is False
    assert gate.failed and gate.status()["failed"] is True
    gate._load_in_background()                      # within RETRY_S: no new attempt
    assert calls == [1]
    gate._failed_at -= speaker.RETRY_S + 1
    gate._load_in_background()
    gate._loader.join(2)
    assert calls == [1, 1]
    monkeypatch.setattr(models, "ensure", lambda m, d, **k: d / m.name)
    assert gate.prepare() is True and not gate.failed


def test_one_scoring_error_accepts_that_capture_but_does_not_latch(gate, caplog):
    gate.enrol([voice(1)] * 3)
    real = gate._model.embed
    gate._model.embed = lambda pcm: (_ for _ in ()).throw(RuntimeError("ort hiccup"))
    assert gate.check(voice(2), "request") == (True, None)
    gate._model.embed = real
    assert gate.check(voice(2), "request")[0] is False
    assert not gate.failed


def test_speaker_model_feeds_normalised_fbanks_and_returns_a_unit_vector(monkeypatch, tmp_path):
    seen = {}

    class Sess:
        def get_inputs(self):
            return [type("I", (), {"name": "x"})()]

        def run(self, _out, feeds):
            seen["x"] = feeds["x"]
            return [np.array([[3.0, 4.0]], dtype=np.float32)]

    monkeypatch.setattr(SpeakerModel, "_session_factory", staticmethod(lambda p: Sess()))
    m = SpeakerModel(tmp_path / "m.onnx")
    rng = np.random.default_rng(0)
    e = m.embed((rng.standard_normal(16000) * 3000).astype(np.int16))
    assert e == pytest.approx([0.6, 0.8])
    x = seen["x"]
    assert x.shape == (1, 98, 80)
    assert np.abs(x[0].mean(axis=0)).max() < 1e-4          # mean-normalised per bin
    m.embed(np.zeros(400, dtype=np.int16))                  # a single frame is padded, not an error
    assert seen["x"].shape == (1, 10, 80)


def test_voiced_drops_the_silence_around_speech():
    speech = (np.sin(np.arange(16000) / 5) * 5000).astype(np.int16)
    pcm = np.concatenate([np.full(8000, 3, np.int16), speech, np.full(19200, 3, np.int16)])
    v = speaker.voiced(pcm)
    assert abs(v.size - speech.size) <= 480
    assert speaker.voiced_s(pcm) == pytest.approx(1.0, abs=0.04)
    assert speaker.voiced(np.zeros(100, np.int16)).size == 100          # under a frame: as is
    quiet = np.zeros(48000, np.int16)
    assert speaker.voiced(quiet).size == 48000       # nothing to embed on its own: all of it...
    assert speaker.voiced_s(quiet) == 0.0            # ...but none of it counts as voice
    hiss = (np.random.default_rng(0).standard_normal(48000) * 5).astype(np.int16)
    assert speaker.voiced_s(hiss) == 0.0
    blip = np.concatenate([np.full(4320, 3000, np.int16), np.zeros(19200, np.int16)])   # 9 frames
    assert speaker.voiced_s(blip) == 0.0


def test_short_answers_get_a_proportionally_lower_bar(gate):
    gate.s.speaker_threshold = 0.4
    assert gate.threshold_for(voice(1, 33600)) == pytest.approx(0.4)
    assert gate.threshold_for(voice(1, 64000)) == pytest.approx(0.4)
    assert gate.threshold_for(voice(1, 16320)) == pytest.approx(0.4 * (0.6 + 0.4 * 0.51))
    assert gate.threshold_for(voice(1, 4800)) == pytest.approx(0.4 * (0.6 + 0.4 * 0.15))
    # more speech never lowers the bar: under ten voiced frames is no speech at all
    assert gate.threshold_for(voice(1, 4320)) == pytest.approx(0.4 * 0.6)
    assert gate.threshold_for(voice(1, 4320)) <= gate.threshold_for(voice(1, 4800))
    # a confirm answer never goes under 80% of the threshold
    assert gate.threshold_for(voice(1, 4800), "confirm") == pytest.approx(0.4 * 0.8)
    assert gate.threshold_for(voice(1, 64000), "confirm") == pytest.approx(0.4)
