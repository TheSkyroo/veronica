import hashlib
from pathlib import Path

import numpy as np
import pytest

from veronica.audio import denoise, models
from veronica.config import Settings

# conftest stubs make_denoiser out for every test; this module tests it.
_REAL_MAKE_DENOISER = denoise.make_denoiser
_REAL_PREPARE_MODEL = denoise.prepare_model
_REAL_ENSURE = models.ensure


class IdentitySession:
    """Stands in for the GTCRN session: hands the spectrum back untouched
    and the caches back bumped, so the STFT plumbing alone is under test."""

    def __init__(self):
        self.runs = 0
        self.seen_caches = []

    def run(self, _outputs, feeds):
        self.runs += 1
        self.seen_caches.append(float(feeds["conv_cache"].flat[0]))
        assert feeds["mix"].shape == (1, 257, 1, 2)
        return [feeds["mix"], feeds["conv_cache"] + 1, feeds["tra_cache"], feeds["inter_cache"]]


def highpassed(x):
    """What the denoiser's own 60 Hz high-pass makes of `x`."""
    from scipy.signal import sosfilt

    return np.round(sosfilt(denoise._HPF, x / 32768.0) * 32768).astype(int)


def tone(n, amp=8000):
    t = np.arange(n) / 16000
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.int16)


@pytest.mark.parametrize("chunk", [480, 1280, 256, 100])
def test_identity_model_reconstructs_the_input_delayed_by_lag(chunk):
    x = tone(16000)
    d = denoise.Denoiser(IdentitySession())
    out = np.concatenate([d.process(x[i:i + chunk]) for i in range(0, x.size, chunk)])
    assert out.size == x.size
    # Sqrt-Hann in and out at a half-window hop is perfect reconstruction
    # (after the first window has filled).
    lag = denoise.LAG
    assert np.abs(out[lag + 512:].astype(int) - highpassed(x)[512:x.size - lag]).max() <= 2
    assert not out[:lag - 256].any()


def test_state_carries_across_calls_and_reset_clears_it():
    s = IdentitySession()
    d = denoise.Denoiser(s)
    d.process(tone(1024))
    assert s.seen_caches[-1] == 3.0          # four hops, each fed the last one's cache
    d.reset()
    d.process(tone(256))
    assert s.seen_caches[-1] == 0.0


def test_process_bytes_keeps_the_frame_size():
    d = denoise.Denoiser(IdentitySession())
    assert len(d.process_bytes(tone(480).tobytes())) == 960
    assert d.process(np.zeros(0, dtype=np.int16)).size == 0


@pytest.fixture
def real_factory(monkeypatch):
    """Undo conftest's hermetic stub for make_denoiser itself (it's the
    function under test here) while keeping the session a fake."""
    monkeypatch.setattr(denoise, "make_denoiser", _REAL_MAKE_DENOISER)
    sessions = []

    def factory(path):
        sessions.append(path)
        return IdentitySession()

    monkeypatch.setattr(denoise, "_session_factory", factory)
    denoise._session.cache_clear()
    yield sessions
    denoise._session.cache_clear()


def test_make_denoiser_off_missing_and_present(real_factory, tmp_home):
    s = Settings()
    assert denoise.make_denoiser(s.model_copy(update={"noise_suppression": False})) is None
    assert denoise.make_denoiser(s) is None                      # model not on disk yet
    s.models_dir.mkdir(parents=True)
    (s.models_dir / models.GTCRN.name).write_bytes(b"onnx")
    a, b = denoise.make_denoiser(s), denoise.make_denoiser(s)
    assert isinstance(a, denoise.Denoiser) and a is not b
    assert real_factory == [str(s.models_dir / models.GTCRN.name)]   # one shared session


def test_prepare_model_never_raises(monkeypatch, tmp_home):
    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr(models, "ensure", boom)
    assert _REAL_PREPARE_MODEL(Settings()) is None
    assert _REAL_PREPARE_MODEL(Settings(noise_suppression=False)) is None


# -- models.ensure -------------------------------------------------------------------
def _fake_fetch(payload: bytes):
    calls = []

    def fetch(url, dest):
        calls.append(url)
        Path(dest).write_bytes(payload)

    return fetch, calls


def test_ensure_downloads_verifies_and_reuses(tmp_path):
    payload = b"model bytes"
    m = models.ModelFile("m.onnx", "https://example/m.onnx", hashlib.sha256(payload).hexdigest())
    fetch, calls = _fake_fetch(payload)
    p = _REAL_ENSURE(m, tmp_path / "models", fetch=fetch)
    assert p.read_bytes() == payload
    assert _REAL_ENSURE(m, tmp_path / "models", fetch=fetch) == p
    assert calls == ["https://example/m.onnx"]


def test_ensure_rejects_a_checksum_mismatch(tmp_path):
    m = models.ModelFile("m.onnx", "https://example/m.onnx", "0" * 64)
    fetch, _ = _fake_fetch(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        _REAL_ENSURE(m, tmp_path, fetch=fetch)
    assert list(tmp_path.iterdir()) == []


def test_pinned_models_have_sha256_pins():
    for m in (models.GTCRN, models.CAMPPLUS):
        assert len(m.sha256) == 64 and m.url.endswith(m.name)


def test_quiet_rooms_reach_the_network_amplified_and_loud_voices_do_not():
    """GTCRN suppresses far less at a real mic's room level; the AGC lifts
    quiet input towards AGC_TARGET (up to AGC_MAX_GAIN) and scales it back
    after, so the output level is the input's."""
    class Probe(IdentitySession):
        def __init__(self):
            super().__init__()
            self.peaks = []

        def run(self, outputs, feeds):
            self.peaks.append(float(np.abs(feeds["mix"]).max()))
            return super().run(outputs, feeds)

    quiet = (np.random.default_rng(0).standard_normal(8000) * 30).astype(np.int16)
    loud = tone(8000, amp=16000)
    p_quiet, p_loud = Probe(), Probe()
    out_quiet = denoise.Denoiser(p_quiet).process(quiet)
    denoise.Denoiser(p_loud).process(loud)
    raw_quiet_peak = float(np.abs(np.fft.rfft(highpassed(quiet)[-512:] / 32768.0 * denoise._WINDOW)).max())
    assert p_quiet.peaks[-1] > 20 * raw_quiet_peak                     # ~32x on the way in
    raw_loud_peak = float(np.abs(np.fft.rfft(highpassed(loud)[-512:] / 32768.0 * denoise._WINDOW)).max())
    assert p_loud.peaks[-1] == pytest.approx(raw_loud_peak, rel=0.05)   # unity for a loud voice
    lag = denoise.LAG
    assert np.abs(out_quiet[lag + 512:].astype(int) - highpassed(quiet)[512:quiet.size - lag]).max() <= 2


def test_rumble_under_60_hz_is_filtered_out():
    t = np.arange(16000) / 16000
    rumble = (4000 * np.sin(2 * np.pi * 15 * t)).astype(np.int16)
    out = denoise.Denoiser(IdentitySession()).process(rumble)
    assert np.abs(out[8000:]).max() < 0.1 * 4000


def test_prepare_model_fetches_and_loads_the_session(real_factory, monkeypatch, tmp_home):
    s = Settings()

    def ensure(m, d, **k):
        d.mkdir(parents=True, exist_ok=True)
        (d / m.name).write_bytes(b"onnx")
        return d / m.name

    monkeypatch.setattr(models, "ensure", ensure)
    assert _REAL_PREPARE_MODEL(s) == s.models_dir / models.GTCRN.name
    assert real_factory == [str(s.models_dir / models.GTCRN.name)]
    denoise.make_denoiser(s)
    assert len(real_factory) == 1                 # the first capture reuses it


def test_a_missing_model_is_fetched_in_the_background_once(real_factory, monkeypatch, tmp_home):
    kicked = []
    monkeypatch.setattr(denoise, "_start_fetch", lambda settings: kicked.append(settings))
    monkeypatch.setattr(denoise, "_fetching", set())
    s = Settings()
    assert denoise.make_denoiser(s) is None
    assert denoise.make_denoiser(s) is None
    assert len(kicked) == 1                  # never on the capture's own thread, never twice at once


def test_disable_turns_suppression_off_until_restart(real_factory, tmp_home):
    s = Settings()
    s.models_dir.mkdir(parents=True)
    (s.models_dir / models.GTCRN.name).write_bytes(b"onnx")
    assert denoise.make_denoiser(s) is not None
    denoise.disable()
    assert denoise.make_denoiser(s) is None


def test_missing_and_failed_to_load_are_logged_separately(real_factory, monkeypatch, tmp_home, caplog):
    monkeypatch.setattr(denoise, "_start_fetch", lambda settings: None)
    monkeypatch.setattr(denoise, "_missing_logged", set())
    monkeypatch.setattr(denoise, "_failed_logged", set())
    s = Settings()
    denoise.make_denoiser(s)
    s.models_dir.mkdir(parents=True)
    (s.models_dir / models.GTCRN.name).write_bytes(b"onnx")
    denoise._session.cache_clear()
    monkeypatch.setattr(denoise, "_session_factory", lambda p: (_ for _ in ()).throw(RuntimeError("corrupt")))
    assert denoise.make_denoiser(s) is None
    assert "is missing" in caplog.text and "failed to load" in caplog.text


def test_downloads_have_a_timeout(monkeypatch, tmp_path):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, n=-1):
            data, self.data = getattr(self, "data", b"abc"), b""
            return data

    def urlopen(url, timeout=None):
        seen["timeout"] = timeout
        return Resp()

    monkeypatch.setattr(models.urllib.request, "urlopen", urlopen)
    models.download("https://example/m.onnx", tmp_path / "m")
    assert seen["timeout"] == models.TIMEOUT_S
    assert (tmp_path / "m").read_bytes() == b"abc"
