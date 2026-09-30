import pytest


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VERONICA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_audio_devices():
    from veronica.audio import devices

    devices.reset()
    yield
    devices.reset()


@pytest.fixture(autouse=True)
def _no_real_audio_models(request, monkeypatch):
    """Hermetic by default: the noise suppressor and speaker model are only
    real in `live` tests, never because the model file happens to be in
    ~/.veronica/models, and nothing is ever downloaded. Tests that want one
    patch in a fake."""
    from veronica.audio import denoise

    monkeypatch.setattr(denoise, "_disabled", False)
    if request.node.get_closest_marker("live") is not None:
        return
    from veronica.audio import models, speaker

    def no_models(*a, **k):
        raise RuntimeError("hermetic test: no real models, no downloads")

    monkeypatch.setattr(denoise, "make_denoiser", lambda settings: None)
    monkeypatch.setattr(denoise, "prepare_model", lambda settings: None)
    monkeypatch.setattr(denoise, "_start_fetch", lambda settings: None)
    monkeypatch.setattr(models, "ensure", no_models)
    monkeypatch.setattr(speaker.SpeakerModel, "_session_factory", staticmethod(no_models))


@pytest.fixture(autouse=True)
def _fake_contacts(monkeypatch):
    """Hermetic: the real Contacts framework is never read. One contact
    ("Priya Shah") unless a test swaps the list; returns it so tests can."""
    from veronica.tools import pim

    book = [("Priya Shah", ["+91 98765 43210"])]

    def search(name):
        n = name.casefold()
        return [c for c in book if any(part.startswith(n) or c[0].casefold().startswith(n)
                                        for part in c[0].casefold().split())]

    monkeypatch.setattr(pim, "_contacts_search", search)
    monkeypatch.setattr(pim, "_resolved", {})
    return book


_EXIT_STATUS = 0


def pytest_sessionfinish(session, exitstatus):
    global _EXIT_STATUS
    _EXIT_STATUS = int(exitstatus)
    # `import sounddevice` initialises PortAudio for the whole test process;
    # terminate it explicitly while the interpreter is still healthy.
    import contextlib
    import sys

    if "sounddevice" in sys.modules:
        with contextlib.suppress(Exception):
            sys.modules["sounddevice"]._terminate()


def pytest_unconfigure(config):
    # Native teardown of CoreAudio's HAL client (touched via ctypes in
    # veronica.audio.devices and by PortAudio) intermittently aborts the
    # process with "recursive_mutex lock failed" (exit 134) *after* every
    # test has passed and the summary has printed. Nothing in the test
    # results depends on that teardown, so leave via os._exit with pytest's
    # real status instead of letting static destructors race.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS)
