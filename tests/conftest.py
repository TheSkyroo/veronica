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
