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
