import logging
import threading
import time

import pytest

from veronica.audio import devices, mic
from veronica.config import Settings


class FakeStream:
    def __init__(self, registry, **kw):
        self.kw = kw
        self.n = 0
        self.closed = False
        registry.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False

    def close(self):
        self.closed = True

    def read(self, n):
        self.n += 1
        time.sleep(0.0005)
        return self.n.to_bytes(4, "little") * (n // 4), False


class FakeSD:
    def __init__(self):
        self.streams = []
        self.calls = []

    def RawInputStream(self, **kw):
        self.calls.append("open")
        return FakeStream(self.streams, **kw)

    def _terminate(self):
        self.calls.append("terminate")

    def _initialize(self):
        self.calls.append("initialize")


@pytest.fixture
def fake_sd(monkeypatch):
    sd = FakeSD()
    monkeypatch.setattr(mic, "sd", sd)
    monkeypatch.setattr(devices, "sd", sd)
    monkeypatch.setattr(devices, "busy", lambda: False)
    return sd


def _drain(gen, n):
    return [next(gen) for _ in range(n)]


def test_mic_frames_without_watch_opens_once_and_stops_thread(fake_sd):
    gen = mic.mic_frames(Settings(), 1280, "wake")
    got = _drain(gen, 5)
    assert len(got) == 5 and got[0] != got[1]
    gen.close()
    time.sleep(0.02)
    assert fake_sd.calls == ["open"]
    assert fake_sd.streams[0].closed
    assert not any(t.name == "wake-mic" and t.is_alive() for t in threading.enumerate())


def test_mic_frames_reopens_stream_when_input_device_changes(fake_sd, caplog):
    ids = iter([1, 1, 1, 2])            # baseline at start, then per frame
    watch = devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids, 2))
    before = []
    with caplog.at_level(logging.INFO, logger="veronica.audio"):
        gen = mic.mic_frames(Settings(), 1280, "wake", watch=watch, before_refresh=lambda: before.append(1))
        got = _drain(gen, 8)
        gen.close()
    time.sleep(0.02)
    assert fake_sd.calls == ["open", "terminate", "initialize", "open"]
    assert fake_sd.streams[0].closed
    assert before == [1]
    assert "input device changed (1 -> 2); reopening mic" in caplog.text
    assert devices.pending is False and devices.initialised_for == 2
    # frames keep flowing through the same generator: the new stream's
    # counter restarts at 1, and nothing was dropped from the old one.
    assert got[:3] == [(i).to_bytes(4, "little") * 320 for i in (1, 2, 3)]
    assert (1).to_bytes(4, "little") * 320 in got[3:]


def test_mic_frames_defers_reopen_while_capture_in_flight(fake_sd, monkeypatch, caplog):
    busy = {"v": True}
    monkeypatch.setattr(devices, "busy", lambda: busy["v"])
    ids = iter([1, 2])
    watch = devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids, 2))
    with caplog.at_level(logging.DEBUG, logger="veronica.audio"):
        gen = mic.mic_frames(Settings(), 1280, "wake", watch=watch)
        _drain(gen, 6)
        assert fake_sd.calls == ["open"]                 # change seen but not acted on
        assert devices.pending is True
        assert caplog.text.count("deferring device refresh: capture in flight") == 1   # once, not per frame
        busy["v"] = False
        _drain(gen, 6)
        gen.close()
    time.sleep(0.02)
    assert fake_sd.calls == ["open", "terminate", "initialize", "open"]


def test_mic_frames_ends_when_reopen_fails(fake_sd, caplog):
    ids = iter([1, 2])
    watch = devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids, 2))
    opens = {"n": 0}
    real = fake_sd.RawInputStream

    def flaky(**kw):
        opens["n"] += 1
        if opens["n"] == 2:
            raise RuntimeError("device gone")
        return real(**kw)

    fake_sd.RawInputStream = flaky
    frames = []
    with caplog.at_level(logging.ERROR, logger="veronica.audio"):
        gen = mic.mic_frames(Settings(), 1280, "wake", watch=watch)
        # the reader's death is re-raised to the consumer (so the wake
        # engine's wait() fails and the orchestrator backs off) rather than
        # the generator quietly ending
        with pytest.raises(RuntimeError, match="device gone"):
            for f in gen:
                frames.append(f)
    assert len(frames) >= 1
    assert "wake mic reader died" in caplog.text


def test_mic_frames_raises_when_first_open_fails(fake_sd):
    def broken(**kw):
        raise OSError("no input device")

    fake_sd.RawInputStream = broken
    gen = mic.mic_frames(Settings(), 1280, "wake")
    with pytest.raises(OSError, match="no input device"):
        next(gen)
    time.sleep(0.02)
    assert not any(t.name == "wake-mic" and t.is_alive() for t in threading.enumerate())


def test_mic_frames_reports_backlog(fake_sd):
    seen = []
    gen = mic.mic_frames(Settings(), 1280, "wake", on_backlog=seen.append)
    next(gen)
    assert len(seen) == 1 and callable(seen[0])
    time.sleep(0.02)
    assert seen[0]() > 0            # reader ran ahead while we stalled
    gen.close()
    assert len(seen) == 2 and seen[1]() == 0


def test_change_seen_with_no_reader_refreshes_before_first_open(fake_sd, caplog):
    """The watch state is module-level: a device change observed while no
    reader is alive (between wait() calls) is applied by the next reader
    before it opens its first stream."""
    ids = iter([1, 2])
    devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids, 2)).check(now=0.0)   # baseline 1
    assert devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids, 2)).check(now=1.0) is True
    assert devices.pending is True
    with caplog.at_level(logging.INFO, logger="veronica.audio"):
        gen = mic.mic_frames(Settings(), 1280, "wake", watch=devices.InputWatch(poll_s=0.0, get_id=lambda: 2))
        _drain(gen, 3)
        gen.close()
    time.sleep(0.02)
    assert fake_sd.calls == ["terminate", "initialize", "open"]
    assert "input device changed (1 -> 2); reopening mic" in caplog.text
    assert devices.pending is False and devices.initialised_for == 2


def test_deferred_refresh_survives_reader_stop_and_start(fake_sd, monkeypatch):
    busy = {"v": True}
    monkeypatch.setattr(devices, "busy", lambda: busy["v"])
    ids = iter([1, 2])
    gen = mic.mic_frames(Settings(), 1280, "wake", watch=devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids, 2)))
    _drain(gen, 4)
    gen.close()                                  # reader stops with the refresh still owed
    time.sleep(0.02)
    assert fake_sd.calls == ["open"] and devices.pending is True

    # next reader starts while still busy: opens anyway, keeps pending...
    gen = mic.mic_frames(Settings(), 1280, "wake", watch=devices.InputWatch(poll_s=0.0, get_id=lambda: 2))
    _drain(gen, 3)
    assert fake_sd.calls == ["open", "open"] and devices.pending is True
    # ...and refreshes at the first non-busy check.
    busy["v"] = False
    _drain(gen, 3)
    gen.close()
    time.sleep(0.02)
    assert fake_sd.calls == ["open", "open", "terminate", "initialize", "open"]
    assert devices.pending is False


def test_switching_back_to_original_device_cancels_pending(fake_sd, monkeypatch):
    monkeypatch.setattr(devices, "busy", lambda: True)
    ids = iter([1, 2, 1])
    gen = mic.mic_frames(Settings(), 1280, "wake", watch=devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids, 1)))
    _drain(gen, 4)
    gen.close()
    time.sleep(0.02)
    assert devices.pending is False and fake_sd.calls == ["open"]


def test_reader_reopens_without_closing_when_portaudio_refreshed_elsewhere(fake_sd, caplog):
    """A refresh from another thread (Player's open-retry path) bumps the
    generation; Pa_Terminate already closed the reader's stream, so the
    reader drops the dead handle (no close()) and opens a fresh one."""
    with caplog.at_level(logging.INFO, logger="veronica.audio"):
        gen = mic.mic_frames(Settings(), 1280, "wake")
        _drain(gen, 2)
        devices.refresh_portaudio()
        # the reader runs ahead of the consumer (frames queue up), so drain
        # until it has actually noticed the refresh and reopened
        deadline = time.monotonic() + 2
        while fake_sd.calls.count("open") < 2 and time.monotonic() < deadline:
            _drain(gen, 1)
        gen.close()
    time.sleep(0.02)
    assert fake_sd.calls == ["open", "terminate", "initialize", "open"]
    assert fake_sd.streams[0].closed is False        # dead handle dropped, not closed
    assert fake_sd.streams[1].closed is True
    assert "wake mic stream invalidated by PortAudio refresh; reopening" in caplog.text
