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
        return self.n.to_bytes(4, "little") * (n // 2), False   # n int16 frames = 2n bytes


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
    assert got[:3] == [(i).to_bytes(4, "little") * 640 for i in (1, 2, 3)]
    assert (1).to_bytes(4, "little") * 640 in got[3:]


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


def test_a_device_that_overflows_every_chunk_does_not_flood_the_log(fake_sd, monkeypatch, caplog):
    """AirPods' hands-free profile overflowed on every chunk: one warning per
    chunk (~15 a second) rotated the whole log away within hours. It is
    summarised at most once per OVERFLOW_LOG_EVERY_S instead."""
    monkeypatch.setattr(FakeStream, "read",
                        lambda self, n: (b"\0" * (2 * n), True))
    with caplog.at_level(logging.WARNING, logger="veronica.audio"):
        gen = mic.mic_frames(Settings(), 1280, "wake")
        _drain(gen, 200)
        gen.close()
    lines = [r for r in caplog.records if "mic overflow" in r.getMessage()]
    assert len(lines) == 1


# -- PortAudio ring size, stalls, persistent overflow, dead consumers -------

class PolledStream(FakeStream):
    """A stream shaped like sounddevice's: `read_available` says how many
    frames are buffered, and `read()` never has to block. `dead_after` reads
    in, it stops delivering for good (what PortAudio's CoreAudio input
    callback does after AudioUnitRender fails: it stops the unit, and a
    blocking read then waits forever)."""

    def __init__(self, registry, chunk=1280, dead_after=None, overflow=False, **kw):
        super().__init__(registry, **kw)
        self.chunk = chunk
        self.dead_after = dead_after
        self.overflow = overflow

    @property
    def dead(self):
        return self.dead_after is not None and self.n >= self.dead_after

    @property
    def read_available(self):
        return 0 if self.dead else self.chunk

    @property
    def active(self):
        return not self.dead

    def read(self, n):
        assert not self.dead, "read() on a dead stream would block forever"
        assert n <= self.read_available, "read() larger than what's buffered would block"
        self.n += 1
        time.sleep(0.0005)
        return self.n.to_bytes(4, "little") * (n // 2), self.overflow


def _polled(fake_sd, *streams_kw):
    """Make fake_sd open PolledStreams, the i-th with streams_kw[i] (the last
    repeats)."""
    def open_(**kw):
        fake_sd.calls.append("open")
        i = min(len(fake_sd.streams), len(streams_kw) - 1)
        return PolledStream(fake_sd.streams, **{**streams_kw[i], **kw})
    fake_sd.RawInputStream = open_


def test_stream_latency_keeps_portaudios_ring_at_least_two_chunks(fake_sd):
    """PortAudio's CoreAudio blocking-read ring is sized from the suggested
    latency and the device's IO buffer, never from our blocksize
    (computeRingBufferSize). A Bluetooth hands-free mic whose snapshotted
    'high' latency is ~30 ms got a 1024-frame ring under a 1280-frame block:
    every callback overflowed and 20% of the audio was dropped. The latency
    we ask for must make the ring hold at least two chunks."""
    fake_sd.query_devices = lambda kind=None: {"default_high_input_latency": 0.02}
    gen = mic.mic_frames(Settings(), 1280, "wake")
    next(gen)
    gen.close()
    # ring >= 2 * latency * rate  ->  latency >= chunk / rate
    assert fake_sd.streams[0].kw["latency"] >= 1280 / 16000


def test_stream_latency_never_lowers_the_devices_own_high_latency(fake_sd):
    fake_sd.query_devices = lambda kind=None: {"default_high_input_latency": 0.3}
    gen = mic.mic_frames(Settings(), 1280, "wake")
    next(gen)
    gen.close()
    assert fake_sd.streams[0].kw["latency"] == pytest.approx(0.3)


def test_reader_reopens_a_stream_that_stopped_delivering(fake_sd, monkeypatch, caplog):
    """The wedge behind two days of a deaf wake word: the device went away
    (AirPods out, sleep), PortAudio stopped the input unit, and read() waited
    forever — holding refresh_lock, so no device refresh could run either.
    The reader must notice no audio arriving, say so, and reopen."""
    monkeypatch.setattr(mic, "STALL_S", 0.1)
    monkeypatch.setattr(devices, "default_input_name", lambda *a, **k: "AirPods - Find My")
    _polled(fake_sd, {"dead_after": 3}, {})
    with caplog.at_level(logging.INFO, logger="veronica.audio"):
        gen = mic.mic_frames(Settings(), 1280, "wake")
        got = _drain(gen, 6)
        gen.close()
    assert len(got) == 6                                  # frames resumed on the new stream
    assert fake_sd.calls == ["open", "open"]              # a plain reopen first
    assert "no audio from AirPods - Find My" in caplog.text
    assert "reopening the stream" in caplog.text


def test_a_second_stall_in_a_row_reinitialises_portaudio(fake_sd, monkeypatch, caplog):
    """If a fresh stream on the same device table is dead too, the table
    itself is stale (the device was re-created under a new id): re-initialise
    PortAudio before opening again."""
    monkeypatch.setattr(mic, "STALL_S", 0.05)
    monkeypatch.setattr(devices, "default_input_name", lambda *a, **k: "Rockerz 480")
    _polled(fake_sd, {"dead_after": 2}, {"dead_after": 0}, {})
    with caplog.at_level(logging.INFO, logger="veronica.audio"):
        gen = mic.mic_frames(Settings(), 1280, "wake")
        got = _drain(gen, 5)
        gen.close()
    assert len(got) == 5
    assert fake_sd.calls == ["open", "open", "terminate", "initialize", "open"]
    assert "re-initialising PortAudio" in caplog.text


def test_a_stalled_reader_does_not_hold_the_refresh_lock(fake_sd, monkeypatch):
    """Waiting for audio must not happen under devices.refresh_lock: the
    Player's output-stream open and every device refresh take it."""
    monkeypatch.setattr(mic, "STALL_S", 30)
    _polled(fake_sd, {"dead_after": 1})
    gen = mic.mic_frames(Settings(), 1280, "wake", on_backlog=lambda f: None)
    next(gen)
    time.sleep(0.05)                                      # reader is now waiting on a dead stream
    got = devices.refresh_lock.acquire(timeout=0.5)
    assert got, "reader held refresh_lock while waiting for audio"
    devices.refresh_lock.release()
    gen.close()


def test_persistent_overflow_reopens_with_a_bigger_buffer_and_says_so(fake_sd, monkeypatch, caplog):
    """Whatever makes a device overflow on every read, spinning on it for
    days is wrong: after OVERFLOW_HEAL_S of it the reader reopens with twice
    the latency (a bigger ring), naming the device, until MAX_LATENCY_S."""
    monkeypatch.setattr(mic, "OVERFLOW_HEAL_S", 0.05)
    monkeypatch.setattr(mic, "MAX_LATENCY_S", 0.4)
    monkeypatch.setattr(devices, "default_input_name", lambda *a, **k: "AirPods - Find My")
    fake_sd.query_devices = lambda kind=None: {"default_high_input_latency": 0.0}
    _polled(fake_sd, {"overflow": True})
    with caplog.at_level(logging.WARNING, logger="veronica.audio"):
        gen = mic.mic_frames(Settings(), 1280, "wake")
        deadline = time.monotonic() + 3
        while "not reopening again" not in caplog.text and time.monotonic() < deadline:
            _drain(gen, 1)
        gen.close()
    lat = [s.kw["latency"] for s in fake_sd.streams]
    assert lat == pytest.approx([0.08, 0.16, 0.32, 0.4])
    assert "overflowing on" in caplog.text and "AirPods - Find My" in caplog.text
    assert "reopening with latency 0.16s" in caplog.text
    assert "not reopening again" in caplog.text


def test_occasional_overflow_does_not_reopen(fake_sd, monkeypatch):
    monkeypatch.setattr(mic, "OVERFLOW_HEAL_S", 0.02)

    class Rare(PolledStream):
        def read(self, n):
            data, _ = super().read(n)
            return data, self.n % 10 == 0

    fake_sd.RawInputStream = lambda **kw: (fake_sd.calls.append("open"), Rare(fake_sd.streams, **kw))[1]
    gen = mic.mic_frames(Settings(), 1280, "wake")
    _drain(gen, 300)
    gen.close()
    assert fake_sd.calls == ["open"]


def test_a_consumer_that_stops_draining_never_blocks_the_reader(fake_sd, monkeypatch, caplog):
    """The queue is bounded and drops its *oldest* frames when full: a wake
    loop that stopped pulling can cost memory for at most QUEUE_MAX frames,
    and the reader keeps reading (and healing) regardless."""
    monkeypatch.setattr(mic, "QUEUE_MAX", 5)
    _polled(fake_sd, {})
    with caplog.at_level(logging.WARNING, logger="veronica.audio"):
        gen = mic.mic_frames(Settings(), 1280, "wake")
        first = next(gen)
        time.sleep(0.1)                                   # consumer stalls
        n_read = fake_sd.streams[0].n
        assert n_read > 20                                # reader kept going
        got = _drain(gen, 5)
        gen.close()
    assert first == (1).to_bytes(4, "little") * 640
    # what's left is the newest audio, not the oldest
    assert int.from_bytes(got[-1][:4], "little") > 20
    assert "not draining" in caplog.text


def test_frames_are_whole_chunks_whatever_the_device_hands_over(fake_sd):
    """read_available can be any size (a Bluetooth mic's IO buffer is a few
    hundred frames); the queue only ever carries whole `chunk`-frame frames."""
    _polled(fake_sd, {"chunk": 300})
    gen = mic.mic_frames(Settings(), 1280, "wake")
    got = _drain(gen, 4)
    gen.close()
    assert {len(f) for f in got} == {2560}
