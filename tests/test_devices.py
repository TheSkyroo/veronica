import logging
import threading
import time

from veronica.audio import devices


# -- default_input_id -------------------------------------------------------

def test_default_input_id_reads_coreaudio_property():
    seen = {}

    def fake_get(obj, addr_ref, qual_size, qual, size_ref, data_ref):
        addr = addr_ref._obj
        seen["obj"] = obj
        seen["selector"] = addr.mSelector.to_bytes(4, "big")
        seen["scope"] = addr.mScope.to_bytes(4, "big")
        seen["element"] = addr.mElement
        seen["size"] = size_ref._obj.value
        data_ref._obj.value = 89
        return 0

    assert devices.default_input_id(get=fake_get) == 89
    assert seen == {"obj": 1, "selector": b"dIn ", "scope": b"glob", "element": 0, "size": 4}


def test_default_input_id_none_on_nonzero_status():
    assert devices.default_input_id(get=lambda *a: -1) is None


def test_default_input_id_none_on_exception():
    def boom(*a):
        raise OSError("no coreaudio")
    assert devices.default_input_id(get=boom) is None


def test_default_input_id_none_when_library_missing(monkeypatch):
    monkeypatch.setattr(devices, "_coreaudio_getter", lambda: None)
    assert devices.default_input_id() is None


# -- refresh_portaudio ------------------------------------------------------

class FakeSD:
    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def _terminate(self):
        self.calls.append("terminate")
        if "terminate" in self.fail:
            raise RuntimeError("already terminated")

    def _initialize(self):
        self.calls.append("initialize")
        if "initialize" in self.fail:
            raise RuntimeError("init failed")


def test_refresh_portaudio_calls_before_then_reinit(monkeypatch):
    sd = FakeSD()
    monkeypatch.setattr(devices, "sd", sd)
    order = []
    devices.refresh_portaudio(before=lambda: order.append("before") or sd.calls.append("before"))
    assert sd.calls == ["before", "terminate", "initialize"]


def test_refresh_portaudio_without_before(monkeypatch):
    sd = FakeSD()
    monkeypatch.setattr(devices, "sd", sd)
    devices.refresh_portaudio()
    assert sd.calls == ["terminate", "initialize"]


def test_refresh_portaudio_swallows_errors(monkeypatch, caplog):
    sd = FakeSD(fail={"terminate", "initialize"})
    monkeypatch.setattr(devices, "sd", sd)

    def bad_before():
        raise RuntimeError("player exploded")

    with caplog.at_level(logging.WARNING, logger="veronica.audio"):
        devices.refresh_portaudio(before=bad_before)
    assert sd.calls == ["terminate", "initialize"]  # every step still attempted
    assert "player exploded" in caplog.text or "init failed" in caplog.text


def test_refresh_portaudio_waits_for_stream_opens_to_finish(monkeypatch):
    """Stream opens (Player/Recorder) run under refresh_lock so PortAudio is
    never terminated between an open starting and the stream being live."""
    sd = FakeSD()
    monkeypatch.setattr(devices, "sd", sd)
    devices.refresh_lock.acquire()          # simulate an open in progress
    t = threading.Thread(target=devices.refresh_portaudio)
    t.start()
    time.sleep(0.02)
    assert sd.calls == []                   # blocked
    devices.refresh_lock.release()
    t.join(1)
    assert sd.calls == ["terminate", "initialize"]


# -- InputWatch -------------------------------------------------------------

def test_input_watch_fires_once_per_change_not_on_first_observation():
    ids = iter([5, 5, 7, 7, 7, None, None, 5])
    changes = []
    w = devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids), on_change=lambda a, b: changes.append((a, b)))
    results = [w.check(now=float(i)) for i in range(8)]
    assert changes == [(5, 7), (7, None), (None, 5)]
    assert results == [False, False, True, False, False, True, False, True]
    assert w.last == 5


def test_input_watch_rate_limits_polls_by_poll_s():
    polls = {"n": 0}

    def get():
        polls["n"] += 1
        return 1

    w = devices.InputWatch(poll_s=2.0, get_id=get)
    w.check(now=10.0)
    w.check(now=10.5)
    w.check(now=11.9)
    assert polls["n"] == 1
    w.check(now=12.0)
    assert polls["n"] == 2


def test_input_watch_default_getter_is_default_input_id(monkeypatch):
    monkeypatch.setattr(devices, "default_input_id", lambda: 3)
    w = devices.InputWatch(poll_s=0.0)
    assert w.check(now=0.0) is False
    assert w.last == 3


def test_input_watch_survives_getter_exception():
    def boom():
        raise RuntimeError("nope")
    w = devices.InputWatch(poll_s=0.0, get_id=boom)
    assert w.check(now=0.0) is False
    assert w.last is None


def test_busy_default_is_false():
    assert devices.busy() is False
