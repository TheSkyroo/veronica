import logging
import threading
import time

import pytest

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


def test_refresh_portaudio_swallows_before_and_terminate_errors(monkeypatch, caplog):
    sd = FakeSD(fail={"terminate"})
    monkeypatch.setattr(devices, "sd", sd)

    def bad_before():
        raise RuntimeError("player exploded")

    with caplog.at_level(logging.WARNING, logger="veronica.audio"):
        devices.refresh_portaudio(before=bad_before)
    assert sd.calls == ["terminate", "initialize"]  # every step still attempted
    assert "player exploded" in caplog.text
    assert "already terminated" in caplog.text


def test_refresh_portaudio_raises_when_initialize_fails(monkeypatch, caplog):
    """PortAudio failing to come back is fatal for audio: log at ERROR and
    raise so the caller (mic reader -> wake.wait -> orchestrator backoff)
    can surface it, instead of silently carrying on with no PortAudio."""
    sd = FakeSD(fail={"initialize"})
    monkeypatch.setattr(devices, "sd", sd)
    devices.observe(5)
    devices.observe(7)
    with caplog.at_level(logging.ERROR, logger="veronica.audio"):
        with pytest.raises(RuntimeError, match="init failed"):
            devices.refresh_portaudio()
    assert sd.calls == ["terminate", "initialize"]
    assert any(r.levelno == logging.ERROR and "PortAudio initialize failed" in r.message for r in caplog.records)
    # state still moves on: the streams are dead either way (Pa_Terminate ran)
    assert devices.generation == 1


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
    # None after a valid baseline is "unknown, keep last", not a change
    assert changes == [(5, 7), (7, 5)]
    assert results == [False, False, True, False, False, False, False, True]
    assert w.last == 5
    assert devices.initialised_for == 5 and devices.pending is False   # back on the baseline device


def test_observe_none_after_valid_baseline_keeps_last():
    assert devices.observe(5) is False
    assert devices.observe(7) is True and devices.pending is True
    assert devices.observe(None) is False
    assert devices.last_input_id == 7 and devices.pending is True     # refresh still owed
    assert devices.observe(7) is False
    assert devices.observe(None) is False and devices.last_input_id == 7


def test_observe_none_baseline_then_real_id_is_a_change():
    # CoreAudio unavailable at first (None baseline): a later real id counts
    assert devices.observe(None) is False
    assert devices.observe(5) is True and devices.pending is True


def test_snapshot_baseline_at_import(monkeypatch):
    """The baseline is taken at import (right after sounddevice initialised
    PortAudio) so a device change during warmup — before any mic reader
    runs — is still seen by the first InputWatch."""
    devices._snapshot_baseline(lambda: 9)
    assert devices._baselined is True and devices.initialised_for == 9 and devices.pending is False
    assert devices.InputWatch(poll_s=0.0, get_id=lambda: 11).check(now=0.0) is True
    assert devices.pending is True


def test_snapshot_baseline_is_guarded():
    def boom():
        raise RuntimeError("no CoreAudio")

    devices._snapshot_baseline(boom)          # must not raise
    assert devices._baselined is False        # nothing observed; first poll baselines instead


def test_module_import_snapshots_baseline():
    # Verified in a fresh interpreter (reloading the module in-process would
    # re-create InputWatch and break isinstance checks elsewhere).
    import subprocess
    import sys

    code = "from veronica.audio import devices; print(devices._baselined)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "True"


def test_pending_tracks_difference_from_initialised_device():
    ids = iter([5, 7, 7])
    w = devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids, 7))
    w.check(now=0.0)
    assert devices.pending is False
    w.check(now=1.0)
    assert devices.pending is True and devices.last_input_id == 7 and devices.initialised_for == 5
    w.check(now=2.0)                        # same device again: still owed, not a new change
    assert devices.pending is True


def test_baseline_is_module_wide_not_per_watch():
    """A fresh InputWatch must not re-baseline: a change between two readers
    is still a change."""
    devices.InputWatch(poll_s=0.0, get_id=lambda: 5).check(now=0.0)
    assert devices.InputWatch(poll_s=0.0, get_id=lambda: 7).check(now=0.0) is True
    assert devices.pending is True


def test_refresh_portaudio_adopts_last_seen_device_and_bumps_generation(monkeypatch):
    monkeypatch.setattr(devices, "sd", FakeSD())
    ids = iter([5, 7])
    devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids, 7)).check(now=0.0)
    devices.InputWatch(poll_s=0.0, get_id=lambda: next(ids, 7)).check(now=0.0)
    assert devices.pending is True and devices.generation == 0
    devices.refresh_portaudio()
    assert devices.pending is False and devices.initialised_for == 7 and devices.generation == 1


def test_register_busy_and_reset():
    assert devices.busy() is False
    flag = {"v": True}
    devices.register_busy(lambda: flag["v"])
    assert devices.busy() is True
    flag["v"] = False
    assert devices.busy() is False
    devices.register_busy(lambda: True)
    devices.reset_busy()
    assert devices.busy() is False


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
