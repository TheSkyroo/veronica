"""Input-volume floor guard: hermetic (fake osascript runner, fake get/set,
fake clock). Never touches osascript or CoreAudio."""
import asyncio
import logging
import subprocess

import pytest

from veronica.audio import input_level
from veronica.audio.input_level import InputLevelGuard, get_input_volume, run_periodic, set_input_volume


class _Proc:
    def __init__(self, stdout="", returncode=0):
        self.stdout, self.returncode = stdout, returncode


# -- get_input_volume ---------------------------------------------------------

def test_get_input_volume_parses_osascript():
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return _Proc("33\n")

    assert get_input_volume(run=run) == 33
    (cmd, kw), = calls
    assert cmd == ["osascript", "-e", "input volume of (get volume settings)"]
    assert kw["timeout"] == 5 and kw["capture_output"] and kw["text"]


@pytest.mark.parametrize("out", ["", "missing value", "abc", "3.5"])
def test_get_input_volume_none_on_non_int(out):
    assert get_input_volume(run=lambda *a, **k: _Proc(out)) is None


def test_get_input_volume_none_on_nonzero_exit():
    assert get_input_volume(run=lambda *a, **k: _Proc("33", returncode=1)) is None


def test_get_input_volume_none_on_exception():
    def boom(*a, **k):
        raise subprocess.TimeoutExpired("osascript", 5)
    assert get_input_volume(run=boom) is None


# -- set_input_volume ---------------------------------------------------------

@pytest.mark.parametrize("level, expected", [(85, 85), (-3, 0), (140, 100)])
def test_set_input_volume_clamps(level, expected):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return _Proc()

    assert set_input_volume(level, run=run) is True
    assert calls == [["osascript", "-e", f"set volume input volume {expected}"]]


def test_set_input_volume_false_on_failure():
    assert set_input_volume(85, run=lambda *a, **k: _Proc(returncode=1)) is False

    def boom(*a, **k):
        raise OSError("no osascript")
    assert set_input_volume(85, run=boom) is False


# -- InputLevelGuard ----------------------------------------------------------

class Fake:
    """Fake get/set/clock/name/on_corrected bundle for the guard."""
    def __init__(self, vol=33, name="AirPods", set_ok=True):
        self.vol, self.name, self.set_ok = vol, name, set_ok
        self.now = 1000.0
        self.gets, self.sets, self.corrected = 0, [], []

    def get(self):
        self.gets += 1
        return self.vol

    def set(self, level):
        self.sets.append(level)
        if self.set_ok:
            self.vol = level
        return self.set_ok

    def guard(self, floor=85, **kw):
        return InputLevelGuard(
            lambda: floor, get=self.get, set=self.set, device_name=lambda: self.name,
            on_corrected=lambda o, n, d: self.corrected.append((o, n, d)),
            clock=lambda: self.now, **kw,
        )


def test_guard_raises_below_floor_and_reports(caplog):
    f = Fake(vol=33)
    g = f.guard()
    with caplog.at_level(logging.INFO, logger="veronica.audio"):
        assert g.check() == 85
    assert f.sets == [85]
    assert f.corrected == [(33, 85, "AirPods")]
    assert "input volume 33 → 85 (AirPods)" in caplog.text


def test_guard_unknown_device_name_in_log(caplog):
    f = Fake(vol=27, name=None)
    g = f.guard()
    with caplog.at_level(logging.INFO, logger="veronica.audio"):
        g.check()
    assert "input volume 27 → 85 (unknown input)" in caplog.text
    assert f.corrected == [(27, 85, "unknown input")]


def test_guard_never_lowers():
    f = Fake(vol=100)
    g = f.guard(floor=85)
    assert g.check() is None
    assert f.sets == [] and f.corrected == []


def test_guard_at_floor_is_noop():
    f = Fake(vol=85)
    assert f.guard(floor=85).check() is None
    assert f.sets == []


def test_guard_floor_zero_disabled_does_not_even_read():
    f = Fake(vol=5)
    assert f.guard(floor=0).check() is None
    assert f.gets == 0 and f.sets == []


def test_guard_floor_read_each_check():
    f = Fake(vol=50)
    floor = {"v": 0}
    g = InputLevelGuard(lambda: floor["v"], get=f.get, set=f.set, device_name=lambda: None,
                        clock=lambda: f.now)
    assert g.check(force=True) is None
    floor["v"] = 80
    assert g.check(force=True) == 80


def test_guard_throttles_by_interval():
    f = Fake(vol=33)
    g = f.guard(interval_s=60.0)
    assert g.check() == 85
    f.vol = 33                       # something lowered it again
    f.now += 30
    assert g.check() is None         # throttled
    assert f.gets == 1
    f.now += 31
    assert g.check() == 85
    assert f.gets == 2


def test_guard_force_bypasses_throttle():
    f = Fake(vol=33)
    g = f.guard(interval_s=60.0)
    g.check()
    f.vol = 33
    assert g.check(force=True) == 85
    assert f.sets == [85, 85]


def test_guard_force_resets_throttle_window():
    f = Fake(vol=33)
    g = f.guard(interval_s=60.0)
    f.now += 100
    g.check(force=True)
    f.vol = 33
    f.now += 1
    assert g.check() is None


def test_guard_get_failure_returns_none():
    g = InputLevelGuard(lambda: 85, get=lambda: None, set=lambda lvl: True,
                        device_name=lambda: "x", clock=lambda: 0.0)
    assert g.check() is None


def test_guard_set_failure_no_report(caplog):
    f = Fake(vol=33, set_ok=False)
    g = f.guard()
    with caplog.at_level(logging.INFO, logger="veronica.audio"):
        assert g.check() is None
    assert f.corrected == []
    assert "input volume 33 → 85 (AirPods)" not in caplog.text
    assert "could not raise input volume 33 → 85" in caplog.text


def test_guard_survives_bad_device_name_and_callback(caplog):
    def bad_name():
        raise RuntimeError("no coreaudio")

    def bad_cb(o, n, d):
        raise RuntimeError("hud gone")
    f = Fake(vol=33)
    g = InputLevelGuard(lambda: 85, get=f.get, set=f.set, device_name=bad_name, on_corrected=bad_cb,
                        clock=lambda: f.now)
    with caplog.at_level(logging.INFO, logger="veronica.audio"):
        assert g.check() == 85
    assert "unknown input" in caplog.text


def test_guard_default_wiring_uses_module_functions():
    g = InputLevelGuard(lambda: 85)
    assert g._get is input_level.get_input_volume
    assert g._set is input_level.set_input_volume
    from veronica.audio import devices
    assert g._device_name is devices.default_input_name


# -- run_periodic ------------------------------------------------------------

class _SpyGuard:
    def __init__(self):
        self.calls = []
        self.interval_s = 0.01

    def check(self, force=False):
        self.calls.append(force)


async def test_run_periodic_forced_once_then_periodic():
    g = _SpyGuard()
    stop = asyncio.Event()
    task = asyncio.ensure_future(run_periodic(g, stop))
    for _ in range(200):
        await asyncio.sleep(0.005)
        if len(g.calls) >= 3:
            break
    stop.set()
    await asyncio.wait_for(task, 1)
    assert g.calls[0] is True
    assert len(g.calls) >= 3 and all(c is False for c in g.calls[1:])


async def test_run_periodic_stops_promptly_and_survives_check_errors(caplog):
    class Bad(_SpyGuard):
        def check(self, force=False):
            super().check(force)
            raise RuntimeError("boom")
    g = Bad()
    g.interval_s = 10.0
    stop = asyncio.Event()
    with caplog.at_level(logging.WARNING, logger="veronica.audio"):
        task = asyncio.ensure_future(run_periodic(g, stop))
        await asyncio.sleep(0.02)
        stop.set()
        await asyncio.wait_for(task, 1)
    assert g.calls == [True]
    assert "input volume check failed" in caplog.text
