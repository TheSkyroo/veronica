"""The UI thread (veronica.ui.dispatch): ordered calls, timers, inline when
already on it. Real threads, short waits."""
import threading

from veronica.ui.dispatch import Dispatcher


def _wait(event: threading.Event) -> bool:
    return event.wait(2)


def test_calls_run_in_order_on_one_thread():
    d = Dispatcher(name="test-ui")
    seen, done = [], threading.Event()
    for i in range(5):
        d.call(lambda i=i: seen.append((i, threading.current_thread().name)))
    d.call(done.set)
    assert _wait(done)
    assert [i for i, _ in seen] == [0, 1, 2, 3, 4]
    assert {name for _, name in seen} == {"test-ui"}
    d.stop()


def test_is_ui_thread_only_on_the_worker():
    d = Dispatcher()
    on, done = [], threading.Event()
    d.call(lambda: (on.append(d.is_ui_thread()), done.set()))
    assert _wait(done)
    assert on == [True] and not d.is_ui_thread()
    d.stop()


def test_a_failing_call_does_not_kill_the_thread(caplog):
    d = Dispatcher()
    done = threading.Event()
    d.call(lambda: 1 / 0)
    d.call(done.set)
    assert _wait(done)
    assert "UI callback failed" in caplog.text
    d.stop()


def test_every_repeats_until_cancelled():
    d = Dispatcher()
    ticks, three = [], threading.Event()

    def tick():
        ticks.append(1)
        if len(ticks) == 3:
            three.set()

    timer = d.every(0.01, tick)
    assert _wait(three)
    timer.cancel()
    n = len(ticks)
    stopped = threading.Event()
    d.every(0.05, stopped.set)
    assert _wait(stopped)
    assert len(ticks) <= n + 1     # at most one already-due tick after cancel
    d.stop()


def test_stop_drops_later_calls():
    d = Dispatcher()
    done = threading.Event()
    d.call(done.set)
    assert _wait(done)
    d.stop()
    ran = []
    d.call(lambda: ran.append(1))
    assert ran == []


def test_on_ui_thread_runs_inline_when_already_there(monkeypatch):
    from veronica.ui import dispatch

    d = Dispatcher()
    monkeypatch.setattr(dispatch, "_default", d)
    order, done = [], threading.Event()

    def outer():
        dispatch.on_ui_thread(lambda: order.append("inner"))   # inline, not queued behind us
        order.append("outer")
        done.set()

    dispatch.on_ui_thread(outer)
    assert _wait(done)
    assert order == ["inner", "outer"]
    d.stop()


def test_icon_images_are_square_rgba_per_state():
    from veronica.ui.icon import STATE_COLORS, orb_image, state_image

    img = orb_image(32)
    assert img.size == (32, 32) and img.mode == "RGBA"
    assert img.getpixel((0, 0))[3] == 0                  # transparent corner
    assert img.getpixel((16, 16))[3] == 255              # opaque middle
    idle, muted = state_image("idle", size=16), state_image("idle", muted=True, size=16)
    assert idle.tobytes() != muted.tobytes()
    assert set(STATE_COLORS) >= {"idle", "listening", "thinking", "speaking", "error", "muted"}
