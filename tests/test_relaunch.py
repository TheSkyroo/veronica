from pathlib import Path

from veronica.ui.relaunch import relaunch


class FakePopen:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return object()


def test_relaunch_from_bundle_schedules_open_then_quits():
    popen = FakePopen()
    order = []
    bundle = Path("/Applications/Veronica.app")

    ok = relaunch(bundle, quit=lambda: order.append("quit"), popen=lambda *a, **k: (order.append("popen"), popen(*a, **k))[1])
    assert ok is True
    assert order == ["popen", "quit"]
    argv, kwargs = popen.calls[0]
    assert argv == ["/bin/sh", "-c", 'sleep 1; open -n "/Applications/Veronica.app"']
    assert kwargs.get("start_new_session") is True


def test_relaunch_without_bundle_just_quits():
    popen = FakePopen()
    quits = []
    ok = relaunch(None, quit=lambda: quits.append(1), popen=popen)
    assert ok is False
    assert quits == [1]
    assert popen.calls == []


def test_relaunch_popen_failure_does_not_quit():
    def broken(*a, **k):
        raise OSError("fork failed")

    quits = []
    ok = relaunch(Path("/Applications/Veronica.app"), quit=lambda: quits.append(1), popen=broken)
    assert ok is False
    assert quits == []
