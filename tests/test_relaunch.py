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

    ok = relaunch(
        bundle, quit=lambda: order.append("quit"), pid=4242,
        popen=lambda *a, **k: (order.append("popen"), popen(*a, **k))[1],
    )
    assert ok is True
    assert order == ["popen", "quit"]
    argv, kwargs = popen.calls[0]
    assert argv == [
        "/bin/sh", "-c",
        'while kill -0 "$2" 2>/dev/null; do sleep 0.2; done; open -n "$1"',
        "sh", "/Applications/Veronica.app", "4242",
    ]
    assert kwargs.get("start_new_session") is True


def test_relaunch_never_interpolates_path_or_pid_into_shell_string():
    popen = FakePopen()
    bundle = Path('/Apps/We"ird $(rm -rf ~)/Veronica.app')
    relaunch(bundle, quit=lambda: None, popen=popen, pid=7)
    argv, _ = popen.calls[0]
    script = argv[2]
    assert "Veronica.app" not in script and "7" not in script
    assert argv[3] == "sh"
    assert argv[4] == str(bundle)
    assert argv[5] == "7"
    # waits for the old process to die before opening (no double instance)
    assert script.index("kill -0") < script.index("open -n")


def test_relaunch_defaults_pid_to_this_process():
    import os

    popen = FakePopen()
    relaunch(Path("/Applications/Veronica.app"), quit=lambda: None, popen=popen)
    argv, _ = popen.calls[0]
    assert argv[5] == str(os.getpid())


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
