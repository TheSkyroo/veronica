import os
from pathlib import Path

from veronica.ui import relaunch as relaunch_mod
from veronica.ui.relaunch import WAIT_THEN_START, relaunch, staged_dir


class FakePopen:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        return object()


EXE = Path("C:/Users/Me/veronica/dist/Veronica/Veronica.exe")


def test_relaunch_from_exe_schedules_helper_then_quits():
    popen = FakePopen()
    order = []

    ok = relaunch(
        EXE, quit=lambda: order.append("quit"), pid=4242, env={"PATH": "x"},
        popen=lambda *a, **k: (order.append("popen"), popen(*a, **k))[1],
    )
    assert ok is True
    assert order == ["popen", "quit"]
    argv, kwargs = popen.calls[0]
    assert argv[0] == "powershell.exe"
    assert argv[-2:] == ["-Command", WAIT_THEN_START]
    assert "-NoProfile" in argv and "-NonInteractive" in argv
    env = kwargs["env"]
    assert env["VERONICA_RELAUNCH_PID"] == "4242"
    assert env["VERONICA_RELAUNCH_EXE"] == str(EXE)
    assert env["VERONICA_RELAUNCH_STAGED"] == str(staged_dir(EXE))
    assert env["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert env["PATH"] == "x"                                   # the rest of the environment is kept
    # detached: no console, survives our exit, not in our Ctrl+C group
    assert kwargs["creationflags"] == relaunch_mod.DETACHED_PROCESS | relaunch_mod.CREATE_NEW_PROCESS_GROUP
    assert kwargs["close_fds"] is True


def test_relaunch_never_interpolates_path_or_pid_into_the_command():
    popen = FakePopen()
    exe = Path("C:/Apps/We'ird $(rm -rf ~); `x`/Veronica.exe")
    relaunch(exe, quit=lambda: None, popen=popen, pid=7, env={})
    argv, kwargs = popen.calls[0]
    assert all("We'ird" not in a and "Veronica.exe" not in a for a in argv)
    assert kwargs["env"]["VERONICA_RELAUNCH_EXE"] == str(exe)
    # waits for the old process to exit before starting the new one (no double instance)
    assert WAIT_THEN_START.index("Wait-Process") < WAIT_THEN_START.index("Start-Process")
    # and swaps a staged build in between the two
    assert WAIT_THEN_START.index("Wait-Process") < WAIT_THEN_START.index("Move-Item") < WAIT_THEN_START.index("Start-Process")


def test_relaunch_defaults_pid_to_this_process_and_env_to_ours():
    popen = FakePopen()
    relaunch(EXE, quit=lambda: None, popen=popen)
    _, kwargs = popen.calls[0]
    assert kwargs["env"]["VERONICA_RELAUNCH_PID"] == str(os.getpid())
    assert kwargs["env"].get("PATH") == os.environ.get("PATH")


def test_relaunch_without_exe_just_quits():
    popen = FakePopen()
    quits = []
    ok = relaunch(None, quit=lambda: quits.append(1), popen=popen)
    assert ok is False
    assert quits == [1]
    assert popen.calls == []


def test_relaunch_popen_failure_does_not_quit():
    def broken(*a, **k):
        raise OSError("powershell not found")

    quits = []
    ok = relaunch(EXE, quit=lambda: quits.append(1), popen=broken)
    assert ok is False
    assert quits == []


def test_staged_dir_is_next_to_the_install():
    assert staged_dir(EXE) == EXE.parent.with_name("Veronica.new")
