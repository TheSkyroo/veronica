"""winproc: npm-shim resolution (never cmd.exe), interrupt/kill of the CLI
tree, and hook command lines that survive cmd.exe and PowerShell. Runs on
any OS: the Win32 calls are injected or absent."""
import subprocess

import pytest

from veronica.brain.backends import winproc

CODEX_CMD = r"""@ECHO off
GOTO start
:find_dp0
SET dp0=%~dp0
EXIT /b
:start
SETLOCAL
CALL :find_dp0

IF EXIST "%dp0%\node.exe" (
  SET "_prog=%dp0%\node.exe"
) ELSE (
  SET "_prog=node"
  SET PATHEXT=%PATHEXT:;.JS;=;%
)

endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & "%_prog%"  "%dp0%\node_modules\@openai\codex\bin\codex.js" %*
"""

COPILOT_PS1 = r"""#!/usr/bin/env pwsh
$basedir=Split-Path $MyInvocation.MyCommand.Definition -Parent
$exe=""
if (Test-Path "$basedir/node$exe") {
  & "$basedir/node$exe"  "$basedir/node_modules/@github/copilot/npm-loader.js" $args
} else {
  & "node$exe"  "$basedir/node_modules/@github/copilot/npm-loader.js" $args
}
"""


@pytest.fixture
def npm(tmp_path):
    """An npm global prefix with codex installed the way npm does it."""
    (tmp_path / "codex.cmd").write_text(CODEX_CMD)
    (tmp_path / "codex").write_text("#!/bin/sh\n")
    script = tmp_path / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    script.parent.mkdir(parents=True)
    script.write_text("")
    return tmp_path


def test_a_cmd_shim_runs_its_script_under_node(npm):
    which = {"codex": str(npm / "codex.cmd"), "node": r"C:\Program Files\nodejs\node.exe"}.get
    argv = winproc.resolve_cli(["codex", "exec", "--json", 'say "hi" & del /q C:\\x %PATH%'], which=which)
    assert argv == [r"C:\Program Files\nodejs\node.exe",
                    str(npm / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"),
                    "exec", "--json", 'say "hi" & del /q C:\\x %PATH%']


def test_the_shims_own_node_is_preferred(npm):
    (npm / "node.exe").write_text("")
    argv = winproc.resolve_cli(["codex"], which={"codex": str(npm / "codex.cmd")}.get)
    assert argv[0] == str(npm / "node.exe")


def test_the_sh_shim_found_instead_of_the_cmd_one_is_followed(npm):
    argv = winproc.resolve_cli(["codex"], which={"codex": str(npm / "codex"), "node": "node.exe"}.get)
    assert argv[1].endswith("codex.js")


def test_a_ps1_shim_is_read_too(tmp_path):
    (tmp_path / "copilot.ps1").write_text(COPILOT_PS1)
    script = tmp_path / "node_modules" / "@github" / "copilot" / "npm-loader.js"
    script.parent.mkdir(parents=True)
    script.write_text("")
    argv = winproc.resolve_cli(["copilot", "-p", "x"], which={"copilot": str(tmp_path / "copilot.ps1"),
                                                             "node": "node.exe"}.get)
    assert argv == ["node.exe", str(script), "-p", "x"]


def test_a_native_exe_is_run_as_is(tmp_path):
    assert winproc.resolve_cli(["agy", "--print="], which=lambda n: r"C:\Tools\agy.exe") == [
        r"C:\Tools\agy.exe", "--print="]


def test_a_name_not_on_path_is_left_alone():
    assert winproc.resolve_cli(["nope", "x"], which=lambda n: None) == ["nope", "x"]


def test_a_batch_file_without_a_script_is_refused(tmp_path):
    bat = tmp_path / "evil.bat"
    bat.write_text("@echo off\r\nsomething.exe %*\r\n")
    with pytest.raises(winproc.UnsafeShim):
        winproc.resolve_cli(["evil", "a&b"], which=lambda n: str(bat))


def test_a_shim_whose_script_is_missing_is_refused(npm):
    (npm / "node_modules" / "@openai" / "codex" / "bin" / "codex.js").unlink()
    with pytest.raises(winproc.UnsafeShim):
        winproc.resolve_cli(["codex"], which={"codex": str(npm / "codex.cmd"), "node": "node"}.get)


def test_a_shim_without_node_is_refused(npm):
    with pytest.raises(winproc.UnsafeShim):
        winproc.resolve_cli(["codex"], which={"codex": str(npm / "codex.cmd")}.get)


# -- interrupt / kill ------------------------------------------------------------

class Proc:
    def __init__(self, pid=4242, fail=None):
        self.pid, self.returncode, self.sent, self.killed, self._fail = pid, None, [], False, fail

    def send_signal(self, sig):
        if self._fail:
            raise self._fail
        self.sent.append(sig)

    def kill(self):
        self.killed = True
        self.returncode = 1


def test_interrupt_sends_ctrl_break():
    p = Proc()
    assert winproc.interrupt(p) and p.sent == [winproc.CTRL_BREAK_EVENT]


@pytest.mark.parametrize("err", [OSError(6, "invalid handle"), ProcessLookupError()])
def test_interrupt_reports_an_undeliverable_break(err):
    assert winproc.interrupt(Proc(fail=err)) is False


def test_kill_tree_runs_taskkill_then_kills():
    runs = []
    p = Proc()
    winproc.kill_tree(p, run=lambda argv, **kw: runs.append((argv, kw)))
    assert runs[0][0] == ["taskkill", "/F", "/T", "/PID", "4242"] and runs[0][1]["timeout"] == 5
    assert p.killed


def test_kill_tree_survives_a_missing_taskkill():
    def missing(argv, **kw):
        raise FileNotFoundError("taskkill")

    p = Proc()
    winproc.kill_tree(p, run=missing)
    assert p.killed


def test_kill_tree_leaves_an_exited_child_alone():
    runs = []
    p = Proc()
    p.returncode = 0
    winproc.kill_tree(p, run=lambda argv, **kw: runs.append(argv))
    assert runs == [] and not p.killed


def test_spawn_flags_off_windows_are_zero(monkeypatch):
    monkeypatch.setattr(winproc.os, "name", "posix")
    assert winproc.spawn_flags() == 0 and winproc.no_window_flags() == 0


def test_spawn_flags_on_windows(monkeypatch):
    monkeypatch.setattr(winproc.os, "name", "nt")
    assert winproc.spawn_flags(lambda: True) == winproc.CREATE_NEW_PROCESS_GROUP
    assert winproc.spawn_flags(lambda: False) == winproc.CREATE_NEW_PROCESS_GROUP | winproc.CREATE_NO_WINDOW
    assert winproc.no_window_flags() == winproc.CREATE_NO_WINDOW


def test_win32_constants_match_the_sdk():
    assert winproc.CREATE_NEW_PROCESS_GROUP == 0x200 and winproc.CREATE_NO_WINDOW == 0x08000000
    assert winproc.CTRL_BREAK_EVENT == 1


# -- hook command lines ----------------------------------------------------------

def test_plain_tokens_are_left_alone():
    assert winproc.neutral_command(["C:\\Python312\\python.exe", "-m", "veronica.brain.hook", "codex"],
                                   shorten=lambda p: 1 / 0) == "C:\\Python312\\python.exe -m veronica.brain.hook codex"


def test_a_spaced_path_becomes_its_short_form(tmp_path):
    d = tmp_path / "Mani Kumar"
    d.mkdir()
    short = lambda p: p.replace("Mani Kumar", "MANIKU~1")
    # the existing part is shortened, the not-yet-written file kept as is
    assert winproc.short_path(d / "hook.log", short) == str(tmp_path / "MANIKU~1" / "hook.log")
    assert winproc.neutral_arg(str(d / "hook.log"), short) == str(tmp_path / "MANIKU~1" / "hook.log")


def test_without_a_short_form_the_path_is_double_quoted(tmp_path):
    d = tmp_path / "Mani Kumar"
    d.mkdir()
    assert winproc.neutral_arg(str(d / "x.log"), lambda p: p) == subprocess.list2cmdline([str(d / "x.log")])
    assert winproc.neutral_arg("not a path", lambda p: p) == '"not a path"'


def test_ps_command_is_all_literal():
    cmd = winproc.ps_command(["C:\\Users\\Mani Kumar\\py.exe", "--log", "C:\\it's $HOME `x`"])
    assert cmd == "& 'C:\\Users\\Mani Kumar\\py.exe' '--log' 'C:\\it''s $HOME `x`'"
    assert winproc.ps_quote("a\u2019b") == "'a\u2019\u2019b'"


def test_children_get_the_console_python_not_pythonw(tmp_path):
    (tmp_path / "pythonw.exe").write_text("")
    (tmp_path / "python.exe").write_text("")
    assert winproc.console_python(str(tmp_path / "pythonw.exe")) == str(tmp_path / "python.exe")
    assert winproc.console_python(str(tmp_path / "python.exe")) == str(tmp_path / "python.exe")
    (tmp_path / "python.exe").unlink()
    assert winproc.console_python(str(tmp_path / "pythonw.exe")) == str(tmp_path / "pythonw.exe")
