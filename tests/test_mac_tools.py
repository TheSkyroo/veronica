import asyncio
import contextlib
import os
import subprocess
import time
from types import SimpleNamespace

import pytest

from veronica.tools import mac


class Done:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


@pytest.fixture
def fake_run(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return Done(out="OUT")

    monkeypatch.setattr(mac.subprocess, "run", run)
    return calls


def text(res):
    return res["content"][0]["text"]


async def test_open_app(fake_run):
    res = await mac.open_app.handler({"name": "Safari"})
    assert fake_run[0][0] == ["open", "-a", "Safari"]
    assert fake_run[0][1]["timeout"] == 10 and "shell" not in fake_run[0][1]
    assert text(res) == "ok" and not res.get("is_error")


async def test_open_app_rejects_paths(fake_run):
    res = await mac.open_app.handler({"name": "/tmp/evil.app"})
    assert res["is_error"] and fake_run == []


async def test_open_app_rejects_hidden(fake_run):
    res = await mac.open_app.handler({"name": ".hidden"})
    assert res["is_error"] and fake_run == []


async def test_open_app_rejects_flag(fake_run):
    res = await mac.open_app.handler({"name": "-e"})
    assert res["is_error"] and fake_run == []


async def test_open_app_missing_name_is_error(fake_run):
    res = await mac.open_app.handler({})
    assert res["is_error"] and fake_run == []


async def test_open_url_rejects_non_http(fake_run):
    res = await mac.open_url.handler({"url": "file:///etc/passwd"})
    assert res["is_error"] and fake_run == []


async def test_open_url(fake_run):
    await mac.open_url.handler({"url": "https://x.y"})
    assert fake_run[0][0] == ["open", "https://x.y"]


async def test_clipboard_read(fake_run):
    assert text(await mac.clipboard_read.handler({})) == "OUT"
    assert fake_run[0][0] == ["pbpaste"]


async def test_clipboard_write(fake_run):
    await mac.clipboard_write.handler({"text": "hello"})
    assert fake_run[0][0] == ["pbcopy"] and fake_run[0][1]["input"] == "hello"


async def test_notify(fake_run):
    await mac.notify.handler({"title": "T", "message": "M"})
    assert fake_run[0][0][:2] == ["osascript", "-e"]
    assert 'display notification "M" with title "T"' in fake_run[0][0][2]


async def test_notify_escapes_quotes(fake_run):
    await mac.notify.handler({"title": 'a"b', "message": "m"})
    assert '\\"' in fake_run[0][0][2]


async def test_volume_set_clamps(fake_run):
    await mac.volume_set.handler({"level": 250})
    assert fake_run[0][0] == ["osascript", "-e", "set volume output volume 100"]
    await mac.volume_set.handler({"level": -5})
    assert fake_run[1][0] == ["osascript", "-e", "set volume output volume 0"]


async def test_volume_set_bad_level_is_error(fake_run):
    res = await mac.volume_set.handler({"level": None})
    assert res["is_error"] and fake_run == []
    res = await mac.volume_set.handler({"level": "abc"})
    assert res["is_error"] and fake_run == []
    await mac.volume_set.handler({"level": "30"})
    assert fake_run[0][0] == ["osascript", "-e", "set volume output volume 30"]


async def test_volume_get(fake_run):
    await mac.volume_get.handler({})
    assert fake_run[0][0] == ["osascript", "-e", "output volume of (get volume settings)"]


async def test_applescript(fake_run):
    await mac.applescript.handler({"script": 'tell application "Music" to play'})
    assert fake_run[0][0] == ["osascript", "-e", 'tell application "Music" to play']


async def test_nonzero_exit_is_error(monkeypatch):
    monkeypatch.setattr(mac.subprocess, "run", lambda *a, **k: Done(rc=1, err="nope"))
    res = await mac.open_app.handler({"name": "Nope"})
    assert res["is_error"] and "nope" in text(res)


async def test_timeout_is_error(monkeypatch):
    def run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=10)

    monkeypatch.setattr(mac.subprocess, "run", run)
    res = await mac.applescript.handler({"script": "delay 100"})
    assert res["is_error"]


async def test_handlers_do_not_block_loop(monkeypatch):
    def blocking_run(argv, **kw):
        time.sleep(0.2)
        return Done(out="OUT")

    monkeypatch.setattr(mac.subprocess, "run", blocking_run)

    ticks = []

    async def ticker():
        while True:
            await asyncio.sleep(0.05)
            ticks.append(time.monotonic())

    ticker_task = asyncio.create_task(ticker())
    try:
        await mac.applescript.handler({"script": "delay 1"})
    finally:
        ticker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker_task
    assert len(ticks) >= 2


# -- dictate_type (A4 dictation; not a Claude tool, called directly) ----------

def test_dictate_type_single_line(fake_run):
    res = mac.dictate_type("hello world")
    assert fake_run[0][0][:2] == ["osascript", "-e"]
    script = fake_run[0][0][2]
    assert 'tell application "System Events"' in script
    assert 'keystroke "hello world"' in script
    assert "keystroke return" not in script
    assert not res.get("is_error")


def test_dictate_type_multiline_uses_keystroke_return(fake_run):
    mac.dictate_type("line one\nline two\nline three")
    script = fake_run[0][0][2]
    assert 'keystroke "line one"' in script
    assert 'keystroke "line two"' in script
    assert 'keystroke "line three"' in script
    assert script.count("keystroke return") == 2
    # ordering: line, return, line, return, line
    idx1 = script.index('keystroke "line one"')
    idxr1 = script.index("keystroke return")
    idx2 = script.index('keystroke "line two"')
    assert idx1 < idxr1 < idx2


def test_dictate_type_escapes_quotes(fake_run):
    mac.dictate_type('she said "hi"')
    script = fake_run[0][0][2]
    assert '\\"hi\\"' in script


def test_dictate_type_error_propagates(monkeypatch):
    monkeypatch.setattr(mac.subprocess, "run", lambda *a, **k: Done(rc=1, err="no access"))
    res = mac.dictate_type("hi")
    assert res["is_error"] and "no access" in res["content"][0]["text"]


def test_server_and_names():
    assert mac.mac_server["name"] == "mac"
    assert set(mac.MAC_TOOL_NAMES) == {
        "open_app", "open_url", "clipboard_read", "clipboard_write",
        "notify", "volume_get", "volume_set", "applescript", "run_shortcut",
    }


@pytest.mark.live
async def test_live_open_finder():
    res = await mac.open_app.handler({"name": "Finder"})
    assert not res.get("is_error")


# -- run_shortcut ---------------------------------------------------------------

@pytest.fixture
def fake_shortcuts(monkeypatch):
    """subprocess.run stubbed per `shortcuts` subcommand: `list` returns the
    installed names, `run` succeeds. Returns the call log plus knobs."""
    calls = []
    state = {"installed": "Morning\nPay Rent\n", "list_rc": 0, "run_rc": 0, "run_err": ""}

    def run(argv, **kw):
        calls.append((argv, kw))
        if argv[:2] == ["shortcuts", "list"]:
            return Done(rc=state["list_rc"], out=state["installed"], err="shortcuts: no access")
        return Done(rc=state["run_rc"], err=state["run_err"])

    monkeypatch.setattr(mac.subprocess, "run", run)
    return SimpleNamespace(calls=calls, state=state)


async def test_run_shortcut_runs_an_installed_one(fake_shortcuts):
    res = await mac.run_shortcut.handler({"name": "Morning"})
    assert [c[0] for c in fake_shortcuts.calls] == [["shortcuts", "list"], ["shortcuts", "run", "Morning"]]
    assert fake_shortcuts.calls[1][1]["timeout"] == mac.SHORTCUT_TIMEOUT_S
    assert text(res) == "Ran Morning" and not res.get("is_error")


async def test_run_shortcut_matches_the_installed_spelling(fake_shortcuts):
    await mac.run_shortcut.handler({"name": "  pay rent "})
    assert fake_shortcuts.calls[1][0] == ["shortcuts", "run", "Pay Rent"]


async def test_run_shortcut_passes_input_through_a_temp_file(fake_shortcuts):
    await mac.run_shortcut.handler({"name": "Morning", "input": "hello"})
    argv = fake_shortcuts.calls[1][0]
    assert argv[:3] == ["shortcuts", "run", "Morning"] and argv[3] == "--input-path"
    path = argv[4]
    assert not os.path.exists(path)          # cleaned up after the run


async def test_run_shortcut_unknown_name_is_a_spoken_error(fake_shortcuts):
    res = await mac.run_shortcut.handler({"name": "Nope"})
    assert res["is_error"] and "no shortcut called 'Nope'" in text(res)
    assert [c[0] for c in fake_shortcuts.calls] == [["shortcuts", "list"]]   # never ran


async def test_run_shortcut_no_shortcuts_installed(fake_shortcuts):
    fake_shortcuts.state["installed"] = ""
    res = await mac.run_shortcut.handler({"name": "Morning"})
    assert res["is_error"] and "no shortcut called" in text(res)


async def test_run_shortcut_reports_a_broken_cli(fake_shortcuts):
    fake_shortcuts.state["list_rc"] = 1
    res = await mac.run_shortcut.handler({"name": "Morning"})
    assert res["is_error"] and "couldn't read the shortcuts list" in text(res)


async def test_run_shortcut_reports_a_failed_run(fake_shortcuts):
    fake_shortcuts.state["run_rc"] = 1
    fake_shortcuts.state["run_err"] = "Error: the shortcut failed"
    res = await mac.run_shortcut.handler({"name": "Morning"})
    assert res["is_error"] and "the shortcut failed" in text(res)


async def test_run_shortcut_requires_a_name(fake_shortcuts):
    res = await mac.run_shortcut.handler({"name": "  "})
    assert res["is_error"] and fake_shortcuts.calls == []


async def test_run_shortcut_rejects_a_flag_name(fake_shortcuts):
    res = await mac.run_shortcut.handler({"name": "--help"})
    assert res["is_error"] and fake_shortcuts.calls == []
