import subprocess

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


def test_server_and_names():
    assert mac.mac_server["name"] == "mac"
    assert set(mac.MAC_TOOL_NAMES) == {
        "open_app", "open_url", "clipboard_read", "clipboard_write",
        "notify", "volume_get", "volume_set", "applescript",
    }


@pytest.mark.live
async def test_live_open_finder():
    res = await mac.open_app.handler({"name": "Finder"})
    assert not res.get("is_error")
