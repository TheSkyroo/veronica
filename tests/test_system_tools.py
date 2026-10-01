import asyncio
import base64
import contextlib
import subprocess
import time

import pytest

from veronica.tools import system


class Done:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


@pytest.fixture
def fake_run(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return Done(out="OUT")

    monkeypatch.setattr(system.subprocess, "run", run)
    return calls


@pytest.fixture
def started(monkeypatch):
    """os.startfile replaced: records what would have been launched."""
    opened = []
    monkeypatch.setattr(system, "_startfile", opened.append)
    return opened


@pytest.fixture
def apps(monkeypatch):
    """A fake Start menu, Get-StartApps list and App Paths registry."""
    state = {
        "lnk": [("Microsoft Edge", r"C:\SM\Microsoft Edge.lnk"),
                ("Spotify", r"C:\Users\u\SM\Spotify.lnk"),
                ("Uninstall Spotify", r"C:\SM\Uninstall Spotify.lnk"),
                ("Visual Studio Code", r"C:\SM\Visual Studio Code\Visual Studio Code.lnk")],
        "uwp": [("Calculator", "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"),
                ("Settings", "windows.immersivecontrolpanel_cw5n1h2txyewy!microsoft.windows.immersivecontrolpanel")],
        "known": {"winword"},
        "uwp_calls": 0,
    }

    def uwp():
        state["uwp_calls"] += 1
        return state["uwp"]

    monkeypatch.setattr(system, "_start_menu_shortcuts", lambda: state["lnk"])
    monkeypatch.setattr(system, "_start_apps", uwp)
    monkeypatch.setattr(system, "_known_bare_app", lambda n: n.casefold() in state["known"])
    return state


def text(res):
    return res["content"][0]["text"]


# -- open_app ---------------------------------------------------------------------

async def test_open_app_start_menu_shortcut(apps, started):
    res = await system.open_app.handler({"name": "spotify"})
    assert started == [r"C:\Users\u\SM\Spotify.lnk"]
    assert text(res) == "ok" and not res.get("is_error")
    assert apps["uwp_calls"] == 0          # Get-StartApps is slow; only asked on a miss


async def test_open_app_prefix_match_prefers_shortest(apps, started):
    apps["lnk"].append(("Microsoft Edge Dev", r"C:\SM\Edge Dev.lnk"))
    await system.open_app.handler({"name": "Microsoft"})
    assert started == [r"C:\SM\Microsoft Edge.lnk"]


async def test_open_app_inner_word_match(apps, started):
    await system.open_app.handler({"name": "code"})
    assert started == [r"C:\SM\Visual Studio Code\Visual Studio Code.lnk"]


async def test_open_app_never_picks_an_uninstaller(apps, started):
    apps["lnk"] = [("Uninstall Spotify", r"C:\SM\Uninstall Spotify.lnk")]
    apps["uwp"] = []
    res = await system.open_app.handler({"name": "Spotify"})
    assert res["is_error"] and started == []


async def test_open_app_uwp_by_aumid(apps, started):
    res = await system.open_app.handler({"name": "calculator"})
    assert started == [r"shell:AppsFolder\Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"]
    assert not res.get("is_error")


async def test_open_app_known_bare_name(apps, started):
    res = await system.open_app.handler({"name": "winword"})
    assert started == ["winword"] and not res.get("is_error")


async def test_open_app_unknown_is_a_spoken_error(apps, started):
    res = await system.open_app.handler({"name": "Nope"})
    assert res["is_error"] and "no app called 'Nope'" in text(res) and started == []


@pytest.mark.parametrize("name", [
    "/tmp/evil.exe", r"C:\Windows\evil.exe", "..\\x", ".hidden", "-e", "a:b", "x|y", "",
])
async def test_open_app_rejects_paths_flags_hidden(apps, started, name):
    res = await system.open_app.handler({"name": name})
    assert res["is_error"] and started == [] and apps["uwp_calls"] == 0


async def test_open_app_missing_name_is_error(apps, started):
    res = await system.open_app.handler({})
    assert res["is_error"] and started == []


def test_best_match_exact_wins_over_prefix():
    cands = [("Notepad++", "a"), ("Notepad", "b")]
    assert system.best_match("notepad", cands) == ("Notepad", "b")


def test_best_match_needs_word_boundary():
    assert system.best_match("edge", [("Knowledge Base", "x")]) is None


def test_start_apps_parses_json(monkeypatch):
    rows = '[{"Name":"Calculator","AppID":"Calc!App"},{"Name":"","AppID":"x"}]'
    monkeypatch.setattr(system, "powershell", lambda script, *a, **k: system._ok(rows))
    assert system._start_apps() == [("Calculator", "Calc!App")]
    monkeypatch.setattr(system, "powershell", lambda script, *a, **k: system._ok('{"Name":"A","AppID":"B"}'))
    assert system._start_apps() == [("A", "B")]
    monkeypatch.setattr(system, "powershell", lambda script, *a, **k: system._err("no"))
    assert system._start_apps() == []


# -- open_url ---------------------------------------------------------------------

@pytest.mark.parametrize("url", ["file:///etc/passwd", "javascript:alert(1)", "https://x.y/a b", "ms-settings:"])
async def test_open_url_rejects_non_http(started, url):
    res = await system.open_url.handler({"url": url})
    assert res["is_error"] and started == []


async def test_open_url(started):
    res = await system.open_url.handler({"url": "https://x.y"})
    assert started == ["https://x.y"] and not res.get("is_error")


# -- clipboard --------------------------------------------------------------------

async def test_clipboard_read(monkeypatch):
    monkeypatch.setattr(system, "_clip_get", lambda: "a\r\nb")
    assert text(await system.clipboard_read.handler({})) == "a\nb"


async def test_clipboard_read_empty(monkeypatch):
    monkeypatch.setattr(system, "_clip_get", lambda: "")
    assert text(await system.clipboard_read.handler({})) == "The clipboard has no text."


async def test_clipboard_write_uses_crlf(monkeypatch):
    got = []
    monkeypatch.setattr(system, "_clip_set", got.append)
    res = await system.clipboard_write.handler({"text": "hello\nworld"})
    assert got == ["hello\r\nworld"] and not res.get("is_error")


async def test_clipboard_busy_is_error(monkeypatch):
    def busy():
        raise RuntimeError("the clipboard is busy in another app")
    monkeypatch.setattr(system, "_clip_get", busy)
    res = await system.clipboard_read.handler({})
    assert res["is_error"] and "busy" in text(res)


# -- notify -----------------------------------------------------------------------

def test_toast_xml_escapes():
    xml = system.toast_xml('a <b> & "c"', "m")
    assert "<text>a &lt;b&gt; &amp; \"c\"</text>" in xml and "<text>m</text>" in xml


async def test_notify_uses_winrt(monkeypatch):
    shown = []
    monkeypatch.setattr(system, "_toast_winrt", lambda t, m: shown.append((t, m)))
    res = await system.notify.handler({"title": "T", "message": "M"})
    assert shown == [("T", "M")] and not res.get("is_error")


async def test_notify_falls_back_to_powershell(monkeypatch, fake_run):
    def no_winrt(t, m):
        raise ImportError("no winrt")
    monkeypatch.setattr(system, "_toast_winrt", no_winrt)
    await system.notify.handler({"title": "T'); Remove-Item x; ('", "message": "M"})
    argv, kw = fake_run[0]
    assert argv == system.POWERSHELL
    # the user's text never appears as script, only inside base64
    assert "Remove-Item" not in kw["input"]
    inner = base64.b64decode(kw["input"].split("FromBase64String('")[1].split("'")[0]).decode()
    assert "Remove-Item" not in inner and "ToastNotificationManager" in inner


# -- volume -----------------------------------------------------------------------

async def test_volume_get(monkeypatch):
    monkeypatch.setattr(system, "_get_volume", lambda: 42)
    assert text(await system.volume_get.handler({})) == "42"


async def test_volume_set_clamps(monkeypatch):
    got = []
    monkeypatch.setattr(system, "_set_volume", got.append)
    await system.volume_set.handler({"level": 250})
    await system.volume_set.handler({"level": -5})
    await system.volume_set.handler({"level": "30"})
    assert got == [100, 0, 30]


async def test_volume_set_bad_level_is_error(monkeypatch):
    got = []
    monkeypatch.setattr(system, "_set_volume", got.append)
    assert (await system.volume_set.handler({"level": None}))["is_error"]
    assert (await system.volume_set.handler({"level": "abc"}))["is_error"]
    assert got == []


async def test_volume_failure_is_error(monkeypatch):
    def broken():
        raise OSError("no audio device")
    monkeypatch.setattr(system, "_get_volume", broken)
    res = await system.volume_get.handler({})
    assert res["is_error"] and "no audio device" in text(res)


# -- powershell -------------------------------------------------------------------

def _decoded(stdin: str) -> str:
    return base64.b64decode(stdin.split("FromBase64String('")[1].split("'")[0]).decode("utf-8")


async def test_powershell_runs_script_on_stdin(fake_run):
    script = 'if ($true) {\n  "héllo"\n}\nelse { "no" }'
    res = await system.powershell_tool.handler({"script": script})
    argv, kw = fake_run[0]
    assert argv == ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                    "-Command", "-"]
    assert kw["timeout"] == system.POWERSHELL_TIMEOUT_S and "shell" not in kw
    assert kw["input"].isascii() and kw["input"].count("\n") == 1
    assert _decoded(kw["input"]) == script
    assert kw["encoding"] == "utf-8"
    assert text(res) == "OUT"


async def test_powershell_requires_script(fake_run):
    res = await system.powershell_tool.handler({"script": "  "})
    assert res["is_error"] and fake_run == []


async def test_powershell_stderr_only_is_error(monkeypatch):
    monkeypatch.setattr(system.subprocess, "run", lambda *a, **k: Done(rc=0, err="Get-Foo : not recognized"))
    res = await system.powershell_tool.handler({"script": "Get-Foo"})
    assert res["is_error"] and "not recognized" in text(res)


async def test_nonzero_exit_is_error(monkeypatch):
    monkeypatch.setattr(system.subprocess, "run", lambda *a, **k: Done(rc=1, err="nope"))
    res = await system.powershell_tool.handler({"script": "exit 1"})
    assert res["is_error"] and "nope" in text(res)


async def test_timeout_is_error(monkeypatch):
    def run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=10)

    monkeypatch.setattr(system.subprocess, "run", run)
    res = await system.powershell_tool.handler({"script": "Start-Sleep 1000"})
    assert res["is_error"] and "timed out" in text(res)


async def test_handlers_do_not_block_loop(monkeypatch):
    def blocking_run(argv, **kw):
        time.sleep(0.2)
        return Done(out="OUT")

    monkeypatch.setattr(system.subprocess, "run", blocking_run)

    ticks = []

    async def ticker():
        while True:
            await asyncio.sleep(0.05)
            ticks.append(time.monotonic())

    ticker_task = asyncio.create_task(ticker())
    try:
        await system.powershell_tool.handler({"script": "Start-Sleep 1"})
    finally:
        ticker_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ticker_task
    assert len(ticks) >= 2


# -- read_battery -----------------------------------------------------------------

@pytest.mark.parametrize("status, expected", [
    ((1, 8, 72), (72, "charging")),
    ((1, 1, 100), (100, "charged")),
    ((0, 1, 35), (35, "discharging")),
    ((0, 2 | 4, 4), (4, "discharging")),
    ((1, 1, 98), (98, None)),               # plugged in, not charging (battery-health hold)
    ((1, 128, 255), (None, None)),          # desktop: no system battery
    ((255, 255, 255), (None, None)),        # unknown
    ((255, 255, 50), (50, None)),
])
def test_read_battery(status, expected):
    assert system.read_battery(status=lambda: status) == expected


def test_read_battery_failure():
    def boom():
        raise OSError("no kernel32")
    assert system.read_battery(status=boom) == (None, None)


# -- dictate_type (A4 dictation; not a Claude tool, called directly) ----------

U = system.KEYEVENTF_UNICODE
UP = system.KEYEVENTF_KEYUP


def test_key_events_single_line():
    assert system.key_events("hi") == [(0, ord("h"), U), (0, ord("h"), U | UP),
                                       (0, ord("i"), U), (0, ord("i"), U | UP)]


def test_key_events_newlines_press_enter():
    ev = system.key_events("a\nb\r\nc")
    enters = [e for e in ev if e[0] == system.VK_RETURN]
    assert enters == [(system.VK_RETURN, 0, 0), (system.VK_RETURN, 0, UP)] * 2
    typed = [chr(e[1]) for e in ev if e[0] == 0 and not e[2] & UP]
    assert typed == ["a", "b", "c"]
    # ordering: a, Enter, b, Enter, c
    assert ev.index((system.VK_RETURN, 0, 0)) > ev.index((0, ord("a"), U))


def test_key_events_non_bmp_is_a_surrogate_pair():
    ev = system.key_events("😀")
    downs = [e[1] for e in ev if not e[2] & UP]
    assert downs == [0xD83D, 0xDE00]


def test_dictate_type_sends_everything():
    got = []

    def send(events):
        got.extend(events)
        return len(events)

    res = system.dictate_type('she said "hi"\nok', send=send)
    assert not res.get("is_error") and got == system.key_events('she said "hi"\nok')


def test_dictate_type_blocked_input_is_error():
    res = system.dictate_type("hi", send=lambda events: 0)
    assert res["is_error"] and "administrator" in text(res)


def test_dictate_type_error_propagates():
    def send(events):
        raise OSError("no access")
    res = system.dictate_type("hi", send=send)
    assert res["is_error"] and "no access" in text(res)


def test_dictate_type_empty_is_a_no_op():
    assert not system.dictate_type("", send=lambda e: pytest.fail("sent")).get("is_error")


def test_server_and_names():
    assert system.system_server["name"] == "system"
    assert set(system.SYSTEM_TOOL_NAMES) == {
        "open_app", "open_url", "clipboard_read", "clipboard_write",
        "notify", "volume_get", "volume_set", "powershell",
    }


@pytest.mark.live
async def test_live_open_notepad():
    res = await system.open_app.handler({"name": "Notepad"})
    assert not res.get("is_error")
