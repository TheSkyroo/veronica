"""Typed Windows actions exposed to Claude as in-process MCP tools.

Windows-only libraries (pywin32, pycaw/comtypes, winrt) are imported inside
the functions that need them, so this module imports anywhere and the tests
can swap each OS call for a fake through the small `_...` seams below."""
import asyncio
import base64
import contextlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from claude_agent_sdk import create_sdk_mcp_server, tool

TIMEOUT_S = 10
# A PowerShell script is a user-written program (it can wait on a network
# call or a device), so it gets its own, far longer budget than the
# one-shot calls.
POWERSHELL_TIMEOUT_S = 120
POWERSHELL = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
              "-Command", "-"]
# No console window flashing up for every helper process.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def run(argv: list[str], stdin: str | None = None, ok_text: str | None = None,
        timeout: int = TIMEOUT_S) -> dict:
    """Run argv (never a shell string) and map the result to MCP content.

    On success, `ok_text` (if given) is returned verbatim instead of stdout —
    used by tools where stdout is not meaningful output. A zero exit with
    nothing on stdout but something on stderr is an error too: PowerShell
    reports a failed cmdlet that way."""
    try:
        done = subprocess.run(argv, input=stdin, check=False, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=timeout, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return _err(f"timed out after {timeout}s")
    except Exception as exc:  # e.g. FileNotFoundError
        return _err(str(exc))
    out, err = (done.stdout or "").strip(), (done.stderr or "").strip()
    if done.returncode != 0 or (err and not out):
        return _err(err or f"exit {done.returncode}")
    if ok_text is not None:
        return _ok(ok_text)
    return _ok(out or "ok")


def _powershell_stdin(script: str) -> str:
    """What goes to `powershell.exe -Command -` on stdin: one ASCII line that
    decodes and runs the real script. Fed raw, `-Command -` reads the script
    line by line like an interactive prompt (an `else` on its own line is a
    syntax error, the last block needs a blank line) and in the console's
    OEM code page (non-ASCII text is mangled); as base64 it arrives as one
    script block, byte-exact. Output is switched to UTF-8 to match `run`."""
    b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    return ("[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "& ([ScriptBlock]::Create([System.Text.Encoding]::UTF8.GetString("
            f"[System.Convert]::FromBase64String('{b64}'))))\n")


def powershell(script: str, timeout: int = TIMEOUT_S, ok_text: str | None = None) -> dict:
    """Run a PowerShell script (synchronous; call via asyncio.to_thread)."""
    return run(POWERSHELL, _powershell_stdin(script), ok_text=ok_text, timeout=timeout)


# -- battery (plain helper, not a tool; used by the orchestrator's quick replies)
def _power_status() -> tuple[int, int, int]:
    """(ACLineStatus, BatteryFlag, BatteryLifePercent) from Win32
    GetSystemPowerStatus; raises OSError when the call fails."""
    import ctypes
    from ctypes import wintypes

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [("ACLineStatus", wintypes.BYTE), ("BatteryFlag", wintypes.BYTE),
                    ("BatteryLifePercent", wintypes.BYTE), ("SystemStatusFlag", wintypes.BYTE),
                    ("BatteryLifeTime", wintypes.DWORD), ("BatteryFullLifeTime", wintypes.DWORD)]

    st = SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
        raise OSError("GetSystemPowerStatus failed")
    return st.ACLineStatus & 0xFF, st.BatteryFlag & 0xFF, st.BatteryLifePercent & 0xFF


def read_battery(status=None) -> tuple[int | None, str | None]:
    """(percent, state), state in charging|discharging|charged, or None when
    the percent is known but the state isn't (plugged in but not charging —
    a battery-health hold, or Windows doesn't say); (None, None) on a
    desktop with no battery or when the call fails.

    Read with GetSystemPowerStatus rather than psutil.sensors_battery():
    psutil only says "plugged in", which can't tell charging from a
    plugged-in hold. `status` replaces the Win32 call in tests."""
    try:
        ac, flag, percent = (status or _power_status)()
    except Exception:
        return None, None
    if percent == 255 or (flag != 255 and flag & 128):     # unknown / no system battery
        return None, None
    percent = max(0, min(100, int(percent)))
    if flag != 255 and flag & 8:
        return percent, "charging"
    if ac == 0:
        return percent, "discharging"
    if ac == 1 and percent >= 100:
        return percent, "charged"
    return percent, None


def _guard(fn):
    """Wrap a handler so malformed args (missing keys, bad types) or an OS
    call that blew up return `_err(...)` instead of raising."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as exc:
            return _err(str(exc))
    return wrapper


@contextlib.contextmanager
def _com():
    """A COM apartment for the calling (worker) thread: pycaw and the clipboard
    are called from asyncio.to_thread workers, which start without one."""
    import comtypes

    comtypes.CoInitialize()
    try:
        yield
    finally:
        comtypes.CoUninitialize()


# -- open_app --------------------------------------------------------------------
# Programs Windows' shell resolves by bare name (App Paths or System32) that
# have no Start-menu entry of their own on current Windows.
_BUILTIN_APPS = frozenset({
    "notepad", "mspaint", "explorer", "taskmgr", "control", "charmap", "osk", "magnify",
    "snippingtool", "wordpad", "write", "regedit", "msinfo32", "resmon", "perfmon",
})
_APP_NAME_RE = re.compile(r"[\w][\w .&+'()-]{0,79}")
_AUMID_RE = re.compile(r"[^\x00-\x1f\"<>|*?]+")


def _start_menu_dirs() -> list[Path]:
    dirs = []
    for var in ("ProgramData", "APPDATA"):
        base = os.environ.get(var)
        if base:
            dirs.append(Path(base) / "Microsoft" / "Windows" / "Start Menu" / "Programs")
    return dirs


def _start_menu_shortcuts() -> list[tuple[str, str]]:
    """(label, path) for every Start-menu shortcut, all users and this user."""
    found = []
    for d in _start_menu_dirs():
        with contextlib.suppress(OSError):
            found += [(p.stem, str(p)) for p in d.rglob("*.lnk")]
    return found


def _start_apps() -> list[tuple[str, str]]:
    """(name, AppUserModelID) for every app `Get-StartApps` lists — this is
    where Store (UWP) apps like Calculator or Settings show up."""
    res = powershell("Get-StartApps | Select-Object Name, AppID | ConvertTo-Json -Compress")
    if res.get("is_error"):
        return []
    try:
        rows = json.loads(res["content"][0]["text"])
    except ValueError:
        return []
    if isinstance(rows, dict):        # ConvertTo-Json unwraps a one-item array
        rows = [rows]
    return [(str(r.get("Name") or ""), str(r.get("AppID") or "")) for r in rows
            if isinstance(r, dict) and r.get("Name") and r.get("AppID")]


def _known_bare_app(name: str) -> bool:
    """`name` is something ShellExecute resolves by itself: registered under
    App Paths (chrome, winword, excel...) or one of a few built-ins."""
    stem = name.casefold().removesuffix(".exe")
    if stem in _BUILTIN_APPS:
        return True
    import winreg

    key = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{stem}.exe"
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        with contextlib.suppress(OSError):
            winreg.CloseKey(winreg.OpenKey(hive, key))
            return True
    return False


def _startfile(target: str) -> None:
    os.startfile(target)   # a resolved shortcut, a known app, or an http(s) URL


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


def best_match(name: str, candidates: list[tuple[str, str]]) -> tuple[str, str] | None:
    """The candidate whose label best matches the spoken `name`: an exact
    (casefolded) label, else the shortest label that starts with it at a
    word boundary, else the shortest that contains it as whole words.
    Uninstallers and help/readme shortcuts are never picked by accident."""
    q = _norm(name)
    if not q:
        return None
    noise = ("uninstall", "readme", "help", "documentation", "release notes")
    pool = [(lbl, tgt) for lbl, tgt in candidates
            if lbl and not any(w in _norm(lbl) and w not in q for w in noise)]
    exact = [c for c in pool if _norm(c[0]) == q]
    if exact:
        return exact[0]
    word = re.compile(rf"(?:^|\W){re.escape(q)}(?:$|\W)")
    prefix = [c for c in pool if _norm(c[0]).startswith(q) and word.search(_norm(c[0]))]
    inner = [c for c in pool if word.search(_norm(c[0]))]
    for group in (prefix, inner):
        if group:
            return min(group, key=lambda c: len(c[0]))
    return None


def _open_app_sync(name: str) -> dict:
    hit = best_match(name, _start_menu_shortcuts())
    if hit is not None:
        _startfile(hit[1])
        return _ok("ok")
    hit = best_match(name, _start_apps())
    if hit is not None:
        aumid = hit[1]
        if os.path.isabs(aumid) and aumid.lower().endswith(".exe"):
            _startfile(aumid)            # a desktop app Get-StartApps lists by path
            return _ok("ok")
        if not _AUMID_RE.fullmatch(aumid) or aumid.startswith("-"):
            return _err(f"can't launch {hit[0]!r}")
        # explorer.exe shell:AppsFolder\<AUMID>, minus explorer's exit code
        # (it reports 1 on success).
        _startfile(f"shell:AppsFolder\\{aumid}")
        return _ok("ok")
    if _known_bare_app(name):
        _startfile(name)
        return _ok("ok")
    return _err(f"there's no app called {name!r} on this PC")


@tool("open_app", "Open a Windows application by name, e.g. Notepad, Microsoft Edge, Spotify",
      {"name": str})
@_guard
async def open_app(args: dict) -> dict:
    name = str(args["name"]).strip()
    # A bare display name only: no paths, drive letters, flags or hidden
    # names, which would let "open an app" run an arbitrary file.
    if (not name or any(c in name for c in "/\\:*?\"<>|") or name.startswith((".", "-"))
            or not _APP_NAME_RE.fullmatch(name)):
        return _err("app name must be a bare application name")
    return await asyncio.to_thread(_open_app_sync, name)


@tool("open_url", "Open an http(s) URL in the default browser", {"url": str})
@_guard
async def open_url(args: dict) -> dict:
    url = str(args.get("url", "")).strip()
    if not url.lower().startswith(("http://", "https://")) or any(ord(c) < 32 or c == " " for c in url):
        return _err("only http(s) URLs are allowed")
    await asyncio.to_thread(_startfile, url)
    return _ok("ok")


# -- clipboard ---------------------------------------------------------------------
@contextlib.contextmanager
def _clipboard():
    """OpenClipboard, retried briefly: another app holding it for a moment
    (a clipboard manager, a copy in progress) is normal."""
    import win32clipboard

    for attempt in range(10):
        try:
            win32clipboard.OpenClipboard()
            break
        except Exception:
            if attempt == 9:
                raise RuntimeError("the clipboard is busy in another app") from None
            time.sleep(0.05)
    try:
        yield win32clipboard
    finally:
        win32clipboard.CloseClipboard()


def _clip_get() -> str:
    with _clipboard() as cb:
        if not cb.IsClipboardFormatAvailable(cb.CF_UNICODETEXT):
            return ""
        return str(cb.GetClipboardData(cb.CF_UNICODETEXT) or "")


def _clip_set(text: str) -> None:
    with _clipboard() as cb:
        cb.EmptyClipboard()
        cb.SetClipboardData(cb.CF_UNICODETEXT, text)


@tool("clipboard_read", "Read the current clipboard text", {})
@_guard
async def clipboard_read(args: dict) -> dict:
    text = await asyncio.to_thread(_clip_get)
    return _ok(text.replace("\r\n", "\n") if text else "The clipboard has no text.")


@tool("clipboard_write", "Replace the clipboard with the given text", {"text": str})
@_guard
async def clipboard_write(args: dict) -> dict:
    await asyncio.to_thread(_clip_set, str(args.get("text", "")).replace("\r\n", "\n").replace("\n", "\r\n"))
    return _ok("ok")


# -- notifications -------------------------------------------------------------
# A toast needs an AppUserModelID that Windows knows. Ours is registered
# under HKCU (DisplayName only — enough for an unpackaged app on Windows 10
# 1903+); if that fails, PowerShell's own, always-registered ID is used.
TOAST_APP_ID = "Veronica.Assistant"
TOAST_APP_NAME = "Veronica"
_POWERSHELL_APP_ID = r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"


def toast_xml(title: str, message: str) -> str:
    return ('<toast><visual><binding template="ToastGeneric">'
            f"<text>{_xml_escape(title)}</text><text>{_xml_escape(message)}</text>"
            "</binding></visual></toast>")


def _toast_app_id() -> str:
    try:
        import winreg

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER,
                              rf"Software\Classes\AppUserModelId\{TOAST_APP_ID}") as k:
            winreg.SetValueEx(k, "DisplayName", 0, winreg.REG_SZ, TOAST_APP_NAME)
        return TOAST_APP_ID
    except Exception:
        return _POWERSHELL_APP_ID


def _toast_winrt(title: str, message: str) -> None:
    from winrt.windows.data.xml.dom import XmlDocument
    from winrt.windows.ui.notifications import (
        ToastNotification,
        ToastNotificationManager,
    )

    doc = XmlDocument()
    doc.load_xml(toast_xml(title, message))
    mgr = ToastNotificationManager
    make = getattr(mgr, "create_toast_notifier_with_id", None) or mgr.create_toast_notifier
    make(_toast_app_id()).show(ToastNotification(doc))


# The same API through PowerShell, for when the winrt packages are missing.
# The text arrives as base64 so nothing the user said is ever parsed as script.
_TOAST_PS = (
    "[void][Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType=WindowsRuntime]\n"
    "[void][Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType=WindowsRuntime]\n"
    "$x = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String('{xml}'))\n"
    "$d = New-Object Windows.Data.Xml.Dom.XmlDocument\n"
    "$d.LoadXml($x)\n"
    "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('{app}')"
    ".Show([Windows.UI.Notifications.ToastNotification]::new($d))\n"
)


def _toast_powershell(title: str, message: str) -> dict:
    xml = base64.b64encode(toast_xml(title, message).encode("utf-8")).decode("ascii")
    return powershell(_TOAST_PS.replace("{xml}", xml).replace("{app}", _POWERSHELL_APP_ID))


def _notify_sync(title: str, message: str) -> dict:
    try:
        _toast_winrt(title, message)
        return _ok("ok")
    except Exception:
        return _toast_powershell(title, message)


@tool("notify", "Show a Windows notification (toast)", {"title": str, "message": str})
@_guard
async def notify(args: dict) -> dict:
    title = str(args.get("title", "") or "")[:200]
    message = str(args.get("message", "") or "")[:1000]
    return await asyncio.to_thread(_notify_sync, title, message)


# -- volume ----------------------------------------------------------------------
def _endpoint_volume():
    """IAudioEndpointVolume of the default output device (call inside _com())."""
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    dev = AudioUtilities.GetSpeakers()
    ev = getattr(dev, "EndpointVolume", None)          # pycaw >= 2025 wraps the device
    if ev is not None:
        return ev
    iface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(iface, POINTER(IAudioEndpointVolume))


def _get_volume() -> int:
    with _com():
        ev = _endpoint_volume()
        return 0 if ev.GetMute() else round(ev.GetMasterVolumeLevelScalar() * 100)


def _set_volume(level: int) -> None:
    with _com():
        ev = _endpoint_volume()
        ev.SetMasterVolumeLevelScalar(level / 100, None)
        ev.SetMute(1 if level == 0 else 0, None)


@tool("volume_get", "Get system output volume (0-100)", {})
@_guard
async def volume_get(args: dict) -> dict:
    return _ok(str(await asyncio.to_thread(_get_volume)))


@tool("volume_set", "Set system output volume (0-100)", {"level": int})
@_guard
async def volume_set(args: dict) -> dict:
    try:
        level = int(float(args.get("level", 0)))
    except (TypeError, ValueError):
        return _err("level must be a number 0-100")
    level = max(0, min(100, level))
    await asyncio.to_thread(_set_volume, level)
    return _ok("ok")


@tool("powershell", "Run a Windows PowerShell script and return its output (powerful; user must confirm)",
      {"script": str})
@_guard
async def powershell_tool(args: dict) -> dict:
    script = str(args.get("script", "") or "")
    if not script.strip():
        return _err("script is required")
    return await asyncio.to_thread(powershell, script, POWERSHELL_TIMEOUT_S)


# -- dictation typing ----------------------------------------------------------
VK_RETURN = 0x0D
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004


def key_events(text: str) -> list[tuple[int, int, int]]:
    """The (vk, scan, flags) keyboard events that type `text`: each UTF-16
    code unit as a KEYEVENTF_UNICODE down/up pair (a character outside the
    BMP is its surrogate pair), and each newline as a real Enter press —
    the same line / Return / line semantics the dictation flow relies on."""
    events = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for i, line in enumerate(lines):
        if i > 0:
            events += [(VK_RETURN, 0, 0), (VK_RETURN, 0, KEYEVENTF_KEYUP)]
        units = line.encode("utf-16-le")
        for j in range(0, len(units), 2):
            cu = int.from_bytes(units[j:j + 2], "little")
            events += [(0, cu, KEYEVENTF_UNICODE), (0, cu, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)]
    return events


def _send_input(events: list[tuple[int, int, int]]) -> int:
    """SendInput the events; returns how many Windows accepted."""
    import ctypes
    from ctypes import wintypes

    ulong_ptr = ctypes.c_size_t

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                    ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr)]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                    ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ulong_ptr)]

    class HARDWAREINPUT(ctypes.Structure):
        _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]

    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]

    class INPUT(ctypes.Structure):
        _fields_ = [("type", wintypes.DWORD), ("u", _U)]

    arr = (INPUT * len(events))()
    for k, (vk, scan, flags) in enumerate(events):
        arr[k].type = 1                                   # INPUT_KEYBOARD
        arr[k].u.ki = KEYBDINPUT(vk, scan, flags, 0, 0)
    send = ctypes.windll.user32.SendInput
    send.argtypes = (wintypes.UINT, ctypes.c_void_p, ctypes.c_int)
    send.restype = wintypes.UINT
    return int(send(len(events), arr, ctypes.sizeof(INPUT)))


def dictate_type(text: str, send=None) -> dict:
    """Type `text` into whatever window is currently focused (dictation, A4).
    Not exposed as a Claude tool — the orchestrator calls this directly for
    the local "dictate"/"stop dictation" intent, never via the brain.
    Synchronous; call via asyncio.to_thread. `send` replaces SendInput in
    tests."""
    events = key_events(text)
    if not events:
        return _ok("ok")
    try:
        sent = (send or _send_input)(events)
    except Exception as exc:
        return _err(str(exc))
    if sent != len(events):
        # UIPI: Windows drops input aimed at a window running elevated.
        return _err("Windows blocked the typing (is the focused app running as administrator?)")
    return _ok("ok")


TOOLS = [open_app, open_url, clipboard_read, clipboard_write, notify, volume_get, volume_set,
         powershell_tool]
SYSTEM_TOOL_NAMES = [t.name for t in TOOLS]
system_server = create_sdk_mcp_server(name="system", version="1.0.0", tools=TOOLS)
