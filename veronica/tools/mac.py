"""Typed macOS actions exposed to Claude as in-process MCP tools."""
import asyncio
import os
import re
import subprocess
import tempfile

from claude_agent_sdk import create_sdk_mcp_server, tool

TIMEOUT_S = 10
# A shortcut is a user-written program (it can wait on a network call or a
# device), so it gets its own, far longer budget than the one-shot commands.
SHORTCUT_TIMEOUT_S = 120


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def run(argv: list[str], stdin: str | None = None, ok_text: str | None = None,
        timeout: int = TIMEOUT_S) -> dict:
    """Run argv (never a shell string) and map the result to MCP content.

    On success, `ok_text` (if given) is returned verbatim instead of stdout —
    used by tools where stdout is not meaningful output (e.g. `open`).
    """
    try:
        done = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return _err(f"timed out after {timeout}s")
    except Exception as exc:  # e.g. FileNotFoundError
        return _err(str(exc))
    if done.returncode != 0:
        return _err(done.stderr.strip() or f"exit {done.returncode}")
    if ok_text is not None:
        return _ok(ok_text)
    out = done.stdout.strip()
    return _ok(out or "ok")


# -- battery (plain helper, not a tool; used by the orchestrator's quick replies)
_BATT_RE = re.compile(r"(\d{1,3})%;\s*(charging|discharging|charged|finishing charge|AC attached)", re.IGNORECASE)


def read_battery(run=subprocess.run) -> tuple[int | None, str | None]:
    """(percent, state) from `pmset -g batt`, state in charging|discharging|
    charged, or None when the percent is known but the state isn't ("AC
    attached; not charging" -- plugged in, battery-health hold); (None, None)
    if pmset is missing, times out, or reports no battery."""
    try:
        p = run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
        m = _BATT_RE.search(p.stdout or "")
    except Exception:
        return None, None
    if not m:
        return None, None
    state = m.group(2).lower()
    if state == "finishing charge":
        state = "charging"
    elif state == "ac attached":
        state = None
    return int(m.group(1)), state


def _q(s: str) -> str:
    """Quote for an AppleScript string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _guard(fn):
    """Wrap a handler so malformed args (missing keys, bad types) return
    `_err(...)` instead of raising — argument extraction happens before
    `run()`'s own error handling, so it needs its own net."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as exc:
            return _err(str(exc))
    return wrapper


@tool("open_app", "Open a macOS application by name, e.g. Safari", {"name": str})
@_guard
async def open_app(args: dict) -> dict:
    name = str(args["name"])
    if "/" in name or name.startswith(".") or name.startswith("-"):
        return _err("app name must be a bare application name")
    return await asyncio.to_thread(run, ["open", "-a", name], None, ok_text="ok")


@tool("open_url", "Open an http(s) URL in the default browser", {"url": str})
@_guard
async def open_url(args: dict) -> dict:
    url = str(args.get("url", ""))
    if not url.startswith(("http://", "https://")):
        return _err("only http(s) URLs are allowed")
    return await asyncio.to_thread(run, ["open", url], None)


@tool("clipboard_read", "Read the current clipboard text", {})
@_guard
async def clipboard_read(args: dict) -> dict:
    return await asyncio.to_thread(run, ["pbpaste"], None)


@tool("clipboard_write", "Replace the clipboard with the given text", {"text": str})
@_guard
async def clipboard_write(args: dict) -> dict:
    return await asyncio.to_thread(run, ["pbcopy"], str(args.get("text", "")))


@tool("notify", "Show a macOS notification banner", {"title": str, "message": str})
@_guard
async def notify(args: dict) -> dict:
    script = f'display notification "{_q(str(args.get("message", "")))}" with title "{_q(str(args.get("title", "")))}"'
    return await asyncio.to_thread(run, ["osascript", "-e", script], None)


@tool("volume_get", "Get system output volume (0-100)", {})
@_guard
async def volume_get(args: dict) -> dict:
    return await asyncio.to_thread(run, ["osascript", "-e", "output volume of (get volume settings)"], None)


@tool("volume_set", "Set system output volume (0-100)", {"level": int})
@_guard
async def volume_set(args: dict) -> dict:
    try:
        level = int(float(args.get("level", 0)))
    except (TypeError, ValueError):
        return _err("level must be a number 0-100")
    level = max(0, min(100, level))
    return await asyncio.to_thread(run, ["osascript", "-e", f"set volume output volume {level}"], None)


@tool("applescript", "Run an AppleScript snippet (powerful; user must confirm)", {"script": str})
@_guard
async def applescript(args: dict) -> dict:
    return await asyncio.to_thread(run, ["osascript", "-e", str(args.get("script", ""))], None)


# -- Shortcuts -----------------------------------------------------------------
def _installed_shortcuts() -> tuple[list[str] | None, str]:
    """(names, error): the shortcuts `shortcuts list` reports, or (None, msg)
    when the CLI itself failed (missing binary, Shortcuts not set up). An
    empty list is a real answer — none are installed."""
    res = run(["shortcuts", "list"])
    if res.get("is_error"):
        return None, res["content"][0]["text"].removeprefix("error: ")
    out = res["content"][0]["text"]
    names = [ln.strip() for ln in out.split("\n") if ln.strip()]
    # `run` turns empty stdout into "ok"; that means "no shortcuts", not one
    # named "ok".
    if names == ["ok"]:
        names = []
    return names, ""


@tool("run_shortcut", "Run a Shortcuts.app shortcut by name, optionally with text input",
      {"name": str, "input": str})
@_guard
async def run_shortcut(args: dict) -> dict:
    name = str(args.get("name", "") or "").strip()
    text_in = str(args.get("input", "") or "")
    if not name:
        return _err("name is required")
    if name.startswith("-"):
        return _err("shortcut name must not start with '-'")
    names, problem = await asyncio.to_thread(_installed_shortcuts)
    if names is None:
        return _err(f"couldn't read the shortcuts list: {problem}")
    match = next((n for n in names if n.lower() == name.lower()), None)
    if match is None:
        # Spoken back to the user, so it has to read as a sentence rather
        # than as the CLI's "Couldn't find shortcut".
        return _err(f"there's no shortcut called {name!r} on this Mac")
    return await asyncio.to_thread(_run_shortcut, match, text_in)


def _run_shortcut(name: str, text_in: str) -> dict:
    """`shortcuts run` has no stdin: input is a file handed over with
    --input-path, so text input goes through a temp file we clean up."""
    argv = ["shortcuts", "run", name]
    path = ""
    try:
        if text_in:
            fd, path = tempfile.mkstemp(prefix="veronica-shortcut-", suffix=".txt")
            with os.fdopen(fd, "w") as fh:
                fh.write(text_in)
            argv += ["--input-path", path]
        # The CLI prints nothing on success (output needs --output-path), so
        # the confirmation is ours.
        return run(argv, None, ok_text=f"Ran {name}", timeout=SHORTCUT_TIMEOUT_S)
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def _keystroke_argv(text: str) -> list[str]:
    """Build the `osascript` argv that types `text` into the frontmost app
    via System Events (Accessibility permission required). Splits on
    newlines into separate `keystroke "<line>"` calls joined by
    `keystroke return`, since a literal newline can't be smuggled through a
    single AppleScript string literal."""
    lines = text.split("\n")
    parts = []
    for i, line in enumerate(lines):
        if i > 0:
            parts.append("keystroke return")
        parts.append(f'keystroke "{_q(line)}"')
    body = "\n".join(parts)
    script = f'tell application "System Events"\n{body}\nend tell'
    return ["osascript", "-e", script]


def dictate_type(text: str) -> dict:
    """Type `text` into whatever app is currently focused (dictation, A4).
    Not exposed as a Claude tool — the orchestrator calls this directly for
    the local "dictate"/"stop dictation" intent, never via the brain.
    Synchronous; call via asyncio.to_thread."""
    return run(_keystroke_argv(text))


TOOLS = [open_app, open_url, clipboard_read, clipboard_write, notify, volume_get, volume_set,
         applescript, run_shortcut]
MAC_TOOL_NAMES = [t.name for t in TOOLS]
mac_server = create_sdk_mcp_server(name="mac", version="1.0.0", tools=TOOLS)
