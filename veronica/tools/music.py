"""Music playback control, exposed to Claude as in-process MCP tools.
Backend is auto-detected per call: Spotify.app if it's running, else
Music.app. All allow-class — playback control only, no filesystem/network
access beyond what Spotify/Music.app already do on their own.
"""
import asyncio
import subprocess

from claude_agent_sdk import create_sdk_mcp_server, tool

TIMEOUT_S = 10
NOTHING_PLAYING = "Nothing is playing."


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def run(argv: list[str], stdin: str | None = None, ok_text: str | None = None) -> dict:
    """Run argv (never a shell string) and map the result to MCP content."""
    try:
        done = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _err(f"timed out after {TIMEOUT_S}s")
    except Exception as exc:
        return _err(str(exc))
    if done.returncode != 0:
        return _err(done.stderr.strip() or f"exit {done.returncode}")
    if ok_text is not None:
        return _ok(ok_text)
    out = done.stdout.strip()
    return _ok(out or "ok")


def _q(s: str) -> str:
    """Quote for an AppleScript string literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _guard(fn):
    """Wrap a handler so malformed args/unexpected failures return
    `_err(...)` instead of raising."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as exc:
            return _err(str(exc))
    return wrapper


def _osa(script: str, ok_text: str | None = None) -> dict:
    return run(["osascript", "-e", script], None, ok_text)


def _app_running(name: str) -> bool:
    res = _osa(f'application "{name}" is running')
    return not res.get("is_error") and res["content"][0]["text"].strip() == "true"


def detect_backend() -> str:
    """'spotify' if Spotify.app is currently running, else 'music'
    (Music.app). Synchronous — call via asyncio.to_thread."""
    return "spotify" if _app_running("Spotify") else "music"


# -- sync implementations (called via asyncio.to_thread from the tool handlers) --

def _play_sync(query: str) -> dict:
    be = detect_backend()
    if not query:
        app = "Spotify" if be == "spotify" else "Music"
        return _osa(f'tell application "{app}" to play', ok_text="Playing.")
    if be == "spotify":
        # Spotify's AppleScript dictionary has no search command; open the
        # spotify:search: URL (which the Spotify app handles itself) then
        # play once it's loaded a result.
        script = (
            f'tell application "Spotify" to open location "spotify:search:{_q(query)}"\n'
            "delay 1\n"
            'tell application "Spotify" to play\n'
        )
    else:
        script = (
            'tell application "Music"\n'
            f'play (first track of playlist "Library" whose name contains "{_q(query)}" '
            f'or artist contains "{_q(query)}")\n'
            "end tell\n"
        )
    return _osa(script, ok_text=f"Playing {query}.")


def _pause_sync() -> dict:
    app = "Spotify" if detect_backend() == "spotify" else "Music"
    return _osa(f'tell application "{app}" to pause', ok_text="Paused.")


def _next_sync() -> dict:
    app = "Spotify" if detect_backend() == "spotify" else "Music"
    return _osa(f'tell application "{app}" to next track', ok_text="Skipped.")


def _prev_sync() -> dict:
    app = "Spotify" if detect_backend() == "spotify" else "Music"
    return _osa(f'tell application "{app}" to previous track', ok_text="Playing previous track.")


def _now_playing_sync() -> dict:
    be = detect_backend()
    app = "Spotify" if be == "spotify" else "Music"
    if not _app_running(app):
        return _ok(NOTHING_PLAYING)
    script = (
        f'tell application "{app}"\n'
        "if player state is playing or player state is paused then\n"
        'return (name of current track) & tab & (artist of current track) & tab & (player state as string)\n'
        "else\n"
        'return ""\n'
        "end if\n"
        "end tell\n"
    )
    res = _osa(script)
    if res.get("is_error"):
        return _ok(NOTHING_PLAYING)
    raw = res["content"][0]["text"].strip()
    if not raw or raw == "ok":
        return _ok(NOTHING_PLAYING)
    parts = raw.split("\t")
    if len(parts) < 3:
        return _ok(NOTHING_PLAYING)
    name, artist, state = parts[0], parts[1], parts[2]
    verb = "Now playing" if "playing" in state else "Paused on"
    return _ok(f"{verb} {name} by {artist}.")


def _volume_sync(level: int) -> dict:
    level = max(0, min(100, level))
    app = "Spotify" if detect_backend() == "spotify" else "Music"
    return _osa(f'tell application "{app}" to set sound volume to {level}', ok_text=f"Volume set to {level}.")


@tool("music_play", "Play music, optionally searching for a track/artist", {"query": str})
@_guard
async def music_play(args: dict) -> dict:
    query = str(args.get("query", "") or "").strip()
    return await asyncio.to_thread(_play_sync, query)


@tool("music_pause", "Pause the currently playing music", {})
@_guard
async def music_pause(args: dict) -> dict:
    return await asyncio.to_thread(_pause_sync)


@tool("music_next", "Skip to the next track", {})
@_guard
async def music_next(args: dict) -> dict:
    return await asyncio.to_thread(_next_sync)


@tool("music_prev", "Go back to the previous track", {})
@_guard
async def music_prev(args: dict) -> dict:
    return await asyncio.to_thread(_prev_sync)


@tool("music_now_playing", "Get the currently playing track, if any", {})
@_guard
async def music_now_playing(args: dict) -> dict:
    return await asyncio.to_thread(_now_playing_sync)


@tool("music_volume", "Set the music player's volume (0-100)", {"level": int})
@_guard
async def music_volume(args: dict) -> dict:
    try:
        level = int(float(args.get("level", 0)))
    except (TypeError, ValueError):
        return _err("level must be a number 0-100")
    return await asyncio.to_thread(_volume_sync, level)


TOOLS = [music_play, music_pause, music_next, music_prev, music_now_playing, music_volume]
MUSIC_TOOL_NAMES = [t.name for t in TOOLS]
music_server = create_sdk_mcp_server(name="music", version="1.0.0", tools=TOOLS)
