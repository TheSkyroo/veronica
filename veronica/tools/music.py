"""Music playback control, exposed to Claude as in-process MCP tools.

Playback goes through Windows' Global System Media Transport Controls (the
same session the media overlay and keyboard media keys drive), so it works
with whatever player currently owns it: Spotify, a browser tab, Groove /
Media Player... A search opens Spotify's own search when Spotify is
installed, else YouTube Music in the browser. All allow-class — playback
control only, no filesystem/network access beyond what the player does.

The winrt and pycaw imports live inside the functions: they exist on
Windows only, and the tests replace `_media_session`, `_spotify_installed`,
`_open` and `_set_app_volume` with fakes.
"""
import asyncio
import contextlib
import os
import urllib.parse
from pathlib import Path

from claude_agent_sdk import create_sdk_mcp_server, tool

NOTHING_PLAYING = "Nothing is playing."
NO_PLAYER = "no music app is open; say what to play"
# GlobalSystemMediaTransportControlsSessionPlaybackStatus
STATUS_PLAYING = 4
STATUS_PAUSED = 5


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def _guard(fn):
    """Wrap a handler so malformed args/unexpected failures return
    `_err(...)` instead of raising."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as exc:
            return _err(str(exc))
    return wrapper


# -- OS seams ---------------------------------------------------------------------
async def _media_session():
    """The current GSMTC session (the player Windows' media keys would
    control), or None when no app has one."""
    from winrt.windows.media.control import (
        GlobalSystemMediaTransportControlsSessionManager as Manager,
    )

    mgr = await Manager.request_async()
    return mgr.get_current_session()


def _spotify_installed() -> bool:
    """Spotify's desktop app (in %APPDATA%), its Store app (an execution
    alias in WindowsApps) or anything registering the spotify: protocol."""
    for var, rel in (("APPDATA", r"Spotify\Spotify.exe"),
                     ("LOCALAPPDATA", r"Microsoft\WindowsApps\Spotify.exe")):
        base = os.environ.get(var)
        if base and Path(base, rel).exists():
            return True
    with contextlib.suppress(Exception):
        import winreg

        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, "spotify"))
        return True
    return False


def _open(target: str) -> None:
    os.startfile(target)   # a spotify:search: URI or an https URL we built


def _app_matches(app_id: str, process_name: str) -> bool:
    """Whether a GSMTC source app id ("Spotify.exe", "chrome",
    "SpotifyAB.SpotifyMusic_zpdnekdrzrea0!Spotify") names the process
    behind an audio session ("Spotify.exe")."""
    stem = process_name.casefold().removesuffix(".exe")
    if len(stem) < 3:
        return False
    app = app_id.casefold()
    parts = {app.removesuffix(".exe"), app.rpartition("!")[2]}
    parts |= {p for p in app.partition("_")[0].split(".") if p}     # package name pieces
    return stem in parts or any(p.startswith(stem) or stem.startswith(p) for p in parts if len(p) >= 4)


def _set_app_volume(app_id: str, level: int) -> str | None:
    """Set the per-app (mixer) volume of every audio session that belongs to
    `app_id`'s process; returns that process name, or None if it has no
    audio session. Synchronous — call via asyncio.to_thread."""
    import comtypes
    from pycaw.pycaw import AudioUtilities

    comtypes.CoInitialize()
    try:
        hit = None
        for s in AudioUtilities.GetAllSessions():
            proc = getattr(s, "Process", None)
            name = proc.name() if proc is not None else ""
            if _app_matches(app_id, name):
                s.SimpleAudioVolume.SetMasterVolume(level / 100, None)
                hit = name
        return hit
    finally:
        comtypes.CoUninitialize()


# -- tools --------------------------------------------------------------------------
async def _session_or_none():
    try:
        return await _media_session()
    except Exception:
        return None


async def _control(method: str, done: str) -> dict:
    session = await _session_or_none()
    if session is None:
        return _err(NO_PLAYER)
    if not await getattr(session, method)():
        return _err("the player didn't accept that")
    return _ok(done)


def _search_target(query: str) -> tuple[str, str]:
    """(URI to open, what to say) for a search. The query is a URI path /
    query component: percent-encode it (spaces, '&', '#', non-ASCII...)."""
    if _spotify_installed():
        return f"spotify:search:{urllib.parse.quote(query, safe='')}", f"Searching Spotify for {query}."
    return (f"https://music.youtube.com/search?q={urllib.parse.quote_plus(query)}",
            f"Searching YouTube Music for {query}.")


@tool("music_play", "Play music, optionally searching for a track/artist", {"query": str})
@_guard
async def music_play(args: dict) -> dict:
    query = " ".join(str(args.get("query", "") or "").split())[:200]
    if not query:
        return await _control("try_play_async", "Playing.")
    target, said = await asyncio.to_thread(_search_target, query)
    await asyncio.to_thread(_open, target)
    return _ok(said)


@tool("music_pause", "Pause the currently playing music", {})
@_guard
async def music_pause(args: dict) -> dict:
    return await _control("try_pause_async", "Paused.")


@tool("music_next", "Skip to the next track", {})
@_guard
async def music_next(args: dict) -> dict:
    return await _control("try_skip_next_async", "Skipped.")


@tool("music_prev", "Go back to the previous track", {})
@_guard
async def music_prev(args: dict) -> dict:
    return await _control("try_skip_previous_async", "Playing previous track.")


@tool("music_now_playing", "Get the currently playing track, if any", {})
@_guard
async def music_now_playing(args: dict) -> dict:
    session = await _session_or_none()
    if session is None:
        return _ok(NOTHING_PLAYING)
    try:
        status = int(session.get_playback_info().playback_status)
        props = await session.try_get_media_properties_async()
    except Exception:
        return _ok(NOTHING_PLAYING)
    if status not in (STATUS_PLAYING, STATUS_PAUSED):
        return _ok(NOTHING_PLAYING)
    name = " ".join(str(getattr(props, "title", "") or "").split())
    artist = " ".join(str(getattr(props, "artist", "") or "").split())
    if not name:
        return _ok(NOTHING_PLAYING)
    verb = "Now playing" if status == STATUS_PLAYING else "Paused on"
    return _ok(f"{verb} {name} by {artist}." if artist else f"{verb} {name}.")


@tool("music_volume", "Set the music player's volume (0-100)", {"level": int})
@_guard
async def music_volume(args: dict) -> dict:
    try:
        level = int(float(args.get("level", 0)))
    except (TypeError, ValueError):
        return _err("level must be a number 0-100")
    level = max(0, min(100, level))
    session = await _session_or_none()
    if session is None:
        return _err(NO_PLAYER)
    app_id = str(session.source_app_user_model_id or "")
    if await asyncio.to_thread(_set_app_volume, app_id, level) is None:
        return _err(f"couldn't find {app_id or 'the player'}'s audio to change its volume")
    return _ok(f"Volume set to {level}.")


TOOLS = [music_play, music_pause, music_next, music_prev, music_now_playing, music_volume]
MUSIC_TOOL_NAMES = [t.name for t in TOOLS]
music_server = create_sdk_mcp_server(name="music", version="1.0.0", tools=TOOLS)
