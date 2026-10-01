"""Music playback control, exposed to Claude as in-process MCP tools.

Pause / next / previous / now-playing / volume go through Windows' Global
System Media Transport Controls (the session the media overlay and keyboard
media keys drive), so they work with whatever player owns it: Spotify, a
browser tab, Media Player...

"Play <something>" actually starts playback:
  1. Spotify, when the user connected their account (veronica.spotify_account):
     Web API search -> pick a device (the active one, else this PC's Spotify
     app, launching it and waiting up to SPOTIFY_LAUNCH_WAIT_S if needed) ->
     start playback there. A query naming an artist exactly ("daft punk")
     plays the artist (their top tracks); "playlist <name>" plays the top
     playlist; anything else plays the top track. Playback control needs
     Spotify Premium: on PREMIUM_REQUIRED (or no device, or any API trouble)
     we fall back to 2.
  2. YouTube, no account needed: the top video of a YouTube search (scraped
     from the results page's ytInitialData, no API key), opened at
     www.youtube.com/watch?v=<id> in the default browser — www rather than
     music.youtube.com because its watch page autoplays reliably for a
     fresh tab. If the search can't be read, the results page is opened.
All allow-class: playback control and opening a media URL only.

The winrt and pycaw imports live inside the functions: they exist on
Windows only, and the tests replace `_media_session`, `_spotify_installed`,
`_open`, `_set_app_volume`, `_spotify_connected`, `_spotify_api`,
`_http_get` and `_sleep` with fakes.
"""
import asyncio
import contextlib
import json
import logging
import os
import platform
import re
import time
import urllib.parse
from pathlib import Path

import httpx
from claude_agent_sdk import create_sdk_mcp_server, tool

from veronica import spotify_account

log = logging.getLogger("veronica.music")

NOTHING_PLAYING = "Nothing is playing."
NO_PLAYER = "no music app is open; say what to play"
# GlobalSystemMediaTransportControlsSessionPlaybackStatus
STATUS_PLAYING = 4
STATUS_PAUSED = 5

SPOTIFY_API = "https://api.spotify.com/v1"
SPOTIFY_LAUNCH_WAIT_S = 10.0
SPOTIFY_POLL_S = 1.0
HTTP_TIMEOUT_S = 8.0
PREMIUM_NOTE = "Spotify Premium is needed for playback control"
YT_RESULTS = "https://www.youtube.com/results?search_query="
YT_WATCH = "https://www.youtube.com/watch?v="
YT_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}
# Pre-answered consent so EU visitors get results, not the consent wall.
YT_COOKIES = {"CONSENT": "YES+cb", "SOCS": "CAI"}


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
    """Hand a URL we built to the shell: http(s) pages and the spotify: app
    URI only, never a path or another protocol."""
    if not target.startswith(("https://", "http://", "spotify:")):
        raise ValueError(f"refusing to open {target!r}")
    os.startfile(target)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _spotify_connected() -> bool:
    return spotify_account.is_connected()


class SpotifyError(Exception):
    """A Spotify Web API failure; `reason` is Spotify's reason code
    (PREMIUM_REQUIRED, NO_ACTIVE_DEVICE...) when it gave one."""

    def __init__(self, status: int, message: str, reason: str = ""):
        super().__init__(message)
        self.status, self.reason = status, reason


def _spotify_api(method: str, path: str, params: dict | None = None,
                 body: dict | None = None) -> dict:
    """Call the Spotify Web API as the connected user; the JSON reply ({}
    for an empty one). Non-2xx raises SpotifyError. Synchronous."""
    token = spotify_account.access_token()
    if not token:
        raise SpotifyError(401, "Spotify isn't connected")
    try:
        r = httpx.request(method, SPOTIFY_API + path, params=params, json=body,
                          headers={"Authorization": f"Bearer {token}"}, timeout=HTTP_TIMEOUT_S)
    except httpx.HTTPError as e:
        raise SpotifyError(0, f"couldn't reach Spotify: {e}") from e
    try:
        data = r.json() if r.content else {}
    except ValueError:
        data = {}
    if r.status_code >= 300:
        err = data.get("error") if isinstance(data, dict) else None
        err = err if isinstance(err, dict) else {}
        raise SpotifyError(r.status_code, str(err.get("message") or f"HTTP {r.status_code}"),
                           str(err.get("reason") or ""))
    return data if isinstance(data, dict) else {}


def _http_get(url: str) -> str:
    """GET a web page as a desktop browser in English would. Synchronous."""
    r = httpx.get(url, headers=YT_HEADERS, cookies=YT_COOKIES, timeout=HTTP_TIMEOUT_S,
                  follow_redirects=True)
    r.raise_for_status()
    return r.text


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


# -- Spotify ----------------------------------------------------------------------------
def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


def _first(items) -> dict | None:
    return next((i for i in items or [] if isinstance(i, dict) and i.get("uri")), None)


def _spotify_pick(query: str) -> tuple[dict, str]:
    """(play body, what's playing) for a query: "playlist X" -> the top
    playlist; a query that *is* an artist's name -> that artist (Spotify
    plays their top tracks); else the top track."""
    words = query.split()
    if len(words) > 1 and words[0].casefold() == "playlist":
        found = _spotify_api("GET", "/search", {"q": " ".join(words[1:]), "type": "playlist", "limit": 1})
        pl = _first(found.get("playlists", {}).get("items"))
        if pl is None:
            raise SpotifyError(404, f"no Spotify playlist for {query}")
        return {"context_uri": pl["uri"]}, f"the playlist {pl.get('name') or query}"
    found = _spotify_api("GET", "/search", {"q": query, "type": "track,artist", "limit": 1,
                                            "market": "from_token"})
    artist = _first(found.get("artists", {}).get("items"))
    if artist is not None and _norm(artist.get("name", "")) == _norm(query):
        return {"context_uri": artist["uri"]}, str(artist.get("name"))
    track = _first(found.get("tracks", {}).get("items"))
    if track is None:
        raise SpotifyError(404, f"nothing on Spotify for {query}")
    by = ", ".join(a.get("name", "") for a in track.get("artists", []) if a.get("name"))
    name = track.get("name") or query
    return {"uris": [track["uri"]]}, f"{name} by {by}" if by else name


def _this_pc() -> str:
    return (os.environ.get("COMPUTERNAME") or platform.node()).casefold()


def _choose_device(devices: list) -> dict | None:
    """The active device, else this PC's Spotify app."""
    usable = [d for d in devices if isinstance(d, dict) and d.get("id") and not d.get("is_restricted")]
    for d in usable:
        if d.get("is_active"):
            return d
    pcs = [d for d in usable if str(d.get("type", "")).casefold() == "computer"]
    me = _this_pc()
    return next((d for d in pcs if str(d.get("name", "")).casefold() == me), pcs[0] if pcs else None)


def _spotify_device() -> dict | None:
    """A device to play on, launching the Spotify app and waiting for it to
    register when there's none."""
    device = _choose_device(_spotify_api("GET", "/me/player/devices").get("devices", []))
    if device is not None or not _spotify_installed():
        return device
    _open("spotify:")
    deadline = time.monotonic() + SPOTIFY_LAUNCH_WAIT_S
    for _ in range(int(SPOTIFY_LAUNCH_WAIT_S / SPOTIFY_POLL_S) + 1):
        _sleep(SPOTIFY_POLL_S)
        device = _choose_device(_spotify_api("GET", "/me/player/devices").get("devices", []))
        if device is not None or time.monotonic() > deadline:
            break
    return device


def _play_spotify(query: str) -> str:
    """Start `query` on Spotify; the reply text. SpotifyError when it can't."""
    body, what = _spotify_pick(query)
    device = _spotify_device()
    if device is None:
        raise SpotifyError(404, "no Spotify device available", "NO_ACTIVE_DEVICE")
    _spotify_api("PUT", "/me/player/play", {"device_id": device["id"]}, body)
    return f"Playing {what} on Spotify."


# -- YouTube ----------------------------------------------------------------------------
_YT_DATA = re.compile(r"(?:var\s+ytInitialData|window\[[\"']ytInitialData[\"']\])\s*=\s*")
_YT_ID = re.compile(r'"videoId":"([\w-]{11})"')


def _yt_text(node) -> str:
    if not isinstance(node, dict):
        return ""
    if node.get("simpleText"):
        return str(node["simpleText"])
    return "".join(str(r.get("text", "")) for r in node.get("runs", []) if isinstance(r, dict))


def _yt_renderers(node):
    """Every videoRenderer in page order (ads and shorts use other
    renderer types, so they're skipped by construction)."""
    stack = [node]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            vr = cur.get("videoRenderer")
            if isinstance(vr, dict):
                yield vr
            stack.extend(reversed(list(cur.values())))
        elif isinstance(cur, list):
            stack.extend(reversed(cur))


def _yt_is_live(vr: dict) -> bool:
    badges = json.dumps(vr.get("badges", [])) + json.dumps(vr.get("thumbnailOverlays", []))
    return "LIVE" in badges or "lengthText" not in vr


def _youtube_top(html: str) -> tuple[str, str] | None:
    """(video id, title) of the first regular video on a YouTube results
    page; title "" when only the regex fallback found an id."""
    m = _YT_DATA.search(html)
    if m:
        try:
            data, _ = json.JSONDecoder().raw_decode(html, m.end())
        except ValueError:
            data = None
        fallback = None
        for vr in _yt_renderers(data):
            vid = str(vr.get("videoId", ""))
            if not re.fullmatch(r"[\w-]{11}", vid):
                continue
            hit = (vid, " ".join(_yt_text(vr.get("title")).split()))
            if not _yt_is_live(vr):
                return hit
            fallback = fallback or hit
        if fallback:
            return fallback
    m = _YT_ID.search(html)
    return (m.group(1), "") if m else None


def _play_youtube(query: str, note: str = "") -> str:
    lead = f"{note}, so " if note else ""
    results = YT_RESULTS + urllib.parse.quote_plus(query)
    try:
        top = _youtube_top(_http_get(results))
    except Exception as e:
        log.info("music: YouTube search failed: %s", e)
        top = None
    if top is None:
        _open(results)
        said = f"I couldn't start a video, so I opened YouTube results for {query}."
        return f"{note}. {said}" if note else said
    vid, title = top
    _open(YT_WATCH + vid)
    said = f"playing {title or query} on YouTube."
    return lead + said if lead else said[0].upper() + said[1:]


def _play_query(query: str) -> str:
    """Play `query`: Spotify when connected, else (or if that fails)
    YouTube. Synchronous (network + polling) — run via asyncio.to_thread."""
    note = ""
    try:
        connected = _spotify_connected()
    except Exception:
        connected = False
    if connected:
        try:
            return _play_spotify(query)
        except Exception as e:
            log.info("music: Spotify play failed: %s", e)
            if isinstance(e, SpotifyError) and e.reason == "PREMIUM_REQUIRED":
                note = PREMIUM_NOTE
    return _play_youtube(query, note)


@tool("music_play", "Play music, optionally searching for a track/artist", {"query": str})
@_guard
async def music_play(args: dict) -> dict:
    query = " ".join(str(args.get("query", "") or "").split())[:200]
    if not query:
        return await _control("try_play_async", "Playing.")
    return _ok(await asyncio.to_thread(_play_query, query))


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
