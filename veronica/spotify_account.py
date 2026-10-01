"""Veronica's link to the user's Spotify account, for the music tools.

Spotify's Web API with the Authorization Code + PKCE flow — no client
secret, so nothing secret ships with Veronica. The user registers their own
app on developer.spotify.com, adds the redirect URI
http://127.0.0.1:<REDIRECT_PORT>/callback (Spotify only accepts the loopback
IP literal for http redirects, not "localhost") and hands Veronica its
Client ID, either as VERONICA_SPOTIFY_CLIENT_ID or in
~/.veronica/spotify_client_id (VERONICA_HOME honoured).

connect() is interactive — it opens the browser on Spotify's consent page
and waits for the redirect on a one-shot loopback http.server — so it is for
the settings UI / an explicit "connect Spotify", never from inside a tool.
The tokens live in ~/.veronica/spotify_token.json (owner-only); access_token()
refreshes them when they are about to expire.

All synchronous (httpx, http.server); async callers use asyncio.to_thread.
The token endpoint goes through `_token_request`, which tests replace.
"""
import base64
import hashlib
import http.server
import json
import logging
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

import httpx

log = logging.getLogger("veronica.spotify")

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPES = "user-read-playback-state user-modify-playback-state"
REDIRECT_PORT = 8898
CONNECT_TIMEOUT_S = 180.0
HTTP_TIMEOUT_S = 10.0
EXPIRY_MARGIN_S = 60.0


class SpotifyAuthError(Exception):
    pass


# -- where things live ----------------------------------------------------------------
def _home() -> Path:
    env = os.environ.get("VERONICA_HOME")
    return Path(env).expanduser() if env else Path.home() / ".veronica"


def _token_path() -> Path:
    return _home() / "spotify_token.json"


def client_id() -> str:
    """The user's Spotify app Client ID: VERONICA_SPOTIFY_CLIENT_ID, else
    the first line of ~/.veronica/spotify_client_id, else the Settings
    field (prefs.json / .env); "" when none."""
    cid = os.environ.get("VERONICA_SPOTIFY_CLIENT_ID", "").strip()
    if cid:
        return cid
    try:
        return (_home() / "spotify_client_id").read_text(encoding="utf-8").strip().split()[0]
    except (OSError, IndexError):
        pass
    return _settings_client_id()


def _settings_client_id() -> str:
    """The Client ID typed into Settings (prefs.json) or set in .env."""
    try:
        from veronica import prefs
        from veronica.config import load_settings

        return load_settings(prefs.load().get("settings")).spotify_client_id.strip()
    except Exception:  # noqa: BLE001  (a broken prefs file means "not configured")
        return ""


def redirect_port() -> int:
    try:
        return int(os.environ.get("VERONICA_SPOTIFY_PORT", "") or REDIRECT_PORT)
    except ValueError:
        return REDIRECT_PORT


def redirect_uri(port: int | None = None) -> str:
    return f"http://127.0.0.1:{port or redirect_port()}/callback"


def _load() -> dict | None:
    try:
        data = json.loads(_token_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("refresh_token") else None


def _save(token: dict) -> None:
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(token, f)
    os.replace(tmp, path)


# -- token endpoint seam -------------------------------------------------------------------
def _token_request(data: dict) -> dict:
    """POST a form to Spotify's token endpoint; the JSON reply, or
    SpotifyAuthError carrying Spotify's error code (e.g. "invalid_grant")."""
    try:
        r = httpx.post(TOKEN_URL, data=data, timeout=HTTP_TIMEOUT_S)
    except httpx.HTTPError as e:
        raise SpotifyAuthError(f"couldn't reach Spotify: {e}") from e
    try:
        body = r.json()
    except ValueError:
        body = {}
    if r.status_code != 200:
        raise SpotifyAuthError(str(body.get("error") or f"HTTP {r.status_code}"))
    return body


def _store_reply(reply: dict, previous: dict | None = None) -> dict:
    """Merge a token-endpoint reply into what we keep (Spotify may omit
    the refresh token on refresh: keep the old one then)."""
    access = reply.get("access_token")
    refresh = reply.get("refresh_token") or (previous or {}).get("refresh_token")
    if not access or not refresh:
        raise SpotifyAuthError("Spotify's reply had no token")
    token = {
        "access_token": access,
        "refresh_token": refresh,
        "expires_at": time.time() + float(reply.get("expires_in", 3600)),
        "scope": reply.get("scope", (previous or {}).get("scope", SCOPES)),
    }
    _save(token)
    return token


# -- public API -------------------------------------------------------------------------
def is_configured() -> bool:
    """A Client ID is set, so connect() can run."""
    return bool(client_id())


def is_connected() -> bool:
    """Configured and holding a (refreshable) token."""
    return is_configured() and _load() is not None


def disconnect() -> None:
    try:
        _token_path().unlink()
    except FileNotFoundError:
        pass


def access_token() -> str | None:
    """A valid access token, refreshing it if needed; None when not
    connected or the user revoked access (the stale token is dropped then).
    Network trouble raises SpotifyAuthError."""
    cid = client_id()
    token = _load()
    if not cid or token is None:
        return None
    if float(token.get("expires_at", 0)) - EXPIRY_MARGIN_S > time.time() and token.get("access_token"):
        return token["access_token"]
    try:
        reply = _token_request({"grant_type": "refresh_token",
                                "refresh_token": token["refresh_token"], "client_id": cid})
    except SpotifyAuthError as e:
        if str(e) in ("invalid_grant", "invalid_client"):
            log.info("spotify: refresh refused (%s); disconnecting", e)
            disconnect()
            return None
        raise
    return _store_reply(reply, token)["access_token"]


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)[:128]
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    return verifier, challenge


def authorize_url(cid: str, challenge: str, state: str, port: int | None = None) -> str:
    return AUTHORIZE_URL + "?" + urllib.parse.urlencode({
        "client_id": cid, "response_type": "code", "redirect_uri": redirect_uri(port),
        "code_challenge_method": "S256", "code_challenge": challenge,
        "state": state, "scope": SCOPES,
    })


_PAGE = ("<!doctype html><meta charset=utf-8><title>Veronica</title>"
         "<body style='font-family:sans-serif;padding:2em'><h2>{}</h2>"
         "<p>You can close this tab.</p>")


def _wait_for_code(port: int, state: str, timeout: float, ready=None) -> str:
    """Serve http://127.0.0.1:<port>/callback until Spotify redirects there
    (or `timeout` passes); the authorization code."""
    result: dict = {}
    done = threading.Event()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            if url.path != "/callback":
                self.send_error(404)
                return
            q = urllib.parse.parse_qs(url.query)
            if q.get("state", [""])[0] != state:
                result["error"] = "state mismatch"
            elif "error" in q:
                result["error"] = q["error"][0]
            else:
                result["code"] = q.get("code", [""])[0]
            ok = bool(result.get("code"))
            body = _PAGE.format("Spotify connected to Veronica." if ok
                                else "Spotify wasn't connected.").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            done.set()

        def log_message(self, *a):
            pass

    try:
        server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        raise SpotifyAuthError(f"port {port} is busy: {e}") from e
    server.timeout = 0.5
    try:
        if ready is not None:
            ready()
        deadline = time.monotonic() + timeout
        while not done.is_set() and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    if result.get("code"):
        return result["code"]
    raise SpotifyAuthError(result.get("error") or "timed out waiting for Spotify")


def connect(open_browser=webbrowser.open, timeout: float = CONNECT_TIMEOUT_S) -> None:
    """Interactive: open Spotify's consent page in the browser, catch the
    redirect on the loopback port, exchange the code and store the token.
    Blocks up to `timeout`; raises SpotifyAuthError on failure."""
    cid = client_id()
    if not cid:
        raise SpotifyAuthError("no Spotify Client ID set (VERONICA_SPOTIFY_CLIENT_ID "
                               "or ~/.veronica/spotify_client_id)")
    port = redirect_port()
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(16)
    url = authorize_url(cid, challenge, state, port)
    code = _wait_for_code(port, state, timeout, ready=lambda: open_browser(url))
    reply = _token_request({"grant_type": "authorization_code", "code": code,
                            "redirect_uri": redirect_uri(port), "client_id": cid,
                            "code_verifier": verifier})
    _store_reply(reply)
    log.info("spotify: connected")
