import json
import os
import threading
import time
import urllib.parse
import urllib.request

import pytest

from veronica import spotify_account as sa


@pytest.fixture
def home(tmp_home, monkeypatch):
    monkeypatch.delenv("VERONICA_SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("VERONICA_SPOTIFY_PORT", raising=False)
    return tmp_home


@pytest.fixture
def token_api(monkeypatch):
    """The token endpoint: records requests, replies from a queue
    (an exception in it is raised)."""
    calls, replies = [], []

    def request(data):
        calls.append(data)
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(sa, "_token_request", request)
    return calls, replies


def write_token(home, **kw):
    tok = {"access_token": "old", "refresh_token": "r1", "expires_at": time.time() + 3600,
           "scope": sa.SCOPES, **kw}
    (home / "spotify_token.json").write_text(json.dumps(tok))


def test_client_id_from_env_or_file(home, monkeypatch):
    assert sa.client_id() == "" and not sa.is_configured()
    (home / "spotify_client_id").write_text("  abc123 \n")
    assert sa.client_id() == "abc123" and sa.is_configured()
    monkeypatch.setenv("VERONICA_SPOTIFY_CLIENT_ID", "fromenv")
    assert sa.client_id() == "fromenv"


def test_redirect_uri_is_loopback_ip(home, monkeypatch):
    assert sa.redirect_uri() == f"http://127.0.0.1:{sa.REDIRECT_PORT}/callback"
    monkeypatch.setenv("VERONICA_SPOTIFY_PORT", "9999")
    assert sa.redirect_uri() == "http://127.0.0.1:9999/callback"


def test_connected_needs_client_id_and_token(home, monkeypatch):
    write_token(home)
    assert not sa.is_connected()
    monkeypatch.setenv("VERONICA_SPOTIFY_CLIENT_ID", "cid")
    assert sa.is_connected()
    sa.disconnect()
    assert not sa.is_connected()
    sa.disconnect()   # idempotent


def test_access_token_fresh_is_returned_without_refresh(home, monkeypatch, token_api):
    monkeypatch.setenv("VERONICA_SPOTIFY_CLIENT_ID", "cid")
    write_token(home)
    assert sa.access_token() == "old" and token_api[0] == []


def test_access_token_not_connected_is_none(home, monkeypatch, token_api):
    monkeypatch.setenv("VERONICA_SPOTIFY_CLIENT_ID", "cid")
    assert sa.access_token() is None


def test_access_token_refreshes_and_keeps_refresh_token(home, monkeypatch, token_api):
    monkeypatch.setenv("VERONICA_SPOTIFY_CLIENT_ID", "cid")
    write_token(home, expires_at=time.time() + 10)
    calls, replies = token_api
    replies.append({"access_token": "new", "expires_in": 3600})
    assert sa.access_token() == "new"
    assert calls == [{"grant_type": "refresh_token", "refresh_token": "r1", "client_id": "cid"}]
    saved = json.loads((home / "spotify_token.json").read_text())
    assert saved["refresh_token"] == "r1" and saved["expires_at"] > time.time() + 3000
    assert sa.access_token() == "new" and len(calls) == 1


def test_access_token_revoked_disconnects(home, monkeypatch, token_api):
    monkeypatch.setenv("VERONICA_SPOTIFY_CLIENT_ID", "cid")
    write_token(home, expires_at=0)
    token_api[1].append(sa.SpotifyAuthError("invalid_grant"))
    assert sa.access_token() is None and not sa.is_connected()


def test_access_token_network_error_raises_and_keeps_token(home, monkeypatch, token_api):
    monkeypatch.setenv("VERONICA_SPOTIFY_CLIENT_ID", "cid")
    write_token(home, expires_at=0)
    token_api[1].append(sa.SpotifyAuthError("couldn't reach Spotify"))
    with pytest.raises(sa.SpotifyAuthError):
        sa.access_token()
    assert sa.is_connected()


def test_connect_without_client_id_raises(home):
    with pytest.raises(sa.SpotifyAuthError, match="Client ID"):
        sa.connect(open_browser=lambda url: None, timeout=1)


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _browser(redirect):
    """A fake browser: once the consent URL is opened, 'the user approves'
    and Spotify redirects to our loopback callback with `redirect(q)`."""
    seen = {}

    def open_browser(url):
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))
        seen.update(q)

        def hit():
            params = urllib.parse.urlencode(redirect(q))
            with urllib.request.urlopen(f"{q['redirect_uri']}?{params}", timeout=5) as r:
                seen["page"] = r.read().decode()
        threading.Thread(target=hit, daemon=True).start()
        return True
    return open_browser, seen


def test_connect_pkce_flow_stores_token(home, monkeypatch, token_api):
    monkeypatch.setenv("VERONICA_SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("VERONICA_SPOTIFY_PORT", str(_free_port()))
    calls, replies = token_api
    replies.append({"access_token": "a", "refresh_token": "r", "expires_in": 3600, "scope": sa.SCOPES})
    browser, seen = _browser(lambda q: {"code": "thecode", "state": q["state"]})
    sa.connect(open_browser=browser, timeout=10)

    assert seen["client_id"] == "cid" and seen["code_challenge_method"] == "S256"
    assert seen["scope"] == "user-read-playback-state user-modify-playback-state"
    assert seen["redirect_uri"].startswith("http://127.0.0.1:")
    (req,) = calls
    assert req["grant_type"] == "authorization_code" and req["code"] == "thecode"
    assert req["redirect_uri"] == seen["redirect_uri"]
    import base64
    import hashlib
    challenge = base64.urlsafe_b64encode(hashlib.sha256(req["code_verifier"].encode()).digest())
    assert challenge.rstrip(b"=").decode() == seen["code_challenge"]
    assert sa.is_connected() and sa.access_token() == "a"
    if os.name == "posix":
        assert (home / "spotify_token.json").stat().st_mode & 0o077 == 0


def test_connect_denied_or_bad_state_raises(home, monkeypatch, token_api):
    monkeypatch.setenv("VERONICA_SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("VERONICA_SPOTIFY_PORT", str(_free_port()))
    browser, _ = _browser(lambda q: {"error": "access_denied", "state": q["state"]})
    with pytest.raises(sa.SpotifyAuthError, match="access_denied"):
        sa.connect(open_browser=browser, timeout=10)
    browser, _ = _browser(lambda q: {"code": "c", "state": "forged"})
    with pytest.raises(sa.SpotifyAuthError, match="state"):
        sa.connect(open_browser=browser, timeout=10)
    assert token_api[0] == [] and not sa.is_connected()


def test_connect_times_out(home, monkeypatch, token_api):
    monkeypatch.setenv("VERONICA_SPOTIFY_CLIENT_ID", "cid")
    monkeypatch.setenv("VERONICA_SPOTIFY_PORT", str(_free_port()))
    with pytest.raises(sa.SpotifyAuthError, match="timed out"):
        sa.connect(open_browser=lambda url: True, timeout=0.6)
