import datetime as dt
import json
import sys

import pytest
import requests
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials

from veronica import google_account as google


@pytest.fixture(autouse=True)
def _fresh(tmp_home):
    google._reset()
    yield
    google._reset()


def token_info(expired=False, **kw):
    when = dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(hours=-1 if expired else 1)
    return {"token": "at-old", "refresh_token": "rt", "client_id": "cid", "client_secret": "cs",
            "token_uri": "https://oauth2.googleapis.com/token", "scopes": google.SCOPES,
            "expiry": when.strftime("%Y-%m-%dT%H:%M:%SZ"), **kw}


def write_token(home, **kw):
    (home / "google_token.json").write_text(json.dumps(token_info(**kw)))


def read_token(home):
    return json.loads((home / "google_token.json").read_text())


# -- files and state -----------------------------------------------------------------

def test_paths_follow_veronica_home(tmp_home):
    assert google.client_file() == tmp_home / "google_client.json"
    assert google.token_file() == tmp_home / "google_token.json"


def test_configured_and_connected(tmp_home):
    assert not google.is_configured() and not google.is_connected()
    (tmp_home / "google_client.json").write_text("{}")
    assert google.is_configured() and not google.is_connected()
    write_token(tmp_home, veronica_email="me@gmail.com")
    assert google.is_connected() and google.account() == "me@gmail.com"
    assert google.status() == {"configured": True, "connected": True, "account": "me@gmail.com",
                               "client_file": str(tmp_home / "google_client.json")}


def test_a_token_without_refresh_token_isnt_connected(tmp_home):
    write_token(tmp_home, refresh_token="")
    assert not google.is_connected()
    (tmp_home / "google_token.json").write_text("not json")
    assert not google.is_connected()


def test_not_configured_message_says_where_the_file_goes(tmp_home):
    with pytest.raises(google.NotConfigured) as exc:
        google.credentials()
    msg = str(exc.value)
    assert msg.startswith("Google isn't set up") and "google_client.json" in msg and "connect Google" in msg


def test_not_connected_when_only_the_client_file_is_there(tmp_home):
    (tmp_home / "google_client.json").write_text("{}")
    with pytest.raises(google.NotConnected, match="Google isn't connected — say 'connect Google'"):
        google.credentials()


# -- credentials / refresh -----------------------------------------------------------

def test_valid_token_is_used_without_refresh(tmp_home, monkeypatch):
    write_token(tmp_home)
    monkeypatch.setattr(Credentials, "refresh", lambda self, req: pytest.fail("refreshed"))
    creds = google.credentials()
    assert creds.token == "at-old" and google.credentials() is creds


def test_expired_token_is_refreshed_and_saved(tmp_home, monkeypatch):
    write_token(tmp_home, expired=True, veronica_email="me@gmail.com")
    requests_made = []

    def refresh(self, req):
        requests_made.append(req)
        self.token = "at-new"
        self.expiry = dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(hours=1)

    monkeypatch.setattr(Credentials, "refresh", refresh)
    monkeypatch.setattr(google, "_refresh_request", lambda: "REQ")
    creds = google.credentials()
    assert creds.token == "at-new" and requests_made == ["REQ"]
    saved = read_token(tmp_home)
    assert saved["token"] == "at-new" and saved["refresh_token"] == "rt"
    assert saved["veronica_email"] == "me@gmail.com"              # kept across a refresh
    if sys.platform != "win32":
        assert (tmp_home / "google_token.json").stat().st_mode & 0o777 == 0o600


def test_revoked_refresh_token_is_not_connected(tmp_home, monkeypatch):
    write_token(tmp_home, expired=True)

    def refresh(self, req):
        raise RefreshError("invalid_grant: Token has been expired or revoked.")

    monkeypatch.setattr(Credentials, "refresh", refresh)
    monkeypatch.setattr(google, "_refresh_request", lambda: None)
    with pytest.raises(google.NotConnected, match="expired or was revoked"):
        google.credentials()


def test_unreadable_token_is_not_connected(tmp_home):
    (tmp_home / "google_token.json").write_text(json.dumps({"refresh_token": "rt"}))   # no client id
    with pytest.raises(google.NotConnected, match="unreadable"):
        google.credentials()


# -- request ---------------------------------------------------------------------

class Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code, self._body = status, body
        self.content = json.dumps(body).encode() if body is not None else text.encode()
        self.text = self.content.decode()

    def json(self):
        if self._body is None:
            raise ValueError("not json")
        return self._body


class Sess:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def request(self, method, url, timeout=None, **kw):
        self.calls.append((method, url, timeout, kw))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


@pytest.fixture
def session(monkeypatch):
    def use(answer):
        s = Sess(answer)
        monkeypatch.setattr(google, "_authorized_session", lambda: s)
        return s
    return use


def test_request_returns_json_and_passes_timeout(session):
    s = session(Resp(200, {"a": 1}))
    assert google.get("https://x/y", params={"q": "z"}) == {"a": 1}
    assert s.calls == [("GET", "https://x/y", google.TIMEOUT, {"params": {"q": "z"}})]


def test_request_empty_body_is_empty_dict(session):
    session(Resp(204))
    assert google.post("https://x/y", {"k": "v"}) == {}


def test_request_http_error_is_concise(session):
    session(Resp(404, {"error": {"code": 404, "message": "Requested entity was not found."}}))
    with pytest.raises(google.ApiError, match="Google said 404: Requested entity was not found.") as exc:
        google.get("https://x/y")
    assert exc.value.status == 404


def test_request_non_json_error(session):
    session(Resp(502, text="<html>Bad Gateway</html>"))
    with pytest.raises(google.ApiError, match="Google said 502: <html>Bad Gateway"):
        google.get("https://x/y")


def test_request_missing_scope_says_reconnect(session):
    session(Resp(403, {"error": {"message": "Request had insufficient authentication scopes.",
                                 "errors": [{"reason": "insufficientPermissions"}]}}))
    with pytest.raises(google.ApiError, match="reconnect Google in Settings"):
        google.get("https://x/y")


def test_request_401_is_not_connected(session):
    session(Resp(401, {"error": {"message": "Invalid Credentials"}}))
    with pytest.raises(google.NotConnected):
        google.get("https://x/y")


def test_request_refresh_failure_is_not_connected(session):
    session(RefreshError("invalid_grant"))
    with pytest.raises(google.NotConnected, match="expired or was revoked"):
        google.get("https://x/y")


def test_request_timeout_is_a_google_error(session):
    session(requests.Timeout("read timed out"))
    with pytest.raises(google.GoogleError, match=r"couldn't reach Google \(timed out\)"):
        google.get("https://x/y")


def test_authorized_session_needs_credentials(tmp_home):
    with pytest.raises(google.NotConfigured):
        google.request("GET", "https://x/y")


def test_authorized_session_is_cached(tmp_home):
    write_token(tmp_home)
    s = google._authorized_session()
    assert s is google._authorized_session() and s.credentials.token == "at-old"


# -- connect / disconnect ------------------------------------------------------------

class FakeFlow:
    def __init__(self, creds):
        self.creds, self.kwargs = creds, None

    def run_local_server(self, **kw):
        self.kwargs = kw
        if isinstance(self.creds, Exception):
            raise self.creds
        return self.creds


def flow_creds(granted=None, refresh_token="rt"):
    return Credentials("at", refresh_token=refresh_token, client_id="cid", client_secret="cs",
                       token_uri="https://oauth2.googleapis.com/token", scopes=google.SCOPES,
                       granted_scopes=granted or google.SCOPES,
                       expiry=dt.datetime.now(dt.UTC).replace(tzinfo=None) + dt.timedelta(hours=1))


def test_connect_without_client_file_is_not_configured(tmp_home):
    with pytest.raises(google.NotConfigured):
        google.connect()


def test_connect_runs_the_loopback_flow_and_saves(tmp_home, monkeypatch):
    (tmp_home / "google_client.json").write_text("{}")
    flow = FakeFlow(flow_creds())
    monkeypatch.setattr(google, "_flow", lambda: flow)
    monkeypatch.setattr(google, "get", lambda url, params=None: {"emailAddress": "me@gmail.com"})
    assert google.connect() == "Google is connected as me@gmail.com."
    assert flow.kwargs["port"] == 0 and flow.kwargs["open_browser"] is True
    assert flow.kwargs["access_type"] == "offline"
    saved = read_token(tmp_home)
    assert saved["refresh_token"] == "rt" and saved["veronica_email"] == "me@gmail.com"
    assert google.is_connected() and google.credentials().token == "at"


def test_connect_reports_unticked_permissions(tmp_home, monkeypatch):
    (tmp_home / "google_client.json").write_text("{}")
    granted = [s for s in google.SCOPES if not s.endswith("gmail.send")]
    monkeypatch.setattr(google, "_flow", lambda: FakeFlow(flow_creds(granted)))
    monkeypatch.setattr(google, "get", lambda url, params=None: {})
    said = google.connect()
    assert said.startswith("Google is connected.") and "gmail.send" in said


def test_connect_failure_is_a_google_error(tmp_home, monkeypatch):
    (tmp_home / "google_client.json").write_text("{}")
    monkeypatch.setattr(google, "_flow", lambda: FakeFlow(RuntimeError("timed out")))
    with pytest.raises(google.GoogleError, match="didn't finish"):
        google.connect()
    assert not google.is_connected()


def test_connect_without_refresh_token_fails(tmp_home, monkeypatch):
    (tmp_home / "google_client.json").write_text("{}")
    monkeypatch.setattr(google, "_flow", lambda: FakeFlow(flow_creds(refresh_token=None)))
    with pytest.raises(google.GoogleError, match="refresh token"):
        google.connect()


def test_disconnect_revokes_and_deletes(tmp_home, monkeypatch):
    write_token(tmp_home)
    revoked = []
    monkeypatch.setattr(google, "_revoke", revoked.append)
    google.disconnect()
    assert revoked == ["rt"] and not (tmp_home / "google_token.json").exists()
    google.disconnect()                                      # nothing left: still fine
    assert revoked == ["rt"]


def test_disconnect_deletes_even_when_revoke_fails(tmp_home, monkeypatch):
    write_token(tmp_home)

    def boom(token):
        raise OSError("offline")

    monkeypatch.setattr(google, "_revoke", boom)
    google.disconnect()
    assert not google.is_connected()
    with pytest.raises(google.NotConfigured):
        google.credentials()
