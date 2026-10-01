"""The user's Google account: OAuth sign-in and authorized REST calls for
the pim tools (Gmail, Calendar, Tasks, People, Drive).

Veronica is an installed (desktop) OAuth client. The user creates their
own "Desktop app" OAuth client in Google Cloud Console and saves its JSON
as `<home>/google_client.json` (home is Settings().home, ~/.veronica unless
VERONICA_HOME says otherwise). `connect()` runs the loopback-redirect flow
once — it opens the browser and blocks until the user consents — and keeps
the resulting refresh token in `<home>/google_token.json`, readable by the
user only where the OS honours that. After that, `credentials()` loads and
refreshes it silently.

Only `connect()` ever opens a browser, and only from an explicit "connect
Google" (Settings button or voice command). A tool call that finds no
usable token gets NotConnected, whose message says what to do, instead.

The google-auth libraries are imported lazily, so importing this module
(and so the pim tools) costs nothing until Google is actually used.
"""
import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger(__name__)

# Least privilege for what the pim tools do; see the README for what each
# one is for. Changing this list means existing users must reconnect.
SCOPES = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/tasks",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/contacts.other.readonly",
    "https://www.googleapis.com/auth/drive.file",
]

CLIENT_FILE = "google_client.json"
TOKEN_FILE = "google_token.json"
CONNECT_TIMEOUT_S = 300           # how long connect() waits for the browser consent
TIMEOUT = (5, 20)                 # (connect, read) seconds for every API call
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
PROFILE_URL = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
_EMAIL_KEY = "veronica_email"     # our own key in the token file: who's connected

_lock = threading.RLock()
_creds = None                     # cached google.oauth2.credentials.Credentials
_session = None                   # cached AuthorizedSession for _creds


class GoogleError(RuntimeError):
    """Anything Google-related a tool should say instead of a result. The
    message is short and speakable."""


class NotConfigured(GoogleError):
    def __init__(self):
        super().__init__(
            f"Google isn't set up — put your OAuth client file at {_shown(client_file())}, "
            "then say 'connect Google' or use Settings.")


class NotConnected(GoogleError):
    def __init__(self, why: str = "Google isn't connected"):
        super().__init__(f"{why} — say 'connect Google' or connect it in Settings.")


class ApiError(GoogleError):
    def __init__(self, status: int, message: str):
        self.status = status
        super().__init__(f"Google said {status}: {message}" if message else f"Google said {status}.")


# -- files -------------------------------------------------------------------------
def home() -> Path:
    try:
        from veronica.config import Settings
        return Path(Settings().home)
    except Exception:
        env = os.environ.get("VERONICA_HOME")
        return Path(env) if env else Path.home() / ".veronica"


def client_file() -> Path:
    return home() / CLIENT_FILE


def token_file() -> Path:
    return home() / TOKEN_FILE


def _shown(p: Path) -> str:
    """A path as it's said: under the profile folder, as ~/..."""
    try:
        return "~/" + p.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(p)


def _read_token() -> dict | None:
    try:
        info = json.loads(token_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return info if isinstance(info, dict) and info.get("refresh_token") else None


def _write_token(info: dict) -> None:
    """Atomically, created owner-only (0600 on POSIX; on Windows the file
    inherits the profile folder's ACL, which is already per-user)."""
    path = token_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(info, f)
    os.replace(tmp, path)


def _save(creds, email: str | None = None) -> None:
    info = json.loads(creds.to_json())
    if email is None:
        email = (_read_token() or {}).get(_EMAIL_KEY, "")
    if email:
        info[_EMAIL_KEY] = email
    _write_token(info)


def _reset() -> None:
    global _creds, _session
    with _lock:
        _creds = _session = None


# -- public API --------------------------------------------------------------------
def is_configured() -> bool:
    """The OAuth client file is there (connect() can run)."""
    return client_file().is_file()


def is_connected() -> bool:
    """A refresh token is saved. Doesn't touch the network: a revoked token
    only shows up as NotConnected on the next call."""
    return _read_token() is not None


def account() -> str:
    """The connected Gmail address, or "" (not connected, or unknown)."""
    return str((_read_token() or {}).get(_EMAIL_KEY, ""))


def status() -> dict:
    """For Settings: {configured, connected, account, client_file}."""
    return {"configured": is_configured(), "connected": is_connected(),
            "account": account(), "client_file": str(client_file())}


def _refresh_request():
    from google.auth.transport.requests import Request
    return Request()


def credentials():
    """Valid credentials, refreshed (and re-saved) when the access token has
    expired. Raises NotConfigured / NotConnected with a speakable message;
    never starts a sign-in."""
    global _creds, _session
    from google.auth.exceptions import RefreshError, TransportError
    from google.oauth2.credentials import Credentials

    with _lock:
        creds = _creds
        if creds is None:
            info = _read_token()
            if info is None:
                raise NotConnected() if is_configured() else NotConfigured()
            try:
                creds = Credentials.from_authorized_user_info(info)
            except ValueError:
                raise NotConnected("Google's saved sign-in is unreadable") from None
        if not creds.valid:
            try:
                creds.refresh(_refresh_request())
            except RefreshError as exc:
                log.info("Google refresh failed: %s", exc)
                _reset()
                raise NotConnected("Google's sign-in has expired or was revoked") from None
            except TransportError as exc:
                raise GoogleError(f"I couldn't reach Google: {exc}") from None
            _save(creds)
        if creds is not _creds:
            _creds, _session = creds, None
        return creds


def _authorized_session():
    """A requests session that adds (and refreshes) the bearer token. Tests
    replace this."""
    global _session
    from google.auth.transport.requests import AuthorizedSession

    with _lock:
        creds = credentials()
        if _session is None:
            _session = AuthorizedSession(creds)
        return _session


def _error_message(resp) -> str:
    try:
        err = resp.json().get("error", {})
    except ValueError:
        return (resp.text or "").strip()[:200]
    if isinstance(err, str):
        return err
    msg = str(err.get("message", "")).strip()
    reasons = {d.get("reason") for d in err.get("errors", []) if isinstance(d, dict)}
    if resp.status_code == 403 and ("insufficientPermissions" in reasons
                                    or "insufficient" in msg.casefold()):
        return ("Veronica doesn't have permission for that — reconnect Google in Settings "
                "and allow everything it asks for.")
    return msg[:200]


def request(method: str, url: str, **kw) -> dict:
    """One authorized API call: the decoded JSON body ({} when there's none).
    Synchronous — callers run it in a thread. Raises ApiError on an HTTP
    error, NotConnected when the token can't be refreshed, GoogleError when
    Google can't be reached."""
    import requests
    from google.auth.exceptions import RefreshError, TransportError

    sess = _authorized_session()
    try:
        resp = sess.request(method, url, timeout=TIMEOUT, **kw)
    except RefreshError:
        _reset()
        raise NotConnected("Google's sign-in has expired or was revoked") from None
    except (requests.RequestException, TransportError) as exc:
        kind = "timed out" if isinstance(exc, requests.Timeout) else type(exc).__name__
        raise GoogleError(f"I couldn't reach Google ({kind}).") from None
    if resp.status_code == 401:
        _reset()
        raise NotConnected("Google rejected the saved sign-in")
    if resp.status_code >= 400:
        raise ApiError(resp.status_code, _error_message(resp))
    if not resp.content:
        return {}
    try:
        return resp.json()
    except ValueError:
        return {}


def get(url: str, params=None) -> dict:
    return request("GET", url, params=params)


def post(url: str, json_body=None, params=None, **kw) -> dict:
    return request("POST", url, json=json_body, params=params, **kw)


def _flow():
    from google_auth_oauthlib.flow import InstalledAppFlow
    return InstalledAppFlow.from_client_secrets_file(str(client_file()), SCOPES)


def connect() -> str:
    """Interactive sign-in: opens the browser on Google's consent page and
    blocks (up to CONNECT_TIMEOUT_S) until the user finishes. Saves the
    token and returns a short spoken summary. Call it off the event loop,
    and only from an explicit user action."""
    global _creds
    if not is_configured():
        raise NotConfigured()
    # Google's granular consent lets the user untick scopes; oauthlib would
    # otherwise raise on a scope set that differs from the one requested.
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
    flow = _flow()
    try:
        creds = flow.run_local_server(
            port=0, open_browser=True, timeout_seconds=CONNECT_TIMEOUT_S,
            authorization_prompt_message="Sign in to Google in your browser: {url}",
            success_message="Veronica is connected to Google. You can close this tab.",
            access_type="offline", prompt="consent")
    except Exception as exc:
        log.warning("Google sign-in failed: %s", exc)
        raise GoogleError("Google sign-in didn't finish.") from None
    if not creds.refresh_token:
        raise GoogleError("Google didn't hand back a refresh token — try connecting again.")
    granted = getattr(creds, "granted_scopes", None) or creds.scopes or SCOPES
    if isinstance(granted, str):
        granted = granted.split()
    _reset()
    _save(creds, email="")
    with _lock:
        _creds = creds
    email = ""
    try:
        email = str(get(PROFILE_URL).get("emailAddress", ""))
    except GoogleError as exc:
        log.info("Gmail profile unavailable after connect: %s", exc)
    if email:
        _save(creds, email=email)
    missing = [s.rsplit("/", 1)[-1] for s in SCOPES if s not in set(granted)]
    said = f"Google is connected as {email}." if email else "Google is connected."
    if missing:
        said += " Some permissions weren't granted (" + ", ".join(missing) + "), so those features won't work."
    return said


def _revoke(token: str) -> None:
    import requests
    requests.post(REVOKE_URL, params={"token": token}, timeout=TIMEOUT,
                  headers={"content-type": "application/x-www-form-urlencoded"})


def disconnect() -> None:
    """Forget the saved sign-in and, best effort, revoke it at Google."""
    info = _read_token()
    _reset()
    if info is not None:
        try:
            _revoke(info.get("refresh_token") or info.get("token") or "")
        except Exception as exc:
            log.info("Google token revoke failed (deleting it anyway): %s", exc)
    try:
        token_file().unlink()
    except FileNotFoundError:
        pass
