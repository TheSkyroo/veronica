"""Personal data via the user's Google account (Google Calendar, Gmail,
Google Tasks, Google Docs) and in-process timers, exposed to Claude as
in-process MCP tools.

Everything goes through Google's REST APIs with the OAuth token kept by
veronica.google_account (no google-api-python-client: plain authorized
requests). Until the user has put an OAuth client file in place and said
"connect Google", every tool answers with that module's NotConfigured /
NotConnected message; a tool never starts a sign-in itself.

The HTTP calls block, so each tool runs its whole exchange on a worker
thread via asyncio.to_thread; every request carries
google_account.TIMEOUT. Tests replace google_account._authorized_session
with a fake session.
"""
import asyncio
import base64
import datetime as dt
import email.utils
import html
import json
import logging
import math
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from email.message import EmailMessage
from urllib.parse import quote

from claude_agent_sdk import create_sdk_mcp_server, tool

from veronica import google_account as google

CALENDAR_DAYS_MAX = 30
REMINDERS_DAYS_MAX = 60
MAIL_LIMIT_MAX = 20
PAGES_MAX = 10                # result pages followed per list call
PREVIEW_CHARS = 200

CALENDAR_API = "https://www.googleapis.com/calendar/v3"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
TASKS_API = "https://tasks.googleapis.com/tasks/v1"
PEOPLE_API = "https://people.googleapis.com/v1"
DRIVE_API = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD = "https://www.googleapis.com/upload/drive/v3/files"
NOTES_FOLDER = "Veronica Notes"
MIME_FOLDER = "application/vnd.google-apps.folder"
MIME_DOC = "application/vnd.google-apps.document"

log = logging.getLogger(__name__)


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def _guard(fn):
    """Wrap a handler so malformed args (missing keys, bad types) and Google
    failures return `_err(...)` instead of raising."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except google.GoogleError as exc:
            log.info("Google call failed: %s", exc)
            return _err(str(exc))
        except Exception as exc:
            return _err(str(exc) or type(exc).__name__)
    return wrapper


async def _google(fn: Callable, *args):
    """Run fn(*args) — synchronous Google calls — on a worker thread."""
    return await asyncio.to_thread(fn, *args)


def _pages(url: str, params: dict, key: str = "items") -> list[dict]:
    """Every `key` entry of a paged list call, following nextPageToken up to
    PAGES_MAX pages."""
    out, params = [], dict(params)
    for _ in range(PAGES_MAX):
        body = google.get(url, params=params)
        out.extend(body.get(key) or [])
        token = body.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return out


# -- helpers -------------------------------------------------------------------------
def _parse_day(day: str) -> dt.date:
    if day == "today":
        return dt.date.today()
    if day == "tomorrow":
        return dt.date.today() + dt.timedelta(days=1)
    return dt.datetime.strptime(day, "%Y-%m-%d").date()


def _parse_start(s: str) -> dt.datetime:
    when = dt.datetime.strptime(s.strip(), "%Y-%m-%d %H:%M")
    if not 1900 <= when.year <= 2999:
        raise ValueError(s)
    return when


def _clamp(value, lo, hi, default):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _clean(s) -> str:
    """Free text (a subject, a title, a mail body) as one line: tabs and
    newlines collapsed, so it can't break the one-record-per-line output."""
    return " ".join(str(s or "").split())


def _rfc3339(local: dt.datetime) -> str:
    """A naive local time as RFC 3339 with this machine's UTC offset."""
    return local.astimezone().isoformat(timespec="seconds")


def _local(stamp: str) -> dt.datetime | None:
    """An RFC 3339 timestamp as a naive local datetime."""
    try:
        d = dt.datetime.fromisoformat(stamp)
    except (TypeError, ValueError):
        return None
    return d.astimezone().replace(tzinfo=None) if d.tzinfo else d


# -- calendar -------------------------------------------------------------------
def _format_events(rows: list[tuple]) -> str:
    """rows: (title, start, end, calendar, location). One line each:
    "HH:MM–HH:MM  title (calendar)" [" @ location"] — proactive.py parses it
    (an all-day event is 00:00–00:00)."""
    out = []
    for title, start, end, cal, loc in rows:
        entry = f"{start:%H:%M}–{end:%H:%M}  {title} ({cal})"
        if loc:
            entry += f" @ {loc}"
        out.append(entry)
    return "\n".join(out) if out else "No events."


def _calendar_name(c: dict) -> str:
    """What the user calls a calendar: their own rename, else its title.
    The primary calendar's title is the account's address, so unrenamed
    it's just "Calendar"."""
    if c.get("summaryOverride"):
        return _clean(c["summaryOverride"])
    if c.get("primary"):
        return "Calendar"
    return _clean(c.get("summary")) or "Calendar"


def _calendars() -> list[dict]:
    return _pages(f"{CALENDAR_API}/users/me/calendarList", {"maxResults": 250})


def _event_times(e: dict) -> tuple[dt.datetime | None, dt.datetime | None]:
    """(start, end) as naive local datetimes; an all-day event (a `date`,
    no time) runs midnight to midnight."""
    out = []
    for key in ("start", "end"):
        t = e.get(key) or {}
        if t.get("dateTime"):
            out.append(_local(t["dateTime"]))
        elif t.get("date"):
            try:
                out.append(dt.datetime.strptime(t["date"], "%Y-%m-%d"))
            except ValueError:
                out.append(None)
        else:
            out.append(None)
    return out[0], out[1]


def _declined(e: dict) -> bool:
    return any(a.get("self") and a.get("responseStatus") == "declined"
               for a in e.get("attendees") or [])


def _events_sync(start: dt.datetime, end: dt.datetime) -> list[tuple]:
    """Events starting in [start, end) on every calendar the user shows in
    Google Calendar (selected, plus the primary), merged by start time."""
    cals = [c for c in _calendars() if c.get("primary") or c.get("selected")]
    labelled = len(cals) > 1
    params = {"singleEvents": "true", "orderBy": "startTime", "maxResults": 250,
              "timeMin": _rfc3339(start), "timeMax": _rfc3339(end)}
    rows = []
    for c in cals:
        url = f"{CALENDAR_API}/calendars/{quote(c['id'], safe='')}/events"
        try:
            items = _pages(url, params)
        except google.ApiError as exc:
            if c.get("primary"):
                raise
            log.info("skipping calendar %r: %s", c.get("summary"), exc)
            continue
        name = _calendar_name(c) if labelled else "Calendar"
        for e in items:
            if e.get("status") == "cancelled" or _declined(e):
                continue
            s, f = _event_times(e)
            # A multi-day event that began before the range isn't listed.
            if s is None or f is None or not start <= s < end:
                continue
            rows.append((_clean(e.get("summary")) or "(no title)", s, f, name,
                         _clean(e.get("location"))))
    return sorted(rows, key=lambda r: r[1])


@tool(
    "calendar_events",
    "List Google Calendar events in a date range across the calendars shown in Google "
    "Calendar (recurring events included): title, start/end time, calendar, location",
    {"day": str, "days": int},
)
@_guard
async def calendar_events(args: dict) -> dict:
    day = str(args.get("day", "today") or "today").strip()
    try:
        first = _parse_day(day)
    except ValueError:
        return _err(f"invalid day: {day!r}")
    days = _clamp(args.get("days", 1), 1, CALENDAR_DAYS_MAX, 1)
    start = dt.datetime.combine(first, dt.time())
    rows = await _google(_events_sync, start, start + dt.timedelta(days=days))
    return _ok(_format_events(rows))


def _find_calendar(name: str) -> dict | None:
    """The writable calendar called `name` (casefolded, by the user's name
    for it or its title)."""
    want = name.casefold()
    for c in _calendars():
        if c.get("accessRole") not in ("owner", "writer"):
            continue
        names = {_calendar_name(c).casefold(), _clean(c.get("summary")).casefold()}
        if want in names:
            return c
    return None


@tool(
    "calendar_create",
    "Create a Google Calendar event (primary calendar unless one is named)",
    {"title": str, "start": str, "minutes": int, "calendar": str},
)
@_guard
async def calendar_create(args: dict) -> dict:
    title = _clean(args.get("title", ""))
    if not title:
        return _err("title is required")
    start_s = str(args.get("start", ""))
    try:
        start = _parse_start(start_s)
    except ValueError:
        return _err(f"invalid start: {start_s!r}, expected 'YYYY-MM-DD HH:MM'")
    minutes = _clamp(args.get("minutes", 60), 1, 24 * 60, 60)
    calendar = _clean(args.get("calendar", ""))

    def go():
        cal_id = "primary"
        if calendar:
            found = _find_calendar(calendar)
            if found is None:
                return _err(f"there's no calendar called {calendar!r}")
            cal_id = found["id"]
        end = start + dt.timedelta(minutes=minutes)
        google.post(f"{CALENDAR_API}/calendars/{quote(cal_id, safe='')}/events",
                    {"summary": title, "start": {"dateTime": _rfc3339(start)},
                     "end": {"dateTime": _rfc3339(end)}})
        return _ok(f"Created '{title}'")
    return await _google(go)


# -- mail -------------------------------------------------------------------
def _format_mail(rows: list[tuple]) -> str:
    """rows: (sender, subject, received, preview). A header line per message
    and the preview indented by two spaces — proactive.count_mail relies on it."""
    out = [f"{when:%Y-%m-%d %H:%M}  {sender} — {subject}\n  {preview}" for sender, subject, when, preview in rows]
    return "\n".join(out) if out else "No messages."


def _sender(raw: str) -> str:
    """A From header as "Name <address>", or whichever of the two it has."""
    name, addr = (_clean(p) for p in email.utils.parseaddr(raw or ""))
    if addr and name and addr.casefold() != name.casefold():
        return f"{name} <{addr}>"
    return name or addr or _clean(raw) or "(unknown sender)"


def _mail_row(m: dict) -> tuple | None:
    headers = {h.get("name", "").casefold(): h.get("value", "")
               for h in (m.get("payload") or {}).get("headers") or []}
    try:
        when = dt.datetime.fromtimestamp(int(m["internalDate"]) / 1000)
    except (KeyError, TypeError, ValueError, OverflowError, OSError):
        when = None
        if headers.get("date"):
            with_tz = email.utils.parsedate_to_datetime(headers["date"])
            when = with_tz.astimezone().replace(tzinfo=None) if with_tz.tzinfo else with_tz
    if when is None:
        return None
    preview = _clean(html.unescape(m.get("snippet", "")))[:PREVIEW_CHARS]
    return (_sender(headers.get("from", "")), _clean(headers.get("subject")) or "(no subject)",
            when, preview)


def _mail_sync(q: str, limit: int) -> list[tuple]:
    """Newest-first messages matching Gmail search `q`, up to `limit`."""
    ids = google.get(f"{GMAIL_API}/messages", params={"q": q, "maxResults": limit}).get("messages") or []
    rows = []
    for ref in ids[:limit]:
        m = google.get(f"{GMAIL_API}/messages/{quote(ref['id'], safe='')}",
                       params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]})
        row = _mail_row(m)
        if row is not None:
            rows.append(row)
    return rows


UNREAD_QUERY = "is:unread in:inbox"


@tool("mail_unread", "List unread Gmail inbox messages: sender, subject, date, preview", {"limit": int})
@_guard
async def mail_unread(args: dict) -> dict:
    limit = _clamp(args.get("limit", 5), 1, MAIL_LIMIT_MAX, 5)
    return _ok(_format_mail(await _google(_mail_sync, UNREAD_QUERY, limit)))


async def mail_unread_count() -> int:
    """The inbox's unread count, straight from the INBOX label — a plain
    helper (not an MCP tool) for the proactive briefing, which wants the
    real number rather than the capped `mail_unread` listing. Raises
    RuntimeError (google.GoogleError is one) when Gmail fails or reports
    something that isn't an integer."""
    try:
        body = await _google(google.get, f"{GMAIL_API}/labels/INBOX")
    except google.GoogleError:
        raise
    except Exception as exc:
        raise RuntimeError(str(exc)) from None
    n = body.get("messagesUnread")
    try:
        return int(n)
    except (TypeError, ValueError):
        raise RuntimeError(f"unexpected unread count: {n!r}") from None


def mail_search_query(query: str) -> str:
    """`query` as one quoted Gmail phrase: quotes, backslashes and control
    characters dropped so it can't close the phrase and add operators."""
    q = " ".join(re.sub(r"[\"\\\x00-\x1f]", " ", query).split())
    return f'"{q}"'


@tool("mail_search",
      "Search Gmail (sender, subject and text; spam and trash excluded) for a phrase",
      {"query": str, "limit": int})
@_guard
async def mail_search(args: dict) -> dict:
    query = _clean(args.get("query", ""))[:200]
    if not query or mail_search_query(query) == '""':
        return _err("query is required")
    limit = _clamp(args.get("limit", 5), 1, MAIL_LIMIT_MAX, 5)
    return _ok(_format_mail(await _google(_mail_sync, mail_search_query(query), limit)))


def _raw_message(rec: "Recipient", subject: str, body: str) -> str:
    """An RFC 2822 message, base64url as Gmail's messages.send wants it.
    Gmail fills in From (the connected account) and Date."""
    msg = EmailMessage()
    msg["To"] = rec.handle if rec.name == rec.handle else email.utils.formataddr((rec.name, rec.handle))
    msg["Subject"] = subject
    msg.set_content(body)
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("ascii")


@tool("mail_send",
      "Compose and send an email from Gmail. `to` is an email address or a contact's name "
      "(looked up in Google Contacts)",
      {"to": str, "subject": str, "body": str})
@_guard
async def mail_send(args: dict) -> dict:
    to = _clean(args.get("to", ""))
    subject = _clean(args.get("subject", ""))
    body = str(args.get("body", "") or "")
    if not to:
        return _err("to is required")
    # A name is resolved to one address first (the same lookup the gate's
    # confirm showed); an ambiguous or unknown one comes back as an error
    # the brain can ask about.
    try:
        rec = await resolve_recipient_async(to)
    except ValueError as exc:
        return _err(str(exc))
    await _google(google.post, f"{GMAIL_API}/messages/send", {"raw": _raw_message(rec, subject, body)})
    return _ok(f"Sent to {rec.name}")


# -- recipients (Google Contacts) -------------------------------------------
RESOLVED_TTL_S = 300          # a confirmed name -> address holds this long
_EMAIL_RE = re.compile(r"[^@\s<>\"']+@[^@\s<>\"']+\.[^@\s<>\"']+")
PEOPLE_MASK = "names,emailAddresses"


@dataclass(frozen=True)
class Recipient:
    name: str
    handle: str


def is_handle(to: str) -> bool:
    """An email address, as opposed to a person's name."""
    return bool(_EMAIL_RE.fullmatch(to.strip()))


def _person_row(p: dict) -> tuple[str, list[str]]:
    """A People API person as (display name, [email addresses...]); an
    "other contact" (someone you've mailed) may have only an address."""
    emails = [_clean(e.get("value")) for e in p.get("emailAddresses") or [] if e.get("value")]
    names = p.get("names") or []
    display = _clean(names[0].get("displayName")) if names else ""
    return display or (emails[0] if emails else ""), emails


_warmed: set[str] = set()


def _people_query(endpoint: str, name: str) -> list[tuple[str, list[str]]]:
    url = f"{PEOPLE_API}/{endpoint}"
    params = {"readMask": PEOPLE_MASK, "pageSize": 10}
    if endpoint not in _warmed:
        # The search cache is only filled by an empty-query request first.
        google.get(url, params={**params, "query": ""})
        _warmed.add(endpoint)
    found = google.get(url, params={**params, "query": name}).get("results") or []
    return [row for row in (_person_row(r.get("person") or {}) for r in found) if row[0]]


def _people_search(name: str) -> list[tuple[str, list[str]]]:
    """Saved contacts matching `name`, else "other contacts" (people you've
    exchanged mail with). Synchronous; raises google.GoogleError."""
    return (_people_query("people:searchContacts", name)
            or _people_query("otherContacts:search", name))


def _contacts_search(name: str) -> list[tuple[str, list[str]]]:
    """People matching `name`, as (display name, [email addresses...]).
    Tests replace this."""
    return _people_search(name)


def _or_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " or " + items[-1]


def _pick(name: str, found: list[tuple[str, list[str]]]) -> Recipient:
    """One person with one address, or a ValueError saying what to ask."""
    if not found:
        raise ValueError(f"No contact named {name}.")
    exact = [f for f in found if f[0].casefold() == name.casefold()]
    if len(exact) == 1:
        found = exact
    if len(found) > 1:
        names = list(dict.fromkeys(f[0] for f in found))[:4]
        raise ValueError(f"Which {name} — {_or_list(names)}?")
    who, handles = found[0]
    handles = list(dict.fromkeys(h for h in handles if is_handle(h)))
    if not handles:
        raise ValueError(f"{who} has no email address in your contacts.")
    if len(handles) > 1:
        raise ValueError(f"{who} has several: {_or_list(handles[:4])}. Which one?")
    return Recipient(who, handles[0])


_resolved: dict[str, tuple[float, Recipient]] = {}


def resolve_recipient(to: str) -> Recipient:
    """A mail recipient from what the brain said: an address as-is, a name
    looked up in Google Contacts. Raises ValueError with a short, speakable
    reason (ambiguous, unknown, Google not connected) instead of guessing.
    A name resolved here is remembered for a few minutes, so the send goes
    to the very address the confirm showed."""
    to = to.strip()
    if is_handle(to):
        return Recipient(to, to)
    if "@" in to or re.fullmatch(r"\+?[\d\s().-]+", to):
        raise ValueError(f"{to} isn't an email address.")
    key = to.casefold()
    hit = _resolved.get(key)
    if hit is not None and time.monotonic() - hit[0] < RESOLVED_TTL_S:
        return hit[1]
    try:
        found = _contacts_search(to)
    except google.GoogleError as exc:
        raise ValueError(f"{exc} Or give me the email address.") from None
    rec = _pick(to, found)
    _resolved[key] = (time.monotonic(), rec)
    log.info("mail recipient %r -> %s", to, rec.name)
    return rec


async def resolve_recipient_async(to: str) -> Recipient:
    return await asyncio.to_thread(resolve_recipient, to)


# -- reminders (Google Tasks) ----------------------------------------------
# Google Tasks keeps only a due *date* (the time part of `due` is always
# midnight UTC and ignored), and never notifies at a time. reminder_create
# therefore writes a given time into the task's notes as "Due at HH:MM",
# and reminders_due reads it back from there; any other task shows 00:00.
TASKLIST = "@default"
_DUE_AT_RE = re.compile(r"\bDue at (\d{1,2}):(\d{2})\b")


def _format_reminders(rows: list[tuple]) -> str:
    """rows: (title, due, list). "YYYY-MM-DD HH:MM  name" [" (list)"] —
    proactive.py parses it."""
    out = []
    for name, when, lst in rows:
        entry = f"{when:%Y-%m-%d %H:%M}  {name}"
        if lst:
            entry += f" ({lst})"
        out.append(entry)
    return "\n".join(out) if out else "No reminders due."


def _task_due(t: dict) -> dt.datetime | None:
    """When a task is due: its date, at the "Due at HH:MM" time from its
    notes when there is one; None for an undated task."""
    try:
        day = dt.datetime.strptime(str(t.get("due", ""))[:10], "%Y-%m-%d")
    except ValueError:
        return None
    m = _DUE_AT_RE.search(t.get("notes") or "")
    if m and int(m[1]) < 24 and int(m[2]) < 60:
        return day.replace(hour=int(m[1]), minute=int(m[2]))
    return day


def _reminders_sync(end: dt.datetime) -> list[tuple]:
    lst = _clean(google.get(f"{TASKS_API}/users/@me/lists/{TASKLIST}").get("title")) or "Tasks"
    items = _pages(f"{TASKS_API}/lists/{TASKLIST}/tasks",
                   {"showCompleted": "false", "showHidden": "false", "maxResults": 100,
                    # due is a date at 00:00Z: the whole last day is asked
                    # for, and the time cut made below.
                    "dueMax": f"{end:%Y-%m-%d}T23:59:59Z"})
    rows = []
    for t in items:
        if t.get("status") == "completed" or t.get("deleted"):
            continue
        due = _task_due(t)
        if due is not None and due < end:
            rows.append((_clean(t.get("title")) or "(no title)", due, lst))
    return sorted(rows, key=lambda r: r[1])


@tool(
    "reminders_due",
    "List incomplete Google Tasks due within N days (including overdue)",
    {"days": int},
)
@_guard
async def reminders_due(args: dict) -> dict:
    days = _clamp(args.get("days", 1), 1, REMINDERS_DAYS_MAX, 1)
    end = dt.datetime.now() + dt.timedelta(days=days)
    return _ok(_format_reminders(await _google(_reminders_sync, end)))


@tool("reminder_create",
      "Create a Google Task, optionally due at a date and time (Google Tasks keeps the date; "
      "the time is noted on the task)",
      {"title": str, "when": str})
@_guard
async def reminder_create(args: dict) -> dict:
    title = _clean(args.get("title", ""))
    if not title:
        return _err("title is required")
    when_s = str(args.get("when", "") or "").strip()
    task = {"title": title}
    if when_s:
        try:
            when = _parse_start(when_s)
        except ValueError:
            return _err(f"invalid when: {when_s!r}, expected 'YYYY-MM-DD HH:MM'")
        task["due"] = f"{when:%Y-%m-%d}T00:00:00.000Z"
        task["notes"] = f"Due at {when:%H:%M}"
    await _google(google.post, f"{TASKS_API}/lists/{TASKLIST}/tasks", task)
    return _ok(f"Created reminder '{title}'")


# -- notes (Google Docs) ------------------------------------------------------
def _notes_folder() -> str:
    """The id of the "Veronica Notes" Drive folder, made on first use. With
    the drive.file scope only folders this app created are visible, so a
    same-named folder of the user's own is never written into."""
    q = f"name = '{NOTES_FOLDER}' and mimeType = '{MIME_FOLDER}' and trashed = false"
    found = google.get(f"{DRIVE_API}/files",
                       params={"q": q, "spaces": "drive", "fields": "files(id)", "pageSize": 1})
    files = found.get("files") or []
    if files:
        return files[0]["id"]
    return google.post(f"{DRIVE_API}/files", {"name": NOTES_FOLDER, "mimeType": MIME_FOLDER},
                       params={"fields": "id"})["id"]


def _multipart(metadata: dict, text: str) -> tuple[bytes, str]:
    """A multipart/related upload body: JSON metadata, then the plain text
    Drive converts into a Google Doc. Returns (body, content type)."""
    boundary = f"veronica-{uuid.uuid4().hex}"
    body = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{json.dumps(metadata)}\r\n"
            f"--{boundary}\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n"
            f"{text}\r\n--{boundary}--\r\n")
    return body.encode("utf-8"), f"multipart/related; boundary={boundary}"


@tool("notes_create", "Create a note as a Google Doc in the 'Veronica Notes' Drive folder",
      {"title": str, "body": str})
@_guard
async def notes_create(args: dict) -> dict:
    title = _clean(args.get("title", ""))
    if not title:
        return _err("title is required")
    body = str(args.get("body", "") or "").replace("\r\n", "\n")

    def go():
        data, ctype = _multipart({"name": title, "mimeType": MIME_DOC, "parents": [_notes_folder()]}, body)
        google.request("POST", DRIVE_UPLOAD, params={"uploadType": "multipart", "fields": "id"},
                       data=data, headers={"Content-Type": ctype})
        return _ok(f"Created note '{title}'")
    return await _google(go)


# -- timers -----------------------------------------------------------------
service = None  # bound by build_orchestrator via bind()


def bind(new_service) -> None:
    """Replace the module-level timer service (called once at startup)."""
    global service
    service = new_service


@tool("timer_set", "Set an in-process timer that speaks and notifies when it fires", {"minutes": float, "label": str})
@_guard
async def timer_set(args: dict) -> dict:
    if service is None:
        return _err("timer service not available")
    try:
        minutes = float(args.get("minutes", 0))
    except (TypeError, ValueError):
        return _err("minutes must be a number")
    if not math.isfinite(minutes) or minutes <= 0:
        return _err("minutes must be a positive, finite number")
    minutes = min(minutes, 24 * 60)
    label = str(args.get("label", "") or "")
    tid = service.set(minutes, label)
    return _ok(f"Timer set for {minutes} min (id {tid})")


@tool("timer_list", "List active timers", {})
@_guard
async def timer_list(args: dict) -> dict:
    if service is None:
        return _err("timer service not available")
    timers = service.list()
    if not timers:
        return _ok("No active timers.")
    lines = [
        f"{t['id']}  {t['label'] or '(no label)'}  {t['remaining_s']:.0f}s left"
        for t in timers
    ]
    return _ok("\n".join(lines))


@tool("timer_cancel", "Cancel a timer by id or label", {"label": str})
@_guard
async def timer_cancel(args: dict) -> dict:
    if service is None:
        return _err("timer service not available")
    label = str(args.get("label", "") or "")
    if not label:
        return _err("label is required")
    ok = service.cancel(label)
    return _ok(f"Cancelled {label}") if ok else _err(f"no timer matching {label!r}")


TOOLS = [
    calendar_events, calendar_create,
    mail_unread, mail_search, mail_send,
    reminder_create, reminders_due,
    notes_create,
    timer_set, timer_list, timer_cancel,
]
PIM_TOOL_NAMES = [t.name for t in TOOLS]
pim_server = create_sdk_mcp_server(name="pim", version="1.0.0", tools=TOOLS)
