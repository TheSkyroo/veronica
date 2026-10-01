"""Personal data via Outlook (calendar, mail, tasks, notes) and in-process
timers, exposed to Claude as in-process MCP tools.

No OAuth: everything goes through the classic Outlook desktop app's COM
interface (`Outlook.Application` / the MAPI namespace), so whatever accounts
Outlook is set up with are the ones used. The new Outlook (and Outlook on
the web) has no COM interface; without classic Outlook every tool here
answers with OUTLOOK_MISSING instead.

COM objects belong to the thread (apartment) that made them, so each call
runs on a worker thread via asyncio.to_thread inside its own
CoInitialize/CoUninitialize (`_outlook_session`) and never lets an Outlook
object escape it. pywin32 is imported there, lazily: tests replace
`_outlook_session` with a fake object model.
"""
import asyncio
import contextlib
import datetime as dt
import logging
import math
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from claude_agent_sdk import create_sdk_mcp_server, tool

CALENDAR_DAYS_MAX = 30
REMINDERS_DAYS_MAX = 60
MAIL_LIMIT_MAX = 20
SCAN_MAX = 2000               # items walked per call, whatever a filter returns
PREVIEW_CHARS = 200

# Outlook constants (OlDefaultFolders / OlItemType).
FOLDER_CALENDAR = 9
FOLDER_CONTACTS = 10
FOLDER_INBOX = 6
FOLDER_NOTES = 12
FOLDER_TASKS = 13
ITEM_MAIL = 0
ITEM_APPOINTMENT = 1
ITEM_TASK = 3
ITEM_NOTE = 5
CLASS_MAIL = 43               # OlObjectClass.olMail
NO_DATE_YEAR = 4000           # Outlook's "None" date is 4501-01-01

OUTLOOK_MISSING = ("I can't reach Outlook. Mail, calendar, tasks and notes need the classic "
                   "Outlook desktop app installed and signed in.")

log = logging.getLogger(__name__)


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


class OutlookUnavailable(RuntimeError):
    """Classic Outlook isn't installed, or its COM server wouldn't start."""

    def __init__(self, detail: str = ""):
        super().__init__(OUTLOOK_MISSING)
        self.detail = detail


def _com_message(exc: Exception) -> str:
    """A pywintypes.com_error's human part (the Outlook exception text when
    there is one), else str(exc)."""
    args = getattr(exc, "args", ())
    if len(args) >= 3 and isinstance(args[2], tuple) and len(args[2]) >= 3 and args[2][2]:
        return str(args[2][2]).strip()
    if len(args) >= 2 and isinstance(args[1], str) and args[1]:
        return args[1]
    return str(exc)


def _guard(fn):
    """Wrap a handler so malformed args (missing keys, bad types) and Outlook
    failures return `_err(...)` instead of raising."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except OutlookUnavailable as exc:
            log.info("Outlook unavailable: %s", exc.detail)
            return _err(OUTLOOK_MISSING)
        except Exception as exc:
            return _err(_com_message(exc))
    return wrapper


@contextlib.contextmanager
def _outlook_session():
    """(application, MAPI namespace) for this thread, inside a COM apartment
    that is torn down on exit. Raises OutlookUnavailable when pywin32 or
    Outlook's COM server is missing."""
    try:
        import pythoncom
        import win32com.client
    except ImportError as exc:
        raise OutlookUnavailable(str(exc)) from None
    pythoncom.CoInitialize()
    try:
        try:
            app = win32com.client.Dispatch("Outlook.Application")
            ns = app.GetNamespace("MAPI")
        except Exception as exc:
            raise OutlookUnavailable(_com_message(exc)) from None
        yield app, ns
    finally:
        pythoncom.CoUninitialize()


def _in_outlook(fn: Callable):
    """Run fn(app, ns) inside an Outlook session. Synchronous."""
    with _outlook_session() as (app, ns):
        return fn(app, ns)


async def _outlook(fn: Callable):
    return await asyncio.to_thread(_in_outlook, fn)


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


def _naive(d) -> dt.datetime | None:
    """An Outlook date as a naive local datetime. pywin32 hands Outlook's
    local times back tagged with a UTC tzinfo; the wall-clock value is the
    one that's right, so the tag is dropped, not converted."""
    if d is None:
        return None
    try:
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    except (AttributeError, TypeError, ValueError):
        return None


def _filter_date(d: dt.datetime) -> str:
    """A date for an Items.Restrict filter. Outlook parses these with the
    user's locale, but an ISO date-time is unambiguous everywhere it's been
    tried; callers still re-check dates in Python."""
    return d.strftime("%Y-%m-%d %H:%M")


def _walk(items, limit: int = SCAN_MAX):
    """Iterate an Items collection with GetFirst/GetNext (the only way that
    works on a recurrence-expanded collection, whose Count is meaningless),
    capped so a filter that matched too much can't run forever."""
    item = items.GetFirst()
    n = 0
    while item is not None and n < limit:
        yield item
        n += 1
        item = items.GetNext()


def _get(obj, name: str, default=""):
    try:
        v = getattr(obj, name)
    except Exception:
        return default
    return default if v is None else v


# -- calendar -------------------------------------------------------------------
def _format_events(rows: list[tuple]) -> str:
    """rows: (title, start, end, calendar, location). One line each:
    "HH:MM–HH:MM  title (calendar)" [" @ location"] — proactive.py parses it."""
    out = []
    for title, start, end, cal, loc in rows:
        entry = f"{start:%H:%M}–{end:%H:%M}  {title} ({cal})"
        if loc:
            entry += f" @ {loc}"
        out.append(entry)
    return "\n".join(out) if out else "No events."


def _events_sync(start: dt.datetime, end: dt.datetime):
    def go(app, ns):
        folder = ns.GetDefaultFolder(FOLDER_CALENDAR)
        cal = _clean(_get(folder, "Name", "Calendar")) or "Calendar"
        items = folder.Items
        # Order matters: Sort, then IncludeRecurrences, then Restrict —
        # otherwise recurring series come back as their master only.
        items.Sort("[Start]")
        items.IncludeRecurrences = True
        hits = items.Restrict(f"[Start] < '{_filter_date(end)}' AND [End] > '{_filter_date(start)}'")
        rows = []
        for it in _walk(hits):
            s, e = _naive(_get(it, "Start", None)), _naive(_get(it, "End", None))
            if s is None or e is None:
                continue
            # Re-checked here, whatever the filter matched: a multi-day event
            # that began before the range isn't listed, as before.
            if not start <= s < end:
                continue
            rows.append((_clean(_get(it, "Subject")) or "(no title)", s, e, cal,
                         _clean(_get(it, "Location"))))
        return sorted(rows, key=lambda r: r[1])
    return go


@tool(
    "calendar_events",
    "List Outlook calendar events in a date range (recurring events included): title, "
    "start/end time, calendar, location",
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
    rows = await _outlook(_events_sync(start, start + dt.timedelta(days=days)))
    return _ok(_format_events(rows))


def _find_calendar(ns, name: str):
    """The calendar folder called `name` (casefolded): the default calendar,
    one of its subfolders, or another account's calendar."""
    want = name.casefold()
    default = ns.GetDefaultFolder(FOLDER_CALENDAR)
    candidates = [default, *list(_get(default, "Folders", []) or [])]
    for store in list(_get(ns, "Stores", []) or []):
        with contextlib.suppress(Exception):
            candidates.append(store.GetDefaultFolder(FOLDER_CALENDAR))
    for f in candidates:
        if _clean(_get(f, "Name")).casefold() == want:
            return f
    return None


@tool(
    "calendar_create",
    "Create an Outlook calendar event (default calendar unless one is named)",
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

    def go(app, ns):
        if calendar:
            folder = _find_calendar(ns, calendar)
            if folder is None:
                return _err(f"there's no calendar called {calendar!r}")
            appt = folder.Items.Add(ITEM_APPOINTMENT)
        else:
            appt = app.CreateItem(ITEM_APPOINTMENT)
        appt.Subject = title
        appt.Start = start
        appt.Duration = minutes
        appt.Save()
        return _ok(f"Created '{title}'")
    return await _outlook(go)


# -- mail -------------------------------------------------------------------
def _format_mail(rows: list[tuple]) -> str:
    """rows: (sender, subject, received, preview). A header line per message
    and the preview indented by two spaces — proactive.count_mail relies on it."""
    out = [f"{when:%Y-%m-%d %H:%M}  {sender} — {subject}\n  {preview}" for sender, subject, when, preview in rows]
    return "\n".join(out) if out else "No messages."


def _sender(m) -> str:
    """"Name <address>" when Outlook has an SMTP address, else the name.
    Exchange senders carry an X.500 path instead; their SMTP address is
    looked up through the address book."""
    name = _clean(_get(m, "SenderName"))
    addr = _clean(_get(m, "SenderEmailAddress"))
    if addr and "@" not in addr:
        addr = ""
        with contextlib.suppress(Exception):
            addr = _clean(m.Sender.GetExchangeUser().PrimarySmtpAddress)
    if addr and name and addr.casefold() != name.casefold():
        return f"{name} <{addr}>"
    return name or addr or "(unknown sender)"


def _mail_rows(items, limit: int) -> list[tuple]:
    rows = []
    for m in _walk(items):
        if len(rows) >= limit:
            break
        if _get(m, "Class", CLASS_MAIL) != CLASS_MAIL:
            continue                    # meeting requests, receipts, ...
        when = _naive(_get(m, "ReceivedTime", None))
        if when is None:
            continue
        preview = _clean(_get(m, "Body"))[:PREVIEW_CHARS]
        rows.append((_sender(m), _clean(_get(m, "Subject")) or "(no subject)", when, preview))
    return rows


def _inbox_items(ns):
    items = ns.GetDefaultFolder(FOLDER_INBOX).Items
    items.Sort("[ReceivedTime]", True)       # newest first
    return items


@tool("mail_unread", "List unread Outlook inbox messages: sender, subject, date, preview", {"limit": int})
@_guard
async def mail_unread(args: dict) -> dict:
    limit = _clamp(args.get("limit", 5), 1, MAIL_LIMIT_MAX, 5)

    def go(app, ns):
        return _mail_rows(_inbox_items(ns).Restrict("[UnRead] = True"), limit)
    return _ok(_format_mail(await _outlook(go)))


async def mail_unread_count() -> int:
    """The inbox's unread count, straight from the folder property — a plain
    helper (not an MCP tool) for the proactive briefing, which wants the
    real number rather than the capped `mail_unread` listing. Raises
    RuntimeError when Outlook fails or reports something that isn't an
    integer."""
    def go(app, ns):
        return ns.GetDefaultFolder(FOLDER_INBOX).UnReadItemCount
    try:
        n = await _outlook(go)
    except OutlookUnavailable:
        raise
    except Exception as exc:
        raise RuntimeError(_com_message(exc)) from None
    try:
        return int(n)
    except (TypeError, ValueError):
        raise RuntimeError(f"unexpected unread count: {n!r}") from None


def _dasl_like(text: str) -> str:
    """`text` as the inside of a DASL LIKE '%...%' literal: quotes doubled,
    and the filter's own wildcard/escape characters dropped."""
    return re.sub(r"[%_\[\]\x00-\x1f]", " ", text).replace("'", "''")


def mail_search_filter(query: str) -> str:
    q = _dasl_like(query)
    fields = ("urn:schemas:httpmail:subject", "urn:schemas:httpmail:fromname",
              "urn:schemas:httpmail:fromemail")
    return "@SQL=" + " OR ".join(f"\"{f}\" LIKE '%{q}%'" for f in fields)


@tool("mail_search", "Search the Outlook inbox by subject/sender substring", {"query": str, "limit": int})
@_guard
async def mail_search(args: dict) -> dict:
    query = _clean(args.get("query", ""))[:200]
    if not query:
        return _err("query is required")
    limit = _clamp(args.get("limit", 5), 1, MAIL_LIMIT_MAX, 5)

    def go(app, ns):
        return _mail_rows(_inbox_items(ns).Restrict(mail_search_filter(query)), limit)
    return _ok(_format_mail(await _outlook(go)))


@tool("mail_send",
      "Compose and send an email from Outlook. `to` is an email address or a contact's name "
      "(looked up in Outlook contacts and the address book)",
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

    def go(app, ns):
        msg = app.CreateItem(ITEM_MAIL)
        msg.To = rec.handle
        msg.Subject = subject
        msg.Body = body
        msg.Send()
        return _ok(f"Sent to {rec.name}")
    return await _outlook(go)


# -- recipients (Outlook contacts / address book) -----------------------------
RESOLVED_TTL_S = 300          # a confirmed name -> address holds this long
_EMAIL_RE = re.compile(r"[^@\s<>\"']+@[^@\s<>\"']+\.[^@\s<>\"']+")


@dataclass(frozen=True)
class Recipient:
    name: str
    handle: str


def is_handle(to: str) -> bool:
    """An email address, as opposed to a person's name."""
    return bool(_EMAIL_RE.fullmatch(to.strip()))


def _contact_row(c) -> tuple[str, list[str]]:
    """An Outlook ContactItem as (display name, [SMTP addresses...])."""
    display = (_clean(_get(c, "FullName")) or _clean(_get(c, "NickName"))
               or _clean(_get(c, "CompanyName")))
    emails = []
    for i in (1, 2, 3):
        addr = _clean(_get(c, f"Email{i}Address"))
        if "@" in addr:                       # an Exchange (EX) entry is X.500, not SMTP
            emails.append(addr)
    return display, emails


def _name_matches(query: str, display: str, extra: list[str]) -> bool:
    q = query.casefold()
    names = [display, *extra]
    return any(n and (n.casefold().startswith(q)
                      or any(part.startswith(q) for part in n.casefold().split()))
               for n in names)


def _search_outlook(ns, name: str) -> list[tuple[str, list[str]]]:
    found = []
    folder = ns.GetDefaultFolder(FOLDER_CONTACTS)
    for c in _walk(folder.Items):
        if _get(c, "Class", 40) != 40:        # olContact (skips distribution lists)
            continue
        display, emails = _contact_row(c)
        extra = [_clean(_get(c, "FirstName")), _clean(_get(c, "LastName")), _clean(_get(c, "NickName"))]
        if display and _name_matches(name, display, extra):
            found.append((display, emails))
    if found:
        return found
    # Not in personal contacts: let Outlook resolve it against the address
    # book (an Exchange GAL, say). Resolve() fails on an ambiguous name.
    rcp = ns.CreateRecipient(name)
    if rcp.Resolve():
        entry = rcp.AddressEntry
        addr = _clean(_get(entry, "Address"))
        if "@" not in addr:
            addr = ""
            with contextlib.suppress(Exception):
                addr = _clean(entry.GetExchangeUser().PrimarySmtpAddress)
        return [(_clean(_get(rcp, "Name")) or name, [addr] if addr else [])]
    return []


def _contacts_search(name: str) -> list[tuple[str, list[str]]]:
    """People matching `name`, as (display name, [email addresses...]), from
    Outlook's contacts and then its address book. Tests replace this.
    Synchronous; raises OutlookUnavailable without Outlook."""
    return _in_outlook(lambda app, ns: _search_outlook(ns, name))


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
        raise ValueError(f"{who} has no email address in Outlook.")
    if len(handles) > 1:
        raise ValueError(f"{who} has several: {_or_list(handles[:4])}. Which one?")
    return Recipient(who, handles[0])


_resolved: dict[str, tuple[float, Recipient]] = {}


def resolve_recipient(to: str) -> Recipient:
    """A mail recipient from what the brain said: an address as-is, a name
    looked up in Outlook. Raises ValueError with a short, speakable reason
    (ambiguous, unknown, no Outlook) instead of guessing. A name resolved
    here is remembered for a few minutes, so the send goes to the very
    address the confirm showed."""
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
    except OutlookUnavailable:
        raise ValueError(OUTLOOK_MISSING + " Or give me the email address.") from None
    rec = _pick(to, found)
    _resolved[key] = (time.monotonic(), rec)
    log.info("mail recipient %r -> %s", to, rec.name)
    return rec


async def resolve_recipient_async(to: str) -> Recipient:
    return await asyncio.to_thread(resolve_recipient, to)


# -- reminders (Outlook tasks) ---------------------------------------------
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


def _task_due(t) -> dt.datetime | None:
    """When a task is due: its reminder time when it has one (that carries
    the hour), else its due date; None for an undated task."""
    for attr, enabled in (("ReminderTime", bool(_get(t, "ReminderSet", False))), ("DueDate", True)):
        if not enabled:
            continue
        d = _naive(_get(t, attr, None))
        if d is not None and d.year < NO_DATE_YEAR:
            return d
    return None


@tool(
    "reminders_due",
    "List incomplete Outlook tasks due within N days (including overdue)",
    {"days": int},
)
@_guard
async def reminders_due(args: dict) -> dict:
    days = _clamp(args.get("days", 1), 1, REMINDERS_DAYS_MAX, 1)
    end = dt.datetime.now() + dt.timedelta(days=days)

    def go(app, ns):
        folder = ns.GetDefaultFolder(FOLDER_TASKS)
        lst = _clean(_get(folder, "Name", "Tasks"))
        rows = []
        for t in _walk(folder.Items.Restrict("[Complete] = False")):
            due = _task_due(t)
            if due is not None and due < end:
                rows.append((_clean(_get(t, "Subject")) or "(no title)", due, lst))
        return sorted(rows, key=lambda r: r[1])
    return _ok(_format_reminders(await _outlook(go)))


@tool("reminder_create", "Create an Outlook task, optionally with a due time and reminder",
      {"title": str, "when": str})
@_guard
async def reminder_create(args: dict) -> dict:
    title = _clean(args.get("title", ""))
    if not title:
        return _err("title is required")
    when_s = str(args.get("when", "") or "").strip()
    when = None
    if when_s:
        try:
            when = _parse_start(when_s)
        except ValueError:
            return _err(f"invalid when: {when_s!r}, expected 'YYYY-MM-DD HH:MM'")

    def go(app, ns):
        task = app.CreateItem(ITEM_TASK)
        task.Subject = title
        if when is not None:
            task.DueDate = dt.datetime.combine(when.date(), dt.time())
            task.ReminderSet = True
            task.ReminderTime = when
        task.Save()
        return _ok(f"Created reminder '{title}'")
    return await _outlook(go)


# -- notes -----------------------------------------------------------------
@tool("notes_create", "Create an Outlook note", {"title": str, "body": str})
@_guard
async def notes_create(args: dict) -> dict:
    title = _clean(args.get("title", ""))
    if not title:
        return _err("title is required")
    body = str(args.get("body", "") or "").replace("\r\n", "\n")
    # A note has no separate title: Outlook shows the body's first line as
    # its subject. Plain text, so nothing needs escaping.
    text = f"{title}\n{body}" if body else title

    def go(app, ns):
        note = app.CreateItem(ITEM_NOTE)
        note.Body = text.replace("\n", "\r\n")
        note.Save()
        return _ok(f"Created note '{title}'")
    return await _outlook(go)


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
