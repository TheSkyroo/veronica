"""Personal data via macOS apps (Calendar, Mail, Reminders) and in-process
timers, exposed to Claude as in-process MCP tools. No OAuth: everything goes
through `osascript` (Apple Events), same argv-only pattern as `tools/mac.py`.
"""
import asyncio
import datetime as dt
import math
import subprocess
from collections.abc import Awaitable, Callable

from claude_agent_sdk import create_sdk_mcp_server, tool

TIMEOUT_S = 30
CALENDAR_DAYS_MAX = 30
REMINDERS_DAYS_MAX = 60
MAIL_LIMIT_MAX = 20


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
    except Exception as exc:  # e.g. FileNotFoundError
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
    """Wrap a handler so malformed args (missing keys, bad types) return
    `_err(...)` instead of raising."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as exc:
            return _err(str(exc))
    return wrapper


async def _osascript(script: str, ok_text: str | None = None) -> dict:
    return await asyncio.to_thread(run, ["osascript", "-e", script], None, ok_text)


# -- date helpers --------------------------------------------------------------
def _parse_day(day: str) -> dt.date:
    if day == "today":
        return dt.date.today()
    if day == "tomorrow":
        return dt.date.today() + dt.timedelta(days=1)
    return dt.datetime.strptime(day, "%Y-%m-%d").date()


def _parse_start(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M")


def _set_date_script(var: str, d: dt.date, hh: int = 0, mm: int = 0) -> str:
    """AppleScript to set `var` (an already-declared `current date`-typed
    variable name) to the given date/time, avoiding locale-fragile `date
    "..."` parsing. Sets day to 1 before year/month to dodge day-overflow
    (e.g. Jan 31 -> Feb) when moving between months."""
    return (
        f"set {var} to current date\n"
        f"set day of {var} to 1\n"
        f"set year of {var} to {d.year}\n"
        f"set month of {var} to {d.month}\n"
        f"set day of {var} to {d.day}\n"
        f"set time of {var} to {hh * 3600 + mm * 60}\n"
    )


# AppleScript handler that collapses tabs/newlines in free text (mail
# subjects/preview, reminder/event titles) so they can't corrupt our
# one-record-per-line, tab-separated wire format.
_SANITIZE_HANDLER = (
    "on sanitize(s)\n"
    "set s to s as string\n"
    "set AppleScript's text item delimiters to tab\n"
    "set s to (text items of s) as string\n"
    'set AppleScript\'s text item delimiters to " "\n'
    "set s to (text items of s) as string\n"
    "set AppleScript's text item delimiters to linefeed\n"
    "set s to (text items of s) as string\n"
    'set AppleScript\'s text item delimiters to " "\n'
    "set s to (text items of s) as string\n"
    "set AppleScript's text item delimiters to return\n"
    "set s to (text items of s) as string\n"
    'set AppleScript\'s text item delimiters to " "\n'
    "set s to (text items of s) as string\n"
    "set AppleScript's text item delimiters to \"\"\n"
    "return s\n"
    "end sanitize\n"
)


def _clamp(value, lo, hi, default):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


# -- calendar -------------------------------------------------------------------
def _format_events(raw: str) -> str:
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    if not lines:
        return "No events."
    out = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 6:
            continue
        title, sh, sm, eh, em, cal, *rest = parts
        loc = rest[0] if rest else ""
        try:
            rng = f"{int(sh):02d}:{int(sm):02d}–{int(eh):02d}:{int(em):02d}"
        except ValueError:
            continue
        entry = f"{rng}  {title} ({cal})"
        if loc.strip():
            entry += f" @ {loc.strip()}"
        out.append(entry)
    return "\n".join(out) if out else "No events."


@tool(
    "calendar_events",
    "List Calendar.app events in a date range: title, start/end time, calendar, location. "
    "Recurring-event instances are not expanded; a recurring series shows only its master event.",
    {"day": str, "days": int},
)
@_guard
async def calendar_events(args: dict) -> dict:
    day = str(args.get("day", "today") or "today")
    try:
        start = _parse_day(day)
    except ValueError:
        return _err(f"invalid day: {day!r}")
    days = _clamp(args.get("days", 1), 1, CALENDAR_DAYS_MAX, 1)

    # Server-side date filtering (`whose start date ≥ … and < …`) is the
    # actual fix for the timeout: the old code fetched every event's start
    # date one at a time via `repeat ... in (events of cal)` and filtered in
    # AppleScript, which is O(every event ever, including recurring
    # instances, in every calendar) rather than O(events in range).
    #
    # NOTE: fetching properties as a batched list (`summary of evs`,
    # `start date of evs`, …) was tried and found to be unreliable against
    # real Calendar.app data — live-tested here, it raised
    # "Can't get summary of {event id ...}" (-1728) for a genuine,
    # just-created event even though `summary of (item 1 of evs)` for the
    # very same event works fine. So each matched event's properties are
    # fetched individually (`item i of evs`), which is one round trip per
    # event rather than one per event per property, but avoids that failure
    # mode entirely.
    script = (
        _SANITIZE_HANDLER
        + 'tell application "Calendar"\n'
        + "set output to \"\"\n"
        + _set_date_script("startDate", start)
        + f"set endDate to startDate + ({days} * days)\n"
        + "repeat with cal in calendars\n"
        + "set evs to (every event of cal whose start date ≥ startDate and start date < endDate)\n"
        + "set evCount to count of evs\n"
        + "set calName to name of cal\n"
        + "repeat with i from 1 to evCount\n"
        + "set evt to item i of evs\n"
        + "set sd to start date of evt\n"
        + "set ed to end date of evt\n"
        + "set loc to \"\"\n"
        + "try\n"
        + "set loc to location of evt\n"
        + "end try\n"
        + "if loc is missing value then set loc to \"\"\n"
        + "set output to output & my sanitize(summary of evt) & tab & (hours of sd) & tab & "
          "(minutes of sd) & tab & (hours of ed) & tab & (minutes of ed) & tab & "
          "my sanitize(calName) & tab & my sanitize(loc) & linefeed\n"
        + "end repeat\n"
        + "end repeat\n"
        + "end tell\n"
        + "return output"
    )
    res = await _osascript(script)
    if res.get("is_error"):
        return res
    return _ok(_format_events(res["content"][0]["text"]))


@tool(
    "calendar_create",
    "Create a Calendar.app event",
    {"title": str, "start": str, "minutes": int, "calendar": str},
)
@_guard
async def calendar_create(args: dict) -> dict:
    title = str(args.get("title", "")).strip()
    if not title:
        return _err("title is required")
    start_s = str(args.get("start", ""))
    try:
        start = _parse_start(start_s)
    except ValueError:
        return _err(f"invalid start: {start_s!r}, expected 'YYYY-MM-DD HH:MM'")
    minutes = _clamp(args.get("minutes", 60), 1, 24 * 60, 60)
    calendar = str(args.get("calendar", "") or "").strip()

    # Calendar.app's scripting dictionary has no "default calendar" property;
    # when none is named, use the first writable calendar (a subscribed/
    # read-only calendar can't accept new events), falling back to the first
    # calendar in the list if that lookup itself fails for any reason.
    script = (
        'tell application "Calendar"\n'
        + _set_date_script("startDate", start.date(), start.hour, start.minute)
        + f"set endDate to startDate + ({minutes} * minutes)\n"
    )
    if calendar:
        script += f'tell calendar "{_q(calendar)}"\n'
    else:
        script += (
            "try\n"
            "set targetCal to first calendar whose writable is true\n"
            "on error\n"
            "set targetCal to calendar 1\n"
            "end try\n"
            "tell targetCal\n"
        )
    script += (
        f'make new event with properties {{summary:"{_q(title)}", start date:startDate, end date:endDate}}\n'
        + "end tell\n"
        + "end tell\n"
    )
    return await _osascript(script, ok_text=f"Created '{title}'")


# -- mail -------------------------------------------------------------------
def _format_mail(raw: str) -> str:
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    if not lines:
        return "No messages."
    out = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 4:
            continue
        sender, subject, when, preview = parts[0], parts[1], parts[2], parts[3]
        out.append(f"{when}  {sender} — {subject}\n  {preview.strip()}")
    return "\n".join(out) if out else "No messages."


_PAD_HANDLER = (
    "on pad(n)\n"
    "if n < 10 then\n"
    "return \"0\" & n\n"
    "else\n"
    "return \"\" & n\n"
    "end if\n"
    "end pad\n"
)


def _mail_script(filter_expr: str, limit: int) -> str:
    """Server-side filter (`messages of inbox whose ...`) does the actual
    perf work (only matching messages come back at all, instead of the old
    code testing every inbox message one at a time), followed by `items 1
    thru n` to cap it at `limit` before touching any properties.
    Properties are fetched per matched message (`item i of msgs`), not as a
    batched property list — batched list fetches (`sender of msgs`, etc.)
    were live-tested against Calendar.app and found to fail for otherwise
    normal data (see calendar_events), so the same batched-list shape is
    avoided here defensively even though it wasn't reproduced against Mail
    directly (no unread mail was available on the test Mac to try it on)."""
    return (
        _SANITIZE_HANDLER
        + _PAD_HANDLER
        + 'tell application "Mail"\n'
        + "set output to \"\"\n"
        + f"set msgs to (messages of inbox whose {filter_expr})\n"
        + "set n to count of msgs\n"
        + f"if n > {limit} then set n to {limit}\n"
        + "repeat with i from 1 to n\n"
        + "set m to item i of msgs\n"
        + "set dt to date received of m\n"
        + "set prev to \"\"\n"
        + "try\n"
        + "set c to my sanitize(content of m as string)\n"
        + "if (length of c) > 200 then\n"
        + "set prev to text 1 thru 200 of c\n"
        + "else\n"
        + "set prev to c\n"
        + "end if\n"
        + "end try\n"
        + "set output to output & my sanitize(sender of m) & tab & my sanitize(subject of m) & tab & "
          "((year of dt) as string) & \"-\" & (my pad(month of dt as integer)) & \"-\" & (my pad(day of dt)) & "
          '" " & (my pad(hours of dt)) & ":" & (my pad(minutes of dt)) & tab & prev & linefeed\n'
        + "end repeat\n"
        + "end tell\n"
        + "return output"
    )


@tool("mail_unread", "List unread Mail.app inbox messages: sender, subject, date, preview", {"limit": int})
@_guard
async def mail_unread(args: dict) -> dict:
    limit = _clamp(args.get("limit", 5), 1, MAIL_LIMIT_MAX, 5)
    script = _mail_script("read status is false", limit)
    res = await _osascript(script)
    if res.get("is_error"):
        return res
    return _ok(_format_mail(res["content"][0]["text"]))


@tool("mail_search", "Search Mail.app inbox by subject/sender substring", {"query": str, "limit": int})
@_guard
async def mail_search(args: dict) -> dict:
    query = str(args.get("query", "")).strip()
    if not query:
        return _err("query is required")
    limit = _clamp(args.get("limit", 5), 1, MAIL_LIMIT_MAX, 5)
    q = f'"{_q(query)}"'
    script = _mail_script(f"(subject contains {q}) or (sender contains {q})", limit)
    res = await _osascript(script)
    if res.get("is_error"):
        return res
    return _ok(_format_mail(res["content"][0]["text"]))


@tool("mail_send", "Compose and send a Mail.app message", {"to": str, "subject": str, "body": str})
@_guard
async def mail_send(args: dict) -> dict:
    to = str(args.get("to", "")).strip()
    subject = str(args.get("subject", ""))
    body = str(args.get("body", ""))
    if not to:
        return _err("to is required")
    script = (
        'tell application "Mail"\n'
        + f'set newMsg to make new outgoing message with properties {{subject:"{_q(subject)}", '
          f'content:"{_q(body)}", visible:false}}\n'
        + "tell newMsg\n"
        + f'make new to recipient at end of to recipients with properties {{address:"{_q(to)}"}}\n'
        + "end tell\n"
        + "send newMsg\n"
        + "end tell\n"
    )
    return await _osascript(script, ok_text=f"Sent to {to}")


# -- reminders ----------------------------------------------------------------
def _format_reminders(raw: str) -> str:
    lines = [ln for ln in raw.split("\n") if ln.strip()]
    if not lines:
        return "No reminders due."
    out = []
    for ln in lines:
        parts = ln.split("\t")
        if len(parts) < 6:
            continue
        name, y, m, d, hh, mm, *rest = parts
        lst = rest[0] if rest else ""
        try:
            when = f"{int(y):04d}-{int(m):02d}-{int(d):02d} {int(hh):02d}:{int(mm):02d}"
        except ValueError:
            continue
        entry = f"{when}  {name}"
        if lst.strip():
            entry += f" ({lst.strip()})"
        out.append(entry)
    return "\n".join(out) if out else "No reminders due."


@tool(
    "reminders_due",
    "List incomplete Reminders.app reminders due within N days (including overdue)",
    {"days": int},
)
@_guard
async def reminders_due(args: dict) -> dict:
    days = _clamp(args.get("days", 1), 1, REMINDERS_DAYS_MAX, 1)
    script = (
        _SANITIZE_HANDLER
        + 'tell application "Reminders"\n'
        + "set output to \"\"\n"
        + f"set endDate to (current date) + ({days} * days)\n"
        + "repeat with lst in lists\n"
        + "repeat with r in (reminders of lst whose completed is false)\n"
        + "set dd to due date of r\n"
        + "if dd is not missing value then\n"
        + "if dd < endDate then\n"
        + "set output to output & my sanitize(name of r) & tab & (year of dd) & tab & "
          "(month of dd as integer) & tab & (day of dd) & tab & (hours of dd) & tab & "
          "(minutes of dd) & tab & my sanitize(name of lst) & linefeed\n"
        + "end if\n"
        + "end if\n"
        + "end repeat\n"
        + "end repeat\n"
        + "end tell\n"
        + "return output"
    )
    res = await _osascript(script)
    if res.get("is_error"):
        return res
    return _ok(_format_reminders(res["content"][0]["text"]))


@tool("reminder_create", "Create a Reminders.app reminder in the default list", {"title": str, "when": str})
@_guard
async def reminder_create(args: dict) -> dict:
    title = str(args.get("title", "")).strip()
    if not title:
        return _err("title is required")
    when_s = str(args.get("when", "") or "").strip()
    due_script = ""
    if when_s:
        try:
            when = _parse_start(when_s)
        except ValueError:
            return _err(f"invalid when: {when_s!r}, expected 'YYYY-MM-DD HH:MM'")
        due_script = (
            _set_date_script("dueDate", when.date(), when.hour, when.minute)
            + "set due date of newRem to dueDate\n"
        )
    script = (
        'tell application "Reminders"\n'
        + "tell default list\n"
        + f'set newRem to make new reminder with properties {{name:"{_q(title)}"}}\n'
        + "end tell\n"
        + due_script
        + "end tell\n"
    )
    return await _osascript(script, ok_text=f"Created reminder '{title}'")


# -- notes -----------------------------------------------------------------
@tool("notes_create", "Create a note in Notes.app", {"title": str, "body": str})
@_guard
async def notes_create(args: dict) -> dict:
    title = str(args.get("title", "")).strip()
    if not title:
        return _err("title is required")
    body = str(args.get("body", "") or "")
    script = (
        'tell application "Notes"\n'
        f'make new note at folder "Notes" with properties {{name:"{_q(title)}", body:"{_q(body)}"}}\n'
        "end tell\n"
    )
    return await _osascript(script, ok_text=f"Created note '{title}'")


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
