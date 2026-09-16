import subprocess

import pytest

from veronica.tools import pim


class Done:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


@pytest.fixture
def fake_run(monkeypatch):
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw))
        return Done(out="")

    monkeypatch.setattr(pim.subprocess, "run", run)
    return calls


def text(res):
    return res["content"][0]["text"]


def set_output(monkeypatch, out):
    def run(argv, **kw):
        return Done(out=out)
    monkeypatch.setattr(pim.subprocess, "run", run)


def argv_of(fake_run):
    assert fake_run[0][0][:2] == ["osascript", "-e"]
    return fake_run[0][0]


def assert_my_sanitize_calls(script: str) -> None:
    """Every call to the `sanitize` handler inside a `tell application` block
    must be `my sanitize(...)` — a bare `sanitize(...)` gets routed to the
    target application instead of the script's own handler and fails at
    runtime with 'Can't continue sanitize (-1708)'."""
    assert "my sanitize(" in script
    for line in script.split("\n"):
        if "sanitize(" in line and "on sanitize(" not in line:
            assert "my sanitize(" in line, f"bare sanitize( call: {line!r}"


# -- calendar_events ----------------------------------------------------------

async def test_calendar_events_argv_shape(fake_run):
    await pim.calendar_events.handler({"day": "today", "days": 1})
    argv = argv_of(fake_run)
    assert len(argv) == 3 and "shell" not in fake_run[0][1]
    assert 'tell application "Calendar"' in argv[2]
    assert "whose start date ≥ startDate and start date < endDate" in argv[2]
    assert_my_sanitize_calls(argv[2])


async def test_calendar_events_formats_output(monkeypatch):
    set_output(monkeypatch, "Standup\t9\t30\t10\t0\tWork\tZoom\n")
    res = await pim.calendar_events.handler({"day": "today"})
    assert text(res) == "09:30–10:00  Standup (Work) @ Zoom"


async def test_calendar_events_no_events(monkeypatch):
    set_output(monkeypatch, "")
    res = await pim.calendar_events.handler({"day": "today"})
    assert text(res) == "No events."


async def test_calendar_events_bad_day_is_error(fake_run):
    res = await pim.calendar_events.handler({"day": "not-a-date"})
    assert res["is_error"] and fake_run == []


async def test_calendar_events_days_clamped(fake_run):
    await pim.calendar_events.handler({"day": "today", "days": 999})
    argv = argv_of(fake_run)
    assert f"({pim.CALENDAR_DAYS_MAX} * days)" in argv[2]
    fake_run.clear()
    await pim.calendar_events.handler({"day": "today", "days": -5})
    argv = argv_of(fake_run)
    assert "(1 * days)" in argv[2]


async def test_calendar_events_error(monkeypatch):
    monkeypatch.setattr(pim.subprocess, "run", lambda *a, **k: Done(rc=1, err="nope"))
    res = await pim.calendar_events.handler({"day": "today"})
    assert res["is_error"]


# -- calendar_create ------------------------------------------------------------

async def test_calendar_create_argv_and_escaping(fake_run):
    res = await pim.calendar_create.handler(
        {"title": 'Team "Sync"', "start": "2026-09-20 10:00", "minutes": 30}
    )
    argv = argv_of(fake_run)
    assert '\\"Sync\\"' in argv[2]
    assert "set year of startDate to 2026" in argv[2]
    assert "set month of startDate to 9" in argv[2]
    assert "set day of startDate to 20" in argv[2]
    assert "set time of startDate to 36000" in argv[2]
    assert "(30 * minutes)" in argv[2]
    assert "first calendar whose writable is true" in argv[2]
    assert "calendar 1" in argv[2]  # on-error fallback
    assert not res.get("is_error")


async def test_calendar_create_named_calendar(fake_run):
    await pim.calendar_create.handler(
        {"title": "X", "start": "2026-09-20 10:00", "calendar": "Work"}
    )
    argv = argv_of(fake_run)
    assert 'calendar "Work"' in argv[2]


async def test_calendar_create_bad_start_is_error(fake_run):
    res = await pim.calendar_create.handler({"title": "X", "start": "nonsense"})
    assert res["is_error"] and fake_run == []


async def test_calendar_create_missing_title_is_error(fake_run):
    res = await pim.calendar_create.handler({"title": "", "start": "2026-09-20 10:00"})
    assert res["is_error"] and fake_run == []


async def test_calendar_create_minutes_clamped(fake_run):
    await pim.calendar_create.handler({"title": "X", "start": "2026-09-20 10:00", "minutes": 10_000})
    argv = argv_of(fake_run)
    assert f"({24 * 60} * minutes)" in argv[2]


# -- mail_unread / mail_search --------------------------------------------------

async def test_mail_unread_argv_and_limit(fake_run):
    await pim.mail_unread.handler({"limit": 3})
    argv = argv_of(fake_run)
    assert 'tell application "Mail"' in argv[2]
    assert "if n > 3 then set n to 3" in argv[2]
    assert "whose read status is false" in argv[2]
    assert_my_sanitize_calls(argv[2])


async def test_mail_unread_limit_clamped(fake_run):
    await pim.mail_unread.handler({"limit": 1000})
    argv = argv_of(fake_run)
    assert f"if n > {pim.MAIL_LIMIT_MAX} then set n to {pim.MAIL_LIMIT_MAX}" in argv[2]


async def test_mail_unread_formats_output(monkeypatch):
    set_output(monkeypatch, "Alice <a@x.com>\tHi there\t2026-09-16 09:05\tPreview text\n")
    res = await pim.mail_unread.handler({})
    out = text(res)
    assert "2026-09-16 09:05" in out and "Alice <a@x.com>" in out and "Hi there" in out
    assert "Preview text" in out


async def test_mail_unread_no_messages(monkeypatch):
    set_output(monkeypatch, "")
    res = await pim.mail_unread.handler({})
    assert text(res) == "No messages."


async def test_mail_unread_count_script_and_value(monkeypatch):
    scripts = []

    async def fake_osascript(script, ok_text=None):
        scripts.append(script)
        return pim._ok("7")

    monkeypatch.setattr(pim, "_osascript", fake_osascript)
    assert await pim.mail_unread_count() == 7
    assert len(scripts) == 1
    assert 'tell application "Mail"' in scripts[0] and "unread count of inbox" in scripts[0]
    assert "whose" not in scripts[0]


async def test_mail_unread_count_zero(monkeypatch):
    async def fake_osascript(script, ok_text=None):
        return pim._ok("0")

    monkeypatch.setattr(pim, "_osascript", fake_osascript)
    assert await pim.mail_unread_count() == 0


async def test_mail_unread_count_raises_on_error(monkeypatch):
    async def fake_osascript(script, ok_text=None):
        return pim._err("Mail got an error: Connection is invalid.")

    monkeypatch.setattr(pim, "_osascript", fake_osascript)
    with pytest.raises(RuntimeError, match="Connection is invalid"):
        await pim.mail_unread_count()


async def test_mail_unread_count_raises_on_garbage(monkeypatch):
    async def fake_osascript(script, ok_text=None):
        return pim._ok("lots")

    monkeypatch.setattr(pim, "_osascript", fake_osascript)
    with pytest.raises(RuntimeError, match="lots"):
        await pim.mail_unread_count()


async def test_mail_search_requires_query(fake_run):
    res = await pim.mail_search.handler({"query": ""})
    assert res["is_error"] and fake_run == []


async def test_mail_search_escapes_query(fake_run):
    await pim.mail_search.handler({"query": 'foo"bar'})
    argv = argv_of(fake_run)
    assert '"foo\\"bar"' in argv[2]
    assert "subject contains" in argv[2] and "sender contains" in argv[2]
    assert_my_sanitize_calls(argv[2])


# -- mail_send ------------------------------------------------------------------

async def test_mail_send_argv_and_escaping(fake_run):
    res = await pim.mail_send.handler({"to": "x@y.com", "subject": 'Hi "there"', "body": "Line1\nLine2"})
    argv = argv_of(fake_run)
    assert 'address:"x@y.com"' in argv[2]
    assert '\\"there\\"' in argv[2]
    assert not res.get("is_error")
    assert "Sent to x@y.com" in text(res)


async def test_mail_send_requires_to(fake_run):
    res = await pim.mail_send.handler({"to": "", "subject": "s", "body": "b"})
    assert res["is_error"] and fake_run == []


# -- reminder_create --------------------------------------------------------------

async def test_reminder_create_no_when(fake_run):
    res = await pim.reminder_create.handler({"title": "Buy milk"})
    argv = argv_of(fake_run)
    assert 'name:"Buy milk"' in argv[2]
    assert "due date of newRem" not in argv[2]
    assert "Created reminder 'Buy milk'" in text(res)


async def test_reminder_create_with_when(fake_run):
    await pim.reminder_create.handler({"title": "Call", "when": "2026-09-20 09:00"})
    argv = argv_of(fake_run)
    assert "set due date of newRem to dueDate" in argv[2]
    assert "set time of dueDate to 32400" in argv[2]


async def test_reminder_create_bad_when_is_error(fake_run):
    res = await pim.reminder_create.handler({"title": "Call", "when": "garbage"})
    assert res["is_error"] and fake_run == []


async def test_reminder_create_missing_title_is_error(fake_run):
    res = await pim.reminder_create.handler({"title": ""})
    assert res["is_error"] and fake_run == []


async def test_reminder_create_escapes_title(fake_run):
    await pim.reminder_create.handler({"title": 'Say "hi"'})
    argv = argv_of(fake_run)
    assert '\\"hi\\"' in argv[2]


# -- notes_create ----------------------------------------------------------------

async def test_notes_create(fake_run):
    res = await pim.notes_create.handler({"title": "Groceries", "body": "milk, eggs"})
    argv = argv_of(fake_run)
    assert 'tell application "Notes"' in argv[2]
    assert 'name:"Groceries"' in argv[2]
    assert 'body:"milk, eggs"' in argv[2]
    assert "Created note 'Groceries'" in text(res)
    # T2: default account/folder, not a hard-coded `at folder "Notes"`
    assert "at folder" not in argv[2]
    assert 'make new note with properties' in argv[2]


async def test_notes_create_escapes_html_in_body_and_title(fake_run):
    """T2: note bodies are HTML — user text must render literally."""
    await pim.notes_create.handler({"title": "A <b>bold</b> & co", "body": "x < y && <script>alert(1)</script>"})
    argv = argv_of(fake_run)
    assert 'name:"A &lt;b&gt;bold&lt;/b&gt; &amp; co"' in argv[2]
    assert 'body:"x &lt; y &amp;&amp; &lt;script&gt;alert(1)&lt;/script&gt;"' in argv[2]
    assert "<b>" not in argv[2] and "<script>" not in argv[2]


async def test_notes_create_ampersand(fake_run):
    await pim.notes_create.handler({"title": "R&D", "body": "fish & chips"})
    argv = argv_of(fake_run)
    assert 'name:"R&amp;D"' in argv[2]
    assert 'body:"fish &amp; chips"' in argv[2]


async def test_notes_create_newlines_become_br(fake_run):
    await pim.notes_create.handler({"title": "Lines", "body": "one\ntwo\nthree"})
    argv = argv_of(fake_run)
    assert 'body:"one<br>two<br>three"' in argv[2]
    assert "\n" not in argv[2].split('body:"')[1].split('"')[0]


async def test_notes_create_missing_title_is_error(fake_run):
    res = await pim.notes_create.handler({"title": "", "body": "x"})
    assert res["is_error"] and fake_run == []


async def test_notes_create_no_body(fake_run):
    await pim.notes_create.handler({"title": "Reminder"})
    argv = argv_of(fake_run)
    assert 'body:""' in argv[2]


async def test_notes_create_escapes_quotes(fake_run):
    await pim.notes_create.handler({"title": 'Say "hi"', "body": 'he said "hi"'})
    argv = argv_of(fake_run)
    # quotes are left for _q() (AppleScript escaping), not turned into &quot;
    assert argv[2].count('\\"hi\\"') == 2
    assert "&quot;" not in argv[2]


# -- _q (AppleScript string literal quoting) ------------------------------------

@pytest.mark.parametrize("raw, quoted", [
    ('say "hi"', 'say \\"hi\\"'),
    ("C:\\path\\to", "C:\\\\path\\\\to"),                        # T10: backslashes doubled
    ('back\\slash "and" quote', 'back\\\\slash \\"and\\" quote'),
    ("trailing backslash\\", "trailing backslash\\\\"),
    ('\\"', '\\\\\\"'),                                         # a backslash-quote pair stays a pair
    ("plain", "plain"),
])
def test_q_escapes_backslashes_before_quotes(raw, quoted):
    assert pim._q(raw) == quoted


async def test_reminder_create_escapes_backslashes(fake_run):
    await pim.reminder_create.handler({"title": "path C:\\temp"})
    argv = argv_of(fake_run)
    assert 'name:"path C:\\\\temp"' in argv[2]


# -- reminders_due --------------------------------------------------------------

async def test_reminders_due_argv_and_days_clamp(fake_run):
    await pim.reminders_due.handler({"days": 200})
    argv = argv_of(fake_run)
    assert f"({pim.REMINDERS_DAYS_MAX} * days)" in argv[2]
    assert_my_sanitize_calls(argv[2])


async def test_reminders_due_formats_output(monkeypatch):
    set_output(monkeypatch, "Buy milk\t2026\t9\t20\t9\t0\tHome\n")
    res = await pim.reminders_due.handler({"days": 3})
    assert text(res) == "2026-09-20 09:00  Buy milk (Home)"


async def test_reminders_due_none(monkeypatch):
    set_output(monkeypatch, "")
    res = await pim.reminders_due.handler({})
    assert text(res) == "No reminders due."


# -- errors / timeouts -------------------------------------------------------------

async def test_timeout_is_error(monkeypatch):
    def run(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=20)

    monkeypatch.setattr(pim.subprocess, "run", run)
    res = await pim.calendar_events.handler({"day": "today"})
    assert res["is_error"]


# -- timers -----------------------------------------------------------------------

async def test_timer_tools_without_bound_service():
    pim.bind(None)
    res = await pim.timer_set.handler({"minutes": 1})
    assert res["is_error"]
    res = await pim.timer_list.handler({})
    assert res["is_error"]
    res = await pim.timer_cancel.handler({"label": "x"})
    assert res["is_error"]


class FakeTimerService:
    def __init__(self):
        self.calls = []

    def set(self, minutes, label=""):
        self.calls.append((minutes, label))
        return "abc123"

    def list(self):
        return [{"id": "abc123", "label": "tea", "minutes": 3.0, "remaining_s": 42.0}]

    def cancel(self, label_or_id):
        return label_or_id == "tea"


async def test_timer_set_list_cancel():
    svc = FakeTimerService()
    pim.bind(svc)
    try:
        res = await pim.timer_set.handler({"minutes": 3, "label": "tea"})
        assert not res.get("is_error") and svc.calls == [(3, "tea")]
        res = await pim.timer_list.handler({})
        assert "tea" in text(res) and "42" in text(res)
        res = await pim.timer_cancel.handler({"label": "tea"})
        assert not res.get("is_error")
        res = await pim.timer_cancel.handler({"label": "nope"})
        assert res.get("is_error")
    finally:
        pim.bind(None)


async def test_timer_set_bad_minutes():
    pim.bind(FakeTimerService())
    try:
        res = await pim.timer_set.handler({"minutes": -1})
        assert res["is_error"]
        res = await pim.timer_set.handler({"minutes": "nope"})
        assert res["is_error"]
    finally:
        pim.bind(None)


@pytest.mark.parametrize("minutes", [float("inf"), float("-inf"), float("nan")])
async def test_timer_set_rejects_non_finite(minutes):
    svc = FakeTimerService()
    pim.bind(svc)
    try:
        res = await pim.timer_set.handler({"minutes": minutes})
        assert res["is_error"]
        assert svc.calls == []
    finally:
        pim.bind(None)


@pytest.mark.parametrize("minutes", [1e6, 1e300])
async def test_timer_set_clamps_to_24h(minutes):
    svc = FakeTimerService()
    pim.bind(svc)
    try:
        res = await pim.timer_set.handler({"minutes": minutes, "label": "long"})
        assert not res.get("is_error")
        assert svc.calls == [(24 * 60, "long")]
    finally:
        pim.bind(None)


def test_server_and_names():
    assert pim.pim_server["name"] == "pim"
    assert set(pim.PIM_TOOL_NAMES) == {
        "calendar_events", "calendar_create",
        "mail_unread", "mail_search", "mail_send",
        "reminder_create", "reminders_due",
        "notes_create",
        "timer_set", "timer_list", "timer_cancel",
    }


@pytest.mark.live
async def test_live_calendar_events_today():
    res = await pim.calendar_events.handler({"day": "today"})
    assert not res.get("is_error")


@pytest.mark.live
async def test_live_reminders_due():
    res = await pim.reminders_due.handler({"days": 1})
    assert not res.get("is_error")
