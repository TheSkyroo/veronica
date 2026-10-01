import base64
import datetime as dt
import email
import email.policy
import json
import re

import pytest

from veronica import google_account as google
from veronica.tools import pim

# -- a fake Google HTTP session ----------------------------------------------------

class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body
        self.content = b"" if body is None else json.dumps(body).encode()
        self.text = self.content.decode()

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class FakeSession:
    """Stands in for google.auth's AuthorizedSession: every request is
    recorded in `calls`, and answered by the first route whose method and
    URL regex match (a dict body, a FakeResponse, or a callable taking the
    call). Unrouted requests are 404s."""

    def __init__(self):
        self.routes = []
        self.calls = []

    def route(self, method, pattern, answer):
        self.routes.insert(0, (method, re.compile(pattern), answer))

    def request(self, method, url, timeout=None, **kw):
        call = {"method": method, "url": url, "timeout": timeout, **kw}
        self.calls.append(call)
        for m, rx, answer in self.routes:
            if m == method and rx.search(url):
                if callable(answer):
                    answer = answer(call)
                return answer if isinstance(answer, FakeResponse) else FakeResponse(200, answer)
        return FakeResponse(404, {"error": {"code": 404, "message": "Not Found"}})

    def find(self, method, pattern):
        return [c for c in self.calls if c["method"] == method and re.search(pattern, c["url"])]


@pytest.fixture
def g(monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(google, "_authorized_session", lambda: fake)
    monkeypatch.setattr(pim, "_warmed", set())
    return fake


def text(res):
    return res["content"][0]["text"]


def local(y, mo, d, h=0, mi=0) -> str:
    """A local wall-clock time as Google sends it: RFC 3339 with an offset."""
    return dt.datetime(y, mo, d, h, mi).astimezone().isoformat()


# -- not set up / not connected ------------------------------------------------------

TOOL_CALLS = [
    ("calendar_events", {}), ("calendar_create", {"title": "x", "start": "2026-09-20 10:00"}),
    ("mail_unread", {}), ("mail_search", {"query": "x"}), ("mail_send", {"to": "a@b.co"}),
    ("reminders_due", {}), ("reminder_create", {"title": "x"}), ("notes_create", {"title": "x"}),
]


@pytest.mark.parametrize("tool, args", TOOL_CALLS)
async def test_every_tool_says_google_isnt_set_up(tmp_home, tool, args):
    res = await getattr(pim, tool).handler(args)
    assert res["is_error"] and "Google isn't set up" in text(res)
    assert "google_client.json" in text(res)


@pytest.mark.parametrize("tool, args", TOOL_CALLS)
async def test_every_tool_says_google_isnt_connected(tmp_home, tool, args):
    (tmp_home / "google_client.json").write_text("{}")
    res = await getattr(pim, tool).handler(args)
    assert res["is_error"] and "Google isn't connected" in text(res) and "connect Google" in text(res)


async def test_http_error_is_reported_with_status(g):
    g.route("GET", r"/calendarList", FakeResponse(500, {"error": {"message": "Backend Error"}}))
    res = await pim.calendar_events.handler({})
    assert res["is_error"] and text(res) == "error: Google said 500: Backend Error"


async def test_every_request_has_a_timeout(g):
    g.route("GET", r"/calendarList", {"items": [{"id": "primary@x", "primary": True}]})
    g.route("GET", r"/events$", {"items": []})
    await pim.calendar_events.handler({})
    assert g.calls and all(c["timeout"] == google.TIMEOUT for c in g.calls)


# -- calendar_events -----------------------------------------------------------------

def ev(title, start, end, **kw):
    def when(v):
        return {"date": v} if len(v) == 10 else {"dateTime": v}
    return {"summary": title, "start": when(start), "end": when(end), **kw}


def calendars(g, *cals):
    g.route("GET", r"/users/me/calendarList$", {"items": list(cals)})


async def test_calendar_events_query_shape(g):
    calendars(g, {"id": "me@gmail.com", "summary": "me@gmail.com", "primary": True, "selected": True})
    g.route("GET", r"/events$", {"items": []})
    await pim.calendar_events.handler({"day": "2026-09-20", "days": 2})
    call, = g.find("GET", r"/events$")
    assert call["url"] == "https://www.googleapis.com/calendar/v3/calendars/me%40gmail.com/events"
    p = call["params"]
    assert (p["singleEvents"], p["orderBy"]) == ("true", "startTime")
    assert dt.datetime.fromisoformat(p["timeMin"]) == dt.datetime(2026, 9, 20).astimezone()
    assert dt.datetime.fromisoformat(p["timeMax"]) == dt.datetime(2026, 9, 22).astimezone()


async def test_calendar_events_formats_output(g):
    calendars(g, {"id": "me", "summary": "me@gmail.com", "primary": True})
    g.route("GET", r"/events$", {"items": [
        ev("Lunch", local(2026, 9, 20, 12), local(2026, 9, 20, 13)),
        ev("Standup", local(2026, 9, 20, 9, 30), local(2026, 9, 20, 10), location="Meet\nRoom"),
    ]})
    res = await pim.calendar_events.handler({"day": "2026-09-20"})
    assert text(res) == ("09:30–10:00  Standup (Calendar) @ Meet Room\n"
                         "12:00–13:00  Lunch (Calendar)")


async def test_calendar_events_converts_other_timezones_to_local(g):
    calendars(g, {"id": "me", "primary": True})
    start = dt.datetime(2026, 9, 20, 9, 0).astimezone().astimezone(dt.UTC)
    g.route("GET", r"/events$", {"items": [
        ev("Call", start.strftime("%Y-%m-%dT%H:%M:%SZ"),
           (start + dt.timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"))]})
    res = await pim.calendar_events.handler({"day": "2026-09-20"})
    assert text(res) == "09:00–09:30  Call (Calendar)"


async def test_calendar_events_all_day_out_of_range_cancelled_declined(g):
    calendars(g, {"id": "me", "primary": True})
    g.route("GET", r"/events$", {"items": [
        ev("Trip", "2026-09-19", "2026-09-22"),                       # began earlier
        ev("Holiday", "2026-09-20", "2026-09-21"),
        ev("Tomorrow", local(2026, 9, 21, 9), local(2026, 9, 21, 10)),
        ev("Gone", local(2026, 9, 20, 9), local(2026, 9, 20, 10), status="cancelled"),
        ev("Nope", local(2026, 9, 20, 9), local(2026, 9, 20, 10),
           attendees=[{"self": True, "responseStatus": "declined"}]),
    ]})
    res = await pim.calendar_events.handler({"day": "2026-09-20"})
    assert text(res) == "00:00–00:00  Holiday (Calendar)"


async def test_calendar_events_merges_selected_calendars(g):
    calendars(g,
              {"id": "me", "summary": "me@gmail.com", "primary": True},
              {"id": "fam#x@group", "summary": "Family", "selected": True},
              {"id": "work", "summary": "Work", "summaryOverride": "Office", "selected": True},
              {"id": "hidden", "summary": "Holidays"})                    # not shown: not read
    g.route("GET", r"/calendars/me/events", {"items": [ev("Gym", local(2026, 9, 20, 18), local(2026, 9, 20, 19))]})
    g.route("GET", r"/calendars/fam%23x%40group/events",
            {"items": [ev("Dinner", local(2026, 9, 20, 20), local(2026, 9, 20, 21))]})
    g.route("GET", r"/calendars/work/events",
            {"items": [ev("Standup", local(2026, 9, 20, 9), local(2026, 9, 20, 9, 15))]})
    res = await pim.calendar_events.handler({"day": "2026-09-20"})
    assert text(res) == ("09:00–09:15  Standup (Office)\n"
                         "18:00–19:00  Gym (Calendar)\n"
                         "20:00–21:00  Dinner (Family)")
    assert not g.find("GET", r"/calendars/hidden/")


async def test_calendar_events_skips_a_failing_secondary_calendar(g):
    calendars(g, {"id": "me", "primary": True}, {"id": "shared", "summary": "Shared", "selected": True})
    g.route("GET", r"/calendars/me/events", {"items": [ev("Gym", local(2026, 9, 20, 18), local(2026, 9, 20, 19))]})
    g.route("GET", r"/calendars/shared/events", FakeResponse(403, {"error": {"message": "Forbidden"}}))
    res = await pim.calendar_events.handler({"day": "2026-09-20"})
    assert text(res) == "18:00–19:00  Gym (Calendar)"


async def test_calendar_events_follows_pages(g):
    calendars(g, {"id": "me", "primary": True})
    g.route("GET", r"/events$", lambda c: (
        {"items": [ev("B", local(2026, 9, 20, 11), local(2026, 9, 20, 12))]} if c["params"].get("pageToken")
        else {"items": [ev("A", local(2026, 9, 20, 10), local(2026, 9, 20, 11))], "nextPageToken": "p2"}))
    res = await pim.calendar_events.handler({"day": "2026-09-20"})
    assert text(res) == "10:00–11:00  A (Calendar)\n11:00–12:00  B (Calendar)"


async def test_calendar_events_no_events(g):
    calendars(g, {"id": "me", "primary": True})
    g.route("GET", r"/events$", {})
    res = await pim.calendar_events.handler({"day": "today"})
    assert text(res) == "No events."


async def test_calendar_events_bad_day_is_error(g):
    res = await pim.calendar_events.handler({"day": "not-a-date"})
    assert res["is_error"] and g.calls == []


async def test_calendar_events_days_clamped(g):
    calendars(g, {"id": "me", "primary": True})
    g.route("GET", r"/events$", {})
    await pim.calendar_events.handler({"day": "2026-09-01", "days": 999})
    p = g.find("GET", r"/events$")[0]["params"]
    assert dt.datetime.fromisoformat(p["timeMax"]) == dt.datetime(2026, 10, 1).astimezone()
    g.calls.clear()
    await pim.calendar_events.handler({"day": "2026-09-01", "days": -5})
    p = g.find("GET", r"/events$")[0]["params"]
    assert dt.datetime.fromisoformat(p["timeMax"]) == dt.datetime(2026, 9, 2).astimezone()


# -- calendar_create -----------------------------------------------------------------

async def test_calendar_create_primary(g):
    g.route("POST", r"/calendars/primary/events$", {"id": "e1"})
    res = await pim.calendar_create.handler({"title": "Dentist\n", "start": "2026-09-20 15:30", "minutes": 45})
    call, = g.calls
    body = call["json"]
    assert body["summary"] == "Dentist"
    assert dt.datetime.fromisoformat(body["start"]["dateTime"]) == dt.datetime(2026, 9, 20, 15, 30).astimezone()
    assert dt.datetime.fromisoformat(body["end"]["dateTime"]) == dt.datetime(2026, 9, 20, 16, 15).astimezone()
    assert text(res) == "Created 'Dentist'"


async def test_calendar_create_named_calendar(g):
    calendars(g, {"id": "me", "primary": True, "accessRole": "owner"},
              {"id": "ro", "summary": "Work", "accessRole": "reader"},
              {"id": "w@group", "summary": "Work", "accessRole": "writer"})
    g.route("POST", r"/events$", {"id": "e1"})
    res = await pim.calendar_create.handler({"title": "Review", "start": "2026-09-20 10:00", "calendar": "work"})
    call, = g.find("POST", r"/events$")
    assert call["url"].endswith("/calendars/w%40group/events") and not res.get("is_error")


async def test_calendar_create_unknown_calendar_is_error(g):
    calendars(g, {"id": "me", "primary": True, "accessRole": "owner"})
    res = await pim.calendar_create.handler({"title": "x", "start": "2026-09-20 10:00", "calendar": "Nope"})
    assert res["is_error"] and "no calendar called 'Nope'" in text(res) and not g.find("POST", "")


@pytest.mark.parametrize("start", ["tomorrow 3pm", "2026-13-01 10:00", "0001-01-01 00:00", ""])
async def test_calendar_create_bad_start_is_error(g, start):
    res = await pim.calendar_create.handler({"title": "x", "start": start})
    assert res["is_error"] and g.calls == []


async def test_calendar_create_missing_title_is_error(g):
    res = await pim.calendar_create.handler({"title": " ", "start": "2026-09-20 10:00"})
    assert res["is_error"] and g.calls == []


async def test_calendar_create_minutes_clamped(g):
    g.route("POST", r"/events$", {})
    await pim.calendar_create.handler({"title": "x", "start": "2026-09-20 10:00", "minutes": 99999})
    body = g.calls[-1]["json"]
    span = dt.datetime.fromisoformat(body["end"]["dateTime"]) - dt.datetime.fromisoformat(body["start"]["dateTime"])
    assert span == dt.timedelta(days=1)


# -- mail_unread / mail_search -------------------------------------------------------

def gmail_message(mid, sender, subject, when: dt.datetime, snippet=""):
    return {"id": mid, "snippet": snippet, "internalDate": str(int(when.timestamp() * 1000)),
            "payload": {"headers": [{"name": "From", "value": sender}, {"name": "Subject", "value": subject},
                                    {"name": "Date", "value": "ignored"}]}}


def mailbox(g, *messages):
    g.route("GET", r"/users/me/messages$", {"messages": [{"id": m["id"]} for m in messages]})
    for m in messages:
        g.route("GET", rf"/users/me/messages/{m['id']}$", m)


async def test_mail_unread_query_and_format(g):
    mailbox(g,
            gmail_message("m2", '"Priya Shah" <priya@x.com>', "Re:\tplans", dt.datetime(2026, 9, 20, 9, 5),
                          "See you &amp; the kids at\n7"),
            gmail_message("m1", "bank@x.com", "", dt.datetime(2026, 9, 19, 18, 0)))
    res = await pim.mail_unread.handler({"limit": 3})
    listing, = g.find("GET", r"/messages$")
    assert listing["url"] == "https://gmail.googleapis.com/gmail/v1/users/me/messages"
    assert listing["params"] == {"q": "is:unread in:inbox", "maxResults": 3}
    get = g.find("GET", r"/messages/m2$")[0]
    assert get["params"] == {"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]}
    assert text(res) == ("2026-09-20 09:05  Priya Shah <priya@x.com> — Re: plans\n  See you & the kids at 7\n"
                         "2026-09-19 18:00  bank@x.com — (no subject)\n  ")


async def test_mail_unread_limit_clamped(g):
    mailbox(g)
    await pim.mail_unread.handler({"limit": 500})
    assert g.calls[0]["params"]["maxResults"] == pim.MAIL_LIMIT_MAX


async def test_mail_preview_is_capped(g):
    mailbox(g, gmail_message("m1", "a@b.co", "s", dt.datetime(2026, 9, 20), "x" * 500))
    res = await pim.mail_unread.handler({})
    assert text(res).splitlines()[1] == "  " + "x" * pim.PREVIEW_CHARS


async def test_mail_unread_no_messages(g):
    g.route("GET", r"/messages$", {"resultSizeEstimate": 0})
    res = await pim.mail_unread.handler({})
    assert text(res) == "No messages."


async def test_mail_unread_count(g):
    g.route("GET", r"/labels/INBOX$", {"id": "INBOX", "messagesUnread": 42})
    assert await pim.mail_unread_count() == 42


async def test_mail_unread_count_raises_when_not_connected(tmp_home):
    with pytest.raises(RuntimeError, match="Google isn't set up"):
        await pim.mail_unread_count()


async def test_mail_unread_count_raises_on_garbage(g):
    g.route("GET", r"/labels/INBOX$", {"id": "INBOX"})
    with pytest.raises(RuntimeError, match="unexpected unread count"):
        await pim.mail_unread_count()


async def test_mail_unread_count_raises_on_http_error(g):
    with pytest.raises(RuntimeError, match="404"):
        await pim.mail_unread_count()


async def test_mail_search_requires_query(g):
    for q in ("", '""'):
        res = await pim.mail_search.handler({"query": q})
        assert res["is_error"] and g.calls == []


async def test_mail_search_quotes_the_query(g):
    mailbox(g)
    await pim.mail_search.handler({"query": 'invoice" OR from:boss \\ x'})
    assert g.calls[0]["params"]["q"] == '"invoice OR from:boss x"'


def test_mail_search_query_cannot_escape_the_phrase():
    q = pim.mail_search_query('a"b\\c\nd')
    assert q == '"a b c d"' and q.count('"') == 2


# -- mail_send ------------------------------------------------------------------

def sent(g):
    call, = g.find("POST", r"/messages/send$")
    raw = base64.urlsafe_b64decode(call["json"]["raw"])
    return email.message_from_bytes(raw, policy=email.policy.default)


async def test_mail_send_to_an_address(g):
    g.route("POST", r"/messages/send$", {"id": "s1"})
    res = await pim.mail_send.handler({"to": "x@y.com", "subject": 'Hi "there"', "body": "Line1\nLine2"})
    msg = sent(g)
    assert g.calls[0]["url"] == "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
    assert (msg["To"], msg["Subject"]) == ("x@y.com", 'Hi "there"')
    assert msg.get_content() == "Line1\nLine2\n"
    assert text(res) == "Sent to x@y.com"


async def test_mail_send_requires_to(g):
    res = await pim.mail_send.handler({"to": "", "subject": "s", "body": "b"})
    assert res["is_error"] and g.calls == []


async def test_mail_send_by_name_resolves_through_contacts(g, _fake_contacts):
    _fake_contacts[0] = ("Priya Shah", ["priya@example.com"])
    g.route("POST", r"/messages/send$", {"id": "s1"})
    res = await pim.mail_send.handler({"to": "Priya", "subject": "s", "body": "on my way"})
    assert sent(g)["To"] == "Priya Shah <priya@example.com>"
    assert not res.get("is_error") and text(res) == "Sent to Priya Shah"


async def test_mail_send_ambiguous_name_asks_and_never_sends(g, _fake_contacts):
    _fake_contacts.append(("Priya Nair", ["pn@x.com"]))
    res = await pim.mail_send.handler({"to": "Priya", "body": "hi"})
    assert res["is_error"] and text(res) == "error: Which Priya — Priya Shah or Priya Nair?"
    assert g.calls == []


async def test_mail_send_unknown_name_says_so(g):
    res = await pim.mail_send.handler({"to": "Zed", "body": "hi"})
    assert res["is_error"] and "No contact named Zed" in text(res) and g.calls == []


async def test_mail_send_to_a_phone_number_is_refused(g):
    res = await pim.mail_send.handler({"to": "+91 98765 43210", "body": "hi"})
    assert res["is_error"] and "isn't an email address" in text(res) and g.calls == []


async def test_mail_send_name_without_google_says_why(tmp_home, monkeypatch):
    monkeypatch.setattr(pim, "_contacts_search", pim._people_search)
    res = await pim.mail_send.handler({"to": "Priya", "body": "hi"})
    assert res["is_error"] and "Google isn't set up" in text(res) and "give me the email address" in text(res)


# -- recipients -------------------------------------------------------------------

def test_resolve_prefers_an_exact_full_name(_fake_contacts):
    _fake_contacts[0] = ("Priya Shah", ["p@x.com"])
    _fake_contacts.append(("Priya Shahani", ["ps@x.com"]))
    assert pim.resolve_recipient("priya shah") == pim.Recipient("Priya Shah", "p@x.com")


def test_resolve_several_addresses_asks_which(_fake_contacts):
    _fake_contacts[0] = ("Priya Shah", ["p@x.com", "priya@y.com"])
    with pytest.raises(ValueError, match="Priya Shah has several: p@x.com or priya@y.com"):
        pim.resolve_recipient("Priya")


def test_resolve_no_email(_fake_contacts):
    _fake_contacts[0] = ("Priya Shah", ["+91 98765 43210"])        # a phone number is no use for mail
    with pytest.raises(ValueError, match="no email address"):
        pim.resolve_recipient("Priya")


def test_resolve_remembers_the_confirmed_address(_fake_contacts, monkeypatch):
    _fake_contacts[0] = ("Priya Shah", ["p@x.com"])
    first = pim.resolve_recipient("Priya")
    _fake_contacts[0] = ("Priya Shah", ["new@x.com"])      # contacts changed after the confirm
    assert pim.resolve_recipient("Priya") == first
    monkeypatch.setattr(pim, "RESOLVED_TTL_S", 0)
    assert pim.resolve_recipient("Priya").handle == "new@x.com"


async def test_resolve_async(_fake_contacts):
    _fake_contacts[0] = ("Priya Shah", ["p@x.com"])
    assert (await pim.resolve_recipient_async("Priya")).handle == "p@x.com"


def person(name, *emails):
    p = {"emailAddresses": [{"value": e} for e in emails]}
    if name:
        p["names"] = [{"displayName": name}]
    return {"person": p}


def test_people_search_warms_up_then_searches_contacts(g):
    g.route("GET", r"/people:searchContacts$", lambda c: (
        {"results": [person("Priya Shah", "p@x.com", "priya@y.com")]} if c["params"]["query"] else {}))
    assert pim._people_search("pri") == [("Priya Shah", ["p@x.com", "priya@y.com"])]
    warm, search = g.calls
    assert warm["url"] == "https://people.googleapis.com/v1/people:searchContacts"
    assert warm["params"] == {"readMask": "names,emailAddresses", "pageSize": 10, "query": ""}
    assert search["params"]["query"] == "pri"
    assert not g.find("GET", r"otherContacts")
    pim._people_search("pri")                                # warmed once per process
    assert len(g.find("GET", r"searchContacts")) == 3


def test_people_search_falls_back_to_other_contacts(g):
    g.route("GET", r"/people:searchContacts$", {})
    g.route("GET", r"/otherContacts:search$",
            {"results": [person("", "dana@corp.com"), person("Dana Lee", "dana.lee@corp.com")]})
    assert pim._people_search("dana") == [("dana@corp.com", ["dana@corp.com"]),
                                          ("Dana Lee", ["dana.lee@corp.com"])]


def test_real_contacts_search_resolves_a_name(g, monkeypatch):
    monkeypatch.setattr(pim, "_contacts_search", pim._people_search)
    g.route("GET", r"/people:searchContacts$", {"results": [person("Priya Shah", "p@x.com")]})
    assert pim.resolve_recipient("Priya") == pim.Recipient("Priya Shah", "p@x.com")


@pytest.mark.parametrize("to, handle", [
    ("a@b.co", True), ("first.last+tag@sub.example.org", True), ("Priya", False),
    ("Priya Shah", False), ("a@b", False), ("+91 98765 43210", False), ("", False),
])
def test_is_handle(to, handle):
    assert pim.is_handle(to) is handle


# -- reminder_create ------------------------------------------------------------

async def test_reminder_create_no_when(g):
    g.route("POST", r"/tasks$", {"id": "t1"})
    res = await pim.reminder_create.handler({"title": "Buy milk"})
    call, = g.calls
    assert call["url"] == "https://tasks.googleapis.com/tasks/v1/lists/@default/tasks"
    assert call["json"] == {"title": "Buy milk"}
    assert text(res) == "Created reminder 'Buy milk'"


async def test_reminder_create_with_when(g):
    g.route("POST", r"/tasks$", {"id": "t1"})
    await pim.reminder_create.handler({"title": "Call mum", "when": "2026-09-21 18:30"})
    assert g.calls[0]["json"] == {"title": "Call mum", "due": "2026-09-21T00:00:00.000Z",
                                  "notes": "Due at 18:30"}


async def test_reminder_create_bad_when_is_error(g):
    res = await pim.reminder_create.handler({"title": "x", "when": "soon"})
    assert res["is_error"] and g.calls == []


async def test_reminder_create_missing_title_is_error(g):
    res = await pim.reminder_create.handler({"title": ""})
    assert res["is_error"] and g.calls == []


# -- reminders_due --------------------------------------------------------------

def gtask(title, due: dt.date | None = None, notes=None, **kw):
    t = {"title": title, "status": "needsAction", **kw}
    if due is not None:
        t["due"] = f"{due:%Y-%m-%d}T00:00:00.000Z"
    if notes is not None:
        t["notes"] = notes
    return t


def tasks(g, *items, title="My Tasks"):
    g.route("GET", r"/users/@me/lists/@default$", {"id": "x", "title": title})
    g.route("GET", r"/lists/@default/tasks$", {"items": list(items)})


async def test_reminders_due_formats_and_filters(g):
    now = dt.datetime.now()
    soon = (now + dt.timedelta(hours=2)).replace(second=0, microsecond=0)
    tasks(g,
          gtask("Someday"),                                                    # undated
          gtask("Far off", dt.date(now.year + 2, 1, 1)),
          gtask("Call\nmum", soon.date(), notes=f"Due at {soon:%H:%M}"),
          gtask("Overdue", dt.date(2020, 1, 2)),
          gtask("Done", dt.date(2020, 1, 1), status="completed"))
    res = await pim.reminders_due.handler({"days": 3})
    call, = g.find("GET", r"/tasks$")
    end = now + dt.timedelta(days=3)
    assert call["params"] == {"showCompleted": "false", "showHidden": "false", "maxResults": 100,
                              "dueMax": f"{end:%Y-%m-%d}T23:59:59Z"}
    assert text(res) == (f"2020-01-02 00:00  Overdue (My Tasks)\n"
                         f"{soon:%Y-%m-%d %H:%M}  Call mum (My Tasks)")


async def test_reminders_due_ignores_a_bogus_time_note(g):
    tasks(g, gtask("Odd", dt.date(2020, 1, 2), notes="Due at 99:99"))
    assert text(await pim.reminders_due.handler({})) == "2020-01-02 00:00  Odd (My Tasks)"


async def test_reminders_due_none(g):
    tasks(g)
    res = await pim.reminders_due.handler({})
    assert text(res) == "No reminders due."


async def test_reminders_due_days_clamped(g):
    far = dt.datetime.now() + dt.timedelta(days=pim.REMINDERS_DAYS_MAX - 1)
    tasks(g, gtask("Later", far.date()))
    assert "Later" in text(await pim.reminders_due.handler({"days": 200}))
    assert "Later" not in text(await pim.reminders_due.handler({"days": -5}))


# -- notes_create ----------------------------------------------------------------

def upload_parts(call):
    ctype = call["headers"]["Content-Type"]
    assert ctype.startswith("multipart/related; boundary=")
    msg = email.message_from_bytes(f"Content-Type: {ctype}\r\n\r\n".encode() + call["data"],
                                   policy=email.policy.default)
    meta, content = msg.iter_parts()
    return json.loads(meta.get_content()), content.get_content().rstrip("\r\n")


async def test_notes_create_makes_the_folder_then_a_doc(g):
    g.route("GET", r"/drive/v3/files$", {"files": []})
    g.route("POST", r"com/drive/v3/files$", {"id": "folder1"})
    g.route("POST", r"/upload/drive/v3/files$", {"id": "doc1"})
    res = await pim.notes_create.handler({"title": "Groceries", "body": "milk\r\neggs"})
    find, = g.find("GET", r"/files$")
    assert find["params"]["q"] == ("name = 'Veronica Notes' and mimeType = "
                                   "'application/vnd.google-apps.folder' and trashed = false")
    mkdir, = g.find("POST", r"com/drive/v3/files$")
    assert mkdir["json"] == {"name": "Veronica Notes", "mimeType": "application/vnd.google-apps.folder"}
    up, = g.find("POST", r"/upload/")
    assert up["params"] == {"uploadType": "multipart", "fields": "id"}
    meta, content = upload_parts(up)
    assert meta == {"name": "Groceries", "mimeType": "application/vnd.google-apps.document",
                    "parents": ["folder1"]}
    assert content.replace("\r\n", "\n") == "milk\neggs"
    assert text(res) == "Created note 'Groceries'"


async def test_notes_create_reuses_the_folder(g):
    g.route("GET", r"/drive/v3/files$", {"files": [{"id": "f9"}]})
    g.route("POST", r"/upload/drive/v3/files$", {"id": "doc1"})
    await pim.notes_create.handler({"title": "Idea"})
    assert not g.find("POST", r"com/drive/v3/files$")
    meta, content = upload_parts(g.find("POST", r"/upload/")[0])
    assert meta["parents"] == ["f9"] and content == ""


async def test_notes_create_missing_title_is_error(g):
    res = await pim.notes_create.handler({"title": "", "body": "x"})
    assert res["is_error"] and g.calls == []


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
