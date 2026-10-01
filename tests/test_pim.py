import contextlib
import datetime as dt
from types import SimpleNamespace

import pytest

from veronica.tools import pim

UTC = dt.timezone.utc


# -- a fake Outlook object model ---------------------------------------------------

class FakeItems:
    """An Outlook Items collection: Sort/Restrict are recorded, Restrict
    keeps the items (tests seed only what Outlook would have matched)
    unless a `matcher` is given, and GetFirst/GetNext walk them."""

    def __init__(self, items=(), log=None, matcher=None):
        self._items = list(items)
        self._i = 0
        self.log = log if log is not None else []
        self.matcher = matcher
        self.IncludeRecurrences = False
        self.added = []

    def Sort(self, key, descending=False):
        self.log.append(("Sort", key, descending, self.IncludeRecurrences))

    def Restrict(self, flt):
        self.log.append(("Restrict", flt, self.IncludeRecurrences))
        keep = [i for i in self._items if self.matcher is None or self.matcher(flt, i)]
        return FakeItems(keep, self.log)

    def GetFirst(self):
        self._i = 0
        return self.GetNext()

    def GetNext(self):
        if self._i >= len(self._items):
            return None
        self._i += 1
        return self._items[self._i - 1]

    def Add(self, kind):
        item = FakeItem(kind=kind)
        self.added.append(item)
        return item


class FakeItem(SimpleNamespace):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.saved = self.sent = False

    def Save(self):
        self.saved = True

    def Send(self):
        self.sent = True


class FakeFolder(SimpleNamespace):
    pass


class FakeRecipient:
    def __init__(self, ok=False, name="", address=""):
        self._ok, self.Name = ok, name
        self.AddressEntry = SimpleNamespace(Address=address)

    def Resolve(self):
        return self._ok


class FakeOutlook:
    def __init__(self):
        self.log = []
        self.folders = {
            pim.FOLDER_CALENDAR: FakeFolder(Name="Calendar", Items=FakeItems(log=self.log), Folders=[]),
            pim.FOLDER_INBOX: FakeFolder(Name="Inbox", Items=FakeItems(log=self.log), UnReadItemCount=0),
            pim.FOLDER_TASKS: FakeFolder(Name="Tasks", Items=FakeItems(log=self.log)),
            pim.FOLDER_CONTACTS: FakeFolder(Name="Contacts", Items=FakeItems(log=self.log)),
        }
        self.Stores = []
        self.created = []
        self.recipient = FakeRecipient()

    # namespace
    def GetDefaultFolder(self, n):
        return self.folders[n]

    def CreateRecipient(self, name):
        self.log.append(("CreateRecipient", name))
        return self.recipient

    # application
    def CreateItem(self, kind):
        item = FakeItem(kind=kind)
        self.created.append(item)
        return item

    def seed(self, folder, items):
        self.folders[folder].Items = FakeItems(items, self.log)


@pytest.fixture
def outlook(monkeypatch):
    fake = FakeOutlook()

    @contextlib.contextmanager
    def session():
        yield fake, fake

    monkeypatch.setattr(pim, "_outlook_session", session)
    return fake


@pytest.fixture
def no_outlook(monkeypatch):
    @contextlib.contextmanager
    def session():
        raise pim.OutlookUnavailable("Invalid class string")
        yield  # pragma: no cover

    monkeypatch.setattr(pim, "_outlook_session", session)


def text(res):
    return res["content"][0]["text"]


def com_time(y, mo, d, h=0, mi=0):
    """What pywin32 hands back: Outlook's local wall clock tagged as UTC."""
    return dt.datetime(y, mo, d, h, mi, tzinfo=UTC)


def event(title, start, end, loc=""):
    return FakeItem(Subject=title, Start=start, End=end, Location=loc)


# -- Outlook missing ---------------------------------------------------------------

@pytest.mark.parametrize("tool, args", [
    ("calendar_events", {}), ("calendar_create", {"title": "x", "start": "2026-09-20 10:00"}),
    ("mail_unread", {}), ("mail_search", {"query": "x"}), ("mail_send", {"to": "a@b.co"}),
    ("reminders_due", {}), ("reminder_create", {"title": "x"}), ("notes_create", {"title": "x"}),
])
async def test_every_tool_says_outlook_is_missing(no_outlook, tool, args):
    res = await getattr(pim, tool).handler(args)
    assert res["is_error"] and text(res) == f"error: {pim.OUTLOOK_MISSING}"


def test_real_session_without_pywin32_is_unavailable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pythoncom", None)     # import fails
    with pytest.raises(pim.OutlookUnavailable), pim._outlook_session():
        pass


async def test_com_error_text_is_reported(outlook, monkeypatch):
    class ComError(Exception):
        pass

    def boom(n):
        raise ComError(-2147352567, "Exception occurred.",
                       (4096, "Microsoft Outlook", "The operation failed.", None, 0, -2147467259), None)
    monkeypatch.setattr(outlook, "GetDefaultFolder", boom)
    res = await pim.mail_unread.handler({})
    assert res["is_error"] and text(res) == "error: The operation failed."


# -- calendar_events ----------------------------------------------------------

async def test_calendar_events_query_shape(outlook):
    await pim.calendar_events.handler({"day": "2026-09-20", "days": 2})
    sort, restrict = outlook.log
    assert sort == ("Sort", "[Start]", False, False)          # sorted before recurrences are expanded
    assert restrict == ("Restrict", "[Start] < '2026-09-22 00:00' AND [End] > '2026-09-20 00:00'", True)


async def test_calendar_events_formats_output(outlook):
    outlook.seed(pim.FOLDER_CALENDAR, [
        event("Lunch", com_time(2026, 9, 20, 12), com_time(2026, 9, 20, 13)),
        event("Standup", com_time(2026, 9, 20, 9, 30), com_time(2026, 9, 20, 10), "Teams\nMeeting"),
    ])
    res = await pim.calendar_events.handler({"day": "2026-09-20"})
    assert text(res) == ("09:30–10:00  Standup (Calendar) @ Teams Meeting\n"
                         "12:00–13:00  Lunch (Calendar)")


async def test_calendar_events_all_day_and_out_of_range(outlook):
    outlook.seed(pim.FOLDER_CALENDAR, [
        event("Trip", com_time(2026, 9, 19), com_time(2026, 9, 22)),          # began earlier
        event("Holiday", com_time(2026, 9, 20), com_time(2026, 9, 21)),
        event("Tomorrow", com_time(2026, 9, 21, 9), com_time(2026, 9, 21, 10)),
    ])
    res = await pim.calendar_events.handler({"day": "2026-09-20"})
    assert text(res) == "00:00–00:00  Holiday (Calendar)"


async def test_calendar_events_no_events(outlook):
    res = await pim.calendar_events.handler({"day": "today"})
    assert text(res) == "No events."


async def test_calendar_events_bad_day_is_error(outlook):
    res = await pim.calendar_events.handler({"day": "not-a-date"})
    assert res["is_error"] and outlook.log == []


async def test_calendar_events_days_clamped(outlook):
    await pim.calendar_events.handler({"day": "2026-09-01", "days": 999})
    assert "[Start] < '2026-10-01 00:00'" in outlook.log[1][1]
    outlook.log.clear()
    await pim.calendar_events.handler({"day": "2026-09-01", "days": -5})
    assert "[Start] < '2026-09-02 00:00'" in outlook.log[1][1]


# -- calendar_create ------------------------------------------------------------

async def test_calendar_create_default_calendar(outlook):
    res = await pim.calendar_create.handler(
        {"title": 'Team "Sync"', "start": "2026-09-20 10:00", "minutes": 30})
    appt, = outlook.created
    assert appt.kind == pim.ITEM_APPOINTMENT and appt.saved
    assert appt.Subject == 'Team "Sync"' and appt.Start == dt.datetime(2026, 9, 20, 10)
    assert appt.Duration == 30
    assert text(res) == "Created 'Team \"Sync\"'"


async def test_calendar_create_named_calendar(outlook):
    work = FakeFolder(Name="Work", Items=FakeItems())
    outlook.folders[pim.FOLDER_CALENDAR].Folders = [work]
    res = await pim.calendar_create.handler({"title": "X", "start": "2026-09-20 10:00", "calendar": "work"})
    assert not res.get("is_error") and outlook.created == []
    assert work.Items.added[0].Subject == "X" and work.Items.added[0].saved


async def test_calendar_create_unknown_calendar_is_error(outlook):
    res = await pim.calendar_create.handler({"title": "X", "start": "2026-09-20 10:00", "calendar": "Nope"})
    assert res["is_error"] and "no calendar called 'Nope'" in text(res) and outlook.created == []


@pytest.mark.parametrize("start", ["nonsense", "2026-13-01 10:00", "20/09/2026 10:00", "0001-01-01 00:00"])
async def test_calendar_create_bad_start_is_error(outlook, start):
    res = await pim.calendar_create.handler({"title": "X", "start": start})
    assert res["is_error"] and outlook.created == []


async def test_calendar_create_missing_title_is_error(outlook):
    res = await pim.calendar_create.handler({"title": " \n", "start": "2026-09-20 10:00"})
    assert res["is_error"] and outlook.created == []


async def test_calendar_create_minutes_clamped(outlook):
    await pim.calendar_create.handler({"title": "X", "start": "2026-09-20 10:00", "minutes": 10_000})
    assert outlook.created[0].Duration == 24 * 60


# -- mail_unread / mail_search --------------------------------------------------

def mail(sender, addr, subject, when, body="", cls=pim.CLASS_MAIL):
    return FakeItem(SenderName=sender, SenderEmailAddress=addr, Subject=subject,
                    ReceivedTime=when, Body=body, Class=cls)


async def test_mail_unread_query_and_format(outlook):
    outlook.seed(pim.FOLDER_INBOX, [
        mail("Alice", "a@x.com", "Hi\tthere", com_time(2026, 9, 16, 9, 5), "Preview\r\ntext"),
        mail("Bob", "/O=EXCHANGE/CN=bob", "Invite", com_time(2026, 9, 16, 8), cls=53),
        mail("Bob", "/O=EXCHANGE/CN=bob", "Notes", com_time(2026, 9, 16, 8)),
    ])
    res = await pim.mail_unread.handler({})
    assert outlook.log[:2] == [("Sort", "[ReceivedTime]", True, False), ("Restrict", "[UnRead] = True", False)]
    assert text(res) == ("2026-09-16 09:05  Alice <a@x.com> — Hi there\n  Preview text\n"
                         "2026-09-16 08:00  Bob — Notes\n  ")


async def test_mail_unread_limit_clamped(outlook):
    outlook.seed(pim.FOLDER_INBOX, [mail("A", "a@x.com", f"m{i}", com_time(2026, 9, 1)) for i in range(50)])
    res = await pim.mail_unread.handler({"limit": 1000})
    assert text(res).count(" — m") == pim.MAIL_LIMIT_MAX
    res = await pim.mail_unread.handler({"limit": 3})
    assert text(res).count(" — m") == 3


async def test_mail_preview_is_capped(outlook):
    outlook.seed(pim.FOLDER_INBOX, [mail("A", "a@x.com", "s", com_time(2026, 9, 1), "x" * 1000)])
    preview = text(await pim.mail_unread.handler({})).split("\n")[1]
    assert preview == "  " + "x" * pim.PREVIEW_CHARS


async def test_mail_unread_no_messages(outlook):
    res = await pim.mail_unread.handler({})
    assert text(res) == "No messages."


async def test_mail_unread_count(outlook):
    outlook.folders[pim.FOLDER_INBOX].UnReadItemCount = 7
    assert await pim.mail_unread_count() == 7
    outlook.folders[pim.FOLDER_INBOX].UnReadItemCount = 0
    assert await pim.mail_unread_count() == 0


async def test_mail_unread_count_raises_without_outlook(no_outlook):
    with pytest.raises(RuntimeError, match="classic Outlook"):
        await pim.mail_unread_count()


async def test_mail_unread_count_raises_on_garbage(outlook):
    outlook.folders[pim.FOLDER_INBOX].UnReadItemCount = "lots"
    with pytest.raises(RuntimeError, match="lots"):
        await pim.mail_unread_count()


async def test_mail_search_requires_query(outlook):
    res = await pim.mail_search.handler({"query": "  "})
    assert res["is_error"] and outlook.log == []


async def test_mail_search_filter_escapes_query(outlook):
    await pim.mail_search.handler({"query": "o'neil"})
    flt = outlook.log[1][1]
    assert flt.startswith("@SQL=")
    assert "\"urn:schemas:httpmail:subject\" LIKE '%o''neil%'" in flt
    assert "fromname" in flt and "fromemail" in flt


def test_mail_search_filter_drops_wildcards():
    assert "%_%" not in pim.mail_search_filter("_") and "[" not in pim.mail_search_filter("[x]")


# -- mail_send ------------------------------------------------------------------

async def test_mail_send_to_an_address(outlook):
    res = await pim.mail_send.handler({"to": "x@y.com", "subject": 'Hi "there"', "body": "Line1\nLine2"})
    msg, = outlook.created
    assert msg.kind == pim.ITEM_MAIL and msg.sent
    assert (msg.To, msg.Subject, msg.Body) == ("x@y.com", 'Hi "there"', "Line1\nLine2")
    assert text(res) == "Sent to x@y.com"


async def test_mail_send_requires_to(outlook):
    res = await pim.mail_send.handler({"to": "", "subject": "s", "body": "b"})
    assert res["is_error"] and outlook.created == []


async def test_mail_send_by_name_resolves_through_contacts(outlook, _fake_contacts):
    _fake_contacts[0] = ("Priya Shah", ["priya@example.com"])
    res = await pim.mail_send.handler({"to": "Priya", "subject": "s", "body": "on my way"})
    assert outlook.created[0].To == "priya@example.com"
    assert not res.get("is_error") and text(res) == "Sent to Priya Shah"


async def test_mail_send_ambiguous_name_asks_and_never_sends(outlook, _fake_contacts):
    _fake_contacts.append(("Priya Nair", ["pn@x.com"]))
    res = await pim.mail_send.handler({"to": "Priya", "body": "hi"})
    assert res["is_error"] and text(res) == "error: Which Priya — Priya Shah or Priya Nair?"
    assert outlook.created == []


async def test_mail_send_unknown_name_says_so(outlook):
    res = await pim.mail_send.handler({"to": "Zed", "body": "hi"})
    assert res["is_error"] and "No contact named Zed" in text(res) and outlook.created == []


async def test_mail_send_to_a_phone_number_is_refused(outlook):
    res = await pim.mail_send.handler({"to": "+1 555 123 4567", "body": "hi"})
    assert res["is_error"] and "isn't an email address" in text(res) and outlook.created == []


async def test_mail_send_name_without_outlook_says_why(outlook, monkeypatch):
    def unavailable(name):
        raise pim.OutlookUnavailable()
    monkeypatch.setattr(pim, "_contacts_search", unavailable)
    res = await pim.mail_send.handler({"to": "Priya", "body": "hi"})
    assert res["is_error"] and "Outlook" in text(res) and "email address" in text(res)
    assert outlook.created == []


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


def contact(full, *emails, first="", last="", nick="", cls=40):
    kw = {f"Email{i + 1}Address": e for i, e in enumerate(emails)}
    return FakeItem(FullName=full, FirstName=first, LastName=last, NickName=nick, Class=cls, **kw)


def test_search_outlook_contacts(outlook):
    outlook.seed(pim.FOLDER_CONTACTS, [
        contact("Priya Shah", "p@x.com", "/O=EXCH/CN=PRIYA", first="Priya", last="Shah"),
        contact("Rahul Verma", "r@x.com"),
        contact("Priya's list", "l@x.com", cls=69),                           # a distribution list
    ])
    assert pim._search_outlook(outlook, "shah") == [("Priya Shah", ["p@x.com"])]
    assert pim._search_outlook(outlook, "pri") == [("Priya Shah", ["p@x.com"])]
    assert ("CreateRecipient", "pri") not in outlook.log


def test_search_outlook_falls_back_to_the_address_book(outlook):
    outlook.recipient = FakeRecipient(ok=True, name="Dana Lee", address="dana@corp.com")
    assert pim._search_outlook(outlook, "Dana") == [("Dana Lee", ["dana@corp.com"])]
    outlook.recipient = FakeRecipient(ok=False)
    assert pim._search_outlook(outlook, "Nobody") == []


def test_contacts_search_runs_in_an_outlook_session(outlook, monkeypatch):
    monkeypatch.undo()                                   # the real _contacts_search, not conftest's fake
    fake = outlook

    @contextlib.contextmanager
    def session():
        yield fake, fake
    monkeypatch.setattr(pim, "_outlook_session", session)
    outlook.seed(pim.FOLDER_CONTACTS, [contact("Priya Shah", "p@x.com")])
    assert pim._contacts_search("priya") == [("Priya Shah", ["p@x.com"])]


@pytest.mark.parametrize("to, handle", [
    ("priya@example.com", True), ("a.b+c@sub.example.co.uk", True),
    ("+15551234567", False), ("Priya", False), ("Priya Shah", False), ("a@b", False), ("@x.com", False),
])
def test_is_handle(to, handle):
    assert pim.is_handle(to) is handle


# -- reminder_create --------------------------------------------------------------

async def test_reminder_create_no_when(outlook):
    res = await pim.reminder_create.handler({"title": "Buy milk"})
    task, = outlook.created
    assert task.kind == pim.ITEM_TASK and task.Subject == "Buy milk" and task.saved
    assert not hasattr(task, "DueDate") and not hasattr(task, "ReminderTime")
    assert text(res) == "Created reminder 'Buy milk'"


async def test_reminder_create_with_when(outlook):
    await pim.reminder_create.handler({"title": "Call", "when": "2026-09-20 09:00"})
    task, = outlook.created
    assert task.DueDate == dt.datetime(2026, 9, 20)
    assert task.ReminderSet is True and task.ReminderTime == dt.datetime(2026, 9, 20, 9)


async def test_reminder_create_bad_when_is_error(outlook):
    res = await pim.reminder_create.handler({"title": "Call", "when": "garbage"})
    assert res["is_error"] and outlook.created == []


async def test_reminder_create_missing_title_is_error(outlook):
    res = await pim.reminder_create.handler({"title": ""})
    assert res["is_error"] and outlook.created == []


# -- reminders_due --------------------------------------------------------------

def task(subject, due=None, reminder=None):
    return FakeItem(Subject=subject, DueDate=due or com_time(4501, 1, 1),
                    ReminderSet=reminder is not None, ReminderTime=reminder or com_time(4501, 1, 1))


async def test_reminders_due_formats_and_filters(outlook):
    now = dt.datetime.now()
    soon = now + dt.timedelta(hours=2)
    soon_utc = com_time(soon.year, soon.month, soon.day, soon.hour, soon.minute)
    outlook.seed(pim.FOLDER_TASKS, [
        task("Someday"),                                                    # undated
        task("Far off", due=com_time(now.year + 2, 1, 1)),
        task("Call\nmum", due=com_time(soon.year, soon.month, soon.day), reminder=soon_utc),
        task("Overdue", due=com_time(2020, 1, 2)),
    ])
    res = await pim.reminders_due.handler({"days": 3})
    assert outlook.log[0] == ("Restrict", "[Complete] = False", False)
    assert text(res) == (f"2020-01-02 00:00  Overdue (Tasks)\n"
                         f"{soon:%Y-%m-%d %H:%M}  Call mum (Tasks)")


async def test_reminders_due_none(outlook):
    res = await pim.reminders_due.handler({})
    assert text(res) == "No reminders due."


async def test_reminders_due_days_clamped(outlook):
    far = dt.datetime.now() + dt.timedelta(days=pim.REMINDERS_DAYS_MAX - 1)
    outlook.seed(pim.FOLDER_TASKS, [task("Later", due=com_time(far.year, far.month, far.day))])
    assert "Later" in text(await pim.reminders_due.handler({"days": 200}))
    assert "Later" not in text(await pim.reminders_due.handler({"days": -5}))


# -- notes_create ----------------------------------------------------------------

async def test_notes_create(outlook):
    res = await pim.notes_create.handler({"title": "Groceries", "body": "milk\neggs"})
    note, = outlook.created
    assert note.kind == pim.ITEM_NOTE and note.saved
    assert note.Body == "Groceries\r\nmilk\r\neggs"
    assert text(res) == "Created note 'Groceries'"


async def test_notes_create_text_is_literal(outlook):
    await pim.notes_create.handler({"title": "R&D <b>", "body": 'say "hi" & <script>'})
    assert outlook.created[0].Body == 'R&D <b>\r\nsay "hi" & <script>'


async def test_notes_create_no_body(outlook):
    await pim.notes_create.handler({"title": "Reminder"})
    assert outlook.created[0].Body == "Reminder"


async def test_notes_create_missing_title_is_error(outlook):
    res = await pim.notes_create.handler({"title": "", "body": "x"})
    assert res["is_error"] and outlook.created == []


# -- helpers ----------------------------------------------------------------------

def test_naive_drops_the_bogus_utc_tag():
    assert pim._naive(com_time(2026, 9, 20, 9, 30)) == dt.datetime(2026, 9, 20, 9, 30)
    assert pim._naive(None) is None


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
