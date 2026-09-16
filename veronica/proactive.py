"""Proactive announcements: a scheduled daily briefing and "heads up"
nudges before calendar events. Composes text from the pim tool outputs
and hands it to Orchestrator.announce(), which only speaks when idle and
not muted — this module never touches audio itself."""
import asyncio
import contextlib
import datetime as dt
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from veronica import prefs

log = logging.getLogger("veronica.proactive")

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


@dataclass
class Schedule:
    briefing_enabled: bool = False
    briefing_time: str = "08:00"   # HH:MM, 24h, local time
    nudges_enabled: bool = False
    nudge_minutes: int = 5

    @classmethod
    def from_prefs(cls, d: dict) -> "Schedule":
        s = cls()
        if not isinstance(d, dict):
            return s
        if isinstance(d.get("briefing_enabled"), bool):
            s.briefing_enabled = d["briefing_enabled"]
        if isinstance(d.get("briefing_time"), str) and _TIME_RE.match(d["briefing_time"]):
            s.briefing_time = d["briefing_time"]
        if isinstance(d.get("nudges_enabled"), bool):
            s.nudges_enabled = d["nudges_enabled"]
        nm = d.get("nudge_minutes")
        if isinstance(nm, int) and not isinstance(nm, bool) and 1 <= nm <= 60:
            s.nudge_minutes = nm
        return s

    def to_prefs(self) -> dict:
        return asdict(self)


def load_schedule(load: Callable[[], dict] = prefs.load) -> Schedule:
    return Schedule.from_prefs((load() or {}).get("proactive", {}))


def save_schedule(s: Schedule, save: Callable[[dict], None] = prefs.save) -> None:
    save({"proactive": s.to_prefs()})


# -- parsing the pim tools' text ---------------------------------------------
# pim._format_events: "HH:MM–HH:MM  title (calendar)" [" @ location"], en dash.
EVENT_LINE_RE = re.compile(r"^(\d{2}):(\d{2})–(\d{2}):(\d{2})  (.+?) \([^()]*\)(?: @ .*)?$")
# pim._format_reminders: "YYYY-MM-DD HH:MM  name" [" (list)"]
REMINDER_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}  (.+?)(?: \([^()]*\))?$")


@dataclass
class Event:
    title: str
    start: dt.datetime | None
    end: dt.datetime | None
    all_day: bool


def parse_events(text: str, today: dt.date) -> list[Event]:
    out: list[Event] = []
    for line in (text or "").splitlines():
        m = EVENT_LINE_RE.match(line.strip())
        if not m:
            continue
        sh, sm, eh, em, title = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5].strip()
        if (sh, sm, eh, em) == (0, 0, 0, 0):
            out.append(Event(title, None, None, True))
            continue
        out.append(Event(
            title,
            dt.datetime.combine(today, dt.time(sh, sm)),
            dt.datetime.combine(today, dt.time(eh, em)),
            False,
        ))
    return out


def parse_reminders(text: str) -> list[str]:
    out = []
    for line in (text or "").splitlines():
        m = REMINDER_LINE_RE.match(line.strip())
        if m:
            out.append(m[1].strip())
    return out


def count_mail(text: str) -> int:
    # pim._format_mail: header line per message, preview line indented by 2.
    if not text or text.startswith("No messages"):
        return 0
    return sum(1 for ln in text.splitlines() if ln and not ln.startswith("  "))


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _clock(t: dt.datetime) -> str:
    return f"{t.hour}:{t.minute:02d}"


# -- the ticker --------------------------------------------------------------
class Proactive:
    TICK_S = 60
    EVENTS_CACHE_S = 300
    USER_NAME = "Manik"
    BRIEFING_MAX_TITLES = 4
    REMINDERS_MAX = 3

    def __init__(
        self,
        schedule: Schedule,
        announce: Callable[[str], Awaitable[None]],
        calendar_events: Callable[[str, int], Awaitable[str]],
        mail_unread_count: Callable[[], Awaitable[int]],
        reminders_due: Callable[[int], Awaitable[str]],
        now: Callable[[], dt.datetime] = dt.datetime.now,
    ) -> None:
        self.schedule = schedule
        self._announce = announce
        self._calendar_events = calendar_events
        self._mail_unread_count = mail_unread_count
        self._reminders_due = reminders_due
        self._now = now
        self._task: asyncio.Task | None = None
        self._last_briefing_date: dt.date | None = None
        self._nudged: set[tuple[str, dt.datetime]] = set()
        self._events_cache: tuple[dt.datetime, list[Event]] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("proactive tick failed")
            await asyncio.sleep(self.TICK_S)

    async def tick(self) -> None:
        now = self._now()
        if self.schedule.briefing_enabled and now.strftime("%H:%M") == self.schedule.briefing_time \
                and self._last_briefing_date != now.date():
            self._last_briefing_date = now.date()
            await self._announce(await self.build_briefing())
        if self.schedule.nudges_enabled:
            await self._check_nudges(now)

    # -- briefing --------------------------------------------------------------
    async def _events_today(self, now: dt.datetime) -> list[Event]:
        if self._events_cache is not None:
            fetched_at, events = self._events_cache
            fresh_by_age = (now - fetched_at).total_seconds() < self.EVENTS_CACHE_S
            has_future_event = any(e.start is not None and e.start > now for e in events)
            if fetched_at.date() == now.date() and (fresh_by_age or has_future_event):
                return events
        events = parse_events(await self._calendar_events("today", 1), now.date())
        self._events_cache = (now, events)
        return events

    async def build_briefing(self) -> str:
        now = self._now()
        hour = now.hour
        greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")
        parts = [f"{greeting}, {self.USER_NAME}."]

        try:
            events = parse_events(await self._calendar_events("today", 1), now.date())
        except Exception:
            log.exception("briefing: calendar fetch failed")
            events = None
        if events is not None:
            if not events:
                parts.append("Nothing on your calendar today.")
            else:
                names = [f"{e.title} at {_clock(e.start)}" if e.start else e.title for e in events]
                shown = names[: self.BRIEFING_MAX_TITLES]
                rest = len(names) - len(shown)
                listed = _join(shown) if rest == 0 else ", ".join(shown) + f" and {rest} more"
                plural = "event" if len(events) == 1 else "events"
                parts.append(f"You have {len(events)} {plural} today: {listed}.")

        try:
            n = int(await self._mail_unread_count())
        except Exception:
            log.exception("briefing: mail fetch failed")
            n = 0
        if n:
            parts.append(f"You have {n} unread email{'s' if n != 1 else ''}.")

        try:
            reminders = parse_reminders(await self._reminders_due(1))
        except Exception:
            log.exception("briefing: reminders fetch failed")
            reminders = []
        if reminders:
            parts.append(f"Reminders due: {_join(reminders[: self.REMINDERS_MAX])}.")
        return " ".join(parts)

    # -- nudges ----------------------------------------------------------------
    async def _check_nudges(self, now: dt.datetime) -> None:
        events = await self._events_today(now)
        window = dt.timedelta(minutes=self.schedule.nudge_minutes)
        self._nudged = {k for k in self._nudged if k[1].date() == now.date()}
        for e in events:
            if e.all_day or e.start is None:
                continue
            delta = e.start - now
            if delta < dt.timedelta(0) or delta > window:
                continue
            key = (e.title, e.start)
            if key in self._nudged:
                continue
            self._nudged.add(key)
            mins = max(1, int(round(delta.total_seconds() / 60)))
            when = "in a minute" if mins == 1 else f"in {mins} minutes"
            await self._announce(f"Heads up, {e.title} starts {when}.")
