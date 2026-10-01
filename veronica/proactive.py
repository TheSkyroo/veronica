"""Proactive announcements: a scheduled daily briefing and "heads up"
nudges before calendar events. Composes text from the pim tool outputs
and hands it to Orchestrator.announce(), which only speaks when idle and
not muted — this module never touches audio itself. A nudge carries
expires_at=<event start> so one that's still queued when the meeting has
already begun is dropped instead of spoken late.

Quiet hours and a spoken snooze hold announcements rather than dropping
them: anything that comes due while held waits here and is spoken when the
hold ends (minus the ones whose moment has passed, by that same
expires_at), the first one prefixed "While you were away:" when more than
one waited."""
import asyncio
import datetime as dt
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Protocol

from veronica import prefs

log = logging.getLogger("veronica.proactive")

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


@dataclass
class Schedule:
    briefing_enabled: bool = False
    briefing_time: str = "08:00"   # HH:MM, 24h, local time
    nudges_enabled: bool = False
    nudge_minutes: int = 5
    # Quiet hours: off by default, like every other proactive feature — the
    # times are what it uses once switched on.
    quiet_enabled: bool = False
    quiet_from: str = "22:00"
    quiet_to: str = "08:00"
    battery_enabled: bool = False
    unread_enabled: bool = False
    unread_time: str = "11:00"

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
        for flag in ("quiet_enabled", "battery_enabled", "unread_enabled"):
            if isinstance(d.get(flag), bool):
                setattr(s, flag, d[flag])
        for key in ("quiet_from", "quiet_to", "unread_time"):
            if isinstance(d.get(key), str) and _TIME_RE.match(d[key]):
                setattr(s, key, d[key])
        return s

    def to_prefs(self) -> dict:
        return asdict(self)


def load_schedule(load: Callable[[], dict] = prefs.load) -> Schedule:
    return Schedule.from_prefs((load() or {}).get("proactive", {}))


def save_schedule(s: Schedule, save: Callable[[dict], None] = prefs.save) -> None:
    save({"proactive": s.to_prefs()})


# -- parsing the pim tools' text ---------------------------------------------
# pim._format_events: "HH:MM–HH:MM  title (calendar)" [" @ location"], en dash.
EVENT_LINE_RE = re.compile(r"^(\d{2}):(\d{2})–(\d{2}):(\d{2})  (.+?) \((.*)\)(?: @ .*)?$")
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


def _hhmm(s: str) -> dt.time:
    return dt.time(int(s[:2]), int(s[3:5]))


def resolve_hold_until(now: dt.datetime, arg: int | str) -> dt.datetime:
    """The snooze intent's payload -> when the hold ends: minutes from now,
    or the next occurrence of a wall-clock "HH:MM". A trailing "?" marks an
    hour spoken without am/pm ("mute nudges until 5"), which means the next
    5 o'clock either way round, so 5 and 17 both count."""
    if isinstance(arg, int):
        return now + dt.timedelta(minutes=arg)
    ambiguous = arg.endswith("?")
    hour, minute = int(arg[:2]), int(arg[3:5])
    hours = [hour, hour + 12] if ambiguous and hour < 12 else [hour]
    candidates = []
    for h in hours:
        t = now.replace(hour=h % 24, minute=minute, second=0, microsecond=0)
        candidates.append(t if t > now else t + dt.timedelta(days=1))
    return min(candidates)


class Announce(Protocol):
    def __call__(self, text: str, *, expires_at: dt.datetime | None = None) -> Awaitable[None]: ...


# -- the ticker --------------------------------------------------------------
class Proactive:
    TICK_S = 60
    EVENTS_CACHE_S = 300
    BRIEFING_GRACE_S = 7200
    USER_NAME = "Manik"
    BATTERY_PCT = 15
    BRIEFING_MAX_TITLES = 4
    HELD_MAX = 10
    REMINDERS_MAX = 3

    def __init__(
        self,
        schedule: Schedule,
        announce: Announce,
        calendar_events: Callable[[str, int], Awaitable[str]],
        mail_unread_count: Callable[[], Awaitable[int]],
        reminders_due: Callable[[int], Awaitable[str]],
        battery: Callable[[], Awaitable[tuple[int | None, str | None]]] | None = None,
        now: Callable[[], dt.datetime] = dt.datetime.now,
    ) -> None:
        self.schedule = schedule
        self._announce = announce
        self._calendar_events = calendar_events
        self._mail_unread_count = mail_unread_count
        self._reminders_due = reminders_due
        self._battery = battery
        self._now = now
        # "snooze notifications for an hour": deliberately not persisted —
        # a restart is a fresh start.
        self.hold_until: dt.datetime | None = None
        self._held: list[tuple[str, dt.datetime | None]] = []
        self._dropped_held = 0
        self._task: asyncio.Task | None = None
        self._last_briefing_date: dt.date | None = None
        self._nudged: set[tuple[str, dt.datetime]] = set()
        self._events_cache: tuple[dt.datetime, list[Event]] | None = None
        self._events_error: str | None = None
        self._battery_warned = False
        self._last_unread_date: dt.date | None = None

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

    # -- holding (quiet hours, snooze) -------------------------------------------
    def holding(self, now: dt.datetime) -> bool:
        if self.hold_until is not None:
            if now < self.hold_until:
                return True
            self.hold_until = None
        s = self.schedule
        if not s.quiet_enabled or s.quiet_from == s.quiet_to:
            return False
        start, end, clock = _hhmm(s.quiet_from), _hhmm(s.quiet_to), now.time()
        # The window normally wraps midnight (22:00 -> 08:00).
        return start <= clock < end if start < end else (clock >= start or clock < end)

    async def _announce_or_hold(self, text: str, expires_at: dt.datetime | None = None) -> None:
        if self.holding(self._now()):
            # A long snooze must not queue up a monologue for the moment it
            # ends: past HELD_MAX she only says how many she sat on.
            if len(self._held) < self.HELD_MAX:
                self._held.append((text, expires_at))
            else:
                self._dropped_held += 1
            return
        await self._announce(text, expires_at=expires_at)

    async def _flush_held(self, now: dt.datetime) -> None:
        held, self._held = self._held, []
        dropped, self._dropped_held = self._dropped_held, 0
        live = [(t, x) for t, x in held if x is None or x >= now]
        for i, (text, expires_at) in enumerate(live):
            if i == 0 and len(live) > 1:
                text = f"While you were away: {text}"
            await self._announce(text, expires_at=expires_at)
        if dropped:
            await self._announce(f"And {dropped} more I held back.")
        if len(live) != len(held):
            log.info("dropped %d held announcement(s) whose moment had passed", len(held) - len(live))

    async def tick(self) -> None:
        now = self._now()
        if self._held and not self.holding(now):
            await self._flush_held(now)
        if self.schedule.briefing_enabled and self._last_briefing_date != now.date():
            hh, mm = (int(p) for p in self.schedule.briefing_time.split(":"))
            due = dt.datetime.combine(now.date(), dt.time(hh, mm))
            if now >= due:
                late_s = (now - due).total_seconds()
                self._last_briefing_date = now.date()
                if late_s <= self.BRIEFING_GRACE_S:
                    await self._announce_or_hold(await self.build_briefing())
                else:
                    log.info("briefing skipped: %.0fs late", late_s)
        if self.schedule.nudges_enabled:
            await self._check_nudges(now)
        if self.schedule.battery_enabled:
            await self._check_battery()
        if self.schedule.unread_enabled:
            await self._check_unread(now)

    # -- briefing --------------------------------------------------------------
    async def _events_today(self, now: dt.datetime) -> list[Event]:
        if self._events_cache is not None:
            fetched_at, events = self._events_cache
            if (now - fetched_at).total_seconds() < self.EVENTS_CACHE_S and fetched_at.date() == now.date():
                return events
        try:
            events = parse_events(await self._calendar_events("today", 1), now.date())
        except Exception as e:
            # Calendar unavailable (Automation denied, timeout): don't hammer
            # it every tick, and don't let the ticker log a traceback a
            # minute — cache "no events" for the usual window and log once.
            if self._events_error is None or self._events_error != str(e):
                log.warning("nudges: calendar fetch failed, retrying in %ss: %s", self.EVENTS_CACHE_S, e)
                self._events_error = str(e)
            events = []
        else:
            self._events_error = None
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
            if delta < -dt.timedelta(seconds=self.TICK_S) or delta > window:
                continue
            key = (e.title, e.start)
            if key in self._nudged:
                continue
            self._nudged.add(key)
            mins = max(1, int(round(delta.total_seconds() / 60)))
            when = "in a minute" if mins == 1 else f"in {mins} minutes"
            await self._announce_or_hold(f"Heads up, {e.title} starts {when}.", expires_at=e.start)

    # -- battery & unread mail (F4) ----------------------------------------------
    async def _check_battery(self) -> None:
        """Warn once per discharge cycle: plugging in (any state that isn't
        "discharging", including the plugged-in-but-not-charging None) arms
        it again."""
        if self._battery is None:
            return
        try:
            percent, state = await self._battery()
        except Exception:
            log.exception("battery read failed")
            return
        if state != "discharging" or percent is None:
            self._battery_warned = False
            return
        if percent >= self.BATTERY_PCT or self._battery_warned:
            return
        self._battery_warned = True
        await self._announce_or_hold(f"Battery's at {percent} percent.")

    async def _check_unread(self, now: dt.datetime) -> None:
        if self._last_unread_date == now.date():
            return
        hh, mm = (int(p) for p in self.schedule.unread_time.split(":"))
        due = dt.datetime.combine(now.date(), dt.time(hh, mm))
        if now < due:
            return
        self._last_unread_date = now.date()
        if (now - due).total_seconds() > self.BRIEFING_GRACE_S:
            return
        try:
            n = int(await self._mail_unread_count())
        except Exception:
            log.exception("unread nudge: mail fetch failed")
            return
        if n:
            await self._announce_or_hold(f"You have {n} unread since this morning.")
