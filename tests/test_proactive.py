import asyncio
import datetime as dt

from veronica import proactive as pr


TODAY = dt.date(2026, 9, 16)


def test_schedule_roundtrip():
    s = pr.Schedule(briefing_enabled=True, briefing_time="07:30", nudges_enabled=True, nudge_minutes=10)
    assert pr.Schedule.from_prefs(s.to_prefs()) == s
    assert pr.Schedule.from_prefs({}) == pr.Schedule()
    assert pr.Schedule.from_prefs({"briefing_time": "bogus", "nudge_minutes": "x"}) == pr.Schedule()


def test_load_save_schedule():
    store = {}
    s = pr.Schedule(briefing_enabled=True)
    pr.save_schedule(s, save=lambda d: store.update(d))
    assert store == {"proactive": s.to_prefs()}
    assert pr.load_schedule(load=lambda: store) == s


def test_parse_events():
    text = "09:30–10:00  Standup (Work) @ Zoom\n00:00–00:00  Holiday (Home)\n13:00–14:00  Lunch with Sam (Personal)\n09:00–09:30  Sync (Work (Shared))"
    evs = pr.parse_events(text, TODAY)
    assert [e.title for e in evs] == ["Standup", "Holiday", "Lunch with Sam", "Sync"]
    assert evs[0].start == dt.datetime(2026, 9, 16, 9, 30) and evs[0].end == dt.datetime(2026, 9, 16, 10, 0)
    assert evs[1].all_day and evs[1].start is None
    assert pr.parse_events("No events.", TODAY) == []
    assert pr.parse_events("", TODAY) == []


def test_parse_reminders_and_mail():
    assert pr.parse_reminders("2026-09-16 09:00  Pay rent (Bills)\n2026-09-16 18:00  Call mum") == ["Pay rent", "Call mum"]
    assert pr.parse_reminders("No reminders due.") == []
    assert pr.count_mail("10:02  Alice — Hi\n  preview\n09:00  Bob — Yo\n  preview") == 2
    assert pr.count_mail("No messages.") == 0


class Clock:
    def __init__(self, t): self.t = t
    def __call__(self): return self.t


def make(schedule, events="No events.", mail=0, reminders="No reminders due.", now=None):
    said = []
    calls = {"events": 0}

    async def announce(t): said.append(t)
    async def cal(day, days):
        calls["events"] += 1
        return events
    async def mail_count(): return mail
    async def rem(days): return reminders

    clock = Clock(now or dt.datetime(2026, 9, 16, 8, 0))
    p = pr.Proactive(schedule, announce, cal, mail_count, rem, now=clock)
    return p, said, clock, calls


async def test_build_briefing_composition():
    p, _, clock, _ = make(pr.Schedule(), events="09:30–10:00  Standup (Work)\n13:00–14:00  Lunch (P)\n15:00–15:30  A (P)\n16:00–16:30  B (P)\n17:00–17:30  C (P)\n18:00–18:30  D (P)", mail=3, reminders="2026-09-16 09:00  Pay rent (Bills)")
    text = await p.build_briefing()
    assert text == ("Good morning, Manik. You have 6 events today: Standup at 9:30, Lunch at 13:00, A at 15:00, "
                    "B at 16:00 and 2 more. You have 3 unread emails. Reminders due: Pay rent.")


async def test_build_briefing_singular_event():
    p, _, _, _ = make(pr.Schedule(), events="09:30–10:00  Standup (Work)")
    text = await p.build_briefing()
    assert "You have 1 event today: Standup at 9:30." in text


async def test_build_briefing_empty_and_greetings():
    p, _, clock, _ = make(pr.Schedule())
    assert await p.build_briefing() == "Good morning, Manik. Nothing on your calendar today."
    clock.t = dt.datetime(2026, 9, 16, 13, 0)
    assert (await p.build_briefing()).startswith("Good afternoon, Manik.")
    clock.t = dt.datetime(2026, 9, 16, 19, 0)
    assert (await p.build_briefing()).startswith("Good evening, Manik.")


async def test_build_briefing_one_email_and_reminder_join():
    p, _, _, _ = make(pr.Schedule(), mail=1, reminders="2026-09-16 09:00  A\n2026-09-16 09:00  B\n2026-09-16 09:00  C\n2026-09-16 09:00  D")
    text = await p.build_briefing()
    assert "You have 1 unread email." in text
    assert text.endswith("Reminders due: A, B and C.")


async def test_build_briefing_tolerates_failing_fetcher():
    p, _, _, _ = make(pr.Schedule(), mail=2)
    async def boom(day, days): raise RuntimeError("no calendar")
    p._calendar_events = boom
    assert await p.build_briefing() == "Good morning, Manik. You have 2 unread emails."


async def test_briefing_fires_once_per_day_at_time():
    p, said, clock, _ = make(pr.Schedule(briefing_enabled=True, briefing_time="08:00"))
    clock.t = dt.datetime(2026, 9, 16, 7, 59)
    await p.tick(); assert said == []
    clock.t = dt.datetime(2026, 9, 16, 8, 1)      # a tick at exactly 08:00 was missed
    await p.tick(); assert len(said) == 1 and said[0].startswith("Good morning")
    await p.tick(); assert len(said) == 1
    clock.t = dt.datetime(2026, 9, 17, 8, 0)
    await p.tick(); assert len(said) == 2


async def test_briefing_skipped_when_far_past_time():
    p, said, clock, _ = make(pr.Schedule(briefing_enabled=True, briefing_time="08:00"))
    clock.t = dt.datetime(2026, 9, 16, 10, 30)     # 2h30m late: past BRIEFING_GRACE_S
    await p.tick(); assert said == []
    clock.t = dt.datetime(2026, 9, 16, 10, 31)
    await p.tick(); assert said == []


async def test_briefing_disabled_never_fires():
    p, said, clock, _ = make(pr.Schedule(briefing_enabled=False))
    await p.tick(); assert said == []


async def test_nudge_fires_once_within_window_and_skips_all_day():
    ev = "09:30–10:00  Standup (Work)\n00:00–00:00  Holiday (Home)\n11:00–12:00  Review (Work)"
    p, said, clock, calls = make(pr.Schedule(nudges_enabled=True, nudge_minutes=5), events=ev)
    clock.t = dt.datetime(2026, 9, 16, 9, 20)
    await p.tick(); assert said == []
    clock.t = dt.datetime(2026, 9, 16, 9, 23)
    await p.tick(); assert said == []
    clock.t = dt.datetime(2026, 9, 16, 9, 24)
    await p.tick(); assert said == []
    assert calls["events"] == 1                      # 9:20, 9:23, 9:24 all within 5 min of the 9:20 fetch
    clock.t = dt.datetime(2026, 9, 16, 9, 25)
    await p.tick(); assert said == ["Heads up, Standup starts in 5 minutes."]
    assert calls["events"] == 2                       # 9:25 is exactly 300s after 9:20: not fresh, refetch
    clock.t = dt.datetime(2026, 9, 16, 9, 26)
    await p.tick(); assert len(said) == 1          # not repeated
    assert calls["events"] == 2                       # 9:26 is within 5 min of the 9:25 fetch
    clock.t = dt.datetime(2026, 9, 16, 10, 59)
    await p.tick(); assert said[-1] == "Heads up, Review starts in a minute."
    assert calls["events"] == 3                       # 10:59 is well past the 9:25 fetch: refetch


async def test_nudge_still_fires_just_after_start():
    ev = "09:30–10:00  Standup (Work)"
    p, said, clock, _ = make(pr.Schedule(nudges_enabled=True, nudge_minutes=5), events=ev)
    clock.t = dt.datetime(2026, 9, 16, 9, 24)
    await p.tick(); assert said == []
    clock.t = dt.datetime(2026, 9, 16, 9, 30, 30)   # start slipped past by < one tick
    await p.tick(); assert said == ["Heads up, Standup starts in a minute."]


async def test_nudge_refetches_after_cache_expiry_and_ignores_past():
    p, said, clock, calls = make(pr.Schedule(nudges_enabled=True), events="09:00–09:30  Old (W)")
    clock.t = dt.datetime(2026, 9, 16, 9, 10)
    await p.tick(); assert said == []              # already started: no nudge
    clock.t = dt.datetime(2026, 9, 16, 9, 16)
    await p.tick(); assert calls["events"] == 2


async def test_start_stop_runs_tick_loop():
    p, said, clock, _ = make(pr.Schedule(briefing_enabled=True, briefing_time="08:00"))
    p.TICK_S = 0.01
    await p.start()
    await asyncio.sleep(0.05)
    p.stop()
    assert len(said) == 1
