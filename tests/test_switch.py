import pytest

from veronica.brain import switch as switch_mod
from veronica.brain.backends import Availability
from veronica.brain.backends.cli import LimitError
from veronica.brain.gate import ToolGate
from veronica.brain.switch import NO_BRAIN_LINE, BrainSwitcher, NoBrain
from veronica.config import Settings


class FakeBrain:
    def __init__(self, name, gate, *, limit=False, reply=("ok.",)):
        self.name, self.gate, self.limit, self.reply = name, gate, limit, list(reply)
        self.closed = False

    async def ask(self, text, images=()):
        if self.limit:
            raise LimitError("usage limit")
        for s in self.reply:
            yield s

    async def interrupt(self):
        pass

    async def close(self):
        self.closed = True


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


HINTS = {
    "not installed": "{label} isn't installed — run install, then login.",
    "not logged in": "{label} isn't logged in — run login in a terminal.",
}
LABELS = {"codex": "Codex", "antigravity": "Antigravity", "claude": "Claude",
          "copilot": "Copilot", "local": "Local"}


def make_switcher(tmp_path, monkeypatch, avail, *, now=None, limits=(), **settings):
    """`avail` maps name -> True | "not installed" | "not logged in" (missing = True).
    Returns (switcher, said, backends, built, saved, clock)."""
    s = Settings(home=tmp_path, **settings)
    gate = ToolGate(s, None)
    said, backends, built, saved = [], [], [], {}
    clock = now or Clock()
    monkeypatch.setattr(switch_mod.prefs, "save_settings_override", lambda k, v: saved.update({k: v}))

    def factory(name, settings, *, gate, on_tool=None, memory=None):
        b = FakeBrain(name, gate, limit=(name in limits))
        built.append(b)
        return b

    def check(name):
        v = avail.get(name, True)
        if v is True:
            return Availability(True, "ok")
        return Availability(False, v, HINTS[v].format(label=LABELS[name]))

    async def say(text):
        said.append(text)

    sw = BrainSwitcher(s, gate=gate, factory=factory, check=check, clock=clock, say=say,
                       on_backend=lambda label, standing_in: backends.append((label, standing_in)))
    return sw, said, backends, built, saved, clock


# -- start ---------------------------------------------------------------------

async def test_init_spawns_nothing_and_start_activates_preferred(tmp_path, monkeypatch):
    sw, said, backends, built, saved, _ = make_switcher(tmp_path, monkeypatch, {})
    assert built == [] and isinstance(sw.brain, NoBrain) and sw.gate is not None
    assert sw.preferred == "codex" and sw.standing_in is False
    await sw.start()
    assert sw.brain.name == "codex" and sw.standing_in is False and said == []
    assert backends == [("Codex", False)] and saved == {}
    assert sw.status_label() == "Codex"


async def test_start_stands_in_when_preferred_unavailable(tmp_path, monkeypatch):
    sw, said, backends, built, saved, _ = make_switcher(
        tmp_path, monkeypatch, {"codex": "not logged in", "antigravity": "not installed"})
    await sw.start()
    assert sw.brain.name == "claude" and sw.standing_in is True and sw.preferred == "codex"
    assert said == ["Codex isn't logged in, so I'm on Claude for now."]
    assert backends == [("Claude (for Codex)", True)] and saved == {}
    assert sw.status_label() == "Claude (for Codex)"


async def test_start_honours_failover_order(tmp_path, monkeypatch):
    sw, said, *_ = make_switcher(tmp_path, monkeypatch, {"codex": "not installed"},
                                 brain_failover_order="copilot,claude")
    await sw.start()
    assert sw.brain.name == "copilot"
    assert said == ["Codex isn't installed, so I'm on Copilot for now."]


async def test_start_with_nothing_available_uses_no_brain(tmp_path, monkeypatch):
    unavailable = {n: "not installed" for n in LABELS}
    sw, said, backends, built, _, _ = make_switcher(tmp_path, monkeypatch, unavailable)
    await sw.start()
    assert isinstance(sw.brain, NoBrain) and sw.brain.name == "none" and built == []
    assert said == [NO_BRAIN_LINE] == ["No brain is ready — log into Codex, Antigravity or Claude."]
    assert [s async for s in sw.brain.ask("hello")] == [NO_BRAIN_LINE]
    await sw.brain.interrupt()
    await sw.brain.close()
    assert sw.brain.gate is sw.gate and sw.status_label() == "No brain"
    assert backends == [("No brain", True)]


async def test_no_brain_proxies_the_gate(tmp_path):
    s = Settings(home=tmp_path)
    g = ToolGate(s, None)
    nb = NoBrain(g)
    nb.begin_turn(3)
    nb.preapprove(3, 99.0)
    nb.clear_trust()
    nb.pending_redirect = "x"
    assert g.pending_redirect == "x" and nb.pending_redirect == "x"
    assert g._current_turn == 3 and g._preapproved_turn == 3


async def test_no_brain_rechecks_and_picks_up_a_login(tmp_path, monkeypatch):
    avail = {n: "not installed" for n in LABELS}
    sw, said, backends, built, _, clock = make_switcher(tmp_path, monkeypatch, avail)
    await sw.start()
    avail["claude"] = True
    await sw.maybe_return()                  # within 60 s: no re-check
    assert isinstance(sw.brain, NoBrain)
    clock.t += 61
    await sw.maybe_return()
    assert sw.brain.name == "claude" and sw.standing_in is True
    assert backends[-1] == ("Claude (for Codex)", True)
    assert said == [NO_BRAIN_LINE]           # silent


# -- manual switch -------------------------------------------------------------

async def test_switch_available_closes_old_and_saves_pref(tmp_path, monkeypatch):
    sw, said, backends, built, saved, _ = make_switcher(tmp_path, monkeypatch, {})
    await sw.start()
    old = sw.brain
    sw.limited_until["claude"] = 5000.0
    a = await sw.switch("claude")
    assert a.ok and sw.brain.name == "claude" and sw.brain is not old and old.closed
    assert sw.preferred == "claude" and sw.s.brain_backend == "claude" and sw.standing_in is False
    assert saved == {"brain_backend": "claude"}
    assert "claude" not in sw.limited_until
    assert backends[-1] == ("Claude", False) and said == []
    assert sw.status_label() == "Claude"


async def test_switch_unavailable_returns_hint_and_keeps_brain(tmp_path, monkeypatch):
    sw, said, backends, built, saved, _ = make_switcher(tmp_path, monkeypatch, {"copilot": "not installed"})
    await sw.start()
    old = sw.brain
    a = await sw.switch("copilot")
    assert a == Availability(False, "not installed", "Copilot isn't installed — run install, then login.")
    assert sw.brain is old and not old.closed and sw.preferred == "codex"
    assert saved == {} and len(backends) == 1 and said == []


async def test_switch_unknown_name(tmp_path, monkeypatch):
    sw, *_ = make_switcher(tmp_path, monkeypatch, {})
    await sw.start()
    a = await sw.switch("qwen")
    assert not a.ok and "qwen" in a.hint and sw.brain.name == "codex"


async def test_manual_switch_back_to_preferred_while_standing_in(tmp_path, monkeypatch):
    sw, said, backends, built, saved, _ = make_switcher(tmp_path, monkeypatch, {}, limits=("codex",))
    await sw.start()
    with pytest.raises(LimitError):
        async for _ in sw.brain.ask("hi"):
            pass
    assert await sw.failover("usage limit") == "antigravity"
    assert sw.standing_in and sw.limited_until["codex"] > 0
    a = await sw.switch("codex")
    assert a.ok and sw.brain.name == "codex" and sw.standing_in is False
    assert sw.limited_until == {} and saved == {"brain_backend": "codex"}


async def test_switch_without_manual_does_not_change_preference(tmp_path, monkeypatch):
    sw, said, backends, built, saved, _ = make_switcher(tmp_path, monkeypatch, {})
    await sw.start()
    a = await sw.switch("claude", manual=False)
    assert a.ok and sw.brain.name == "claude" and sw.preferred == "codex" and sw.standing_in is True
    assert saved == {} and sw.s.brain_backend == "codex"
    assert backends[-1] == ("Claude (for Codex)", True)


# -- failover ------------------------------------------------------------------

async def test_failover_picks_next_in_order_and_cools_down(tmp_path, monkeypatch):
    sw, said, backends, built, saved, clock = make_switcher(tmp_path, monkeypatch, {}, brain_limit_cooldown_min=30)
    await sw.start()
    old = sw.brain
    new = await sw.failover("usage limit")
    assert new == "antigravity" and sw.brain.name == "antigravity" and old.closed
    assert sw.limited_until["codex"] == clock.t + 30 * 60
    assert said == ["Codex hit its usage limit — switching to Antigravity."]
    assert sw.standing_in is True and sw.preferred == "codex" and saved == {}
    assert sw.status_label() == "Antigravity (for Codex)"
    assert backends[-1] == ("Antigravity (for Codex)", True)


async def test_failover_skips_unavailable_and_cooled_down(tmp_path, monkeypatch):
    sw, said, *_ , clock = make_switcher(tmp_path, monkeypatch, {"antigravity": "not logged in"})
    await sw.start()
    sw.limited_until["claude"] = clock.t + 10
    assert await sw.failover("usage limit") == "copilot"
    assert said == ["Codex hit its usage limit — switching to Copilot."]


async def test_failover_uses_custom_order(tmp_path, monkeypatch):
    sw, said, *_ = make_switcher(tmp_path, monkeypatch, {}, brain_failover_order="claude, copilot, bogus")
    await sw.start()
    assert await sw.failover("usage limit") == "claude"
    # the rest of BACKENDS trail the configured order
    assert await sw.failover("usage limit") == "copilot"
    assert await sw.failover("usage limit") == "antigravity"


async def test_second_failover_never_returns_to_a_cooled_brain(tmp_path, monkeypatch):
    sw, said, *_ , clock = make_switcher(
        tmp_path, monkeypatch, {"copilot": "not installed", "local": "not installed"})
    await sw.start()
    assert await sw.failover("usage limit") == "antigravity"
    assert await sw.failover("usage limit") == "claude"
    assert sw.limited_until["antigravity"] == clock.t + 60 * 60
    assert said[-1] == "Antigravity hit its usage limit — switching to Claude."
    assert sw.status_label() == "Claude (for Codex)"
    assert await sw.failover("usage limit") is None
    assert said[-1] == "Claude hit its usage limit and no other brain is ready."
    assert sw.brain.name == "claude" and sw.limited_until["claude"] == clock.t + 60 * 60


async def test_failover_none_available_says_so(tmp_path, monkeypatch):
    sw, said, backends, built, _, _ = make_switcher(
        tmp_path, monkeypatch, {n: "not installed" for n in LABELS if n != "codex"})
    await sw.start()
    old = sw.brain
    assert await sw.failover("usage limit") is None
    assert said == ["Codex hit its usage limit and no other brain is ready."]
    assert sw.brain is old and not old.closed and len(backends) == 1


async def test_failover_disabled_returns_none_without_switching(tmp_path, monkeypatch):
    sw, said, backends, built, _, _ = make_switcher(tmp_path, monkeypatch, {}, brain_failover=False)
    await sw.start()
    old = sw.brain
    assert await sw.failover("usage limit") is None
    assert sw.brain is old and not old.closed and sw.limited_until == {} and len(backends) == 1
    assert said == ["Codex hit its usage limit."]


async def test_failover_without_say(tmp_path, monkeypatch):
    sw, *_ = make_switcher(tmp_path, monkeypatch, {})
    sw._say = None
    await sw.start()
    assert await sw.failover("usage limit") == "antigravity"


# -- return --------------------------------------------------------------------

async def test_maybe_return_waits_for_cooldown_then_returns_silently(tmp_path, monkeypatch):
    sw, said, backends, built, saved, clock = make_switcher(tmp_path, monkeypatch, {})
    await sw.start()
    await sw.failover("usage limit")
    said.clear()
    standin = sw.brain
    await sw.maybe_return()
    assert sw.brain is standin and sw.standing_in
    clock.t = sw.limited_until["codex"] - 1
    await sw.maybe_return()
    assert sw.brain is standin
    clock.t = sw.limited_until["codex"]
    await sw.maybe_return()
    assert sw.brain.name == "codex" and sw.brain is not standin and standin.closed
    assert sw.standing_in is False and said == [] and saved == {}
    assert backends[-1] == ("Codex", False) and sw.status_label() == "Codex"


async def test_maybe_return_does_nothing_when_not_standing_in(tmp_path, monkeypatch):
    sw, said, backends, built, *_ = make_switcher(tmp_path, monkeypatch, {})
    await sw.start()
    await sw.maybe_return()
    assert len(built) == 1 and len(backends) == 1


async def test_maybe_return_after_cooldown_keeps_standin_if_preferred_now_unavailable(tmp_path, monkeypatch):
    avail = {}
    sw, said, backends, built, _, clock = make_switcher(tmp_path, monkeypatch, avail)
    await sw.start()
    await sw.failover("usage limit")
    standin = sw.brain
    avail["codex"] = "not logged in"
    clock.t = sw.limited_until["codex"] + 1
    await sw.maybe_return()
    assert sw.brain is standin and sw.standing_in
    avail["codex"] = True
    await sw.maybe_return()                  # re-check throttled to once a minute
    assert sw.brain is standin
    clock.t += 61
    await sw.maybe_return()
    assert sw.brain.name == "codex" and sw.standing_in is False


async def test_maybe_return_rechecks_availability_every_minute(tmp_path, monkeypatch):
    avail = {"codex": "not logged in"}
    checks = []
    sw, said, backends, built, _, clock = make_switcher(tmp_path, monkeypatch, avail)
    inner = sw._check
    sw._check = lambda n: (checks.append(n), inner(n))[1]
    await sw.start()
    assert sw.brain.name == "antigravity"
    checks.clear()
    await sw.maybe_return()
    assert checks == []                      # too soon
    clock.t += 60
    await sw.maybe_return()
    assert checks == ["codex"] and sw.brain.name == "antigravity"
    avail["codex"] = True
    checks.clear()
    await sw.maybe_return()
    assert checks == []
    clock.t += 60
    standin = sw.brain
    await sw.maybe_return()
    assert sw.brain.name == "codex" and standin.closed and sw.standing_in is False
    assert said == ["Codex isn't logged in, so I'm on Antigravity for now."]
    assert backends[-1] == ("Codex", False)


async def test_clock_is_reachable_for_which_brain_wording(tmp_path, monkeypatch):
    sw, *_, clock = make_switcher(tmp_path, monkeypatch, {})
    assert sw._clock() == clock.t


# -- offline (F1) ---------------------------------------------------------------

def wire_up(sw, wire):
    """Point the switcher's reachability probe at a dict the test flips."""
    async def probe(name):
        return wire["up"]

    sw._is_online = probe


async def test_no_internet_stands_the_local_model_in(tmp_path, monkeypatch):
    sw, said, backends, _, _, _ = make_switcher(tmp_path, monkeypatch, {})
    wire = {"up": True}
    wire_up(sw, wire)
    await sw.start()
    await sw.maybe_offline()
    assert sw.brain.name == "codex" and said == []
    wire["up"] = False
    await sw.maybe_offline()
    assert sw.brain.name == "local" and sw.standing_in is True and sw.preferred == "codex"
    assert said == [switch_mod.OFFLINE_LINE]
    assert backends[-1] == ("Local (for Codex)", True)


async def test_the_local_model_hands_back_silently_when_the_wire_returns(tmp_path, monkeypatch):
    sw, said, _, _, saved, _ = make_switcher(tmp_path, monkeypatch, {})
    wire = {"up": False}
    wire_up(sw, wire)
    await sw.start()
    await sw.maybe_offline()
    assert sw.brain.name == "local"
    await sw.maybe_return()                  # still offline: stay put
    assert sw.brain.name == "local"
    wire["up"] = True
    await sw.maybe_return()
    assert sw.brain.name == "codex" and sw.standing_in is False
    assert said == [switch_mod.OFFLINE_LINE] and saved == {}


async def test_offline_fallback_can_be_turned_off(tmp_path, monkeypatch):
    sw, said, *_ = make_switcher(tmp_path, monkeypatch, {}, brain_offline_fallback=False)
    wire_up(sw, {"up": False})
    await sw.start()
    await sw.maybe_offline()
    assert sw.brain.name == "codex" and said == []


async def test_offline_without_a_local_model_keeps_the_current_brain(tmp_path, monkeypatch):
    sw, said, *_ = make_switcher(tmp_path, monkeypatch, {"local": "not installed"})
    wire_up(sw, {"up": False})
    await sw.start()
    await sw.maybe_offline()
    assert sw.brain.name == "codex" and said == []


async def test_already_local_never_re_announces(tmp_path, monkeypatch):
    sw, said, *_ = make_switcher(tmp_path, monkeypatch, {})
    wire_up(sw, {"up": False})
    await sw.start()
    await sw.maybe_offline()
    await sw.maybe_offline()
    assert said == [switch_mod.OFFLINE_LINE]


async def test_a_manual_local_preference_is_left_alone(tmp_path, monkeypatch):
    sw, said, *_ = make_switcher(tmp_path, monkeypatch, {}, brain_backend="local")
    wire_up(sw, {"up": False})
    await sw.start()
    await sw.maybe_offline()
    assert sw.brain.name == "local" and sw.standing_in is False and said == []


async def test_a_start_stand_in_never_drops_onto_the_local_model(tmp_path, monkeypatch):
    """The local model is for a dead wire only. A vendor brain that isn't
    logged in yet must not quietly put the 3B model in its place."""
    sw, said, *_ = make_switcher(
        tmp_path, monkeypatch, {n: "not logged in" for n in LABELS if n != "local"})
    await sw.start()
    assert sw.brain.name == "none" and said == [NO_BRAIN_LINE]


async def test_failover_never_lands_on_the_local_model(tmp_path, monkeypatch):
    """A usage limit on a fully online session is not a reason to go local."""
    sw, said, *_ = make_switcher(
        tmp_path, monkeypatch,
        {n: "not installed" for n in ("antigravity", "claude", "copilot")},
        limits=("codex",))
    await sw.start()
    assert await sw.failover("usage limit") is None
    assert sw.brain.name == "codex"
    assert said == ["Codex hit its usage limit and no other brain is ready."]


async def test_local_is_still_reachable_by_asking_for_it(tmp_path, monkeypatch):
    sw, *_ = make_switcher(tmp_path, monkeypatch, {})
    assert (await sw.switch("local")).ok
    assert sw.brain.name == "local" and sw.preferred == "local"


async def test_online_candidate_skips_the_local_model(tmp_path, monkeypatch):
    sw, *_ = make_switcher(tmp_path, monkeypatch, {"codex": "not installed"})
    assert sw.online_candidate() == "antigravity"
    sw2, *_ = make_switcher(tmp_path, monkeypatch,
                            {n: "not installed" for n in LABELS if n != "local"})
    assert sw2.online_candidate() is None
