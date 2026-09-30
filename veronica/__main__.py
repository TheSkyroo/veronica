import argparse
import asyncio
import logging
import os
import sys
import threading

from veronica import prefs, proactive
from veronica.audio import denoise
from veronica.audio.input_level import InputLevelGuard
from veronica.audio.play import Player, register_for_refresh
from veronica.audio.record import Recorder
from veronica.audio.speaker import SpeakerGate
from veronica.audio.wake import make_wake
from veronica.brain.gate import ToolGate
from veronica.brain.switch import BrainSwitcher
from veronica.config import Settings, settings, setup_logging
from veronica.memory.store import MemoryStore
from veronica.orchestrator import ConfirmResult, Orchestrator
from veronica.speech import voices
from veronica.speech.stt import Transcriber, stt_spec
from veronica.speech.tts import Synthesizer
from veronica.tools import mac as mac_tools, memory_tools, pim
from veronica.tools.timers import TimerService


def build_orchestrator(s: Settings, on_state=None, on_event=None, *, audio: bool = True, on_quit=None,
                       updater_check=None, updater_update=None, relaunch=None, can_relaunch=None,
                       version_describe=None) -> Orchestrator:
    """`updater_check`/`updater_update`/`relaunch`/`can_relaunch` are the
    menu bar app's self-update hooks (see Orchestrator); None (text mode)
    disables the "update yourself" turn. `version_describe` is a cached
    "Veronica x.y.z (sha, date)" for the version turn (default: git, on a
    thread)."""
    holder: dict = {}

    async def confirm(summary: str, detail: str = "") -> ConfirmResult:
        return await holder["orch"].confirm(summary, detail)

    on_level = (lambda v: on_event("mic", v)) if (on_event and audio) else None
    # Routed through the orchestrator rather than straight to on_event so the
    # gate's own cards (auto, trusted, pre-approved) join the turn's plan card
    # alongside the ones confirm() emits; it only ever fires while a brain is
    # running a tool, long after holder["orch"] is set.
    on_tool = (lambda su, d: holder["orch"].tool_card(su, d)) if on_event else None
    # Memory is built for both voice and text mode: text mode still runs
    # local remember/forget intents and logs turns, and the brain still
    # wants facts/recent injected into its system prompt.
    store = MemoryStore(s.memory_path) if s.memory_enabled else None
    # Voice/speed chosen at runtime ("use a british voice", "speak faster")
    # outlive the process via prefs.json; Settings only supplies the default.
    saved = prefs.load()
    saved_voice = saved.get("tts_voice")
    if saved_voice not in voices.VOICE_IDS:
        if saved_voice:
            logging.getLogger("veronica").warning("unknown saved voice %r; using %s", saved_voice, s.kokoro_voice)
        saved_voice = s.kokoro_voice
    try:
        saved_speed = voices.clamp_speed(saved.get("tts_speed", voices.DEFAULT_SPEED))
    except (TypeError, ValueError):
        saved_speed = voices.DEFAULT_SPEED
    saved_hindi_voice = saved.get("tts_hindi_voice")
    if saved_hindi_voice not in voices.HINDI_VOICE_IDS:
        if saved_hindi_voice:
            logging.getLogger("veronica").warning(
                "unknown saved hindi voice %r; using %s", saved_hindi_voice, voices.DEFAULT_HINDI_VOICE
            )
        saved_hindi_voice = voices.DEFAULT_HINDI_VOICE
    # Language mode ("speak hindi" / "switch to english" / "dono bhasha")
    # persists the same way; it decides which whisper models load now, and
    # make_stt is how the orchestrator swaps them on a later switch.
    language = saved.get("language")
    if language not in ("en", "hi", "auto"):
        if language:
            logging.getLogger("veronica").warning("unknown saved language %r; using %s", language, s.language)
        language = s.language
    main_model, stt_language, partial_model = stt_spec(s, language)

    def make_stt(model: str, lang: str | None) -> Transcriber:
        return Transcriber(model, language=lang)

    stt = partial_stt = None
    if audio:
        try:
            stt = make_stt(main_model, stt_language)
            partial_stt = make_stt(partial_model, stt_language) if s.partial_stt else None
        except Exception:
            if language == "en":
                raise
            # The multilingual models are downloaded on first use; offline
            # (or a corrupt cache) must not stop Veronica from starting.
            # Fall back to the English pair for this session only -- the
            # saved pref is left alone so the next online launch restores it.
            logging.getLogger("veronica").exception(
                "multilingual whisper models failed to load; falling back to English for this session"
            )
            language = "en"
            main_model, stt_language, partial_model = stt_spec(s, "en")
            stt = make_stt(main_model, stt_language)
            partial_stt = make_stt(partial_model, stt_language) if s.partial_stt else None

    # Proactive briefings/nudges read the same pim tools the brain uses,
    # just without going through Claude: the ticker gets the tools' text
    # (or a mail count) and composes the announcement itself.
    # A failed fetch (timeout, Automation denied) raises so build_briefing's
    # guarded fetch logs it and drops the sentence, rather than reading the
    # error text as "Nothing on your calendar today."
    def _text_or_raise(res: dict) -> str:
        text = res["content"][0]["text"]
        if res.get("is_error"):
            raise RuntimeError(text)
        return text

    async def _cal(day: str, days: int) -> str:
        return _text_or_raise(await pim.calendar_events.handler({"day": day, "days": days}))

    async def _mail_count() -> int:
        # Mail's own unread count is the real number; the listing is capped
        # at MAIL_LIMIT_MAX, so counting it is only a best-effort fallback.
        try:
            return await pim.mail_unread_count()
        except Exception as exc:
            logging.getLogger("veronica").warning("mail unread count failed, counting the listing: %s", exc)
        res = await pim.mail_unread.handler({"limit": 50})
        return proactive.count_mail(res["content"][0]["text"]) if not res.get("is_error") else 0

    async def _rem(days: int) -> str:
        return _text_or_raise(await pim.reminders_due.handler({"days": days}))

    async def _battery() -> tuple[int | None, str | None]:
        # pmset shells out: off the loop, like the battery quick reply.
        return await asyncio.to_thread(mac_tools.read_battery)

    pro = None
    if audio:
        # holder["orch"] is set right after construction, and announce() is
        # only called from ticks that start in run_forever, so the lambda
        # never runs before the orchestrator exists.
        pro = proactive.Proactive(
            proactive.Schedule.from_prefs(saved.get("proactive", {})),
            announce=lambda t, expires_at=None: holder["orch"].announce(t, expires_at=expires_at),
            calendar_events=_cal, mail_unread_count=_mail_count, reminders_due=_rem, battery=_battery,
        )
    guard = None
    if audio:
        # Reads the floor live so a Settings change applies without a
        # restart; the HUD hint goes through the orchestrator (once per
        # session), which exists by the time the first check runs.
        guard = InputLevelGuard(
            floor=lambda: s.input_volume_floor,
            on_corrected=lambda old, new, name: holder["orch"].input_volume_corrected(old, new, name),
        )
    speaker = None
    if audio:
        # "Only my voice": inert until a profile is enrolled. Both models
        # are fetched (if missing) and loaded in the background so the first
        # capture doesn't wait; until the suppressor lands the mic audio is
        # used as is, and a speaker check that can't load accepts everyone.
        speaker = SpeakerGate(s)

        def prepare_models() -> None:
            denoise.prepare_model(s)
            if speaker.active:
                speaker.prepare()

        threading.Thread(target=prepare_models, name="veronica-models", daemon=True).start()
    player = Player()
    register_for_refresh(player)
    # One confirm gate for every brain; the switcher builds the backends on
    # it lazily (orch.start_brain activates the preferred one) and speaks
    # through the orchestrator when it fails over or can't start.
    gate = ToolGate(s, confirm, on_tool=on_tool, say=lambda t: holder["orch"].say(t))
    switcher = BrainSwitcher(
        s, gate=gate, on_tool=on_tool, memory=store,
        say=lambda t: holder["orch"].say(t),
        on_backend=lambda label, standing_in: holder["orch"].backend_changed(label, standing_in),
    )
    orch = Orchestrator(
        s,
        wake=make_wake(s, verify=speaker.check_wake) if audio else None,
        recorder=Recorder(s, on_level=on_level) if audio else None,
        stt=stt,
        partial_stt=partial_stt,
        brain=switcher.brain,
        switcher=switcher,
        tts=Synthesizer(saved_voice, s.models_dir, speed=saved_speed, hindi_voice=saved_hindi_voice),
        player=player,
        store=store,
        on_state=on_state,
        on_event=on_event,
        on_quit=on_quit,
        proactive=pro,
        input_guard=guard,
        stt_factory=make_stt,
        language=language,
        updater_check=updater_check,
        updater_update=updater_update,
        relaunch=relaunch,
        can_relaunch=can_relaunch,
        version_describe=version_describe,
        speaker=speaker,
    )
    holder["orch"] = orch
    pim.bind(TimerService(on_fire=orch.announce))
    memory_tools.bind(store)
    return orch


def _ask_stdin(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return "n"


def _quit_noop() -> None:
    # --text mode has no running app/menu bar to tear down; just acknowledge.
    print("[quit] Veronica isn't running as a background app in --text mode.")


async def _text_mode(text: str) -> None:
    orch = build_orchestrator(settings, audio=False, on_quit=_quit_noop)
    # no mic in text mode: risky (confirm-class) tools ask y/N on stdin.
    async def confirm(summary: str, detail: str = "", *, question: str | None = None) -> bool:
        answer = await asyncio.to_thread(_ask_stdin, f"{question or Orchestrator.confirm_prompt(summary)} [{detail}] [y/N] ")
        ok = answer.strip().lower() in ("y", "yes")
        print(f"[tool] {summary} -> {'allowed' if ok else 'declined'}")
        return ok
    orch.gate._confirm = confirm
    try:
        print("[text mode] safe tools run automatically; risky tools ask y/N on this terminal")
        # Picks the brain (or says which stand-in it's on) and opens the
        # gate socket the external brains ask on — run_forever's job in
        # the app.
        await orch.start_brain()
        # _brain_turn, not handle_text: a usage limit fails over to the next
        # brain here exactly as it does for a spoken turn.
        for sent in await orch._brain_turn(text):
            print(sent)
    finally:
        await orch.stop_brain()
        await orch.brain.close()
        orch.player.close()
        store = getattr(orch, "store", None)
        if store is not None:
            store.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="veronica")
    p.add_argument("--text", help="ask once via text, no audio input (speaks the reply)")
    args = p.parse_args(argv)
    setup_logging()
    if os.environ.get("ANTHROPIC_API_KEY"):
        os.environ.pop("ANTHROPIC_API_KEY")
        logging.getLogger("veronica").warning(
            "ANTHROPIC_API_KEY ignored — Veronica uses your Claude Code login"
        )
    if args.text:
        asyncio.run(_text_mode(args.text))
        return
    lock = acquire_instance_lock(Settings().home / "veronica.lock")
    if lock is None:
        logging.getLogger("veronica").error("another Veronica is already running; exiting")
        return
    from veronica.ui.menubar import run_app
    run_app()


def acquire_instance_lock(path):
    """Hold an exclusive advisory lock on `path` for the life of the
    process so two Veronicas can't fight over the microphone (the second
    one would hear nothing and the wake word would look broken). Returns
    the open file (keep it referenced) or None when another instance holds
    it."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    f.write(str(os.getpid()))
    f.flush()
    return f


if __name__ == "__main__":
    main(sys.argv[1:])
