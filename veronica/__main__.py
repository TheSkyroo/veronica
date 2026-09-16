import argparse
import asyncio
import logging
import os
import sys

from veronica import prefs, proactive
from veronica.audio.play import Player
from veronica.audio.record import Recorder
from veronica.audio.wake import make_wake
from veronica.brain.agent import Brain
from veronica.config import Settings, settings, setup_logging
from veronica.memory.store import MemoryStore
from veronica.orchestrator import Orchestrator
from veronica.speech import voices
from veronica.speech.stt import Transcriber
from veronica.speech.tts import Synthesizer
from veronica.tools import memory_tools, pim
from veronica.tools.timers import TimerService


def build_orchestrator(s: Settings, on_state=None, on_event=None, *, audio: bool = True, on_quit=None) -> Orchestrator:
    holder: dict = {}

    async def confirm(summary: str, detail: str = "") -> bool:
        return await holder["orch"].confirm(summary, detail)

    on_level = (lambda v: on_event("mic", v)) if (on_event and audio) else None
    on_tool = (lambda su, d: on_event("tool", {"summary": su, "decision": d})) if on_event else None
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

    pro = None
    if audio:
        # holder["orch"] is set right after construction, and announce() is
        # only called from ticks that start in run_forever, so the lambda
        # never runs before the orchestrator exists.
        pro = proactive.Proactive(
            proactive.Schedule.from_prefs(saved.get("proactive", {})),
            announce=lambda t, expires_at=None: holder["orch"].announce(t, expires_at=expires_at),
            calendar_events=_cal, mail_unread_count=_mail_count, reminders_due=_rem,
        )
    orch = Orchestrator(
        s,
        wake=make_wake(s) if audio else None,
        recorder=Recorder(s, on_level=on_level) if audio else None,
        stt=Transcriber(s.whisper_model) if audio else None,
        partial_stt=Transcriber(s.partial_stt_model) if (audio and s.partial_stt) else None,
        brain=Brain(s, confirm=confirm, on_tool=on_tool, memory=store),
        tts=Synthesizer(saved_voice, s.models_dir, speed=saved_speed),
        player=Player(),
        store=store,
        on_state=on_state,
        on_event=on_event,
        on_quit=on_quit,
        proactive=pro,
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
    async def confirm(summary: str, detail: str = "") -> bool:
        answer = await asyncio.to_thread(_ask_stdin, f"Run {summary}? [{detail}] [y/N] ")
        ok = answer.strip().lower() in ("y", "yes")
        print(f"[tool] {summary} -> {'allowed' if ok else 'declined'}")
        return ok
    orch.brain._confirm = confirm
    try:
        print("[text mode] safe tools run automatically; risky tools ask y/N on this terminal")
        for sent in await orch.handle_text(text):
            print(sent)
    finally:
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
    from veronica.ui.menubar import run_app
    run_app()


if __name__ == "__main__":
    main(sys.argv[1:])
