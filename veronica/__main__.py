import argparse
import asyncio
import logging
import os
import sys

from veronica.audio.play import Player
from veronica.audio.record import Recorder
from veronica.audio.wake import WakeWord
from veronica.brain.agent import Brain
from veronica.config import Settings, settings, setup_logging
from veronica.orchestrator import Orchestrator
from veronica.speech.stt import Transcriber
from veronica.speech.tts import Synthesizer


def build_orchestrator(s: Settings, on_state=None, on_event=None, *, audio: bool = True) -> Orchestrator:
    holder: dict = {}

    async def confirm(summary: str) -> bool:
        return await holder["orch"].confirm(summary)

    on_level = (lambda v: on_event("mic", v)) if (on_event and audio) else None
    on_tool = (lambda su, d: on_event("tool", {"summary": su, "decision": d})) if on_event else None
    orch = Orchestrator(
        s,
        wake=WakeWord(s) if audio else None,
        recorder=Recorder(s, on_level=on_level) if audio else None,
        stt=Transcriber(s.whisper_model) if audio else None,
        brain=Brain(s, confirm=confirm, on_tool=on_tool),
        tts=Synthesizer(s.kokoro_voice, s.models_dir),
        player=Player(),
        on_state=on_state,
        on_event=on_event,
    )
    holder["orch"] = orch
    return orch


def _ask_stdin(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return "n"


async def _text_mode(text: str) -> None:
    orch = build_orchestrator(settings, audio=False)
    # no mic in text mode: risky (confirm-class) tools ask y/N on stdin.
    async def confirm(summary: str) -> bool:
        answer = await asyncio.to_thread(_ask_stdin, f"Run {summary}? [y/N] ")
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
