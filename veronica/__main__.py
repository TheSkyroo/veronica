import argparse
import asyncio
import sys

from veronica.audio.play import Player
from veronica.audio.record import Recorder
from veronica.audio.wake import WakeWord
from veronica.brain.agent import Brain
from veronica.config import Settings, settings, setup_logging
from veronica.orchestrator import Orchestrator
from veronica.speech.stt import Transcriber
from veronica.speech.tts import Synthesizer


def build_orchestrator(s: Settings, on_state=None, *, audio: bool = True) -> Orchestrator:
    holder: dict = {}

    async def confirm(summary: str) -> bool:
        return await holder["orch"].confirm(summary)

    orch = Orchestrator(
        s,
        wake=WakeWord(s) if audio else None,
        recorder=Recorder(s) if audio else None,
        stt=Transcriber(s.whisper_model) if audio else None,
        brain=Brain(s, confirm=confirm),
        tts=Synthesizer(s.kokoro_voice, s.models_dir),
        player=Player(),
        on_state=on_state,
    )
    holder["orch"] = orch
    return orch


async def _text_mode(text: str) -> None:
    orch = build_orchestrator(settings, audio=False)
    # no mic in text mode: auto-approve tool calls and print them
    async def confirm(summary: str) -> bool:
        print(f"[tool] {summary} -> allowed")
        return True
    orch.brain._confirm = confirm
    for sent in await orch.handle_text(text):
        print(sent)
    await orch.brain.close()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="veronica")
    p.add_argument("--text", help="ask once via text, no audio input (speaks the reply)")
    args = p.parse_args(argv)
    setup_logging()
    if args.text:
        asyncio.run(_text_mode(args.text))
        return
    from veronica.ui.menubar import run_app
    run_app()


if __name__ == "__main__":
    main(sys.argv[1:])
