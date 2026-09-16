# Veronica

macOS voice assistant. Say "Hey Veronica" once you have trained the custom model (see scripts/train_wakeword.md); until then say "Hey Jarvis", ask, listen.
Brain = Claude via your Claude Code login. Speech = local (faster-whisper + Kokoro).

## Setup
    brew install uv portaudio
    uv venv --python 3.12 && uv pip install -e ".[dev]"
    uv run python scripts/download_models.py
    claude auth login        # if not already

Wake word: say "Veronica" or "hey Veronica" (whisper engine, default). To use the lighter openwakeword engine set VERONICA_WAKE_ENGINE=openwakeword (falls back to "hey jarvis" until you train a custom model — see scripts/train_wakeword.md).

Wake word not triggering? Run `uv run python scripts/wake_scores.py`, say the phrase, and (openwakeword engine) set VERONICA_WAKE_THRESHOLD in .env just below the scores you see.

Do not set `ANTHROPIC_API_KEY` — Veronica uses your Claude Code login (it is ignored if set).

- macOS will ask for Microphone access for your terminal app on first run (System Settings → Privacy & Security → Microphone).
- First run downloads the whisper `small.en` model (~470 MB).
- Say the wake word while Veronica is talking to interrupt her (barge-in).
- Risky actions (writing files, shell commands that change things, AppleScript, clipboard writes) ask "Run …?" — answer "yes" or "no".
- A floating HUD appears at the top-right when Veronica wakes (orb + transcript + tool activity) and fades after 3 s of idle. Disable with VERONICA_HUD_ENABLED=false.

## Using Veronica

Say "Veronica …" in one breath — e.g. "Veronica, what time is it" — rather than pausing after the wake word. She
buffers the tail of the wake-word audio and hands it straight to the recorder, so a command spoken in the same
breath as the wake word isn't lost and the wake chime is skipped when she can already hear you talking.

The HUD's status line under the orb shows what she's doing:

- **Warming up…** — models are loading (first run only).
- **Listening…** — she's capturing your voice (also shown while a follow-up or a confirmation reply is expected);
  the bar next to it tracks the live mic level.
- **Thinking…** — Claude is working on a reply.
- **Speaking** — she's talking.
- **Say yes or no** — she's asked for confirmation before a risky action.
- **Error** — something went wrong; check the log.

While she's listening, the HUD also shows a live partial transcript of what you're saying (in italics), which is
replaced by the final transcript once you finish talking. This costs a bit of CPU; disable it with
`VERONICA_PARTIAL_STT=false` in `.env` to save power on slower Macs.

She stops listening rather than eavesdropping indefinitely: the follow-up window after a reply is short (4 s by
default — set `VERONICA_FOLLOWUP_WINDOW_S` in `.env` to change it), and you can end the conversation immediately by
saying "thanks Veronica" / "thank you Veronica" (she replies "Okay.") or "that's all", "stop", "goodbye", "never
mind" (she just goes quiet, no follow-up window).

## Run
    uv run python -m veronica                 # menu bar app
    uv run python -m veronica --text "hello"  # no audio, debug

## Test
    uv run pytest            # unit
    uv run pytest -m live    # needs mic/speaker/models/login
    uv run playwright install chromium   # once, for the live HUD tests
