# Veronica

macOS voice assistant. Say "Hey Veronica" once you have trained the custom model (see scripts/train_wakeword.md); until then say "Hey Jarvis", ask, listen.
Brain = Claude via your Claude Code login. Speech = local (faster-whisper + Kokoro).

## Setup
    brew install uv portaudio
    uv venv --python 3.12 && uv pip install -e ".[dev]"
    uv run python scripts/download_models.py
    claude auth login        # if not already

## Run
    uv run python -m veronica                 # menu bar app
    uv run python -m veronica --text "hello"  # no audio, debug

## Test
    uv run pytest            # unit
    uv run pytest -m live    # needs mic/speaker/models/login
