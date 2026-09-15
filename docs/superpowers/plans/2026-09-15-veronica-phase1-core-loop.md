# Veronica Phase 1 — Core Voice Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wake word → record → local STT → Claude (via Claude Code subscription) → local TTS, in a macOS menu-bar app, with a `--text` debug mode.

**Architecture:** One Python process. `rumps` owns the main thread (AppKit); an `asyncio` loop runs in a background thread and drives a state machine (idle → listening → thinking → speaking → follow-up). Each stage is a small module with one function-sized interface so it can be swapped or tested alone. Brain is `claude-agent-sdk` `ClaudeSDKClient` with a `can_use_tool` gate; Phase 1 gates *every* built-in tool behind a spoken confirmation.

**Tech Stack:** Python 3.12, `uv`, `claude-agent-sdk`, `openwakeword`, `webrtcvad-wheels`, `sounddevice`, `numpy`, `faster-whisper`, `kokoro-onnx`, `soundfile`, `rumps`, `pydantic-settings`, `pytest`, `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-09-15-veronica-voice-agent-design.md`

## Global Constraints

- macOS only, Apple Silicon target. Python ≥ 3.12.
- No Anthropic API key. Brain auth = Claude Code CLI login (`claude` on PATH, logged in). `ANTHROPIC_API_KEY` must NOT be set in the env Veronica runs in (it would override the login).
- Speech is local: `faster-whisper` `base.en` (int8), Kokoro ONNX. Audio is 16 kHz mono int16 for capture; Kokoro outputs 24 kHz float32.
- System prompt (verbatim from spec): "You are Veronica, a voice assistant running on Mani's Mac. Reply in one to three spoken sentences. No markdown, no lists, no code unless asked. For long answers, give the short version and offer to say more. Today is {date}."
- Agent options: `effort="low"`, `max_turns=8`, `permission_mode="default"`, `can_use_tool` set. Session id persisted at `~/.veronica/session`.
- Turn-taking numbers: VAD silence end 700 ms, max utterance 15 s, min speech 300 ms, follow-up window 8 s, confirm listen 5 s, brain timeout 60 s.
- Confirm words: `yes`, `yeah`, `yep`, `do it`, `go`, `go ahead`, `confirm`, `sure`. Anything else = deny with message `user declined`.
- Deviations from spec: `webrtcvad` replaces silero VAD (avoids torch). Wake word uses openwakeword pretrained `hey_jarvis` until Task 12 trains `hey_veronica`.
- Logs: `~/.veronica/logs/veronica.log`, `RotatingFileHandler(maxBytes=5_000_000, backupCount=5)`.
- Commit after every task. No `Co-Authored-By` trailers.

---

## File structure

```
pyproject.toml
README.md
scripts/download_models.py        # Kokoro model + voices, openwakeword models
veronica/
  __init__.py
  __main__.py                     # `python -m veronica [--text "..."]`
  config.py                       # Settings (paths, thresholds), logging setup
  audio/__init__.py
  audio/play.py                   # Player: queue of float32 chunks, stop()
  audio/record.py                 # Recorder: VAD-endpointed capture
  audio/wake.py                   # WakeWord: blocks until detection
  speech/__init__.py
  speech/stt.py                   # Transcriber.transcribe(pcm16) -> str
  speech/tts.py                   # Synthesizer.synth(text) -> (np.float32, sr)
  brain/__init__.py
  brain/sentences.py              # SentenceSplitter (stream text → sentences)
  brain/prompts.py                # system_prompt(date) -> str
  brain/agent.py                  # Brain.ask(text) -> AsyncIterator[str]
  orchestrator.py                 # Orchestrator state machine
  ui/__init__.py
  ui/menubar.py                   # rumps app + background asyncio thread
tests/
  conftest.py
  test_sentences.py
  test_play.py
  test_tts.py
  test_stt.py
  test_record.py
  test_wake.py
  test_agent.py
  test_orchestrator.py
  fixtures/speech_1s.wav          # generated in Task 5
```

---

### Task 1: Project scaffold + config

**Files:**
- Create: `pyproject.toml`, `veronica/__init__.py`, `veronica/config.py`, `tests/conftest.py`, `tests/test_config.py`, `README.md`

**Interfaces:**
- Produces: `veronica.config.Settings` (pydantic-settings) with fields below; `veronica.config.setup_logging() -> logging.Logger`; `veronica.config.settings` module-level instance.

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "veronica"
version = "0.1.0"
description = "Voice assistant for macOS powered by Claude Code"
requires-python = ">=3.12"
dependencies = [
  "claude-agent-sdk>=0.1.0",
  "openwakeword>=0.6.0",
  "webrtcvad-wheels>=2.0.14",
  "sounddevice>=0.5.1",
  "soundfile>=0.12.1",
  "numpy>=1.26",
  "faster-whisper>=1.1.0",
  "kokoro-onnx>=0.4.0",
  "rumps>=0.4.0",
  "pydantic-settings>=2.6",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-asyncio>=0.24", "ruff>=0.6"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
markers = ["live: needs real audio devices, models, or Claude Code login"]
addopts = "-m 'not live'"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["veronica"]
```

- [ ] **Step 2: Create venv and install**

Run:
```bash
cd /Users/manikchandsahu/Github/veronica
uv venv --python 3.12 && uv pip install -e ".[dev]"
```
Expected: installs without error. (If `uv` missing: `brew install uv`.)

- [ ] **Step 3: Write failing config test**

`tests/conftest.py`:
```python
import pytest


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VERONICA_HOME", str(tmp_path))
    return tmp_path
```

`tests/test_config.py`:
```python
from veronica.config import Settings


def test_defaults(tmp_home):
    s = Settings()
    assert s.home == tmp_home
    assert s.sample_rate == 16000
    assert s.vad_silence_ms == 700
    assert s.max_utterance_s == 15
    assert s.min_speech_ms == 300
    assert s.followup_window_s == 8
    assert s.confirm_listen_s == 5
    assert s.brain_timeout_s == 60
    assert s.wake_model == "hey_jarvis"
    assert s.session_file == tmp_home / "session"
    assert s.log_file == tmp_home / "logs" / "veronica.log"


def test_dirs_created(tmp_home):
    s = Settings()
    s.ensure_dirs()
    assert (tmp_home / "logs").is_dir()
    assert (tmp_home / "models").is_dir()
```

- [ ] **Step 4: Run test, verify fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL `ModuleNotFoundError: No module named 'veronica.config'`

- [ ] **Step 5: Implement config**

`veronica/__init__.py`: empty file.

`veronica/config.py`:
```python
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VERONICA_", env_file=".env", extra="ignore")

    home: Path = Field(default_factory=lambda: Path.home() / ".veronica")

    # audio
    sample_rate: int = 16000
    frame_ms: int = 30                 # webrtcvad frame size
    vad_aggressiveness: int = 2        # 0-3
    vad_silence_ms: int = 700
    max_utterance_s: int = 15
    min_speech_ms: int = 300
    followup_window_s: int = 8
    confirm_listen_s: int = 5

    # wake word
    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.5

    # speech
    whisper_model: str = "base.en"
    kokoro_voice: str = "af_sarah"

    # brain
    brain_timeout_s: int = 60
    effort: str = "low"
    max_turns: int = 8

    @property
    def session_file(self) -> Path:
        return self.home / "session"

    @property
    def log_file(self) -> Path:
        return self.home / "logs" / "veronica.log"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    def ensure_dirs(self) -> None:
        (self.home / "logs").mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    settings.ensure_dirs()
    log = logging.getLogger("veronica")
    if log.handlers:
        return log
    log.setLevel(level)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(settings.log_file, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log
```

- [ ] **Step 6: Run test, verify passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: 2 PASS

- [ ] **Step 7: README stub**

`README.md`:
```markdown
# Veronica

macOS voice assistant. Say "Hey Veronica" (Phase 1: "Hey Jarvis"), ask, listen.
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
```

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml README.md veronica tests && git commit -m "feat: project scaffold and settings"
```

---

### Task 2: Sentence splitter

**Files:**
- Create: `veronica/brain/__init__.py`, `veronica/brain/sentences.py`, `tests/test_sentences.py`

**Interfaces:**
- Produces: `SentenceSplitter` with `feed(text: str) -> list[str]` (returns complete sentences found so far) and `flush() -> list[str]` (returns remaining tail as one sentence if non-empty).

- [ ] **Step 1: Write failing tests**

`tests/test_sentences.py`:
```python
from veronica.brain.sentences import SentenceSplitter


def test_splits_on_terminal_punctuation():
    s = SentenceSplitter()
    assert s.feed("Hello there. How are") == ["Hello there."]
    assert s.feed(" you? Fine!") == ["How are you?", "Fine!"]


def test_holds_incomplete_tail():
    s = SentenceSplitter()
    assert s.feed("It is 3 p") == []
    assert s.feed(".m. now") == []          # "3 p.m." not split: no space after period
    assert s.flush() == ["It is 3 p.m. now"]


def test_flush_empty():
    s = SentenceSplitter()
    assert s.flush() == []


def test_strips_markdown_noise():
    s = SentenceSplitter()
    assert s.feed("**Bold** and `code`. ") == ["Bold and code."]


def test_does_not_split_decimal():
    s = SentenceSplitter()
    assert s.feed("Pi is 3.14 roughly. Ok.") == ["Pi is 3.14 roughly.", "Ok."]
```

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_sentences.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`veronica/brain/__init__.py`: empty.

`veronica/brain/sentences.py`:
```python
import re

# sentence ends at . ! ? followed by whitespace (so "3.14" and "p.m." don't split)
_END = re.compile(r"(?<=[.!?])\s+")
_MD = re.compile(r"[*_`#]+")


class SentenceSplitter:
    """Accumulates streamed text and emits complete sentences."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> list[str]:
        self._buf += _MD.sub("", text)
        parts = _END.split(self._buf)
        if len(parts) <= 1:
            return []
        self._buf = parts[-1]
        return [p.strip() for p in parts[:-1] if p.strip()]

    def flush(self) -> list[str]:
        tail = self._buf.strip()
        self._buf = ""
        return [tail] if tail else []
```

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest tests/test_sentences.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add veronica/brain tests/test_sentences.py && git commit -m "feat: streaming sentence splitter"
```

---

### Task 3: Audio player

**Files:**
- Create: `veronica/audio/__init__.py`, `veronica/audio/play.py`, `tests/test_play.py`

**Interfaces:**
- Produces: `Player(sample_rate: int = 24000)` with `async play(samples: np.ndarray) -> None` (float32 mono, blocks until done or stopped), `stop() -> None` (cancels current + drains queue), `is_playing: bool`. Uses `sounddevice.play`/`sounddevice.wait` via `asyncio.to_thread` so tests can monkeypatch `sounddevice`.

- [ ] **Step 1: Write failing tests**

`tests/test_play.py`:
```python
import asyncio
import numpy as np
import pytest

from veronica.audio import play as play_mod


class FakeSD:
    def __init__(self):
        self.played = []
        self.stopped = 0

    def play(self, samples, samplerate):
        self.played.append((len(samples), samplerate))

    def wait(self):
        pass

    def stop(self):
        self.stopped += 1


@pytest.fixture
def fake_sd(monkeypatch):
    sd = FakeSD()
    monkeypatch.setattr(play_mod, "sd", sd)
    return sd


async def test_play_sends_samples(fake_sd):
    p = play_mod.Player(sample_rate=24000)
    await p.play(np.zeros(2400, dtype=np.float32))
    assert fake_sd.played == [(2400, 24000)]
    assert p.is_playing is False


async def test_stop_calls_sd_stop(fake_sd):
    p = play_mod.Player()
    p.stop()
    assert fake_sd.stopped == 1


async def test_stop_skips_queued(fake_sd):
    p = play_mod.Player()
    p.stop()  # sets stopped flag before play
    await p.play(np.zeros(10, dtype=np.float32))
    assert fake_sd.played == []
```

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_play.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`veronica/audio/__init__.py`: empty.

`veronica/audio/play.py`:
```python
import asyncio

import numpy as np
import sounddevice as sd


class Player:
    """Plays float32 mono chunks sequentially; stop() cancels playback."""

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.is_playing = False
        self._stopped = False

    def reset(self) -> None:
        """Clear a previous stop() so new playback is accepted."""
        self._stopped = False

    async def play(self, samples: np.ndarray) -> None:
        if self._stopped:
            return
        self.is_playing = True
        try:
            await asyncio.to_thread(self._blocking_play, samples)
        finally:
            self.is_playing = False

    def _blocking_play(self, samples: np.ndarray) -> None:
        sd.play(samples, samplerate=self.sample_rate)
        sd.wait()

    def stop(self) -> None:
        self._stopped = True
        sd.stop()
```

Note: the test for `stop_skips_queued` relies on `_stopped` staying set until `reset()`; the orchestrator calls `reset()` at the start of every speaking turn.

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest tests/test_play.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add veronica/audio tests/test_play.py && git commit -m "feat: cancellable audio player"
```

---

### Task 4: Model download script + TTS adapter

**Files:**
- Create: `scripts/download_models.py`, `veronica/speech/__init__.py`, `veronica/speech/tts.py`, `tests/test_tts.py`

**Interfaces:**
- Produces: `Synthesizer(voice: str, models_dir: Path)` with `synth(text: str) -> tuple[np.ndarray, int]` (float32 samples, sample rate) and `async asynth(text) -> tuple[np.ndarray, int]` (runs `synth` in a thread). Class attribute `_kokoro_cls` for test injection.

- [ ] **Step 1: Download script**

`scripts/download_models.py`:
```python
"""Fetch Kokoro TTS model/voices and openwakeword base models into ~/.veronica/models."""
import urllib.request

from veronica.config import settings

KOKORO = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}


def main() -> None:
    settings.ensure_dirs()
    for name, url in KOKORO.items():
        dest = settings.models_dir / name
        if dest.exists():
            print(f"ok      {dest}")
            continue
        print(f"fetch   {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"saved   {dest}")

    import openwakeword
    openwakeword.utils.download_models()
    print("openwakeword models ready")


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/download_models.py`
Expected: two files in `~/.veronica/models`, openwakeword prints download progress.

- [ ] **Step 2: Write failing tests**

`tests/test_tts.py`:
```python
import numpy as np
import pytest

from veronica.speech.tts import Synthesizer


class FakeKokoro:
    def __init__(self, model_path, voices_path):
        self.model_path = model_path
        self.calls = []

    def create(self, text, voice, speed, lang):
        self.calls.append((text, voice))
        return np.zeros(240, dtype=np.float32), 24000


def test_synth_returns_samples(tmp_path):
    Synthesizer._kokoro_cls = FakeKokoro
    s = Synthesizer(voice="af_sarah", models_dir=tmp_path)
    samples, sr = s.synth("hello")
    assert sr == 24000
    assert samples.dtype == np.float32
    assert s._engine.calls == [("hello", "af_sarah")]


async def test_asynth(tmp_path):
    Synthesizer._kokoro_cls = FakeKokoro
    s = Synthesizer(voice="af_sarah", models_dir=tmp_path)
    samples, sr = await s.asynth("hi")
    assert len(samples) == 240


@pytest.mark.live
def test_real_kokoro_speaks():
    from veronica.config import settings
    s = Synthesizer(voice=settings.kokoro_voice, models_dir=settings.models_dir)
    samples, sr = s.synth("Hello, I am Veronica.")
    assert sr == 24000 and len(samples) > sr  # > 1 s of audio
```

- [ ] **Step 3: Run, verify fails**

Run: `uv run pytest tests/test_tts.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 4: Implement**

`veronica/speech/__init__.py`: empty.

`veronica/speech/tts.py`:
```python
import asyncio
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro


class Synthesizer:
    """Kokoro ONNX text-to-speech. Output: float32 mono at 24 kHz."""

    _kokoro_cls = Kokoro  # swapped in tests

    def __init__(self, voice: str, models_dir: Path) -> None:
        self.voice = voice
        self._engine = self._kokoro_cls(
            str(models_dir / "kokoro-v1.0.onnx"),
            str(models_dir / "voices-v1.0.bin"),
        )

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        samples, sr = self._engine.create(text, voice=self.voice, speed=1.0, lang="en-us")
        return np.asarray(samples, dtype=np.float32), sr

    async def asynth(self, text: str) -> tuple[np.ndarray, int]:
        return await asyncio.to_thread(self.synth, text)
```

- [ ] **Step 5: Run, verify passes**

Run: `uv run pytest tests/test_tts.py -v` → 2 PASS (live skipped).
Run: `uv run pytest tests/test_tts.py -m live -v` → 1 PASS (needs models).

- [ ] **Step 6: Commit**

```bash
git add scripts veronica/speech tests/test_tts.py && git commit -m "feat: Kokoro TTS adapter and model download script"
```

---

### Task 5: STT adapter + speech fixture

**Files:**
- Create: `veronica/speech/stt.py`, `tests/test_stt.py`, `tests/fixtures/speech_1s.wav`, `scripts/make_fixture.py`

**Interfaces:**
- Produces: `Transcriber(model_name: str)` with `transcribe(pcm16: np.ndarray) -> str` (int16 mono 16 kHz in, stripped text out; `""` when nothing) and `async atranscribe(pcm16) -> str`. Class attribute `_model_cls` for test injection.

- [ ] **Step 1: Generate fixture**

`scripts/make_fixture.py` (uses the real Kokoro to produce a known-utterance WAV, resampled to 16 kHz int16):
```python
import numpy as np
import soundfile as sf

from veronica.config import settings
from veronica.speech.tts import Synthesizer

s = Synthesizer(voice=settings.kokoro_voice, models_dir=settings.models_dir)
samples, sr = s.synth("What time is it")
# naive resample 24k -> 16k
idx = np.arange(0, len(samples), sr / 16000)
pcm = np.interp(idx, np.arange(len(samples)), samples)
sf.write("tests/fixtures/speech_1s.wav", (pcm * 32767).astype(np.int16), 16000)
print("wrote tests/fixtures/speech_1s.wav")
```

Run: `mkdir -p tests/fixtures && uv run python scripts/make_fixture.py`
Expected: file exists, ~1–2 s.

- [ ] **Step 2: Write failing tests**

`tests/test_stt.py`:
```python
import numpy as np
import pytest
import soundfile as sf

from veronica.speech.stt import Transcriber


class FakeSeg:
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self, name, device, compute_type):
        self.name = name

    def transcribe(self, audio, beam_size, language, vad_filter):
        return iter([FakeSeg(" hello "), FakeSeg("world")]), None


def test_transcribe_joins_segments():
    Transcriber._model_cls = FakeModel
    t = Transcriber("base.en")
    assert t.transcribe(np.zeros(16000, dtype=np.int16)) == "hello world"


async def test_atranscribe():
    Transcriber._model_cls = FakeModel
    t = Transcriber("base.en")
    assert await t.atranscribe(np.zeros(16000, dtype=np.int16)) == "hello world"


@pytest.mark.live
def test_real_whisper_on_fixture():
    from faster_whisper import WhisperModel
    Transcriber._model_cls = WhisperModel
    pcm, sr = sf.read("tests/fixtures/speech_1s.wav", dtype="int16")
    text = Transcriber("base.en").transcribe(pcm).lower()
    assert "time" in text
```

- [ ] **Step 3: Run, verify fails**

Run: `uv run pytest tests/test_stt.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 4: Implement**

`veronica/speech/stt.py`:
```python
import asyncio

import numpy as np
from faster_whisper import WhisperModel


class Transcriber:
    """faster-whisper wrapper. Input: int16 mono 16 kHz."""

    _model_cls = WhisperModel  # swapped in tests

    def __init__(self, model_name: str) -> None:
        self._model = self._model_cls(model_name, device="cpu", compute_type="int8")

    def transcribe(self, pcm16: np.ndarray) -> str:
        audio = pcm16.astype(np.float32) / 32768.0
        segments, _ = self._model.transcribe(
            audio, beam_size=1, language="en", vad_filter=False
        )
        return " ".join(s.text.strip() for s in segments).strip()

    async def atranscribe(self, pcm16: np.ndarray) -> str:
        return await asyncio.to_thread(self.transcribe, pcm16)
```

- [ ] **Step 5: Run, verify passes**

Run: `uv run pytest tests/test_stt.py -v` → 2 PASS.
Run: `uv run pytest tests/test_stt.py -m live -v` → 1 PASS (first run downloads `base.en`).

- [ ] **Step 6: Commit**

```bash
git add veronica/speech/stt.py tests/test_stt.py tests/fixtures scripts/make_fixture.py && git commit -m "feat: faster-whisper STT adapter with speech fixture"
```

---

### Task 6: VAD-endpointed recorder

**Files:**
- Create: `veronica/audio/record.py`, `tests/test_record.py`

**Interfaces:**
- Produces: `Recorder(settings)` with `async capture(max_s: int | None = None) -> np.ndarray | None` — returns int16 mono 16 kHz utterance, or `None` if speech shorter than `min_speech_ms` / nothing before `max_s`. Constructor accepts `frames: Callable[[], Iterator[bytes]] | None` — a frame source yielding `frame_ms` chunks of int16 bytes; default reads from `sounddevice.RawInputStream`. Class attribute `_vad_cls` for injection.

- [ ] **Step 1: Write failing tests**

`tests/test_record.py`:
```python
import numpy as np

from veronica.audio.record import Recorder
from veronica.config import Settings

FRAME = 480  # 30 ms @ 16 kHz


class FakeVad:
    """Speech if frame is non-zero."""

    def __init__(self, level):
        pass

    def is_speech(self, frame_bytes, sample_rate):
        return any(frame_bytes)


def frames(pattern):
    """pattern: string of 's' (speech) / '.' (silence), one char per 30 ms frame."""
    for ch in pattern:
        val = 1000 if ch == "s" else 0
        yield np.full(FRAME, val, dtype=np.int16).tobytes()
    while True:
        yield np.zeros(FRAME, dtype=np.int16).tobytes()


def make(pattern, **over):
    Recorder._vad_cls = FakeVad
    s = Settings(vad_silence_ms=90, min_speech_ms=60, max_utterance_s=1, **over)
    return Recorder(s, frames=lambda: frames(pattern))


async def test_returns_speech_then_stops_on_silence():
    r = make("....ssssss...........")
    pcm = await r.capture()
    assert pcm is not None
    # 6 speech frames + 3 silence frames (90 ms) captured
    assert len(pcm) == FRAME * 9


async def test_too_short_speech_returns_none():
    r = make("..s....")
    assert await r.capture() is None


async def test_no_speech_before_max_returns_none():
    r = make(".........................................")
    assert await r.capture(max_s=1) is None


async def test_max_utterance_cap():
    r = make("s" * 200)  # 6 s of speech, cap 1 s
    pcm = await r.capture()
    assert len(pcm) <= 16000 + FRAME
```

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_record.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`veronica/audio/record.py`:
```python
import asyncio
from collections.abc import Callable, Iterator

import numpy as np
import sounddevice as sd
import webrtcvad

from veronica.config import Settings


class Recorder:
    """Captures one utterance, endpointed by webrtcvad silence."""

    _vad_cls = webrtcvad.Vad  # swapped in tests

    def __init__(self, settings: Settings, frames: Callable[[], Iterator[bytes]] | None = None) -> None:
        self.s = settings
        self._frames = frames or self._mic_frames
        self._vad = self._vad_cls(settings.vad_aggressiveness)

    def _mic_frames(self) -> Iterator[bytes]:
        n = self.s.sample_rate * self.s.frame_ms // 1000
        with sd.RawInputStream(samplerate=self.s.sample_rate, channels=1, dtype="int16", blocksize=n) as stream:
            while True:
                data, _ = stream.read(n)
                yield bytes(data)

    async def capture(self, max_s: int | None = None) -> np.ndarray | None:
        return await asyncio.to_thread(self._capture, max_s)

    def _capture(self, max_s: int | None) -> np.ndarray | None:
        fm = self.s.frame_ms
        silence_frames_needed = self.s.vad_silence_ms // fm
        min_speech_frames = self.s.min_speech_ms // fm
        max_frames = (max_s or self.s.max_utterance_s) * 1000 // fm
        wait_frames = (max_s * 1000 // fm) if max_s else None

        buf: list[bytes] = []
        speech_frames = 0
        silence_run = 0
        started = False
        waited = 0

        for frame in self._frames():
            is_speech = self._vad.is_speech(frame, self.s.sample_rate)
            if not started:
                waited += 1
                if is_speech:
                    started = True
                elif wait_frames is not None and waited >= wait_frames:
                    return None
                else:
                    continue
            buf.append(frame)
            if is_speech:
                speech_frames += 1
                silence_run = 0
            else:
                silence_run += 1
            if silence_run >= silence_frames_needed or len(buf) >= max_frames:
                break

        if speech_frames < min_speech_frames:
            return None
        return np.frombuffer(b"".join(buf), dtype=np.int16)
```

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest tests/test_record.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
git add veronica/audio/record.py tests/test_record.py && git commit -m "feat: VAD-endpointed recorder"
```

---

### Task 7: Wake word listener

**Files:**
- Create: `veronica/audio/wake.py`, `tests/test_wake.py`

**Interfaces:**
- Produces: `WakeWord(settings, frames=None)` with `async wait() -> None` (returns on detection). Same `frames` injection pattern as Recorder (80 ms frames = 1280 samples, openwakeword's expected chunk). Class attribute `_model_cls` for injection.

- [ ] **Step 1: Write failing tests**

`tests/test_wake.py`:
```python
import numpy as np

from veronica.audio.wake import WakeWord
from veronica.config import Settings

CHUNK = 1280


class FakeModel:
    def __init__(self, wakeword_models, inference_framework):
        self.name = wakeword_models[0]

    def predict(self, chunk):
        return {self.name: 0.9 if chunk.max() > 0 else 0.0}

    def reset(self):
        pass


def frames(pattern):
    for ch in pattern:
        yield np.full(CHUNK, 1000 if ch == "w" else 0, dtype=np.int16).tobytes()
    while True:
        yield np.zeros(CHUNK, dtype=np.int16).tobytes()


async def test_wait_returns_on_detection():
    WakeWord._model_cls = FakeModel
    w = WakeWord(Settings(), frames=lambda: frames("...w"))
    await w.wait()  # must return, not hang


async def test_threshold_respected():
    WakeWord._model_cls = FakeModel
    seen = []

    def f():
        for ch in "..w":
            seen.append(ch)
            yield np.full(CHUNK, 1000 if ch == "w" else 0, dtype=np.int16).tobytes()

    w = WakeWord(Settings(wake_threshold=0.5), frames=f)
    await w.wait()
    assert seen == [".", ".", "w"]
```

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_wake.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`veronica/audio/wake.py`:
```python
import asyncio
from collections.abc import Callable, Iterator

import numpy as np
import sounddevice as sd
from openwakeword.model import Model

from veronica.config import Settings

CHUNK = 1280  # 80 ms @ 16 kHz, openwakeword's native chunk


class WakeWord:
    """Blocks until the configured wake word scores above threshold."""

    _model_cls = Model  # swapped in tests

    def __init__(self, settings: Settings, frames: Callable[[], Iterator[bytes]] | None = None) -> None:
        self.s = settings
        self._frames = frames or self._mic_frames
        self._model = self._model_cls(wakeword_models=[settings.wake_model], inference_framework="onnx")

    def _mic_frames(self) -> Iterator[bytes]:
        with sd.RawInputStream(samplerate=self.s.sample_rate, channels=1, dtype="int16", blocksize=CHUNK) as stream:
            while True:
                data, _ = stream.read(CHUNK)
                yield bytes(data)

    async def wait(self) -> None:
        await asyncio.to_thread(self._wait)

    def _wait(self) -> None:
        self._model.reset()
        for frame in self._frames():
            chunk = np.frombuffer(frame, dtype=np.int16)
            scores = self._model.predict(chunk)
            if scores[self.s.wake_model] >= self.s.wake_threshold:
                return
```

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest tests/test_wake.py -v`
Expected: 2 PASS

- [ ] **Step 5: Live check**

Run: `uv run python -c "
import asyncio
from veronica.audio.wake import WakeWord
from veronica.config import settings
print('say: hey jarvis'); asyncio.run(WakeWord(settings).wait()); print('DETECTED')"`
Expected: prints DETECTED after you say "hey jarvis".

- [ ] **Step 6: Commit**

```bash
git add veronica/audio/wake.py tests/test_wake.py && git commit -m "feat: openwakeword listener"
```

---

### Task 8: Brain (Agent SDK wrapper) with confirmation gate

**Files:**
- Create: `veronica/brain/prompts.py`, `veronica/brain/agent.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: `SentenceSplitter` (Task 2).
- Produces:
  - `prompts.system_prompt(today: datetime.date) -> str`
  - `Brain(settings, confirm: Callable[[str], Awaitable[bool]])` with `async ask(text: str) -> AsyncIterator[str]` yielding sentences; `async close()`. `confirm(summary)` is awaited for every tool call; `True` → allow, `False` → deny `user declined`.
  - `summarize_tool(tool_name: str, input: dict) -> str` — one-line spoken summary, e.g. `Bash: ls -la`, `Write file notes.txt`.
  - Session id read from / written to `settings.session_file`.
  - Class attribute `_client_cls` for injection (defaults to `ClaudeSDKClient`).

- [ ] **Step 1: Write failing tests**

`tests/test_agent.py`:
```python
import datetime as dt

import pytest

from veronica.brain import agent as agent_mod
from veronica.brain.agent import Brain, summarize_tool
from veronica.brain.prompts import system_prompt
from veronica.config import Settings


def test_system_prompt_has_date_and_rules():
    p = system_prompt(dt.date(2026, 9, 15))
    assert "You are Veronica" in p
    assert "2026-09-15" in p
    assert "one to three spoken sentences" in p


def test_summarize_tool():
    assert summarize_tool("Bash", {"command": "ls -la"}) == "Bash: ls -la"
    assert summarize_tool("Write", {"file_path": "/x/notes.txt"}) == "Write file /x/notes.txt"
    assert summarize_tool("Edit", {"file_path": "/x/a.py"}) == "Edit file /x/a.py"
    assert summarize_tool("WebSearch", {"query": "weather"}) == "WebSearch: weather"
    assert summarize_tool("Foo", {"a": 1}) == "Foo"


# ---- fake SDK client -------------------------------------------------------

class _Text:
    def __init__(self, text): self.text = text

class _Assistant:
    def __init__(self, *texts): self.content = [_Text(t) for t in texts]

class _Result:
    def __init__(self, sid): self.session_id = sid; self.result = None; self.terminal_reason = "success"


class FakeClient:
    instances = []

    def __init__(self, options=None):
        self.options = options
        self.queries = []
        self.script = [_Assistant("Hello there. How "), _Assistant("are you?"), _Result("sess-1")]
        FakeClient.instances.append(self)

    async def connect(self): pass
    async def disconnect(self): pass
    async def query(self, prompt): self.queries.append(prompt)

    async def receive_response(self):
        for m in self.script:
            yield m


@pytest.fixture
def brain(tmp_home, monkeypatch):
    monkeypatch.setattr(agent_mod, "AssistantMessage", _Assistant)
    monkeypatch.setattr(agent_mod, "TextBlock", _Text)
    monkeypatch.setattr(agent_mod, "ResultMessage", _Result)
    Brain._client_cls = FakeClient
    FakeClient.instances.clear()

    async def confirm(summary): return summary.startswith("Bash")

    return Brain(Settings(), confirm=confirm)


async def test_ask_yields_sentences_and_saves_session(brain, tmp_home):
    out = [s async for s in brain.ask("hi")]
    assert out == ["Hello there.", "How are you?"]
    assert (tmp_home / "session").read_text() == "sess-1"
    assert FakeClient.instances[0].queries == ["hi"]


async def test_options_wired(brain):
    [s async for s in brain.ask("x")]
    o = FakeClient.instances[0].options
    assert o.effort == "low"
    assert o.max_turns == 8
    assert o.permission_mode == "default"
    assert "You are Veronica" in o.system_prompt
    assert o.can_use_tool is not None


async def test_resume_from_saved_session(brain, tmp_home):
    (tmp_home / "session").write_text("old-sess")
    [s async for s in brain.ask("x")]
    assert FakeClient.instances[0].options.resume == "old-sess"


async def test_can_use_tool_gate(brain):
    [s async for s in brain.ask("x")]
    gate = FakeClient.instances[0].options.can_use_tool
    allow = await gate("Bash", {"command": "ls"}, None)
    deny = await gate("Write", {"file_path": "a"}, None)
    assert allow.behavior == "allow"
    assert deny.behavior == "deny" and deny.message == "user declined"


async def test_timeout_yields_message(brain, monkeypatch):
    import asyncio

    async def slow(self):
        await asyncio.sleep(10)
        yield _Result("s")

    monkeypatch.setattr(FakeClient, "receive_response", slow)
    brain.s = Settings(brain_timeout_s=0)
    out = [s async for s in brain.ask("x")]
    assert out == ["Taking too long, cancelled."]
```

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_agent.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement prompts**

`veronica/brain/prompts.py`:
```python
import datetime as dt


def system_prompt(today: dt.date) -> str:
    return (
        "You are Veronica, a voice assistant running on Mani's Mac. "
        "Reply in one to three spoken sentences. No markdown, no lists, no code unless asked. "
        "For long answers, give the short version and offer to say more. "
        f"Today is {today.isoformat()}."
    )
```

- [ ] **Step 4: Implement agent**

`veronica/brain/agent.py`:
```python
import asyncio
import datetime as dt
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)
from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

from veronica.brain.prompts import system_prompt
from veronica.brain.sentences import SentenceSplitter
from veronica.config import Settings

log = logging.getLogger("veronica.brain")

Confirm = Callable[[str], Awaitable[bool]]


def summarize_tool(tool_name: str, input: dict) -> str:
    if tool_name in ("Write", "Edit") and "file_path" in input:
        return f"{tool_name} file {input['file_path']}"
    for key in ("command", "query", "url", "pattern", "file_path"):
        if key in input:
            return f"{tool_name}: {input[key]}"
    return tool_name


class Brain:
    """One resumable Claude Code session; every tool call goes through `confirm`."""

    _client_cls = ClaudeSDKClient  # swapped in tests

    def __init__(self, settings: Settings, confirm: Confirm) -> None:
        self.s = settings
        self._confirm = confirm
        self._client = None

    # -- session persistence --------------------------------------------------
    def _load_session(self) -> str | None:
        f = self.s.session_file
        return f.read_text().strip() or None if f.exists() else None

    def _save_session(self, sid: str) -> None:
        self.s.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.s.session_file.write_text(sid)

    # -- permission gate ------------------------------------------------------
    async def _can_use_tool(self, tool_name: str, input: dict, context):
        summary = summarize_tool(tool_name, input)
        log.info("tool request: %s", summary)
        if await self._confirm(summary):
            return PermissionResultAllow(updated_input=input)
        return PermissionResultDeny(message="user declined")

    def _options(self) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            system_prompt=system_prompt(dt.date.today()),
            effort=self.s.effort,
            max_turns=self.s.max_turns,
            permission_mode="default",
            can_use_tool=self._can_use_tool,
            resume=self._load_session(),
        )

    async def _ensure_client(self):
        if self._client is None:
            self._client = self._client_cls(options=self._options())
            await self._client.connect()
        return self._client

    # -- public ---------------------------------------------------------------
    async def ask(self, text: str) -> AsyncIterator[str]:
        client = await self._ensure_client()
        splitter = SentenceSplitter()
        await client.query(text)
        try:
            async with asyncio.timeout(self.s.brain_timeout_s):
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                for sent in splitter.feed(block.text):
                                    yield sent
                    elif isinstance(msg, ResultMessage):
                        self._save_session(msg.session_id)
        except TimeoutError:
            log.warning("brain timeout after %ss", self.s.brain_timeout_s)
            await self.close()
            yield "Taking too long, cancelled."
            return
        for sent in splitter.flush():
            yield sent

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            finally:
                self._client = None
```

- [ ] **Step 5: Run, verify passes**

Run: `uv run pytest tests/test_agent.py -v`
Expected: 7 PASS

- [ ] **Step 6: Live smoke (needs `claude` logged in)**

Run: `uv run python -c "
import asyncio
from veronica.brain.agent import Brain
from veronica.config import settings
async def yes(s): print('CONFIRM?', s); return True
async def main():
    b = Brain(settings, confirm=yes)
    async for s in b.ask('Say hi in one sentence.'): print('>', s)
    await b.close()
asyncio.run(main())"`
Expected: one or two spoken-style sentences printed. If `CLIError`/auth error: run `claude auth login` and ensure `ANTHROPIC_API_KEY` is unset.

- [ ] **Step 7: Commit**

```bash
git add veronica/brain tests/test_agent.py && git commit -m "feat: Claude Agent SDK brain with confirmation gate"
```

---

### Task 9: Orchestrator state machine

**Files:**
- Create: `veronica/orchestrator.py`, `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `WakeWord.wait()`, `Recorder.capture(max_s)`, `Transcriber.atranscribe(pcm)`, `Brain.ask(text)`, `Synthesizer.asynth(text)`, `Player.play/stop/reset`.
- Produces: `Orchestrator(settings, wake, recorder, stt, brain, tts, player, on_state: Callable[[str], None] | None = None)` with `async run_forever()`, `async handle_text(text: str) -> list[str]` (returns spoken sentences; used by `--text` mode and tests), `async confirm(summary: str) -> bool` (speaks question, listens, matches confirm words), `state: str`. States: `idle`, `listening`, `thinking`, `speaking`, `followup`. `Orchestrator.CONFIRM_WORDS` frozenset.

- [ ] **Step 1: Write failing tests**

`tests/test_orchestrator.py`:
```python
import numpy as np
import pytest

from veronica.config import Settings
from veronica.orchestrator import Orchestrator


class Wake:
    async def wait(self): pass

class Rec:
    def __init__(self, pcms): self.pcms = list(pcms)
    async def capture(self, max_s=None): return self.pcms.pop(0) if self.pcms else None

class STT:
    def __init__(self, texts): self.texts = list(texts)
    async def atranscribe(self, pcm): return self.texts.pop(0)

class Brain:
    def __init__(self): self.asked = []
    async def ask(self, text):
        self.asked.append(text)
        yield "Sure."
        yield "Done."

class TTS:
    def __init__(self): self.said = []
    async def asynth(self, text):
        self.said.append(text)
        return np.zeros(10, dtype=np.float32), 24000

class Player:
    def __init__(self): self.played = 0; self.stops = 0; self.resets = 0
    async def play(self, s): self.played += 1
    def stop(self): self.stops += 1
    def reset(self): self.resets += 1


def build(rec_pcms=(), stt_texts=()):
    states = []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec(rec_pcms), stt=STT(stt_texts),
        brain=Brain(), tts=TTS(), player=Player(), on_state=states.append,
    )
    return o, states


async def test_handle_text_speaks_each_sentence():
    o, states = build()
    out = await o.handle_text("hello")
    assert out == ["Sure.", "Done."]
    assert o.tts.said == ["Sure.", "Done."]
    assert o.player.played == 2 and o.player.resets == 1
    assert states[:2] == ["thinking", "speaking"]


async def test_confirm_yes_and_no():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16)], stt_texts=["Yes, do it", "nah"])
    assert await o.confirm("Bash: ls") is True
    assert await o.confirm("Bash: rm") is False
    assert o.tts.said[0] == "Run Bash: ls?"


async def test_confirm_no_speech_is_deny():
    o, _ = build(rec_pcms=[None])
    assert await o.confirm("Write file a") is False


async def test_empty_transcript_prompts_retry():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16)], stt_texts=[""])
    await o.one_turn()
    assert o.tts.said == ["Sorry, didn't catch that."]
    assert o.brain.asked == []


async def test_full_turn():
    o, states = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what time is it"])
    await o.one_turn()
    assert o.brain.asked == ["what time is it"]
    assert states == ["listening", "thinking", "speaking", "followup", "idle"]
```

- [ ] **Step 2: Run, verify fails**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`veronica/orchestrator.py`:
```python
import asyncio
import logging
import re
import time
from collections.abc import Callable

from veronica.config import Settings

log = logging.getLogger("veronica.orchestrator")


class Orchestrator:
    CONFIRM_WORDS = frozenset({"yes", "yeah", "yep", "do it", "go", "go ahead", "confirm", "sure"})

    def __init__(self, settings: Settings, *, wake, recorder, stt, brain, tts, player,
                 on_state: Callable[[str], None] | None = None) -> None:
        self.s = settings
        self.wake, self.recorder, self.stt = wake, recorder, stt
        self.brain, self.tts, self.player = brain, tts, player
        self._on_state = on_state or (lambda _: None)
        self.state = "idle"

    def _set(self, state: str) -> None:
        self.state = state
        log.info("state=%s", state)
        self._on_state(state)

    # -- speaking -------------------------------------------------------------
    async def say(self, text: str) -> None:
        samples, _ = await self.tts.asynth(text)
        await self.player.play(samples)

    async def handle_text(self, text: str) -> list[str]:
        """Ask the brain and speak each sentence as it arrives. Returns sentences."""
        self._set("thinking")
        t0 = time.monotonic()
        spoken: list[str] = []
        self.player.reset()
        async for sent in self.brain.ask(text):
            if not spoken:
                self._set("speaking")
                log.info("latency first-sentence=%.2fs", time.monotonic() - t0)
            spoken.append(sent)
            await self.say(sent)
        if not spoken:
            await self.say("I have nothing to say to that.")
        return spoken

    # -- confirmation gate ----------------------------------------------------
    async def confirm(self, summary: str) -> bool:
        self.player.stop()
        self.player.reset()
        await self.say(f"Run {summary}?")
        pcm = await self.recorder.capture(max_s=self.s.confirm_listen_s)
        if pcm is None:
            return False
        heard = (await self.stt.atranscribe(pcm)).lower()
        words = re.sub(r"[^a-z ]", " ", heard).split()
        text = " ".join(words)
        ok = any(text.startswith(w) or f" {w}" in f" {text}" for w in self.CONFIRM_WORDS)
        log.info("confirm heard=%r -> %s", heard, ok)
        return ok

    # -- one interaction ------------------------------------------------------
    async def one_turn(self) -> None:
        """Called after wake word: listen, answer, then follow-up window."""
        self._set("listening")
        pcm = await self.recorder.capture()
        while True:
            if pcm is None:
                break
            text = await self.stt.atranscribe(pcm)
            log.info("heard=%r", text)
            if not text:
                self.player.reset()
                await self.say("Sorry, didn't catch that.")
            else:
                await self.handle_text(text)
            self._set("followup")
            pcm = await self.recorder.capture(max_s=self.s.followup_window_s)
        self._set("idle")

    async def run_forever(self) -> None:
        self._set("idle")
        while True:
            await self.wake.wait()
            try:
                await self.one_turn()
            except Exception:
                log.exception("turn failed")
                self.player.reset()
                await self.say("Something went wrong, check the log.")
                self._set("idle")
```

Note: `followup_window_s=0` in tests makes `capture(max_s=0)` → `wait_frames=None` path in Recorder; the fake Rec ignores it. Real Recorder with `max_s=0` would block forever — Settings default is 8, and Task 1 config forbids 0 in practice; add a guard: in `one_turn`, use `max_s=max(1, self.s.followup_window_s)`. Apply that edit now (the fakes don't care).

- [ ] **Step 4: Run, verify passes**

Run: `uv run pytest tests/test_orchestrator.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add veronica/orchestrator.py tests/test_orchestrator.py && git commit -m "feat: orchestrator state machine with confirm gate and follow-up window"
```

---

### Task 10: Entry point with `--text` mode

**Files:**
- Create: `veronica/__main__.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `build_orchestrator(settings, on_state=None) -> Orchestrator` (wires real components; `Brain.confirm` bound to `Orchestrator.confirm`); `main(argv)`.

- [ ] **Step 1: Implement**

`veronica/__main__.py`:
```python
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
```

- [ ] **Step 2: Live smoke**

Run: `uv run python -m veronica --text "what is two plus two"`
Expected: prints a one-sentence answer and speaks it through the speaker. Log line `latency first-sentence=…` in `~/.veronica/logs/veronica.log`.

- [ ] **Step 3: Commit**

```bash
git add veronica/__main__.py && git commit -m "feat: CLI entry point with --text debug mode"
```

---

### Task 11: Menu bar app

**Files:**
- Create: `veronica/ui/__init__.py`, `veronica/ui/menubar.py`

**Interfaces:**
- Consumes: `build_orchestrator` (Task 10).
- Produces: `run_app()` — starts rumps on main thread, asyncio loop in a daemon thread, updates title from orchestrator state; menu items Mute (stops wake-word loop), Quit.

- [ ] **Step 1: Implement**

`veronica/ui/__init__.py`: empty.

`veronica/ui/menubar.py`:
```python
import asyncio
import threading

import rumps

from veronica.__main__ import build_orchestrator
from veronica.config import settings

ICONS = {"idle": "◯", "listening": "◉", "thinking": "…", "speaking": "♪", "followup": "◎", "error": "✕"}


class VeronicaApp(rumps.App):
    def __init__(self) -> None:
        super().__init__("V ◯", quit_button=None)
        self._state = "idle"
        self._muted = False
        self.menu = [rumps.MenuItem("Mute", callback=self.toggle_mute), None, rumps.MenuItem("Quit", callback=self.quit)]
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._timer = rumps.Timer(self._refresh, 0.25)
        self._timer.start()

    # asyncio side (background thread)
    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._orch = build_orchestrator(settings, on_state=self._on_state)
            self._loop.run_until_complete(self._orch.run_forever())
        except Exception as e:  # surface startup failures (no mic, not logged in)
            self._state = "error"
            self._error = str(e)

    def _on_state(self, state: str) -> None:
        self._state = state

    # AppKit side (main thread)
    def _refresh(self, _timer) -> None:
        self.title = f"V {ICONS.get(self._state, '?')}" if not self._muted else "V zz"

    def toggle_mute(self, item: rumps.MenuItem) -> None:
        self._muted = not self._muted
        item.state = self._muted
        # stop any speech; the wake loop keeps running but muting is honoured in _refresh only.
        if self._muted:
            self._orch.player.stop()

    def quit(self, _item) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        rumps.quit_application()


def run_app() -> None:
    VeronicaApp().run()
```

- [ ] **Step 2: Live check**

Run: `uv run python -m veronica`
Expected: "V ◯" in menu bar. Say "hey jarvis" → icon "◉" → ask "what day is it" → "…" then "♪" while it speaks → "◎" for 8 s → "◯". Quit works. If macOS prompts for microphone permission, allow it (Terminal/iTerm needs Microphone in System Settings → Privacy).

- [ ] **Step 3: Commit**

```bash
git add veronica/ui && git commit -m "feat: rumps menu bar app with state indicator"
```

---

### Task 12: Custom "Hey Veronica" wake word

**Files:**
- Create: `scripts/train_wakeword.md`
- Modify: `veronica/config.py` default `wake_model`, `README.md`

- [ ] **Step 1: Train model**

openwakeword's custom-model notebook: https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb (Colab). Set target phrase `hey veronica`, keep defaults, export `hey_veronica.onnx`. Copy to `~/.veronica/models/hey_veronica.onnx`.

Write `scripts/train_wakeword.md` with exactly those instructions plus the copy command.

- [ ] **Step 2: Point config at it**

In `veronica/config.py` change `wake_model: str = "hey_jarvis"` → `wake_model: str = "hey_veronica"`, and in `veronica/audio/wake.py` change the model constructor to pass a path when the model is custom:

```python
        model_ref = settings.wake_model
        custom = settings.models_dir / f"{model_ref}.onnx"
        if custom.exists():
            model_ref = str(custom)
        self._model = self._model_cls(wakeword_models=[model_ref], inference_framework="onnx")
```
and match the score key: openwakeword keys custom models by file stem, so keep `scores[self.s.wake_model]`.

Update `tests/test_config.py` expected default to `"hey_veronica"`. Update README "Hey Jarvis" note.

- [ ] **Step 3: Run tests + live check**

Run: `uv run pytest -v` → all PASS.
Run: `uv run python -m veronica` → say "hey veronica" → listening icon.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "feat: custom hey veronica wake word"
```

---

## Self-review

- **Spec coverage (Phase 1):** wake word (T7, T12), record+VAD (T6), STT (T5), brain w/ resume + effort/max_turns + system prompt (T8), tool gate w/ spoken confirm + confirm words (T8, T9), streaming sentence→TTS (T2, T4, T9), follow-up window (T9), barge-in — **gap**: wake-word-during-TTS barge-in not implemented in Phase 1 (wake listener is not running while speaking). Deferred to Phase 2 plan; noted here so it is not lost. Error handling: STT empty (T9), brain timeout (T8), generic turn failure (T9), startup failure → red icon (T11). Usage-limit-specific message: deferred to Phase 2 (needs the SDK's error type observed live). Logging + latency log (T1, T9). Menu bar (T11). `--text` mode (T10). Tests unit + live markers (all tasks).
- **Placeholders:** none.
- **Type consistency:** `capture(max_s)` returns `np.ndarray | None` everywhere; `atranscribe -> str`; `asynth -> (np.ndarray, int)`; `ask -> AsyncIterator[str]`; `confirm(summary) -> bool`; `Player.reset/stop/play` used identically in T3, T9, T10, T11.
