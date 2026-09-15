import asyncio
import logging

import pytest

import veronica.__main__ as main_mod
from veronica.config import Settings


class _FakeSynthesizer:
    def __init__(self, voice, models_dir):
        self.voice = voice
        self.models_dir = models_dir


class _FakeBrain:
    def __init__(self, settings, confirm):
        self.s = settings
        self._confirm = confirm


async def test_build_orchestrator_text_mode(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "Brain", _FakeBrain)

    orch = main_mod.build_orchestrator(Settings(), audio=False)

    assert orch.wake is None
    assert orch.recorder is None
    assert orch.stt is None

    calls = []

    async def fake_confirm(summary):
        calls.append(summary)
        return True

    monkeypatch.setattr(orch, "confirm", fake_confirm)

    result = await orch.brain._confirm("Bash: ls")
    assert result is True
    assert calls == ["Bash: ls"]


def test_main_text_mode_parses(monkeypatch):
    calls = []

    async def fake_text_mode(text):
        calls.append(text)

    monkeypatch.setattr(main_mod, "_text_mode", fake_text_mode)
    monkeypatch.setattr(main_mod, "setup_logging", lambda: None)

    main_mod.main(["--text", "hi"])

    assert calls == ["hi"]


def test_main_scrubs_anthropic_api_key(monkeypatch, caplog):
    async def fake_text_mode(text):
        pass

    monkeypatch.setattr(main_mod, "_text_mode", fake_text_mode)
    monkeypatch.setattr(main_mod, "setup_logging", lambda: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-used")

    with caplog.at_level(logging.WARNING, logger="veronica"):
        main_mod.main(["--text", "hi"])

    assert "ANTHROPIC_API_KEY" not in main_mod.os.environ
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("ANTHROPIC_API_KEY" in r.message and "ignored" in r.message for r in warnings)


class _StubBrain:
    def __init__(self):
        self._confirm = None
        self.closed = False

    async def close(self):
        self.closed = True


class _StubOrchestrator:
    def __init__(self, handle_text):
        self.brain = _StubBrain()
        self._handle_text = handle_text

    async def handle_text(self, text):
        return await self._handle_text(self, text)


def test_text_mode_closes_brain_on_error(monkeypatch, tmp_home, capsys):
    async def raising_handle_text(orch, text):
        raise RuntimeError("boom")

    stub = _StubOrchestrator(raising_handle_text)
    monkeypatch.setattr(main_mod, "build_orchestrator", lambda s, audio=False: stub)

    with pytest.raises(RuntimeError):
        asyncio.run(main_mod._text_mode("x"))

    assert stub.brain.closed is True
    out = capsys.readouterr().out
    assert "[text mode] all tool calls are auto-approved — no voice confirmation" in out


def test_text_mode_prints_sentences_and_tools(monkeypatch, tmp_home, capsys):
    async def handle_text(orch, text):
        await orch.brain._confirm("Bash: ls")
        return ["Hi."]

    stub = _StubOrchestrator(handle_text)
    monkeypatch.setattr(main_mod, "build_orchestrator", lambda s, audio=False: stub)

    asyncio.run(main_mod._text_mode("x"))

    assert stub.brain.closed is True
    out = capsys.readouterr().out
    assert "[text mode] all tool calls are auto-approved — no voice confirmation" in out
    assert "[tool] Bash: ls -> allowed" in out
    assert "Hi." in out
