import asyncio
import logging

import pytest

import veronica.__main__ as main_mod
from veronica.config import Settings
from veronica.memory.store import MemoryStore
from veronica.tools import memory_tools


class _FakeSynthesizer:
    def __init__(self, voice, models_dir):
        self.voice = voice
        self.models_dir = models_dir


class _FakeBrain:
    def __init__(self, settings, confirm, on_tool=None, memory=None):
        self.s = settings
        self._confirm = confirm
        self.on_tool = on_tool
        self.memory = memory


async def test_build_orchestrator_text_mode(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "Brain", _FakeBrain)

    orch = main_mod.build_orchestrator(Settings(), audio=False)

    assert orch.wake is None
    assert orch.recorder is None
    assert orch.stt is None
    assert isinstance(orch.store, MemoryStore)
    assert orch.brain.memory is orch.store
    assert memory_tools.store is orch.store
    orch.store.close()
    memory_tools.bind(None)

    calls = []

    async def fake_confirm(summary, detail=""):
        calls.append(summary)
        return True

    monkeypatch.setattr(orch, "confirm", fake_confirm)

    result = await orch.brain._confirm("Bash: ls")
    assert result is True
    assert calls == ["Bash: ls"]


async def test_build_orchestrator_no_store_when_memory_disabled(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "Brain", _FakeBrain)

    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=False)

    assert orch.store is None
    assert orch.brain.memory is None
    assert memory_tools.store is None


class _FakeWakeWord:
    def __init__(self, settings):
        self.s = settings


def _fake_make_wake(settings, frames=None):
    return _FakeWakeWord(settings)


class _FakeTranscriber:
    def __init__(self, model):
        self.model = model


class _FakeRecorder:
    def __init__(self, settings, on_level=None):
        self.s = settings
        self.on_level = on_level


class _FakeBrainWithOnTool:
    def __init__(self, settings, confirm, on_tool=None, memory=None):
        self.s = settings
        self._confirm = confirm
        self.on_tool = on_tool
        self.memory = memory


async def test_build_orchestrator_emits_mic_and_tool_events(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "Brain", _FakeBrainWithOnTool)
    monkeypatch.setattr(main_mod, "make_wake", _fake_make_wake)
    monkeypatch.setattr(main_mod, "Transcriber", _FakeTranscriber)
    monkeypatch.setattr(main_mod, "Recorder", _FakeRecorder)

    seen = []
    orch = main_mod.build_orchestrator(
        Settings(), on_event=lambda k, p: seen.append((k, p)), audio=True
    )

    orch.recorder.on_level(0.5)
    orch.brain.on_tool("Read: /x", "auto")

    assert seen == [("mic", 0.5), ("tool", {"summary": "Read: /x", "decision": "auto"})]
    orch.store.close()
    memory_tools.bind(None)


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


class _StubPlayer:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _StubOrchestrator:
    def __init__(self, handle_text):
        self.brain = _StubBrain()
        self.player = _StubPlayer()
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
    assert stub.player.closed is True
    out = capsys.readouterr().out
    assert "[text mode] safe tools run automatically; risky tools ask y/N on this terminal" in out


def test_text_mode_prints_sentences_and_tools(monkeypatch, tmp_home, capsys):
    monkeypatch.setattr(main_mod, "_ask_stdin", lambda prompt: "y")

    async def handle_text(orch, text):
        await orch.brain._confirm("Bash: ls")
        return ["Hi."]

    stub = _StubOrchestrator(handle_text)
    monkeypatch.setattr(main_mod, "build_orchestrator", lambda s, audio=False: stub)

    asyncio.run(main_mod._text_mode("x"))

    assert stub.brain.closed is True
    assert stub.player.closed is True
    out = capsys.readouterr().out
    assert "[text mode] safe tools run automatically; risky tools ask y/N on this terminal" in out
    assert "[tool] Bash: ls -> allowed" in out
    assert "Hi." in out


def test_text_mode_prompts_for_confirm_class(monkeypatch, tmp_home, capsys):
    prompts = []
    monkeypatch.setattr(main_mod, "_ask_stdin", lambda prompt: (prompts.append(prompt), "y")[1])

    async def handle_text(orch, text):
        return ["Hi."]

    stub = _StubOrchestrator(handle_text)
    monkeypatch.setattr(main_mod, "build_orchestrator", lambda s, audio=False: stub)

    asyncio.run(main_mod._text_mode("x"))

    ok = asyncio.run(stub.brain._confirm("Bash: rm x", "Bash: rm -rf x"))
    assert ok is True
    assert prompts == ["Run Bash: rm x? [Bash: rm -rf x] [y/N] "]

    prompts.clear()
    monkeypatch.setattr(main_mod, "_ask_stdin", lambda prompt: (prompts.append(prompt), "n")[1])
    ok = asyncio.run(stub.brain._confirm("Bash: rm x"))
    assert ok is False
    assert prompts == ["Run Bash: rm x? [] [y/N] "]


def test_ask_stdin_closed_stdin_declines(monkeypatch, tmp_home, capsys):
    def raising_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", raising_input)

    async def handle_text(orch, text):
        return ["Hi."]

    stub = _StubOrchestrator(handle_text)
    monkeypatch.setattr(main_mod, "build_orchestrator", lambda s, audio=False: stub)

    asyncio.run(main_mod._text_mode("x"))

    ok = asyncio.run(stub.brain._confirm("Bash: rm x"))
    assert ok is False
    out = capsys.readouterr().out
    assert "[tool] Bash: rm x -> declined" in out
