import asyncio
import logging

import pytest

import veronica.__main__ as main_mod
from veronica.config import Settings
from veronica.memory.store import MemoryStore
from veronica.tools import memory_tools


class _FakeSynthesizer:
    def __init__(self, voice, models_dir, speed=1.0, hindi_voice="hf_alpha"):
        self.voice = voice
        self.models_dir = models_dir
        self.speed = speed
        self.hindi_voice = hindi_voice


class _FakeBrain:
    def __init__(self, settings, confirm, on_tool=None, memory=None):
        self.s = settings
        self._confirm = confirm
        self.on_tool = on_tool
        self.memory = memory


class _FakeSwitcher:
    """Stands in for BrainSwitcher: exposes the gate it was given and a
    `_FakeBrain` built on that gate's confirm, the way make_brain would."""
    def __init__(self, settings, *, gate, on_tool=None, memory=None, say=None, on_backend=None):
        self.s = settings
        self.gate = gate
        self.say = say
        self.on_backend = on_backend
        self.brain = _FakeBrain(settings, gate._confirm, on_tool=on_tool, memory=memory)
        self.started = 0

    async def start(self):
        self.started += 1
        if self.on_backend is not None:
            self.on_backend("Claude", False)

    def status_label(self):
        return "Claude"


async def test_build_orchestrator_text_mode(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {})
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)

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


async def test_build_orchestrator_wires_the_switcher(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {})
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    seen = []

    orch = main_mod.build_orchestrator(Settings(), on_event=lambda k, p: seen.append((k, p)), audio=False)
    try:
        sw = orch.switcher
        assert isinstance(sw, _FakeSwitcher)
        assert orch.brain is sw.brain and orch.gate is sw.gate
        assert isinstance(sw.gate, main_mod.ToolGate) and sw.gate.s is orch.s
        # the switcher's spoken lines and label changes go through the orchestrator
        said = []

        async def fake_say(text, *, lang=None):
            said.append(text)

        monkeypatch.setattr(orch, "say", fake_say)
        await sw.say("Codex hit its usage limit — switching to Claude.")
        assert said == ["Codex hit its usage limit — switching to Claude."]
        sw.on_backend("Claude (for Codex)", True)
        assert ("hud", {"backend": "Claude (for Codex)"}) in seen
    finally:
        orch.store.close()
        memory_tools.bind(None)


async def test_build_orchestrator_no_store_when_memory_disabled(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {})
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)

    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=False)

    assert orch.store is None
    assert orch.brain.memory is None
    assert memory_tools.store is None


async def test_build_orchestrator_applies_saved_voice_prefs(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {"tts_voice": "am_adam", "tts_speed": 9})

    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=False)

    assert orch.tts.voice == "am_adam"
    assert orch.tts.speed == 1.5   # clamped to SPEED_MAX


async def test_build_orchestrator_defaults_without_voice_prefs(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {})

    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=False)

    assert orch.tts.voice == Settings().kokoro_voice
    assert orch.tts.speed == 1.0


async def test_build_orchestrator_ignores_unknown_voice_pref(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {"tts_voice": "zz_nobody"})

    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=False)

    assert orch.tts.voice == Settings().kokoro_voice


async def test_build_orchestrator_ignores_bad_speed_pref(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {"tts_speed": "fast"})

    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=False)

    assert orch.tts.speed == 1.0


class _FakeWakeWord:
    def __init__(self, settings):
        self.s = settings


def _fake_make_wake(settings, frames=None):
    return _FakeWakeWord(settings)


class _FakeTranscriber:
    def __init__(self, model, language="en"):
        self.model = model
        self.model_name = model
        self.language = language


class _FakeRecorder:
    def __init__(self, settings, on_level=None):
        self.s = settings
        self.on_level = on_level


async def test_build_orchestrator_emits_mic_and_tool_events(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {})
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
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


async def test_build_orchestrator_wires_proactive(monkeypatch, tmp_home):
    from veronica.proactive import Schedule

    saved = {"proactive": {"briefing_enabled": True, "briefing_time": "07:45", "nudge_minutes": 12}}
    monkeypatch.setattr(main_mod.prefs, "load", lambda: saved)
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    monkeypatch.setattr(main_mod, "make_wake", _fake_make_wake)
    monkeypatch.setattr(main_mod, "Transcriber", _FakeTranscriber)
    monkeypatch.setattr(main_mod, "Recorder", _FakeRecorder)

    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=True)
    assert orch.proactive is not None
    assert orch.proactive.schedule == Schedule.from_prefs(saved["proactive"])
    assert orch.proactive.schedule.briefing_time == "07:45"
    memory_tools.bind(None)


async def test_build_orchestrator_no_proactive_in_text_mode(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {})
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=False)
    assert orch.proactive is None
    memory_tools.bind(None)


async def _build_audio_orch(monkeypatch):
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {})
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    monkeypatch.setattr(main_mod, "make_wake", _fake_make_wake)
    monkeypatch.setattr(main_mod, "Transcriber", _FakeTranscriber)
    monkeypatch.setattr(main_mod, "Recorder", _FakeRecorder)
    return main_mod.build_orchestrator(Settings(memory_enabled=False), audio=True)


async def test_proactive_adapters_read_pim_tool_text(monkeypatch, tmp_home):
    """Calendar/reminder adapters hand the pim tools' text to Proactive; the
    mail adapter uses Mail's real unread count, not the (capped) listing."""
    from veronica.tools import pim as pim_tools

    calls = []

    async def cal(args):
        calls.append(("cal", args))
        return {"content": [{"type": "text", "text": "09:00–09:30  Standup (Work)"}]}

    async def listing(args):
        calls.append(("mail_listing", args))
        return {"content": [{"type": "text", "text": "A  Subject\n  preview\nB  Other\n  preview"}]}

    async def count():
        calls.append(("mail_count", None))
        return 42

    async def rem(args):
        calls.append(("rem", args))
        return {"content": [{"type": "text", "text": "2026-09-16 10:00  Pay rent"}]}

    monkeypatch.setattr(pim_tools.calendar_events, "handler", cal)
    monkeypatch.setattr(pim_tools.mail_unread, "handler", listing)
    monkeypatch.setattr(pim_tools, "mail_unread_count", count)
    monkeypatch.setattr(pim_tools.reminders_due, "handler", rem)

    pro = (await _build_audio_orch(monkeypatch)).proactive
    assert await pro._calendar_events("today", 1) == "09:00–09:30  Standup (Work)"
    assert await pro._mail_unread_count() == 42
    assert await pro._reminders_due(1) == "2026-09-16 10:00  Pay rent"
    assert calls == [
        ("cal", {"day": "today", "days": 1}),
        ("mail_count", None),
        ("rem", {"days": 1}),
    ]
    memory_tools.bind(None)


async def test_proactive_mail_count_falls_back_to_listing(monkeypatch, tmp_home, caplog):
    """If Mail's unread-count property fails, fall back to counting the
    (capped) unread listing; a listing error counts as zero."""
    from veronica.tools import pim as pim_tools

    async def count():
        raise RuntimeError("Mail got an error: Connection is invalid.")

    async def listing(args):
        assert args == {"limit": 50}
        return {"content": [{"type": "text", "text": "A  Subject\n  preview\nB  Other\n  preview"}]}

    async def listing_err(args):
        return {"content": [{"type": "text", "text": "error: Mail isn't running"}], "is_error": True}

    monkeypatch.setattr(pim_tools, "mail_unread_count", count)
    monkeypatch.setattr(pim_tools.mail_unread, "handler", listing)
    pro = (await _build_audio_orch(monkeypatch)).proactive
    with caplog.at_level(logging.WARNING, logger="veronica"):
        assert await pro._mail_unread_count() == 2
    assert any("unread count" in r.getMessage() for r in caplog.records)

    monkeypatch.setattr(pim_tools.mail_unread, "handler", listing_err)
    assert await pro._mail_unread_count() == 0
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


class _StubGate:
    def __init__(self):
        self._confirm = None


class _StubOrchestrator:
    def __init__(self, handle_text):
        self.brain = _StubBrain()
        self.gate = _StubGate()
        self.player = _StubPlayer()
        self._handle_text = handle_text
        self.brain_started = self.brain_stopped = 0

    async def start_brain(self):
        self.brain_started += 1

    async def stop_brain(self):
        self.brain_stopped += 1

    async def handle_text(self, text):
        return await self._handle_text(self, text)


def test_text_mode_closes_brain_on_error(monkeypatch, tmp_home, capsys):
    async def raising_handle_text(orch, text):
        raise RuntimeError("boom")

    stub = _StubOrchestrator(raising_handle_text)
    monkeypatch.setattr(main_mod, "build_orchestrator", lambda s, audio=False, on_quit=None: stub)

    with pytest.raises(RuntimeError):
        asyncio.run(main_mod._text_mode("x"))

    assert stub.brain.closed is True
    assert stub.player.closed is True
    out = capsys.readouterr().out
    assert "[text mode] safe tools run automatically; risky tools ask y/N on this terminal" in out


def test_text_mode_prints_sentences_and_tools(monkeypatch, tmp_home, capsys):
    monkeypatch.setattr(main_mod, "_ask_stdin", lambda prompt: "y")

    async def handle_text(orch, text):
        await orch.gate._confirm("Bash: ls")
        return ["Hi."]

    stub = _StubOrchestrator(handle_text)
    monkeypatch.setattr(main_mod, "build_orchestrator", lambda s, audio=False, on_quit=None: stub)

    asyncio.run(main_mod._text_mode("x"))

    assert stub.brain.closed is True
    assert stub.player.closed is True
    # the brain is activated (and the gate socket opened) before the ask,
    # and the socket closed after it
    assert stub.brain_started == 1 and stub.brain_stopped == 1
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
    monkeypatch.setattr(main_mod, "build_orchestrator", lambda s, audio=False, on_quit=None: stub)

    asyncio.run(main_mod._text_mode("x"))

    ok = asyncio.run(stub.gate._confirm("Bash: rm x", "Bash: rm -rf x"))
    assert ok is True
    assert prompts == ["Run Bash: rm x? [Bash: rm -rf x] [y/N] "]

    prompts.clear()
    monkeypatch.setattr(main_mod, "_ask_stdin", lambda prompt: (prompts.append(prompt), "n")[1])
    ok = asyncio.run(stub.gate._confirm("Bash: rm x"))
    assert ok is False
    assert prompts == ["Run Bash: rm x? [] [y/N] "]


def test_ask_stdin_closed_stdin_declines(monkeypatch, tmp_home, capsys):
    def raising_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", raising_input)

    async def handle_text(orch, text):
        return ["Hi."]

    stub = _StubOrchestrator(handle_text)
    monkeypatch.setattr(main_mod, "build_orchestrator", lambda s, audio=False, on_quit=None: stub)

    asyncio.run(main_mod._text_mode("x"))

    ok = asyncio.run(stub.gate._confirm("Bash: rm x"))
    assert ok is False
    out = capsys.readouterr().out
    assert "[tool] Bash: rm x -> declined" in out


async def test_proactive_adapters_raise_on_pim_error(monkeypatch, tmp_home):
    """A Calendar/Reminders failure (timeout, Automation denied) must reach
    Proactive as an exception, not as 'Nothing on your calendar today.'"""
    from veronica.tools import pim as pim_tools

    async def boom(args):
        return {"content": [{"type": "text", "text": "boom"}], "is_error": True}

    async def count():
        return 0

    monkeypatch.setattr(pim_tools.calendar_events, "handler", boom)
    monkeypatch.setattr(pim_tools.reminders_due, "handler", boom)
    monkeypatch.setattr(pim_tools, "mail_unread_count", count)

    pro = (await _build_audio_orch(monkeypatch)).proactive
    with pytest.raises(RuntimeError, match="boom"):
        await pro._calendar_events("today", 1)
    with pytest.raises(RuntimeError, match="boom"):
        await pro._reminders_due(1)

    text = await pro.build_briefing()
    assert "Nothing on your calendar" not in text
    assert "Reminders due" not in text
    assert text.startswith("Good ")
    memory_tools.bind(None)


# -- batch C: language mode ---------------------------------------------------

def _patch_audio_fakes(monkeypatch, saved):
    monkeypatch.setattr(main_mod.prefs, "load", lambda: saved)
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    monkeypatch.setattr(main_mod, "make_wake", _fake_make_wake)
    monkeypatch.setattr(main_mod, "Transcriber", _FakeTranscriber)
    monkeypatch.setattr(main_mod, "Recorder", _FakeRecorder)


async def test_build_orchestrator_hindi_language_pref_picks_multilingual_models(monkeypatch, tmp_home):
    _patch_audio_fakes(monkeypatch, {"language": "hi"})
    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=True)
    assert orch.language == "hi"
    assert (orch.stt.model_name, orch.stt.language) == ("small", "hi")
    assert (orch.partial_stt.model_name, orch.partial_stt.language) == ("tiny", "hi")
    memory_tools.bind(None)


async def test_build_orchestrator_auto_language_pref_lets_whisper_detect(monkeypatch, tmp_home):
    _patch_audio_fakes(monkeypatch, {"language": "auto"})
    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=True)
    assert orch.language == "auto"
    assert (orch.stt.model_name, orch.stt.language) == ("small", None)
    assert (orch.partial_stt.model_name, orch.partial_stt.language) == ("tiny", None)
    memory_tools.bind(None)


async def test_build_orchestrator_default_language_is_english_models(monkeypatch, tmp_home):
    _patch_audio_fakes(monkeypatch, {})
    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=True)
    assert orch.language == "en"
    assert (orch.stt.model_name, orch.stt.language) == ("small.en", "en")
    assert (orch.partial_stt.model_name, orch.partial_stt.language) == ("tiny.en", "en")
    memory_tools.bind(None)


async def test_build_orchestrator_ignores_unknown_language_pref(monkeypatch, tmp_home):
    _patch_audio_fakes(monkeypatch, {"language": "fr"})
    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=True)
    assert orch.language == "en"
    assert orch.stt.model_name == "small.en"
    memory_tools.bind(None)


async def test_build_orchestrator_falls_back_to_english_when_multilingual_stt_fails(monkeypatch, tmp_home, caplog):
    _patch_audio_fakes(monkeypatch, {"language": "hi"})
    saved_calls = []
    monkeypatch.setattr(main_mod.prefs, "save", lambda d: saved_calls.append(d))

    class _Flaky(_FakeTranscriber):
        def __init__(self, model, language="en"):
            if model in ("small", "tiny"):
                raise RuntimeError("model download failed")
            super().__init__(model, language)

    monkeypatch.setattr(main_mod, "Transcriber", _Flaky)
    with caplog.at_level("ERROR", logger="veronica"):
        orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=True)
    assert orch.language == "en"
    assert (orch.stt.model_name, orch.stt.language) == ("small.en", "en")
    assert (orch.partial_stt.model_name, orch.partial_stt.language) == ("tiny.en", "en")
    assert any("model download failed" in r.getMessage() or "download failed" in (r.exc_text or "") for r in caplog.records)
    assert saved_calls == []     # the saved "hi" pref is left intact for the next launch
    memory_tools.bind(None)


async def test_build_orchestrator_english_stt_failure_still_raises(monkeypatch, tmp_home):
    _patch_audio_fakes(monkeypatch, {})

    def boom(model, language="en"):
        raise RuntimeError("no models at all")

    monkeypatch.setattr(main_mod, "Transcriber", boom)
    with pytest.raises(RuntimeError, match="no models at all"):
        main_mod.build_orchestrator(Settings(memory_enabled=False), audio=True)
    memory_tools.bind(None)


async def test_build_orchestrator_stt_factory_builds_transcribers(monkeypatch, tmp_home):
    _patch_audio_fakes(monkeypatch, {})
    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=True)
    assert callable(orch.stt_factory)
    t = orch.stt_factory("small", "hi")
    assert isinstance(t, _FakeTranscriber) and (t.model_name, t.language) == ("small", "hi")
    memory_tools.bind(None)


async def test_build_orchestrator_applies_saved_hindi_voice(monkeypatch, tmp_home):
    _patch_audio_fakes(monkeypatch, {"tts_hindi_voice": "hm_omega"})
    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=False)
    assert orch.tts.hindi_voice == "hm_omega"
    memory_tools.bind(None)


async def test_build_orchestrator_ignores_bad_hindi_voice(monkeypatch, tmp_home):
    _patch_audio_fakes(monkeypatch, {"tts_hindi_voice": "af_sarah"})
    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=False)
    assert orch.tts.hindi_voice == "hf_alpha"
    memory_tools.bind(None)


async def test_build_orchestrator_passes_updater_hooks(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {})
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    check, update, relaunch = dict, (lambda st: ""), (lambda: True)

    orch = main_mod.build_orchestrator(
        Settings(), audio=False, updater_check=check, updater_update=update, relaunch=relaunch,
    )
    assert (orch.updater_check, orch.updater_update, orch.relaunch) == (check, update, relaunch)

    plain = main_mod.build_orchestrator(Settings(), audio=False)
    assert (plain.updater_check, plain.updater_update, plain.relaunch) == (None, None, None)


async def test_build_orchestrator_wires_input_guard(monkeypatch, tmp_home):
    from veronica.audio.input_level import InputLevelGuard

    orch = await _build_audio_orch(monkeypatch)
    g = orch.input_guard
    assert isinstance(g, InputLevelGuard)
    assert g._floor() == 85
    orch.s.input_volume_floor = 60
    assert g._floor() == 60
    memory_tools.bind(None)


async def test_build_orchestrator_input_guard_hint_reaches_hud_once(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {})
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    monkeypatch.setattr(main_mod, "make_wake", _fake_make_wake)
    monkeypatch.setattr(main_mod, "Transcriber", _FakeTranscriber)
    monkeypatch.setattr(main_mod, "Recorder", _FakeRecorder)
    seen = []
    orch = main_mod.build_orchestrator(
        Settings(memory_enabled=False), on_event=lambda k, p: seen.append((k, p)), audio=True
    )
    cb = orch.input_guard._on_corrected
    cb(33, 85, "AirPods")
    cb(20, 85, "AirPods")
    assert seen == [("tool", {"summary": "Input volume 33 → 85 (AirPods)", "decision": "auto"})]
    memory_tools.bind(None)


async def test_build_orchestrator_no_input_guard_in_text_mode(monkeypatch, tmp_home):
    monkeypatch.setattr(main_mod.prefs, "load", lambda: {})
    monkeypatch.setattr(main_mod, "Synthesizer", _FakeSynthesizer)
    monkeypatch.setattr(main_mod, "BrainSwitcher", _FakeSwitcher)
    orch = main_mod.build_orchestrator(Settings(memory_enabled=False), audio=False)
    assert orch.input_guard is None
    memory_tools.bind(None)
