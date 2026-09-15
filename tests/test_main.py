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
