import logging
import pathlib
import sys

import pytest

from pydantic import ValidationError

from veronica import config as config_mod
from veronica.config import EDITABLE_SETTINGS, Settings, coerce_setting, load_settings, log_level_from_env, setup_logging


def test_defaults(tmp_home):
    s = Settings()
    assert s.home == tmp_home
    assert s.sample_rate == 16000
    assert s.vad_silence_ms == 1200
    assert s.max_utterance_s == 15
    assert s.min_speech_ms == 300
    assert s.followup_window_s == 4
    assert s.followup_skip_ms == 300
    assert s.confirm_listen_s == 10
    assert s.listen_wait_s == 6
    assert s.capture_extra_s == 3.0
    assert s.wake_retry_s == 10
    assert s.wake_threshold == 0.35
    assert s.wake_hits == 2
    assert s.brain_timeout_s == 60
    assert s.max_turns is None
    assert s.brain_cwd == pathlib.Path.home()
    assert s.wake_model == "hey_veronica"
    assert s.wake_engine == "whisper"
    assert s.wake_whisper_model == "tiny.en"
    assert s.wake_window_s == 1.2
    assert s.wake_hop_s == 0.25
    assert s.wake_min_rms == 0.003
    assert s.wake_phrases == ["veronica", "veronika", "hey veronica", "hi veronica"]
    assert s.session_file == tmp_home / "session"
    assert s.log_file == tmp_home / "logs" / "veronica.log"
    assert s.whisper_model == "small.en"
    assert s.partial_stt is True
    assert s.partial_stt_model == "tiny.en"
    assert s.partial_hop_s == 0.7
    assert s.hud_enabled is True and s.hud_hide_after_s == 3.0
    assert (s.hud_width, s.hud_height, s.hud_margin) == (540, 300, 24)
    assert s.hud_mode == "full" and s.hud_mini_width == 400 and s.hud_mini_height == 72
    assert s.vad_silence_ms == 1200
    assert s.barge_threshold == 0.8
    assert s.chime_wake_hz == 880 and s.chime_followup_hz == 660
    assert s.memory_enabled is True
    assert s.memory_recent_turns == 6
    assert s.memory_path == tmp_home / "memory.db"
    assert s.ptt_enabled is True
    assert s.ptt_keycode == 61
    assert s.dictation_max_s == 60
    assert s.language == "en"
    assert s.whisper_multilingual_model == "small"
    assert s.partial_stt_multilingual_model == "tiny"


def test_dirs_created(tmp_home):
    s = Settings()
    s.ensure_dirs()
    assert (tmp_home / "logs").is_dir()
    assert (tmp_home / "models").is_dir()


def _fresh_veronica_logger():
    log = logging.getLogger("veronica")
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    return log


def test_setup_logging_drops_stream_handler_when_not_a_tty(tmp_home, monkeypatch):
    log = _fresh_veronica_logger()
    monkeypatch.setattr(sys.stderr, "isatty", lambda: False)
    try:
        log = setup_logging()
        assert not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                        for h in log.handlers)
        assert any(isinstance(h, logging.FileHandler) for h in log.handlers)
    finally:
        _fresh_veronica_logger()


def test_setup_logging_keeps_stream_handler_when_tty(tmp_home, monkeypatch):
    log = _fresh_veronica_logger()
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    try:
        log = setup_logging()
        assert any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                    for h in log.handlers)
    finally:
        _fresh_veronica_logger()


def test_load_settings_applies_overrides(tmp_home):
    s = load_settings({"effort": "high"})
    assert s.effort == "high"


def test_load_settings_ignores_unknown_key(tmp_home):
    s = load_settings({"not_a_real_field": 123})
    assert not hasattr(s, "not_a_real_field")


def test_load_settings_ignores_invalid_value(tmp_home):
    s = load_settings({"followup_window_s": "abc"})
    assert s.followup_window_s == 4


def test_load_settings_reraises_when_env_field_invalid(tmp_home, monkeypatch):
    # A bad NON-override field (env) can't be fixed by dropping overrides:
    # must raise (not spin forever popping keys that aren't there).
    monkeypatch.setenv("VERONICA_SAMPLE_RATE", "abc")
    with pytest.raises(ValidationError):
        load_settings({})
    with pytest.raises(ValidationError):
        load_settings({"effort": "high"})


def test_load_settings_drops_bad_override_but_reraises_env_error(tmp_home, monkeypatch):
    monkeypatch.setenv("VERONICA_SAMPLE_RATE", "abc")
    with pytest.raises(ValidationError):
        load_settings({"followup_window_s": "abc"})


def test_load_settings_loop_is_bounded(tmp_home, monkeypatch):
    calls = []
    real = Settings

    class Counting(real):
        def __init__(self, **kw):
            calls.append(dict(kw))
            super().__init__(**kw)

    monkeypatch.setattr(config_mod, "Settings", Counting)
    monkeypatch.setenv("VERONICA_SAMPLE_RATE", "abc")
    with pytest.raises(ValidationError):
        load_settings({"followup_window_s": "abc", "effort": "high"})
    assert len(calls) <= 3  # len(filtered) + 1


def test_load_settings_none_overrides(tmp_home):
    s = load_settings(None)
    assert s.effort == "low"


def test_coerce_setting_clamps_int():
    assert coerce_setting("followup_window_s", 99) == 15
    assert coerce_setting("followup_window_s", 0) == 1


def test_coerce_setting_clamps_float():
    assert coerce_setting("wake_min_rms", 0.0001) == 0.001
    assert coerce_setting("wake_min_rms", 999) == 0.05


def test_wake_timing_fields_editable_with_restart():
    for name, lo, hi in (("wake_min_rms", 0.001, 0.05), ("wake_window_s", 0.8, 2.5), ("wake_hop_s", 0.15, 0.6)):
        field = EDITABLE_SETTINGS[name]
        assert field.kind == "float" and field.restart is True
        assert (field.min, field.max) == (lo, hi)
        assert coerce_setting(name, 0.0) == lo
        assert coerce_setting(name, 99) == hi


def test_log_level_from_env(monkeypatch):
    monkeypatch.delenv("VERONICA_LOG_LEVEL", raising=False)
    assert log_level_from_env() == logging.INFO
    monkeypatch.setenv("VERONICA_LOG_LEVEL", "debug")
    assert log_level_from_env() == logging.DEBUG
    monkeypatch.setenv("VERONICA_LOG_LEVEL", "loud")
    assert log_level_from_env() == logging.INFO


def test_setup_logging_honours_env_level(tmp_home, monkeypatch):
    _fresh_veronica_logger()
    monkeypatch.setenv("VERONICA_LOG_LEVEL", "DEBUG")
    try:
        assert setup_logging().level == logging.DEBUG
    finally:
        _fresh_veronica_logger()


def test_coerce_setting_clamps_vad_silence_ms():
    assert coerce_setting("vad_silence_ms", 100) == 300
    assert coerce_setting("vad_silence_ms", 9999) == 3000


def test_coerce_setting_clamps_max_utterance_s():
    assert coerce_setting("max_utterance_s", 0) == 5
    assert coerce_setting("max_utterance_s", 999) == 60


def test_coerce_setting_splits_list():
    assert coerce_setting("wake_phrases", "veronica, hey veronica") == ["veronica", "hey veronica"]


def test_coerce_setting_invalid_choice_raises():
    with pytest.raises(ValueError):
        coerce_setting("effort", "ludicrous")


def test_coerce_setting_valid_choice():
    assert coerce_setting("effort", "medium") == "medium"


def test_coerce_setting_unknown_field_raises():
    with pytest.raises(ValueError):
        coerce_setting("not_a_real_field", 1)


def test_every_editable_setting_is_a_settings_field():
    fields = Settings.model_fields
    for name in EDITABLE_SETTINGS:
        assert name in fields, f"{name} is not a Settings field"


def test_validate_assignment_rejects_bad_type(tmp_home):
    s = Settings()
    with pytest.raises(Exception):
        s.followup_window_s = "x"


def test_input_volume_floor_default_and_clamp():
    assert Settings().input_volume_floor == 85
    f = EDITABLE_SETTINGS["input_volume_floor"]
    assert f.kind == "int" and f.restart is False and (f.min, f.max) == (0, 100)
    assert coerce_setting("input_volume_floor", -5) == 0
    assert coerce_setting("input_volume_floor", 250) == 100
    assert coerce_setting("input_volume_floor", "70") == 70


def test_computer_trust_s_default_and_clamp():
    assert Settings().computer_trust_s == 90
    f = EDITABLE_SETTINGS["computer_trust_s"]
    assert f.kind == "int" and f.restart is False and (f.min, f.max) == (0, 600)
    assert f.label == "Screen-control trust window (seconds)"
    assert coerce_setting("computer_trust_s", -5) == 0
    assert coerce_setting("computer_trust_s", 5000) == 600
    assert coerce_setting("computer_trust_s", "30") == 30


def test_hud_particles_and_intensity_are_live_editable():
    s = Settings()
    assert s.hud_particles == 4000 and s.hud_intensity == 1.0
    f = EDITABLE_SETTINGS["hud_particles"]
    assert f.kind == "int" and f.restart is False and (f.min, f.max) == (500, 8000)
    assert f.label == "HUD particles"
    g = EDITABLE_SETTINGS["hud_intensity"]
    assert g.kind == "float" and g.restart is False and (g.min, g.max) == (0.2, 2.0)
    assert g.label == "HUD intensity"
    assert coerce_setting("hud_particles", 10) == 500
    assert coerce_setting("hud_particles", 99999) == 8000
    assert coerce_setting("hud_intensity", 5) == 2.0
    assert coerce_setting("hud_intensity", "0.5") == 0.5


def test_preapprove_by_wording_is_a_live_bool():
    assert Settings().preapprove_by_wording is True
    f = EDITABLE_SETTINGS["preapprove_by_wording"]
    assert f.kind == "bool" and f.restart is False
    assert f.label == 'Pre-approve when I say "do it"'
    assert "never for sending mail" in f.help
    assert coerce_setting("preapprove_by_wording", "off") is False
    assert coerce_setting("preapprove_by_wording", True) is True


def test_gate_socket_and_backend_dirs(tmp_home):
    s = Settings()
    assert s.gate_socket == tmp_home / "gate.sock"
    d = s.backend_dir("codex")
    assert d == tmp_home / "backends" / "codex" and d.is_dir()
    assert d.stat().st_mode & 0o777 == 0o700
    assert s.session_file_for("claude") == s.session_file
    assert s.session_file_for("codex") == d / "session"


def test_antigravity_native_tools_is_a_live_bool():
    assert Settings().antigravity_native_tools is True
    f = EDITABLE_SETTINGS["antigravity_native_tools"]
    assert f.kind == "bool" and f.restart is False
    assert f.label == "Antigravity: allow its own shell"
    assert coerce_setting("antigravity_native_tools", "off") is False
