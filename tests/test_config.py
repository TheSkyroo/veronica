import logging
import pathlib
import sys

import pytest

from veronica.config import EDITABLE_SETTINGS, Settings, coerce_setting, load_settings, setup_logging


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
    assert s.wake_window_s == 1.6
    assert s.wake_hop_s == 0.4
    assert s.wake_min_rms == 0.01
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


def test_load_settings_none_overrides(tmp_home):
    s = load_settings(None)
    assert s.effort == "low"


def test_coerce_setting_clamps_int():
    assert coerce_setting("followup_window_s", 99) == 15
    assert coerce_setting("followup_window_s", 0) == 1


def test_coerce_setting_clamps_float():
    assert coerce_setting("wake_min_rms", 0.0001) == 0.002
    assert coerce_setting("wake_min_rms", 999) == 0.05


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
