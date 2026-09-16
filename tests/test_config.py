import pathlib

from veronica.config import Settings


def test_defaults(tmp_home):
    s = Settings()
    assert s.home == tmp_home
    assert s.sample_rate == 16000
    assert s.vad_silence_ms == 600
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
    assert s.vad_silence_ms == 600
    assert s.barge_threshold == 0.8
    assert s.chime_wake_hz == 880 and s.chime_followup_hz == 660
    assert s.memory_enabled is True
    assert s.memory_recent_turns == 6
    assert s.memory_path == tmp_home / "memory.db"


def test_dirs_created(tmp_home):
    s = Settings()
    s.ensure_dirs()
    assert (tmp_home / "logs").is_dir()
    assert (tmp_home / "models").is_dir()
