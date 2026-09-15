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
