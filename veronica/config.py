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
    vad_silence_ms: int = 600
    max_utterance_s: int = 15
    min_speech_ms: int = 300
    followup_window_s: int = 4
    confirm_listen_s: int = 10
    listen_wait_s: int = 6

    # wake word
    wake_engine: str = "whisper"
    wake_model: str = "hey_veronica"
    wake_threshold: float = 0.35
    wake_hits: int = 2
    wake_retry_s: int = 10
    barge_threshold: float = 0.8  # openwakeword engine only; the whisper engine uses own-speech suppression instead
    wake_whisper_model: str = "tiny.en"
    wake_window_s: float = 1.6
    wake_hop_s: float = 0.4
    wake_min_rms: float = 0.01
    wake_phrases: list[str] = Field(
        default_factory=lambda: ["veronica", "veronika", "hey veronica", "hi veronica"]
    )

    # speech
    whisper_model: str = "small.en"
    kokoro_voice: str = "af_sarah"
    partial_stt: bool = True
    partial_stt_model: str = "tiny.en"
    partial_hop_s: float = 0.7

    # chimes (Hz)
    chime_wake_hz: int = 880
    chime_followup_hz: int = 660

    # brain
    brain_timeout_s: int = 60
    interrupt_drain_s: int = 3
    effort: str = "low"
    max_turns: int = 8

    # HUD
    hud_enabled: bool = True
    hud_hide_after_s: float = 3.0
    hud_width: int = 400
    hud_height: int = 230
    hud_margin: int = 24

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
