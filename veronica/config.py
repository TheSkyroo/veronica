import logging
import os
import sys
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from veronica import prefs

log = logging.getLogger("veronica.config")


# The brains Veronica can run on, in the user's default order. The registry
# in veronica.brain.backends is the source of truth for everything else.
BRAIN_BACKENDS: tuple[str, ...] = ("codex", "antigravity", "claude", "copilot", "local")

# The offline brain's defaults: the llama.cpp server and weights the user
# already keeps on this Mac. Both are plain paths, editable in Settings.
LOCAL_SERVER_BIN = Path.home() / "Github/sih/manas/runtime/bin/llama-server"
LOCAL_MODEL = Path.home() / "Github/sih/manas/models/granite-4.2-3b-q4_k_m.gguf"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VERONICA_", env_file=".env", extra="ignore", validate_assignment=True
    )

    home: Path = Field(default_factory=lambda: Path.home() / ".veronica")

    # audio
    sample_rate: int = 16000
    frame_ms: int = 30                 # webrtcvad frame size
    vad_aggressiveness: int = 2        # 0-3
    vad_silence_ms: int = 1200
    max_utterance_s: int = 15
    min_speech_ms: int = 300
    followup_window_s: int = 4
    followup_skip_ms: int = 300
    confirm_listen_s: int = 10
    listen_wait_s: int = 6
    capture_extra_s: float = 3.0
    # Voice isolation (docs/superpowers/specs/2026-10-01-veronica-voice-
    # isolation-design.md). Noise suppression cleans what the recorder's VAD
    # sees — so clatter and music stop opening captures — while whisper still
    # gets the raw audio (it transcribes that better); the wake check is
    # left alone.
    # vad_min_rms is the level of the cleaned audio a 30 ms frame must reach
    # to count as speech (so it only applies while suppression runs): faint
    # background talk stays out, you from across the room still get in.
    # 0 = off.
    noise_suppression: bool = True
    vad_min_rms: float = 0.001
    # Only my voice: with a profile enrolled ("learn my voice"), a request,
    # follow-up or confirm answer whose speaker score is under the threshold
    # (lower for short answers, SpeakerGate.threshold_for) is ignored as if
    # nothing was said. The wake word stays open to anyone unless
    # speaker_verification_wake is on too.
    speaker_verification: bool = True
    speaker_verification_wake: bool = False
    speaker_threshold: float = 0.35
    # How long a turn may stay silent before she says a short "On it." so a
    # slow brain doesn't feel like a dropped question. 0 turns the line off.
    ack_after_s: float = 8.0   # only for an unusually long wait; see Orchestrator.ACK_MIN_GAP_S

    # wake word
    wake_engine: str = "whisper"
    wake_model: str = "hey_veronica"
    wake_threshold: float = 0.35
    wake_hits: int = 2
    wake_retry_s: int = 10
    barge_threshold: float = 0.8  # openwakeword engine only; the whisper engine uses own-speech suppression instead
    wake_whisper_model: str = "tiny.en"
    wake_window_s: float = 2.0   # long enough to hold a slowly spoken "Ve-ro-ni-ca"
    wake_hop_s: float = 0.25
    wake_min_rms: float = 0.003   # far-field speech sits around 0.003-0.01; near-field 0.012-0.05
    wake_phrases: list[str] = Field(
        default_factory=lambda: ["veronica", "veronika", "hey veronica", "hi veronica"]
    )
    # macOS input volume floor (0-100): call apps' auto-gain and device
    # switches keep dropping it (27, 33 seen), which kills far-field wake
    # detection. The guard raises it back to this; 0 = off.
    input_volume_floor: int = 85

    # speech
    language: str = "en"
    whisper_model: str = "small.en"
    whisper_multilingual_model: str = "small"
    kokoro_voice: str = "af_heart"
    partial_stt: bool = True
    partial_stt_model: str = "tiny.en"
    partial_stt_multilingual_model: str = "tiny"
    partial_hop_s: float = 0.7

    # chimes (Hz)
    chime_wake_hz: int = 880
    chime_followup_hz: int = 660

    # brain
    # Which backend does the thinking (the *preferred* one; failover may
    # stand another in at runtime, see veronica.brain.switch).
    brain_backend: str = "codex"
    brain_failover: bool = True
    brain_failover_order: str = "codex,antigravity,claude,copilot"
    brain_limit_cooldown_min: int = 60
    brain_timeout_s: float = 60
    interrupt_drain_s: float = 3
    effort: str = "low"
    max_turns: int | None = None
    # A Claude session is resumed until it reaches this age; the CLI replays
    # the whole transcript on resume, so an old one gets slow. 0 = never retire.
    brain_session_max_age_h: int = 48
    brain_cwd: Path = Field(default_factory=Path.home)
    # Seconds after one approved screen action during which further
    # confirm-class computer actions in the SAME app are auto-allowed. 0 = off.
    computer_trust_s: int = 90
    # "Copy this, just do it": a request whose wording already says go
    # ahead skips the yes/no for the ONE confirm-class action it produces
    # (never for always-confirm tools, see policy.always_confirm).
    preapprove_by_wording: bool = True
    # Shortcuts the user has marked safe: `run_shortcut` runs these without
    # asking. Everything else is confirm-class, so an empty list (the
    # default) means every shortcut is asked about.
    shortcut_allowlist: list[str] = Field(default_factory=list)
    # Confirm-class tools the user has approved for good, by ticking them in
    # Settings or answering a confirm with "always". Only the names in
    # policy.AUTO_ALLOWABLE take effect: anything else here is ignored, so a
    # hand-typed mail_send still asks every single time.
    auto_allow_tools: list[str] = Field(default_factory=lambda: ["mcp__mac__clipboard_write"])
    # External brains: may the vendor CLI use its own shell/file tools
    # (each call still asked through Veronica's hook)? Off = only our
    # MCP tools. Flipped off automatically when the hook canary trips.
    codex_native_tools: bool = True
    antigravity_native_tools: bool = True
    copilot_native_tools: bool = True
    # Offline brain (veronica.brain.backends.local): a llama.cpp server on
    # this Mac. Started lazily, on a port of ours, and never asked to reach
    # the network.
    local_server_bin: Path = Field(default_factory=lambda: LOCAL_SERVER_BIN)
    local_model: Path = Field(default_factory=lambda: LOCAL_MODEL)
    local_ctx: int = 8192
    local_port: int = 8749
    # Stand the local brain in when the active brain needs the network and
    # there isn't any (see veronica.net / BrainSwitcher.maybe_offline).
    brain_offline_fallback: bool = True

    # memory
    memory_enabled: bool = True
    memory_recent_turns: int = 6
    # How many remembered facts the system prompt carries, most-recently-used
    # first; the block is still byte-capped on top of this (brain/prompts.py).
    memory_facts_max: int = 40

    # push-to-talk
    ptt_enabled: bool = True
    ptt_keycode: int = 61   # Right Option
    ptt_max_s: int = 30     # hard cap on one held capture (onset wait + recording)

    # dictation
    dictation_max_s: int = 60

    # HUD
    hud_enabled: bool = True
    hud_hide_after_s: float = 3.0
    hud_width: int = 540
    hud_height: int = 300
    hud_margin: int = 24
    hud_mode: str = "full"    # "full" | "mini"; runtime pref, see veronica.prefs
    hud_mini_width: int = 400
    hud_mini_height: int = 72
    hud_particles: int = 4000   # orb particle count (live-editable)
    hud_intensity: float = 1.0  # orb glow/brightness multiplier (live-editable)

    @field_validator("brain_backend")
    @classmethod
    def _known_backend(cls, v: str) -> str:
        if v not in BRAIN_BACKENDS:
            raise ValueError(f"brain_backend must be one of {', '.join(BRAIN_BACKENDS)}; got {v!r}")
        return v

    @property
    def session_file(self) -> Path:
        return self.home / "session"

    @property
    def log_file(self) -> Path:
        return self.home / "logs" / "veronica.log"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def memory_path(self) -> Path:
        return self.home / "memory.db"

    @property
    def gate_socket(self) -> Path:
        """Unix socket the external brains' processes ask for tool permission on."""
        return self.home / "gate.sock"

    def backend_dir(self, name: str) -> Path:
        """Per-brain workspace (session id, hook config, hook log); private."""
        d = self.home / "backends" / name
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        return d

    def session_file_for(self, name: str) -> Path:
        return self.session_file if name == "claude" else self.backend_dir(name) / "session"

    def ensure_dirs(self) -> None:
        (self.home / "logs").mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class EditableField:
    kind: Literal["bool", "int", "float", "str", "choice", "list"]
    label: str
    help: str = ""
    choices: tuple[str, ...] | None = None
    min: float | None = None
    max: float | None = None
    restart: bool = True


EDITABLE_SETTINGS: dict[str, EditableField] = {
    "followup_window_s": EditableField(
        "int", "Follow-up window (seconds)", "How long she keeps listening after answering.",
        min=1, max=15, restart=False,
    ),
    "confirm_listen_s": EditableField(
        "int", "Confirmation timeout (seconds)", "How long she waits for yes/no.",
        min=3, max=30, restart=False,
    ),
    "ack_after_s": EditableField(
        "float", "\"On it\" after (seconds)",
        "How long a slow answer may stay silent before she says she's on it. 0 = never.",
        min=0, max=15, restart=False,
    ),
    "vad_silence_ms": EditableField(
        "int", "End-of-speech silence (ms)",
        "How long you can pause before Veronica decides you're done talking. "
        "Raise if she cuts you off mid-sentence.",
        min=300, max=3000, restart=False,
    ),
    "max_utterance_s": EditableField(
        "int", "Max utterance length (seconds)", "Hard cap on one spoken command.",
        min=5, max=60, restart=False,
    ),
    "hud_hide_after_s": EditableField(
        "float", "Hide HUD after (seconds)", "", min=1, max=30, restart=False,
    ),
    "hud_particles": EditableField(
        "int", "HUD particles", "More looks richer, costs CPU.", min=500, max=8000, restart=False,
    ),
    "hud_intensity": EditableField(
        "float", "HUD intensity", "Glow/brightness multiplier.", min=0.2, max=2.0, restart=False,
    ),
    "noise_suppression": EditableField(
        "bool", "Reduce background noise",
        "Keeps clatter, music and fan noise from being taken for speech.", restart=False,
    ),
    "vad_min_rms": EditableField(
        "float", "Speech level floor",
        "How loud (after noise reduction) a sound must be to count as speech; only applies while noise "
        "reduction is on. Raise if room noise keeps her listening; lower if she misses you from across the "
        "room. 0 = off.",
        min=0.0, max=0.01, restart=False,
    ),
    "speaker_verification": EditableField(
        "bool", "Only listen to my voice",
        "After \"learn my voice\", she ignores requests and yes/no answers in other voices.", restart=False,
    ),
    "speaker_threshold": EditableField(
        "float", "Voice match strictness",
        "Higher ignores more of other people and may miss you in noise. Recent scores are listed below.",
        min=0.2, max=0.7, restart=False,
    ),
    "speaker_verification_wake": EditableField(
        "bool", "Only wake for my voice",
        "The wake word must be your voice too. Off by default: a missed wake is worse than a stray one. "
        "While it's off, someone else saying \"Veronica\" can still interrupt her.",
        restart=False,
    ),
    "wake_min_rms": EditableField(
        "float", "Wake sensitivity (min level)",
        "Lower = hears you from farther away, more false wakes.", min=0.001, max=0.05, restart=True,
    ),
    "wake_window_s": EditableField(
        "float", "Wake window (seconds)", "How much audio each wake check listens to.",
        min=0.8, max=2.5, restart=True,
    ),
    "wake_hop_s": EditableField(
        "float", "Wake hop (seconds)", "How often the wake check runs; lower = faster, more CPU.",
        min=0.15, max=0.6, restart=True,
    ),
    "wake_phrases": EditableField("list", "Wake phrases", "Comma-separated; 'veronica' is recommended."),
    "input_volume_floor": EditableField(
        "int", "Input volume floor",
        "Raise the Mac's input volume back to this when a call app or device switch lowers it. 0 = off.",
        min=0, max=100, restart=False,
    ),
    "ptt_enabled": EditableField("bool", "Push-to-talk (hold Right Option)"),
    "effort": EditableField("choice", "Brain effort", "Higher is smarter and slower.",
                             choices=("low", "medium", "high")),
    "memory_enabled": EditableField("bool", "Remember conversations"),
    "memory_facts_max": EditableField(
        "int", "Facts she carries into a new conversation",
        "The most recently used facts go first; the rest stay in memory and still come back "
        "via her memory tools. 0 = none.",
        min=0, max=200, restart=False,
    ),
    "brain_cwd": EditableField("str", "Working folder", "Where shell commands run."),
    "brain_session_max_age_h": EditableField(
        "int", "Start a fresh conversation after (hours)",
        "A long-running conversation gets slower to resume. 0 = keep it forever.",
        min=0, max=720, restart=False),
    "computer_trust_s": EditableField(
        "int", "Screen-control trust window (seconds)",
        "After you approve one click/type, further screen actions in the same app are allowed "
        "for this long. 0 = ask every time. Never covers pressing Enter, terminals or system dialogs.",
        min=0, max=600, restart=False,
    ),
    "preapprove_by_wording": EditableField(
        "bool", 'Pre-approve when I say "do it"',
        "If your request already says do it / go ahead, skip the yes/no for that one action "
        "(never for sending mail, deleting, shutdown, or Enter).",
        restart=False,
    ),
    "shortcut_allowlist": EditableField(
        "list", "Shortcuts she may run without asking",
        "Comma-separated shortcut names, exactly as they're named in Shortcuts. "
        "Anything not listed still asks first.",
        restart=False,
    ),
    "auto_allow_tools": EditableField(
        "list", "Tools she may use without asking",
        "Ticked above. Comma-separated tool names; clear one to start asking again. "
        "Only the tools listed here can ever be added — sending, screen control and the shell always ask.",
        restart=False,
    ),
    "brain_backend": EditableField(
        "choice", "Brain", "Which assistant runs the thinking. Each uses its own login.",
        choices=BRAIN_BACKENDS, restart=False,
    ),
    "brain_failover": EditableField(
        "bool", "Switch brains on usage limits",
        "When the current brain hits its usage limit, hand the request to the next available one "
        "and come back later.",
        restart=False,
    ),
    "brain_failover_order": EditableField(
        "str", "Failover order", "Comma-separated backend names, tried in order.", restart=False,
    ),
    "brain_limit_cooldown_min": EditableField(
        "int", "Limit cooldown (minutes)", "How long to wait before trying a brain that hit its limit again.",
        min=5, max=1440, restart=False,
    ),
    "codex_native_tools": EditableField(
        "bool", "Codex: allow its own shell",
        "Off = only Veronica's tools; on = its shell and file edits too, each asked through Veronica.",
        restart=False,
    ),
    "antigravity_native_tools": EditableField(
        "bool", "Antigravity: allow its own shell",
        "Off = only Veronica's tools; on = its shell and file edits too, each asked through Veronica.",
        restart=False,
    ),
    "copilot_native_tools": EditableField(
        "bool", "Copilot: allow its own shell",
        "Off = only Veronica's tools; on = its shell and file edits too, each asked through Veronica.",
        restart=False,
    ),
    "brain_offline_fallback": EditableField(
        "bool", "Use the local model when offline",
        "No internet? Answer on the model running on this Mac, and go back when it returns.",
        restart=False,
    ),
    "local_server_bin": EditableField(
        "str", "Local server", "Path to llama-server.", restart=False,
    ),
    "local_model": EditableField(
        "str", "Local model", "Path to the .gguf weights she thinks with offline.", restart=False,
    ),
    "local_ctx": EditableField(
        "int", "Local context (tokens)", "Bigger remembers more and loads slower.",
        min=1024, max=131072, restart=False,
    ),
    "local_port": EditableField(
        "int", "Local port", "Where the local server listens, on this Mac only.",
        min=1024, max=65535, restart=False,
    ),
}


def coerce_setting(name: str, value: Any) -> Any:
    """Coerce a raw (e.g. UI-supplied) value to what Settings expects for
    field `name`: clamps numbers to their min/max, splits/strips comma
    lists, validates choice membership. Raises ValueError on bad input."""
    if name not in EDITABLE_SETTINGS:
        raise ValueError(f"{name!r} is not an editable setting")
    field = EDITABLE_SETTINGS[name]

    if field.kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    if field.kind == "int":
        v = int(value)
        if field.min is not None:
            v = max(v, int(field.min))
        if field.max is not None:
            v = min(v, int(field.max))
        return v

    if field.kind == "float":
        v = float(value)
        if field.min is not None:
            v = max(v, field.min)
        if field.max is not None:
            v = min(v, field.max)
        return v

    if field.kind == "choice":
        s = str(value)
        if not field.choices or s not in field.choices:
            raise ValueError(f"{value!r} is not a valid choice for {name!r} ({field.choices})")
        return s

    if field.kind == "list":
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return list(value)

    # kind == "str"
    return str(value)


def load_settings(overrides: dict | None = None) -> Settings:
    """Build a Settings instance from env/.env defaults with `overrides`
    (a dict of field name -> value, as loaded from prefs.json's "settings"
    key) applied on top. Unknown keys and values that fail validation are
    dropped (logged, not raised) so a corrupt prefs.json never blocks
    startup. A validation error on a non-override field (env/.env) is
    re-raised: nothing here can fix it."""
    overrides = overrides or {}
    filtered = {k: v for k, v in overrides.items() if k in EDITABLE_SETTINGS}
    for k in overrides:
        if k not in EDITABLE_SETTINGS:
            log.warning("ignoring unknown settings override %r", k)

    # Bounded: each iteration drops at least one override, so at most
    # len(filtered) + 1 attempts. A validation error on a field that is NOT
    # an override (e.g. VERONICA_SAMPLE_RATE=abc in the env) can't be fixed
    # here and is re-raised as a clean startup crash.
    for _ in range(len(filtered) + 1):
        try:
            return Settings(**filtered)
        except ValidationError as e:
            bad_fields = {err["loc"][0] for err in e.errors() if err.get("loc")}
            bad = bad_fields & set(filtered)
            if not bad:
                raise
            for f in bad:
                log.warning("ignoring invalid settings override %r=%r", f, filtered.get(f))
                filtered.pop(f, None)
    return Settings(**filtered)


settings = load_settings(prefs.load().get("settings"))


def log_level_from_env(default: int = logging.INFO) -> int:
    """Log level named by VERONICA_LOG_LEVEL (e.g. DEBUG), or `default` if
    unset/unrecognised. DEBUG turns on per-hop wake rms lines and the like."""
    name = os.environ.get("VERONICA_LOG_LEVEL", "").strip().upper()
    if not name:
        return default
    level = logging.getLevelName(name)
    if not isinstance(level, int):
        log.warning("ignoring unknown VERONICA_LOG_LEVEL=%r", name)
        return default
    return level


def setup_logging(level: int | None = None) -> logging.Logger:
    settings.ensure_dirs()
    log = logging.getLogger("veronica")
    if log.handlers:
        return log
    log.setLevel(level if level is not None else log_level_from_env())
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = RotatingFileHandler(settings.log_file, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    log.addHandler(fh)
    # When launched from the .app bundle (no controlling terminal), stderr
    # (logging.StreamHandler's default stream) isn't a TTY: skip the
    # StreamHandler so nothing tries to write to a closed/redirected stream,
    # and rely on the log file alone.
    if sys.stderr is not None and sys.stderr.isatty():
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(sh)
    return log
