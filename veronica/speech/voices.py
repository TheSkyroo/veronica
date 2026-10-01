"""Kokoro voice table and the spoken-request → voice-id resolver behind
"use a British male voice" / "switch to Adam"."""

# spoken name -> Kokoro voice id (all shipped in voices-v1.0.bin).
# Prefix: a=American, b=British; f=female, m=male.
VOICES: dict[str, str] = {
    "sarah": "af_sarah",
    "bella": "af_bella",
    "nicole": "af_nicole",
    "sky": "af_sky",
    "adam": "am_adam",
    "michael": "am_michael",
    "emma": "bf_emma",
    "isabella": "bf_isabella",
    "george": "bm_george",
    "lewis": "bm_lewis",
    # Kokoro's top-graded English voice and the most natural by predicted MOS
    # (docs/superpowers/specs/2026-10-01-veronica-voice-model-options.md) —
    # the default. Last, so "an American woman" still means Sarah.
    "heart": "af_heart",
}
VOICE_IDS: list[str] = list(VOICES.values())

# Hindi voices (hf_=Hindi female, hm_=Hindi male), shipped in the same bin.
HINDI_VOICES: dict[str, str] = {
    "alpha": "hf_alpha",
    "beta": "hf_beta",
    "omega": "hm_omega",
    "psi": "hm_psi",
}
HINDI_VOICE_IDS: list[str] = list(HINDI_VOICES.values())
# hf_beta: lower and steadier than hf_alpha (median 186 Hz vs 214 Hz), which
# users heard as a child's voice, and the most intelligible in the
# read-back test (CER 0.02 vs 0.05).
DEFAULT_HINDI_VOICE = "hf_beta"

ALL_VOICE_IDS: list[str] = VOICE_IDS + HINDI_VOICE_IDS

DEFAULT_VOICE = "af_sarah"
DEFAULT_SPEED = 1.0
SPEED_STEP = 0.15
SPEED_MIN = 0.7
SPEED_MAX = 1.5

_GENDER = {"male": "m", "man": "m", "guy": "m", "female": "f", "woman": "f", "lady": "f"}
_ACCENT = {
    "british": "b", "english": "b", "uk": "b", "american": "a", "us": "a",
    "hindi": "h", "indian": "h",
}
_DEFAULT_WORDS = {"default", "normal", "usual", "original"}


def is_hindi_voice(voice_id: str) -> bool:
    return voice_id in HINDI_VOICE_IDS


def resolve_voice(request: str) -> str | None:
    """Map a spoken request to a voice id: an exact name ("adam"), a
    descriptor combo ("british male", "female", "hindi male"), or "default".
    Descriptors pick the first table entry whose id prefix matches; unknown
    → None."""
    words = request.lower().replace("-", " ").split()
    if not words:
        return None
    if len(words) == 1 and words[0] in VOICES:
        return VOICES[words[0]]
    if len(words) == 1 and words[0] in HINDI_VOICES:
        return HINDI_VOICES[words[0]]
    if any(w in _DEFAULT_WORDS for w in words):
        return DEFAULT_VOICE
    accent = next((_ACCENT[w] for w in words if w in _ACCENT), None)
    gender = next((_GENDER[w] for w in words if w in _GENDER), None)
    if accent is None and gender is None:
        return None
    for vid in ALL_VOICE_IDS:
        if (accent is None or vid[0] == accent) and (gender is None or vid[1] == gender):
            return vid
    return None


def display_name(voice_id: str) -> str:
    return voice_id.split("_", 1)[-1].capitalize()


def next_voice(current: str) -> str:
    if current not in VOICE_IDS:
        return VOICE_IDS[0]
    return VOICE_IDS[(VOICE_IDS.index(current) + 1) % len(VOICE_IDS)]


def clamp_speed(x: float) -> float:
    return max(SPEED_MIN, min(SPEED_MAX, float(x)))
