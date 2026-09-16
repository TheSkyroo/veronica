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
}
VOICE_IDS: list[str] = list(VOICES.values())

DEFAULT_VOICE = "af_sarah"
DEFAULT_SPEED = 1.0
SPEED_STEP = 0.15
SPEED_MIN = 0.7
SPEED_MAX = 1.5

_GENDER = {"male": "m", "man": "m", "guy": "m", "female": "f", "woman": "f", "lady": "f"}
_ACCENT = {"british": "b", "english": "b", "uk": "b", "american": "a", "us": "a"}
_DEFAULT_WORDS = {"default", "normal", "usual", "original"}


def resolve_voice(request: str) -> str | None:
    """Map a spoken request to a voice id: an exact name ("adam"), a
    descriptor combo ("british male", "female"), or "default". Descriptors
    pick the first table entry whose id prefix matches; unknown → None."""
    words = request.lower().replace("-", " ").split()
    if not words:
        return None
    if len(words) == 1 and words[0] in VOICES:
        return VOICES[words[0]]
    if any(w in _DEFAULT_WORDS for w in words):
        return DEFAULT_VOICE
    accent = next((_ACCENT[w] for w in words if w in _ACCENT), None)
    gender = next((_GENDER[w] for w in words if w in _GENDER), None)
    if accent is None and gender is None:
        return None
    for vid in VOICE_IDS:
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
