"""Local voice intents: phrases handled entirely in the orchestrator, without
ever going to the brain (end the conversation, resize/hide the HUD)."""
import re
import string
from typing import Literal

Intent = Literal["end", "hud_mini", "hud_full", "hud_hide", "hud_reset", "mute", "unmute", "quit"]

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, drop ASCII punctuation and the Devanagari danda (।).
    Devanagari letters survive (string.punctuation is ASCII-only), so the
    Hindi-script phrases in the tables below match in pinned Hindi mode."""
    return (text or "").lower().replace("’", "").replace("।", "").translate(_PUNCT_TABLE).strip()


END_PHRASES = frozenset({
    "thanks veronica", "thank you veronica", "that's all", "thats all",
    "that is all", "stop", "goodbye", "never mind", "nevermind",
    "go idle", "turn yourself off", "go to sleep", "sleep", "go away",
    "bye", "dismiss",
    # Hinglish (no bare "bas": too common mid-sentence, "bas ek minute")
    "bas karo", "theek hai bas", "chup", "chup raho",
    "band karo", "ruko", "ruk jao",
    # Devanagari
    "बस", "बस करो", "चुप", "रुको", "बंद करो",
})

HUD_MINI_PHRASES = frozenset({
    "make yourself small", "make yourself smaller", "shrink", "shrink yourself",
    "minimize", "minimise", "mini mode", "small mode", "go small",
    # Hinglish
    "chhoti ho jao", "chota karo",
})

HUD_FULL_PHRASES = frozenset({
    "expand", "expand yourself", "make yourself big", "make yourself bigger",
    "full mode", "show details", "go big",
    # Hinglish
    "badi ho jao", "bada karo",
})

HUD_HIDE_PHRASES = frozenset({
    "hide", "hide yourself", "hide the hud", "hide the panel",
})

# "The HUD vanished" (typically after a monitor change left it on a
# display that's gone): forget the saved position and show it at the
# default spot on the main screen.
HUD_RESET_PHRASES = frozenset({
    "where are you", "show yourself", "come back",
    "reset the hud", "reset hud", "hud reset",
    # Hinglish
    "kahan ho", "wapas aao",
    # Devanagari
    "कहाँ हो", "वापस आओ",
})

MUTE_PHRASES = frozenset({
    "mute", "mute yourself", "be quiet", "silence",
    # Hinglish
    "mute karo", "awaaz band karo",
    # Devanagari
    "म्यूट करो", "आवाज़ बंद करो",
})

UNMUTE_PHRASES = frozenset({
    "unmute", "unmute yourself", "you can talk", "speak again",
    # Hinglish
    "unmute karo", "awaaz chalu karo",
    # Devanagari
    "अनम्यूट करो", "आवाज़ चालू करो",
})

QUIT_PHRASES = frozenset({
    "quit", "quit veronica", "shut down", "shut yourself down", "exit", "turn off completely",
    # Hinglish
    "quit karo", "band ho jao",
})

# Screen-awareness fast path: matched exactly like the other local intents
# (whole-utterance, then clause-by-clause), but kept separate from
# match_intent's Intent enum since it doesn't end the turn — it feeds the
# brain a screenshot instead of skipping it.
SCREEN_PHRASES = frozenset({
    "whats on my screen", "what is on my screen", "what's on my screen",
    "look at my screen", "look at the screen",
    "summarize this page", "summarize this screen", "summarize my screen",
    "what does this error say", "what does this say",
})

# Language-switch intent (B1-adjacent): "speak hindi" / "switch to
# english" / "understand both". Carries a payload (the requested
# LanguageMode) so it has its own function, like match_voice_intent.
LanguageMode = Literal["en", "hi", "auto"]

_LANG_PHRASES: dict[str, LanguageMode] = {
    "speak hindi": "hi", "talk in hindi": "hi", "switch to hindi": "hi", "hindi mein bolo": "hi",
    "hindi me bolo": "hi", "hindi mein baat karo": "hi", "hindi me baat karo": "hi", "speak in hindi": "hi",
    "speak english": "en", "talk in english": "en", "switch to english": "en", "english mein bolo": "en",
    "english me bolo": "en", "angrezi mein bolo": "en", "speak in english": "en",
    "understand both": "auto", "both languages": "auto", "hindi and english": "auto", "hindi aur english": "auto",
    "auto language": "auto", "dono bhasha": "auto",
}

# Hinglish keys of _LANG_PHRASES (the rest are English phrasings).
_LANG_PHRASES_HINGLISH = frozenset({
    "hindi mein bolo", "hindi me bolo", "hindi mein baat karo", "hindi me baat karo",
    "english mein bolo", "english me bolo", "angrezi mein bolo",
    "hindi aur english", "dono bhasha",
})


def match_language_intent(text: str) -> LanguageMode | None:
    """Match a heard utterance against the language-switch phrase table
    (see _LANG_PHRASES). Whole-utterance candidates only — no clause
    split, so e.g. "translate this to hindi" stays with the brain."""
    for candidate in _candidates_for(normalize(text)):
        if candidate in _LANG_PHRASES:
            return _LANG_PHRASES[candidate]
    return None


# Settings window / history tab / version / self-update (Batch D). All
# whole-utterance only (no clause split): "open safari settings", "history
# of rome" and "update my calendar" must stay with the brain.
SettingsTab = Literal["general", "history"]

_SETTINGS_PHRASES: dict[str, SettingsTab] = {
    "open settings": "general", "show settings": "general", "settings": "general",
    "preferences": "general", "open preferences": "general", "open the settings": "general",
    "settings kholo": "general", "setting kholo": "general",
    "show history": "history", "show my history": "history", "what did i ask you": "history",
    "what did i ask you earlier": "history", "history": "history", "history dikhao": "history",
    "conversation history": "history", "show conversation history": "history",
}
_SETTINGS_PHRASES_HINGLISH = frozenset({"settings kholo", "setting kholo", "history dikhao"})

VERSION_PHRASES = frozenset({
    "what version are you", "which version are you", "what version", "version",
    "your version", "whats your version", "what's your version", "which version",
    "kaunsa version hai",
})
_VERSION_PHRASES_HINGLISH = frozenset({"kaunsa version hai"})

UPDATE_PHRASES = frozenset({
    "update yourself", "update now", "check for updates", "check for an update",
    "check for update", "apna update karo", "update karo",
})
_UPDATE_PHRASES_HINGLISH = frozenset({"apna update karo", "update karo"})


def match_settings_intent(text: str) -> SettingsTab | None:
    """"open settings" / "settings kholo" -> "general"; "show history" /
    "what did i ask you" / "history dikhao" -> "history". Whole-utterance
    candidates only, so "open safari settings" is None."""
    for candidate in _candidates_for(normalize(text)):
        if candidate in _SETTINGS_PHRASES:
            return _SETTINGS_PHRASES[candidate]
    return None


def match_version_intent(text: str) -> bool:
    """True for "what version are you" / "version" / "kaunsa version hai"
    (whole utterance only: "what version of python is installed" is False)."""
    return any(c in VERSION_PHRASES for c in _candidates_for(normalize(text)))


def match_update_intent(text: str) -> bool:
    """True for "update yourself" / "check for updates" / "apna update karo"
    (whole utterance only: "update my calendar" is False)."""
    return any(c in UPDATE_PHRASES for c in _candidates_for(normalize(text)))


# Hinglish "which brain" phrases (the switch forms are regexes, see
# match_brain_intent below).
_WHICH_BRAIN_PHRASES_HINGLISH = frozenset({
    "kaunsa brain hai", "kaun sa brain hai", "kaunsa brain chal raha hai", "kaunsa model hai",
    "kaun sa model hai", "kaunsa brain use kar rahi ho",
})

# Romanized-Hindi (Hinglish) forms of the local intents above. Filled in by
# the Hinglish intents work; quick.is_hinglish_phrase() unions this with its
# own phrase tables so a whole-utterance Hinglish command is treated as
# Hindi even when the transcriber labels it "en".
HINGLISH_INTENT_PHRASES: frozenset[str] = frozenset({
    "bas karo", "theek hai bas", "chup", "chup raho",
    "band karo", "ruko", "ruk jao",
    "mute karo", "awaaz band karo", "unmute karo", "awaaz chalu karo",
    "chhoti ho jao", "chota karo", "badi ho jao", "bada karo",
    "kahan ho", "wapas aao",
    "quit karo", "band ho jao",
}) | _LANG_PHRASES_HINGLISH | _SETTINGS_PHRASES_HINGLISH | _VERSION_PHRASES_HINGLISH | _UPDATE_PHRASES_HINGLISH \
    | _WHICH_BRAIN_PHRASES_HINGLISH

_LEAD_PREFIXES = ("hey veronica ", "veronica ")
_TRAIL_SUFFIX = " please"

# Leading filler words/phrases stripped before matching a clause — spoken
# hedging ("I think...") or politeness ("please...") that shouldn't stop an
# otherwise-exact phrase from matching. "okay"/"ok" are included so e.g.
# "Okay stop." still ends the conversation.
FILLERS = frozenset({"i think", "please", "can you", "could you", "just", "okay", "ok"})
_FILLERS_BY_LEN = tuple(sorted(FILLERS, key=len, reverse=True))

# Clauses are split on sentence/list punctuation, so a single utterance
# carrying multiple thoughts ("Make yourself small. I can't see you.") can
# still match on its first clause. Deliberately NOT split on " and " --
# that swallowed compound phrases like "hide and seek" into a false-positive
# "hide" match.
_CLAUSE_SPLIT_RE = re.compile(r"[.,!?;।]+")  # danda = Hindi full stop


def _strip_wrapper(norm: str) -> str:
    """Strip an optional leading "veronica"/"hey veronica" and an optional
    trailing "please" from an already-normalized utterance."""
    for prefix in _LEAD_PREFIXES:
        if norm.startswith(prefix):
            norm = norm[len(prefix):]
            break
    if norm.endswith(_TRAIL_SUFFIX):
        norm = norm[: -len(_TRAIL_SUFFIX)]
    return norm.strip()


def _strip_leading_filler(norm: str) -> str:
    """Strip a single leading filler word/phrase (see FILLERS) from an
    already-normalized string."""
    for filler in _FILLERS_BY_LEN:
        if norm == filler:
            return ""
        if norm.startswith(filler + " "):
            return norm[len(filler) + 1:].strip()
    return norm


def _candidates_for(norm: str):
    stripped = _strip_wrapper(norm)
    seen = set()
    for candidate in (norm, stripped, _strip_leading_filler(norm), _strip_leading_filler(stripped)):
        if candidate and candidate not in seen:
            seen.add(candidate)
            yield candidate


def _match_candidate(candidate: str) -> Intent | None:
    if candidate in END_PHRASES:
        return "end"
    if candidate in HUD_MINI_PHRASES:
        return "hud_mini"
    if candidate in HUD_FULL_PHRASES:
        return "hud_full"
    if candidate in HUD_HIDE_PHRASES:
        return "hud_hide"
    if candidate in HUD_RESET_PHRASES:
        return "hud_reset"
    if candidate in MUTE_PHRASES:
        return "mute"
    if candidate in UNMUTE_PHRASES:
        return "unmute"
    if candidate in QUIT_PHRASES:
        return "quit"
    return None


_MEMORY_LEAD_RE = re.compile(r"^(?:hey\s+veronica|veronica)[,\s]+", re.IGNORECASE)
_REMEMBER_RE = re.compile(r"^remember\s+(?:that\s+)?(.+)$", re.IGNORECASE)
_FORGET_RE = re.compile(r"^forget\s+(?:that\s+)?(.+)$", re.IGNORECASE)

# "remember when/what/how/if/why ..." reads as a recall question ("remember
# when we went to Paris?"), not a fact to store — fall through to Claude
# instead of stashing the literal question text as a fact.
_REMEMBER_QUESTION_LEADS = frozenset({"when", "what", "how", "if", "why"})


def match_memory_intent(text: str) -> tuple[str, str] | None:
    """Match "remember (that) X" / "forget (that) X" against a heard
    utterance. Unlike match_intent (exact-phrase matching against a
    normalized string), this carries a payload, so it preserves the
    original casing/punctuation of X rather than normalizing it — only an
    optional leading "veronica"/"hey veronica" and a trailing sentence-ending
    period are stripped. Returns ("remember", X) or ("forget", X), or None
    if the utterance doesn't start with "remember"/"forget"."""
    raw = (text or "").strip()
    raw = _MEMORY_LEAD_RE.sub("", raw, count=1).strip()
    for kind, pattern in (("remember", _REMEMBER_RE), ("forget", _FORGET_RE)):
        m = pattern.match(raw)
        if m:
            arg = m.group(1).strip().rstrip(".!?").strip()
            if not arg or arg.lower() == "that":
                return None
            if kind == "remember" and arg.split()[0].lower() in _REMEMBER_QUESTION_LEADS:
                return None
            return (kind, arg)
    return None


# Music playback fast path: matched exactly like the other local intents.
# Kept as its own function (rather than folded into Intent) since it carries
# no payload beyond which action to take, and never touches the brain.
MusicAction = Literal["play", "pause", "next", "prev", "now_playing"]

_MUSIC_PLAY_PHRASES = frozenset({"resume", "resume music", "play music", "unpause", "unpause music"})
_MUSIC_PAUSE_PHRASES = frozenset({"pause", "pause music", "stop the music", "stop music"})
_MUSIC_NEXT_PHRASES = frozenset({"next song", "next track", "skip", "skip song", "skip track"})
# "go back" deliberately absent: far too generic (navigation, undo, "go
# back to what you were saying") to hijack as a music command.
_MUSIC_PREV_PHRASES = frozenset({"previous song", "previous track", "previous", "last song"})
_MUSIC_NOW_PLAYING_PHRASES = frozenset({
    "whats playing", "what is playing", "what's playing",
    "what song is this", "what song is playing", "whats this song",
})


def _match_music_candidate(candidate: str) -> MusicAction | None:
    if candidate in _MUSIC_PLAY_PHRASES:
        return "play"
    if candidate in _MUSIC_PAUSE_PHRASES:
        return "pause"
    if candidate in _MUSIC_NEXT_PHRASES:
        return "next"
    if candidate in _MUSIC_PREV_PHRASES:
        return "prev"
    if candidate in _MUSIC_NOW_PLAYING_PHRASES:
        return "now_playing"
    return None


def match_music_intent(text: str) -> MusicAction | None:
    """Match a heard utterance against the music-control phrase sets (see
    A3): "pause"/"pause music", "resume"/"play music", "next song"/"skip",
    "previous"/"previous song", "what's playing". Matched the same way as
    match_intent (whole utterance, then each clause)."""
    norm_whole = normalize(text)
    for candidate in _candidates_for(norm_whole):
        result = _match_music_candidate(candidate)
        if result is not None:
            return result
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        clause_norm = normalize(clause)
        if not clause_norm:
            continue
        for candidate in _candidates_for(clause_norm):
            result = _match_music_candidate(candidate)
            if result is not None:
                return result
    return None


# Voice / speed fast path (B1): "use a british voice", "switch to adam
# voice", "change your voice", "speak faster", "normal speed". Carries a
# payload (the requested voice descriptor, or which way to nudge speed) so
# it has its own function like match_memory_intent; resolving the
# descriptor to an actual Kokoro voice id is veronica.speech.voices' job.
VoiceAction = tuple[Literal["voice", "speed"], str]

_VOICE_PICK_RE = re.compile(
    r"^(?:use|switch to|change to|speak (?:in|with))\s+(?:a |an |the )?(.+?)\s+voice$"
)
_ARTICLES = frozenset({"a", "an", "the"})
_VOICE_NEXT_PHRASES = frozenset({
    "change your voice", "different voice", "use a different voice",
    "change voice", "another voice", "use another voice",
})
# Bare "faster"/"slower" are here because normalize()+_strip_wrapper turn
# "faster please" into "faster".
_SPEED_PHRASES: dict[str, str] = {
    "speak faster": "faster", "talk faster": "faster", "faster please": "faster",
    "faster": "faster", "speed up": "faster", "speak quicker": "faster",
    "speak slower": "slower", "talk slower": "slower", "slower please": "slower",
    "slower": "slower", "slow down": "slower",
    "normal speed": "normal", "default speed": "normal", "reset speed": "normal",
    "reset your speed": "normal", "speak normally": "normal",
}


def _match_voice_candidate(candidate: str) -> VoiceAction | None:
    if candidate in _VOICE_NEXT_PHRASES:
        return ("voice", "next")
    if candidate in _SPEED_PHRASES:
        return ("speed", _SPEED_PHRASES[candidate])
    m = _VOICE_PICK_RE.match(candidate)
    if m:
        req = m.group(1).strip()
        # "use a voice": the optional-article group backtracks so the
        # descriptor is just the article — no voice was actually named.
        if not req or req in _ARTICLES:
            return None
        if req in ("different", "another"):
            return ("voice", "next")
        return ("voice", req)
    return None


def match_voice_intent(text: str) -> VoiceAction | None:
    """"use a british voice" / "switch to adam voice" / "speak faster" ...
    Same candidate strategy as match_intent: whole normalized utterance,
    then each clause. Returns ("voice", <descriptor or "next">) or
    ("speed", "faster"|"slower"|"normal"), or None."""
    for candidate in _candidates_for(normalize(text)):
        result = _match_voice_candidate(candidate)
        if result is not None:
            return result
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        clause_norm = normalize(clause)
        if not clause_norm:
            continue
        for candidate in _candidates_for(clause_norm):
            result = _match_voice_candidate(candidate)
            if result is not None:
                return result
    return None


# Proactive briefings & nudges (B2): "brief me", "give me a briefing every
# morning at 8", "stop the morning briefing", "warn me 10 minutes before my
# meetings", "turn off nudges". Carries a payload (the briefing time as
# "HH:MM", or the nudge lead in minutes) so it has its own function.
ProactiveAction = tuple[
    Literal["brief_now", "briefing_on", "briefing_off", "nudges_on", "nudges_off"], str | int | None
]

_BRIEF_NOW_PHRASES = frozenset({
    "brief me", "give me a briefing", "give me my briefing", "morning briefing", "my briefing",
    "whats my day look like", "what does my day look like", "what does my day look like today",
    "how does my day look", "whats my day like",
})
_BRIEFING_ON_RE = re.compile(
    r"^(?:give me|start|turn on|enable|set up)\s+(?:a |the |my )?(?:morning |daily )?briefings?"
    r"(?:\s+(?:every day|every morning|daily|each morning))?(?:\s+at\s+(.+))?$"
)
_BRIEFING_OFF_RE = re.compile(
    r"^(?:stop|turn off|cancel|disable)\s+(?:the |my )?(?:morning |daily )?briefings?$"
)
_NUDGES_ON_RE = re.compile(
    r"^(?:remind me|warn me|nudge me|tell me|alert me|turn on nudges|enable nudges)"
    r"(?:\s+(\d{1,2})\s+minutes?)?(?:\s+before\s+(?:my |the )?(?:meetings?|events?|calendar events?))?$"
)
_NUDGES_OFF_RE = re.compile(
    r"^(?:stop|turn off|disable|cancel)\s+(?:the |my )?"
    r"(?:meeting nudges|nudges|meeting reminders|reminders before (?:my )?meetings)$"
)
# "8", "8 am", "8:30", "7 30 am", "6:15 pm"
_CLOCK_RE = re.compile(r"^(\d{1,2})(?:[:\s](\d{2}))?\s*(am|pm)?$")
# normalize() strips the colon, so a time reaching the intent regexes looks
# like "730 am" / "1845": 3-4 digits, last two are the minutes.
_CLOCK_COMPACT_RE = re.compile(r"^(\d{1,2})(\d{2})\s*(am|pm)?$")


def parse_clock_time(s: str) -> str | None:
    """"8", "8 am", "8:30", "7 30 am", "730", "6 pm", "noon" -> "HH:MM"
    (24h) or None if it isn't a clock time."""
    s = (s or "").strip().lower().replace(".", "")
    if s == "noon":
        return "12:00"
    if s == "midnight":
        return "00:00"
    m = _CLOCK_RE.match(s) or _CLOCK_COMPACT_RE.match(s)
    if not m:
        return None
    h, mm, ap = int(m[1]), int(m[2] or 0), m[3]
    if mm > 59:
        return None
    if ap:
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if ap == "pm" else 0)
    elif not 0 <= h <= 23:
        return None
    return f"{h:02d}:{mm:02d}"


def _match_proactive_candidate(candidate: str) -> ProactiveAction | None:
    if candidate in _BRIEF_NOW_PHRASES:
        return ("brief_now", None)
    if _BRIEFING_OFF_RE.match(candidate):
        return ("briefing_off", None)
    m = _BRIEFING_ON_RE.match(candidate)
    if m:
        if m.group(1):
            t = parse_clock_time(m.group(1))
            return ("briefing_on", t) if t else None
        return ("briefing_on", None)
    if _NUDGES_OFF_RE.match(candidate):
        return ("nudges_off", None)
    m = _NUDGES_ON_RE.match(candidate)
    if m:
        # "remind me" alone (no "before meetings") is a reminder request, not nudges
        if "before" not in candidate and "nudges" not in candidate:
            return None
        return ("nudges_on", int(m.group(1)) if m.group(1) else None)
    return None


def match_proactive_intent(text: str) -> ProactiveAction | None:
    """"brief me" / "give me a briefing every morning at 8" / "stop the
    morning briefing" / "warn me 10 minutes before my meetings" / "turn off
    nudges". Same candidate strategy as match_intent: whole normalized
    utterance, then each clause. Returns (kind, payload) where payload is
    the "HH:MM" briefing time, the nudge lead in minutes, or None."""
    for candidate in _candidates_for(normalize(text)):
        result = _match_proactive_candidate(candidate)
        if result is not None:
            return result
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        clause_norm = normalize(clause)
        if not clause_norm:
            continue
        for candidate in _candidates_for(clause_norm):
            result = _match_proactive_candidate(candidate)
            if result is not None:
                return result
    return None


# Brains: "switch to codex" / "use copilot" / "back to claude" / "which
# brain are you on". Carries the backend name, so it has its own function;
# the orchestrator's BrainSwitcher does the actual switch (and says why it
# can't). Names are the BACKENDS keys — kept literal here so this module
# stays import-light.
BrainAction = tuple[Literal["switch", "which"], str | None]

_BRAIN_NAMES = r"(claude|codex|antigravity|copilot)"
_BRAIN_SWITCH_RE = re.compile(
    rf"^(?:switch(?: brains?)? to|use|change(?: brains?)? to|switch(?: the)? brain to)\s+(?:the )?{_BRAIN_NAMES}(?: brain)?$"
)
_BRAIN_BACK_RE = re.compile(rf"^(?:go )?back to {_BRAIN_NAMES}(?: brain)?$")
# Hinglish: "codex pe switch karo", "copilot use karo", "claude pe wapas jao".
_BRAIN_SWITCH_HINGLISH_RE = re.compile(
    rf"^{_BRAIN_NAMES}\s+(?:(?:pe|par)\s+(?:switch karo|jao|wapas jao)|use karo|chalao)$"
)
_WHICH_BRAIN_PHRASES = frozenset({
    "which brain are you on", "which brain are you using", "which brain is this", "which brain is it",
    "what brain are you on", "what brain are you using", "what brain is this",
    "which model are you on", "which model are you using", "which model is this", "what model are you using",
    "who am i talking to", "who is this", "which ai is this", "which ai are you",
})


def _match_brain_candidate(candidate: str) -> BrainAction | None:
    if candidate in _WHICH_BRAIN_PHRASES or candidate in _WHICH_BRAIN_PHRASES_HINGLISH:
        return ("which", None)
    for pattern in (_BRAIN_SWITCH_RE, _BRAIN_BACK_RE, _BRAIN_SWITCH_HINGLISH_RE):
        m = pattern.match(candidate)
        if m:
            return ("switch", m.group(1))
    return None


def match_brain_intent(text: str) -> BrainAction | None:
    """"switch to codex" / "use copilot" / "go back to claude" -> ("switch",
    name); "which brain are you on" -> ("which", None). Same candidate
    strategy as match_intent: whole normalized utterance, then each clause.
    Unknown names ("use gemini") don't match, so they reach the brain."""
    for candidate in _candidates_for(normalize(text)):
        result = _match_brain_candidate(candidate)
        if result is not None:
            return result
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        clause_norm = normalize(clause)
        if not clause_norm:
            continue
        for candidate in _candidates_for(clause_norm):
            result = _match_brain_candidate(candidate)
            if result is not None:
                return result
    return None


def match_screen_intent(text: str) -> bool:
    """True if `text` (as-spoken) asks Veronica to look at the screen —
    matched the same way as match_intent (whole utterance, then each
    clause), against SCREEN_PHRASES."""
    norm_whole = normalize(text)
    for candidate in _candidates_for(norm_whole):
        if candidate in SCREEN_PHRASES:
            return True
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        clause_norm = normalize(clause)
        if not clause_norm:
            continue
        for candidate in _candidates_for(clause_norm):
            if candidate in SCREEN_PHRASES:
                return True
    return False


# Note-taking and dictation (A4) — matched with their own functions (like
# match_memory_intent) since they carry a payload / aren't in the plain
# Intent enum.
_TAKE_NOTE_RE = re.compile(r"^take a note[:,]?\s+(.+)$", re.IGNORECASE)
_NOTE_THAT_RE = re.compile(r"^note that\s+(.+)$", re.IGNORECASE)

DICTATE_PHRASES = frozenset({"dictate", "start dictation", "begin dictation"})
STOP_DICTATION_PHRASES = frozenset({"stop dictation", "stop dictating", "end dictation"})


def match_note_intent(text: str) -> str | None:
    """Match "take a note: X" / "take a note X" / "note that X" against a
    heard utterance, returning X (original casing/punctuation preserved,
    only a trailing sentence-ending period stripped) or None."""
    raw = (text or "").strip()
    raw = _MEMORY_LEAD_RE.sub("", raw, count=1).strip()
    for pattern in (_TAKE_NOTE_RE, _NOTE_THAT_RE):
        m = pattern.match(raw)
        if m:
            arg = m.group(1).strip().rstrip(".!?").strip()
            if arg:
                return arg
    return None


def match_dictation_intent(text: str) -> bool:
    """True if `text` asks Veronica to start dictating — matched the same
    way as match_intent (whole utterance, then each clause)."""
    norm_whole = normalize(text)
    for candidate in _candidates_for(norm_whole):
        if candidate in DICTATE_PHRASES:
            return True
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        clause_norm = normalize(clause)
        if not clause_norm:
            continue
        for candidate in _candidates_for(clause_norm):
            if candidate in DICTATE_PHRASES:
                return True
    return False


def is_stop_dictation(text: str) -> bool:
    """True if `text` is the "stop dictation" utterance that ends an
    in-progress dictation capture."""
    return normalize(text) in STOP_DICTATION_PHRASES


def match_intent(text: str) -> Intent | None:
    """Match a heard utterance (as-spoken, not yet normalized) against the
    local intent phrase sets. Tries the whole normalized utterance first
    (with an optional leading "veronica"/"hey veronica", trailing "please",
    and leading filler stripped), then each clause of the raw utterance in
    order — split on '.', ',', '!', '?', ';' — so the first clause that
    exactly matches a phrase wins."""
    norm_whole = normalize(text)
    for candidate in _candidates_for(norm_whole):
        result = _match_candidate(candidate)
        if result is not None:
            return result

    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        clause_norm = normalize(clause)
        if not clause_norm:
            continue
        for candidate in _candidates_for(clause_norm):
            result = _match_candidate(candidate)
            if result is not None:
                return result
    return None
