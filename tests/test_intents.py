import pytest

from veronica.brain.intents import (
    is_stop_dictation,
    match_dictation_intent,
    match_intent,
    match_language_intent,
    match_memory_intent,
    match_music_intent,
    match_note_intent,
    match_screen_intent,
    match_settings_intent,
    match_update_intent,
    match_version_intent,
    normalize,
)


@pytest.mark.parametrize(
    "heard, expected",
    [
        # end
        ("thanks veronica", "end"),
        ("thank you veronica", "end"),
        ("that's all", "end"),
        ("thats all", "end"),
        ("that is all", "end"),
        ("stop", "end"),
        ("Stop.", "end"),
        ("goodbye", "end"),
        ("never mind", "end"),
        ("nevermind", "end"),
        ("go idle", "end"),
        ("turn yourself off", "end"),
        ("go to sleep", "end"),
        ("sleep", "end"),
        ("go away", "end"),
        ("bye", "end"),
        ("dismiss", "end"),
        ("Veronica, dismiss", "end"),
        ("hey veronica, go idle", "end"),
        ("bye please", "end"),
        # hud_mini
        ("make yourself small", "hud_mini"),
        ("make yourself smaller", "hud_mini"),
        ("shrink", "hud_mini"),
        ("shrink yourself", "hud_mini"),
        ("minimize", "hud_mini"),
        ("minimise", "hud_mini"),
        ("mini mode", "hud_mini"),
        ("small mode", "hud_mini"),
        ("go small", "hud_mini"),
        ("Veronica, shrink", "hud_mini"),
        ("hey veronica shrink please", "hud_mini"),
        # hud_full
        ("expand", "hud_full"),
        ("expand yourself", "hud_full"),
        ("make yourself big", "hud_full"),
        ("make yourself bigger", "hud_full"),
        ("full mode", "hud_full"),
        ("show details", "hud_full"),
        ("go big", "hud_full"),
        ("expand please", "hud_full"),
        # hud_reset
        ("where are you", "hud_reset"),
        ("Where are you?", "hud_reset"),
        ("show yourself", "hud_reset"),
        ("come back", "hud_reset"),
        ("reset the hud", "hud_reset"),
        ("reset hud", "hud_reset"),
        ("hud reset", "hud_reset"),
        ("Veronica, where are you?", "hud_reset"),
        ("hey veronica come back please", "hud_reset"),
        # hud_hide
        ("hide", "hud_hide"),
        ("hide yourself", "hud_hide"),
        ("hide the hud", "hud_hide"),
        ("hide the panel", "hud_hide"),
        ("veronica hide", "hud_hide"),
        # mute
        ("mute", "mute"),
        ("mute yourself", "mute"),
        ("be quiet", "mute"),
        ("silence", "mute"),
        ("Veronica, mute yourself", "mute"),
        ("mute please", "mute"),
        # unmute
        ("unmute", "unmute"),
        ("unmute yourself", "unmute"),
        ("you can talk", "unmute"),
        ("speak again", "unmute"),
        # quit
        ("quit", "quit"),
        ("quit veronica", "quit"),
        ("shut down", "quit"),
        ("shut yourself down", "quit"),
        ("exit", "quit"),
        ("turn off completely", "quit"),
        ("hey veronica, quit", "quit"),
        # no match
        ("what time is it", None),
        ("", None),
        (None, None),
        ("shrinking violet", None),
        ("hide and seek", None),
        # clause splitting: first matching clause wins
        ("Make yourself small. I can't see you.", "hud_mini"),
        ("I think make yourself small. Assalamu alaikum.", "hud_mini"),
        ("Veronica, go idle please", "end"),
        ("Okay stop.", "end"),
        ("expand on that idea.", None),
        ("hide my files, please", None),
    ],
)
def test_match_intent(heard, expected):
    assert match_intent(heard) == expected


def test_normalize_strips_punctuation_and_case():
    assert normalize("Stop.") == "stop"
    assert normalize("That's all!") == "thats all"
    assert normalize(None) == ""


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("remember that I like tea", ("remember", "I like tea")),
        ("remember I like tea", ("remember", "I like tea")),
        ("Remember that my birthday is in June.", ("remember", "my birthday is in June")),
        ("forget that I like tea", ("forget", "I like tea")),
        ("forget I like tea", ("forget", "I like tea")),
        ("Veronica, remember I work at Acme", ("remember", "I work at Acme")),
        ("hey veronica remember that I'm allergic to peanuts", ("remember", "I'm allergic to peanuts")),
        ("remember that", None),
        ("remember", None),
        ("forget", None),
        ("remembering things is hard", None),
        ("remember when we went to Paris", None),
        ("remember what I told you", None),
        ("remember how to make pasta", None),
        ("remember if I locked the door", None),
        ("remember why I called", None),
        ("forget when we went to Paris", ("forget", "when we went to Paris")),
        ("what time is it", None),
        ("", None),
        (None, None),
    ],
)
def test_match_memory_intent(heard, expected):
    assert match_memory_intent(heard) == expected


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("what's on my screen", True),
        ("what is on my screen", True),
        ("Veronica, what's on my screen?", True),
        ("look at my screen", True),
        ("look at the screen", True),
        ("summarize this page", True),
        ("summarize this screen", True),
        ("summarize my screen", True),
        ("what does this error say", True),
        ("what does this say", True),
        ("can you look at my screen", True),
        ("what time is it", False),
        ("", False),
        (None, False),
    ],
)
def test_match_screen_intent(heard, expected):
    assert match_screen_intent(heard) == expected


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("pause", "pause"),
        ("pause music", "pause"),
        ("stop the music", "pause"),
        ("resume", "play"),
        ("resume music", "play"),
        ("play music", "play"),
        ("unpause", "play"),
        ("next song", "next"),
        ("skip", "next"),
        ("skip song", "next"),
        ("previous", "prev"),
        ("previous song", "prev"),
        ("go back", None),   # T12: too generic to be a music command
        ("what's playing", "now_playing"),
        ("whats playing", "now_playing"),
        ("what song is this", "now_playing"),
        ("Veronica, pause please", "pause"),
        ("what time is it", None),
        ("", None),
        (None, None),
    ],
)
def test_match_music_intent(heard, expected):
    assert match_music_intent(heard) == expected


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("take a note: buy milk", "buy milk"),
        ("take a note buy milk", "buy milk"),
        ("Take a note, call mom tomorrow.", "call mom tomorrow"),
        ("note that the wifi password is abc123", "the wifi password is abc123"),
        ("Veronica, note that I owe Sam $20", "I owe Sam $20"),
        ("take a note", None),
        ("note that", None),
        ("what time is it", None),
        ("", None),
        (None, None),
    ],
)
def test_match_note_intent(heard, expected):
    assert match_note_intent(heard) == expected


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("dictate", True),
        ("start dictation", True),
        ("begin dictation", True),
        ("Veronica, start dictation", True),
        ("dictation", False),
        ("what time is it", False),
        ("", False),
        (None, False),
    ],
)
def test_match_dictation_intent(heard, expected):
    assert match_dictation_intent(heard) == expected


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("stop dictation", True),
        ("stop dictating", True),
        ("end dictation", True),
        ("Stop dictation.", True),
        ("stop", False),
        ("", False),
        (None, False),
    ],
)
def test_is_stop_dictation(heard, expected):
    assert is_stop_dictation(heard) == expected


# -- batch B: voice / speed --------------------------------------------------

from veronica.brain.intents import match_voice_intent


@pytest.mark.parametrize("text,expected", [
    ("use a male voice", ("voice", "male")),
    ("Veronica, use a british voice please", ("voice", "british")),
    ("switch to Adam's voice", ("voice", "adams")),
    ("switch to adam voice", ("voice", "adam")),
    ("change to the british male voice", ("voice", "british male")),
    ("speak in a female voice", ("voice", "female")),
    ("speak with a british voice", ("voice", "british")),
    ("use the default voice", ("voice", "default")),
    ("change your voice", ("voice", "next")),
    ("different voice", ("voice", "next")),
    ("use a different voice", ("voice", "next")),
    ("speak faster", ("speed", "faster")),
    ("talk faster", ("speed", "faster")),
    ("faster please", ("speed", "faster")),
    ("speed up", ("speed", "faster")),
    ("speak slower", ("speed", "slower")),
    ("talk slower", ("speed", "slower")),
    ("slow down", ("speed", "slower")),
    ("normal speed", ("speed", "normal")),
    ("default speed", ("speed", "normal")),
    ("reset speed", ("speed", "normal")),
    ("what's your voice like", None),
    ("faster internet please", None),
    ("use a voice", None),
    ("the voice of reason", None),
])
def test_match_voice_intent(text, expected):
    assert match_voice_intent(text) == expected


# -- batch B: proactive briefings & nudges ------------------------------------

from veronica.brain.intents import match_proactive_intent, parse_clock_time


@pytest.mark.parametrize("s,expected", [
    ("8", "08:00"), ("8 am", "08:00"), ("8am", "08:00"), ("8:30", "08:30"), ("8:30 am", "08:30"),
    ("7 30 am", "07:30"), ("6 pm", "18:00"), ("6:15 pm", "18:15"), ("12 pm", "12:00"), ("12 am", "00:00"),
    ("noon", "12:00"), ("midnight", "00:00"), ("18:45", "18:45"), ("25", None), ("8:75", None), ("", None),
    # normalize() strips the colon before the intent regex sees the time
    ("730", "07:30"), ("730 am", "07:30"), ("1845", "18:45"), ("2500", None), ("875", None),
])
def test_parse_clock_time(s, expected):
    assert parse_clock_time(s) == expected


@pytest.mark.parametrize("text,expected", [
    ("brief me", ("brief_now", None)),
    ("Veronica, give me a briefing", ("brief_now", None)),
    ("morning briefing", ("brief_now", None)),
    ("what's my day look like", ("brief_now", None)),
    ("what does my day look like", ("brief_now", None)),
    ("give me a briefing every morning at 8", ("briefing_on", "08:00")),
    ("give me a morning briefing at 7:30 am", ("briefing_on", "07:30")),
    ("start the briefing every day at 6 pm", ("briefing_on", "18:00")),
    ("turn on the morning briefing", ("briefing_on", None)),
    ("stop the morning briefing", ("briefing_off", None)),
    ("turn off briefings", ("briefing_off", None)),
    ("cancel the briefing", ("briefing_off", None)),
    ("remind me before my meetings", ("nudges_on", None)),
    ("warn me 10 minutes before my meetings", ("nudges_on", 10)),
    ("nudge me before events", ("nudges_on", None)),
    ("turn on nudges", ("nudges_on", None)),
    ("tell me 15 minutes before meetings", ("nudges_on", 15)),
    ("stop the meeting nudges", ("nudges_off", None)),
    ("turn off nudges", ("nudges_off", None)),
    ("turn off reminders before meetings", ("nudges_off", None)),
    ("what's on my calendar", None),
    ("brief history of rome", None),
    ("remind me to call mum", None),
    ("give me a briefing every morning at 25", None),
])
def test_match_proactive_intent(text, expected):
    assert match_proactive_intent(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("bas karo", "end"), ("theek hai bas", "end"), ("chup", "end"), ("chup raho", "end"),
    ("band karo", "end"), ("ruko", "end"), ("ruk jao", "end"),
    ("mute karo", "mute"), ("awaaz band karo", "mute"), ("unmute karo", "unmute"), ("awaaz chalu karo", "unmute"),
    ("chhoti ho jao", "hud_mini"), ("chota karo", "hud_mini"), ("badi ho jao", "hud_full"), ("bada karo", "hud_full"),
    ("quit karo", "quit"), ("band ho jao", "quit"),
    ("kahan ho", "hud_reset"), ("wapas aao", "hud_reset"),
    ("Veronica, bas karo please", "end"),
    # bare "bas" is too common mid-sentence ("bas ek minute") to end the turn
    ("bas", None), ("bas ek minute", None),
    # Devanagari (pinned hi mode)
    ("बस", "end"), ("बस करो", "end"), ("चुप", "end"), ("रुको", "end"), ("बंद करो", "end"), ("बस करो।", "end"),
    ("म्यूट करो", "mute"), ("आवाज़ बंद करो", "mute"), ("अनम्यूट करो", "unmute"), ("आवाज़ चालू करो", "unmute"),
    ("कहाँ हो", "hud_reset"), ("वापस आओ", "hud_reset"), ("कहाँ हो?", "hud_reset"),
])
def test_hinglish_local_intents(text, expected):
    assert match_intent(text) == expected


def test_normalize_keeps_devanagari_and_strips_danda():
    assert normalize("समय क्या है?") == "समय क्या है"
    assert normalize("बस करो।") == "बस करो"
    assert normalize("आवाज़ बंद करो") == "आवाज़ बंद करो"


@pytest.mark.parametrize("text,expected", [
    ("speak hindi", "hi"), ("talk in hindi", "hi"), ("switch to hindi", "hi"), ("hindi mein bolo", "hi"),
    ("hindi me bolo", "hi"), ("hindi mein baat karo", "hi"),
    ("speak english", "en"), ("talk in english", "en"), ("switch to english", "en"), ("english mein bolo", "en"),
    ("english me bolo", "en"), ("angrezi mein bolo", "en"),
    ("understand both", "auto"), ("both languages", "auto"), ("hindi and english", "auto"), ("hindi aur english", "auto"),
    ("auto language", "auto"), ("dono bhasha", "auto"),
    ("what is hindi for hello", None), ("translate this to hindi", None), ("speak faster", None),
])
def test_match_language_intent(text, expected):
    assert match_language_intent(text) == expected


def test_clause_split_on_danda():
    assert match_intent("ठीक है। बस।") == "end"
    assert match_intent("मीटिंग बंद करो।") is None


# -- Batch D: settings / history / version / update -----------------------------

@pytest.mark.parametrize("text,expected", [
    ("open settings", "general"), ("show settings", "general"), ("settings", "general"),
    ("preferences", "general"), ("open preferences", "general"),
    ("settings kholo", "general"), ("setting kholo", "general"),
    ("Veronica, open settings please", "general"), ("Open settings.", "general"),
    ("show history", "history"), ("show my history", "history"), ("what did i ask you", "history"),
    ("what did I ask you earlier?", "history"), ("history", "history"), ("history dikhao", "history"),
    ("conversation history", "history"),
    # whole-utterance only: never hijack a longer request
    ("open safari settings", None), ("history of rome", None), ("what did i ask you to buy", None),
    ("settings for the hud", None), ("", None),
])
def test_match_settings_intent(text, expected):
    assert match_settings_intent(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("what version are you", True), ("which version are you", True), ("what version", True),
    ("version", True), ("your version", True), ("kaunsa version hai", True),
    ("Veronica, what version are you?", True),
    ("what version of python is installed", False), ("version control", False), ("", False),
])
def test_match_version_intent(text, expected):
    assert match_version_intent(text) is expected


@pytest.mark.parametrize("text,expected", [
    ("update yourself", True), ("update now", True), ("check for updates", True),
    ("check for an update", True), ("apna update karo", True), ("update karo", True),
    ("Veronica, update yourself please.", True),
    ("update my calendar", False), ("update the note", False), ("update", False), ("", False),
])
def test_match_update_intent(text, expected):
    assert match_update_intent(text) is expected


def test_hinglish_settings_phrases_count_as_hinglish():
    from veronica.brain.intents import HINGLISH_INTENT_PHRASES
    for phrase in ("settings kholo", "setting kholo", "history dikhao", "kaunsa version hai",
                   "apna update karo", "update karo"):
        assert phrase in HINGLISH_INTENT_PHRASES


def test_hud_reset_hinglish_phrases_count_as_hinglish():
    from veronica.brain.intents import HINGLISH_INTENT_PHRASES
    assert {"kahan ho", "wapas aao"} <= HINGLISH_INTENT_PHRASES


# -- brains: "switch to codex" / "which brain are you on" ------------------------
from veronica.brain.intents import match_brain_intent


@pytest.mark.parametrize("text,expected", [
    ("switch to codex", ("switch", "codex")), ("use copilot", ("switch", "copilot")),
    ("switch brain to antigravity", ("switch", "antigravity")), ("change to claude", ("switch", "claude")),
    ("switch to the codex brain", ("switch", "codex")), ("Veronica, switch to Codex please.", ("switch", "codex")),
    ("back to claude", ("switch", "claude")), ("go back to claude", ("switch", "claude")),
    ("codex pe switch karo", ("switch", "codex")), ("copilot use karo", ("switch", "copilot")),
    ("claude pe wapas jao", ("switch", "claude")), ("antigravity chalao", ("switch", "antigravity")),
    ("which brain are you on", ("which", None)), ("which model are you using", ("which", None)),
    ("who am i talking to", ("which", None)), ("which brain is this", ("which", None)),
    # not a which-brain phrase: it is usually about a photo, a caller or a name
    ("who is this", None), ("who is this?", None),
    ("kaunsa brain hai", ("which", None)),
    ("use qwen please", None), ("use gemini", None), ("switch to spanish", None),
    ("use a british voice", None), ("open codex", None), ("use codex to write a poem", None),
    ("", None),
])
def test_match_brain_intent(text, expected):
    assert match_brain_intent(text) == expected


def test_hinglish_which_brain_phrases_count_as_hinglish():
    from veronica.brain.intents import HINGLISH_INTENT_PHRASES
    assert {"kaunsa brain hai", "kaun sa model hai"} <= HINGLISH_INTENT_PHRASES
