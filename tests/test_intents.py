import pytest

from veronica.brain.intents import (
    is_stop_dictation,
    match_dictation_intent,
    match_intent,
    match_memory_intent,
    match_music_intent,
    match_note_intent,
    match_screen_intent,
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
        ("go back", "prev"),
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
