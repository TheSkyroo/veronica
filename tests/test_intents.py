import pytest

from veronica.brain.intents import match_intent, normalize


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
        # no match
        ("what time is it", None),
        ("", None),
        (None, None),
        ("shrinking violet", None),
        ("hide and seek", None),
    ],
)
def test_match_intent(heard, expected):
    assert match_intent(heard) == expected


def test_normalize_strips_punctuation_and_case():
    assert normalize("Stop.") == "stop"
    assert normalize("That's all!") == "thats all"
    assert normalize(None) == ""
