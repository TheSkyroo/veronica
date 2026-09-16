import pytest

from veronica.speech import voices as v


@pytest.mark.parametrize("req,expected", [
    ("adam", "am_adam"),
    ("Adam", "am_adam"),
    ("  sarah ", "af_sarah"),
    ("male", "am_adam"),
    ("a man's", None),            # possessive not handled: falls to None
    ("man", "am_adam"),
    ("female", "af_sarah"),
    ("woman", "af_sarah"),
    ("british", "bf_emma"),
    ("english", "bf_emma"),
    ("uk", "bf_emma"),
    ("american", "af_sarah"),
    ("us", "af_sarah"),
    ("british male", "bm_george"),
    ("male british", "bm_george"),
    ("british female", "bf_emma"),
    ("american man", "am_adam"),
    ("american woman", "af_sarah"),
    ("default", "af_sarah"),
    ("normal", "af_sarah"),
    ("robot", None),
    ("", None),
])
def test_resolve_voice(req, expected):
    assert v.resolve_voice(req) == expected


def test_display_name():
    assert v.display_name("af_sarah") == "Sarah"
    assert v.display_name("bm_george") == "George"
    assert v.display_name("zz_unknown") == "Unknown"


def test_next_voice_cycles_in_table_order():
    ids = v.VOICE_IDS
    assert v.next_voice(ids[0]) == ids[1]
    assert v.next_voice(ids[-1]) == ids[0]
    assert v.next_voice("zz_unknown") == ids[0]


def test_clamp_speed():
    assert v.clamp_speed(0.1) == v.SPEED_MIN
    assert v.clamp_speed(9) == v.SPEED_MAX
    assert v.clamp_speed(1.0) == 1.0


def test_all_ten_voices_present():
    assert len(v.VOICES) == 10
    assert set(v.VOICES) == {"sarah", "bella", "nicole", "sky", "adam", "michael",
                             "emma", "isabella", "george", "lewis"}
