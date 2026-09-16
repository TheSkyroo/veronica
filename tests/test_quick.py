import datetime as dt
import random

import pytest

from veronica.brain import quick as q

NOW = dt.datetime(2026, 9, 16, 15, 42)


def at(text, lang="en"):
    return q.match_quick(text, now=lambda: NOW, lang=lang)


@pytest.mark.parametrize("text,expected", [
    ("what time is it", ("time", "It's 3:42 pm.")),
    ("Veronica, what's the time please", ("time", "It's 3:42 pm.")),
    ("kitne baje hain", ("time", "Abhi 3:42 pm hain.")),
    ("what's the date", ("date", "It's Wednesday, September 16th.")),
    ("aaj kya tareekh hai", ("date", "Aaj Wednesday, 16 September hai.")),
    ("what day is it today", ("day", "It's Wednesday.")),
    ("aaj kaun sa din hai", ("day", "Aaj Wednesday hai.")),
    ("hello", ("social", None)),
    ("good evening", ("social", "Good evening, Manik.")),
    ("shukriya", ("social", None)),
    ("who are you", ("social", "I'm Veronica, your voice assistant on this Mac.")),
    ("tum kaun ho", ("social", "Main Veronica hoon, is Mac par aapki voice assistant.")),
    ("what can you do", ("social", "I can answer questions, control this Mac, read your calendar and mail, play music, take notes, control your browser, and remember things for you.")),
    ("battery level", ("battery", "en")),
    ("battery kitni hai", ("battery", "hi")),
    ("what's the volume", ("volume", "en")),
])
def test_match_quick_tables(text, expected):
    got = at(text)
    assert got is not None and got[0] == expected[0]
    if expected[1] is not None:
        assert got[1] == expected[1]


def test_hi_lang_uses_hindi_copy():
    assert at("what time is it", lang="hi") == ("time", "Abhi 3:42 pm hain.")
    assert at("what day is it", lang="hi") == ("day", "Aaj Wednesday hai.")


def test_social_choice_is_seeded():
    q._rng = random.Random(1)
    a = at("hello")[1]
    q._rng = random.Random(1)
    assert at("hello")[1] == a
    assert a in {"Hi Manik.", "Hello. What can I do for you?", "Hey there."}
    q._rng = random.Random(0)
    assert at("shukriya")[1] in {"Koi baat nahi.", "Hamesha."}


@pytest.mark.parametrize("text,reply", [
    ("what's 12 times 8", "12 times 8 is 96."),
    ("12 x 8", "12 times 8 is 96."),
    ("calculate 100 divided by 7", "100 divided by 7 is 14.2857."),
    ("what is 15 percent of 80", "15 percent of 80 is 12."),
    ("square root of 144", "square root of 144 is 12."),
    ("2 to the power of 10", "2 to the power of 10 is 1024."),
    ("what's 7 squared", "7 squared is 49."),
    ("what's 3 plus 4 times 2", "3 plus 4 times 2 is 11."),
    ("what's 10 minus 15", "10 minus 15 is -5."),
    ("12 guna 8 kitna hota hai", "12 guna 8, 96 hota hai."),
    ("100 bhaag 4 kya hota hai", "100 bhaag 4, 25 hota hai."),
    # symbol operators survive now that math runs on the raw text
    ("what's 10 - 3", "10 minus 3 is 7."),
    ("whats 10-3", "10 minus 3 is 7."),
    ("5 + 5", "5 plus 5 is 10."),
    ("what's 5 + 5?", "5 plus 5 is 10."),
    ("15% of 80", "15 percent of 80 is 12."),
    ("12 * 8", "12 times 8 is 96."),
    ("12 × 8", "12 times 8 is 96."),
    ("100 / 8", "100 divided by 8 is 12.5."),
    ("2^10", "2 to the power of 10 is 1024."),
    ("Veronica, what's 5 + 5, please?", "5 plus 5 is 10."),
    ("Okay, what's 6 times 7.", "6 times 7 is 42."),
])
def test_math(text, reply):
    assert at(text) == ("math", reply)


@pytest.mark.parametrize("text", [
    # integers only: a digit next to . , : is a decimal / thousands / clock time
    "3.14 times 2", "12.5 plus 1", "1,000 plus 1", "at 5:30 plus 10", "what's 3.14 times 2?",
    # bare numbers (follow-up answers like "3") are not arithmetic
    "5", "2024", "3", "what's 5", "5.", "(5)",
    "twenty-five plus 1",
])
def test_math_no_match(text):
    assert at(text) is None


@pytest.mark.parametrize("text", [
    "set a timer for 5 minutes", "what time is my next meeting", "whats the date of the meeting",
    "what's 5 plus 5 in binary", "hello can you open safari", "what's 10 divided by 0",
    "what is 99999999999999 times 99999999999999", "1 plus 2 plus 3 plus 4 plus 5 plus 6 plus 7 plus 8",
    "time to go", "battery is low", "how are you going to do that",
])
def test_no_match(text):
    assert at(text) is None


def test_evaluate_and_format():
    assert q.evaluate("2 + 3 * 4") == 14
    assert q.evaluate("(2 + 3) * 4") == 20
    assert q.evaluate("-3 + 5") == 2
    assert q.evaluate("1 / 0") is None
    assert q.format_number(96.0) == "96"
    assert q.format_number(14.285714) == "14.2857"
    assert q.format_number(0.5) == "0.5"
    assert q.format_number(-5.0) == "-5"


@pytest.mark.parametrize("n,s", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"), (13, "13th"), (21, "21st"), (22, "22nd"), (23, "23rd"), (31, "31st")])
def test_ordinal(n, s):
    assert q.ordinal(n) == s


def test_is_hinglish_phrase():
    assert q.is_hinglish_phrase("Veronica, shukriya") and q.is_hinglish_phrase("kitne baje hain")
    assert not q.is_hinglish_phrase("thanks") and not q.is_hinglish_phrase("kal meeting hai")


def test_reply_for_battery_and_volume():
    assert q.reply_for("battery", "en", percent=72, state="charging") == "Battery is at 72 percent and charging."
    assert q.reply_for("battery", "en", percent=72, state="discharging") == "Battery is at 72 percent and not charging."
    assert q.reply_for("battery", "en", percent=100, state="charged") == "Battery is at 100 percent and fully charged."
    assert q.reply_for("battery", "hi", percent=72, state="charging") == "Battery 72 percent hai aur charge ho rahi hai."
    assert q.reply_for("battery", "hi", percent=72, state="discharging") == "Battery 72 percent hai aur charge nahi ho rahi."
    assert q.reply_for("battery", "hi", percent=100, state="charged") == "Battery 100 percent hai aur full charge hai."
    assert q.reply_for("battery", "en", percent=None, state=None) == "I couldn't read the battery level."
    assert q.reply_for("battery", "hi", percent=None, state=None) == "Battery level nahi mil paaya."
    assert q.reply_for("volume", "en", percent=40) == "Volume is at 40 percent."
    assert q.reply_for("volume", "hi", percent=40) == "Volume 40 percent hai."
    assert q.reply_for("volume", "en", percent=None) == "I couldn't read the volume."
    assert q.reply_for("volume", "hi", percent=None) == "Volume nahi mil paaya."
