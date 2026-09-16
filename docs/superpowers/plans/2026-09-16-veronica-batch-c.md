# Veronica Batch C Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Sub-second local answers for trivial questions, and Hindi/Hinglish understanding + replies with a Hindi voice.

**Architecture:** C1 adds a pure matcher/evaluator module (`veronica/brain/quick.py`) plus one orchestrator local turn. C2 threads a per-utterance language through STT → orchestrator → brain prompt → TTS: `Transcriber` reports the detected language, the orchestrator derives `_utterance_lang`, `Synthesizer.synth(text, lang)` picks the Hindi voice + Kokoro `lang="hi"`, and a language-mode pref swaps STT models at runtime through a factory injected from `__main__`.

**Tech Stack:** Python 3.12, uv, pytest (asyncio auto), faster-whisper (`small.en` / multilingual `small`, `tiny.en` / `tiny`), Kokoro ONNX (voices `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`, lang `hi`), rumps/PyObjC.

**Spec:** `docs/superpowers/specs/2026-09-16-veronica-batch-c-design.md`

## Global Constraints

- Brain stays on the user's Claude Code subscription login via `claude-agent-sdk`. **No API key**, ever.
- Confirm-gate stays strict: `veronica/brain/policy.py` `classify()` is the only thing that may auto-allow a tool. Never set `allowed_tools`.
- Speech stays local/zero-key (faster-whisper STT, Kokoro TTS).
- Never add `Co-Authored-By` trailers or "Generated with Claude Code" to commits.
- Tests: `uv run pytest -q` (asyncio auto, `live` marker deselected). No test touches the real mic, speakers, network, models or AppleScript.
- Local fast paths fire only on unambiguous whole-utterance matches after `normalize()`; `match_quick` uses the whole utterance only (no clause splitting). Everything else goes to the brain as today.
- Copy strings are exactly as written in the spec/plan.
- Branch `batch-c` (created; spec committed).

## File map

| File | Responsibility |
|---|---|
| `veronica/brain/quick.py` (new) | `match_quick`, math parser/evaluator, phrase tables, reply copy (en/hi) |
| `veronica/orchestrator.py` | `_quick_turn` (+ battery/volume resolution), `_utterance_lang`, language-switch turn, `lang` plumbing into `say`/`handle_text` |
| `veronica/speech/stt.py` | `Transcriber(model_name, language)`, `transcribe_detailed`, `set_language` |
| `veronica/speech/tts.py` | `synth(text, lang=None)`, `hindi_voice` |
| `veronica/speech/voices.py` | Hindi voices table + resolution |
| `veronica/brain/sentences.py` | `।` + Devanagari/digit boundaries |
| `veronica/brain/intents.py` | Hinglish phrases for existing intents, `match_language_intent` |
| `veronica/brain/prompts.py` | language sentence |
| `veronica/config.py` | `language`, `whisper_multilingual_model`, `partial_stt_multilingual_model` |
| `veronica/__main__.py` | STT factory, language pref at startup |
| `veronica/ui/menubar.py` | Hindi voices in Voice submenu |
| `veronica/ui/hud/hud.css` | Devanagari font fallback |
| `scripts/download_models.py` | `--hindi` prefetch |
| `README.md` | Quick replies + Hindi & Hinglish sections |

---

### Task 1: `quick.py` — phrase tables, math evaluator, `match_quick`

**Files:**
- Create: `veronica/brain/quick.py`
- Test: `tests/test_quick.py`

**Interfaces:**
- Produces: `QuickReply = tuple[str, str]`; `match_quick(text: str, *, now: Callable[[], dt.datetime] = dt.datetime.now, lang: str = "en") -> QuickReply | None`; `evaluate(expr: str) -> float | None`; `format_number(x: float) -> str`; `ordinal(n: int) -> str`; `reply_for(kind: str, lang: str, **kw) -> str` (used by the orchestrator for battery/volume copy: kinds `"battery"` with `percent:int|None, state:"charging"|"discharging"|"charged"|None`, `"volume"` with `percent:int|None`); module-level `_rng = random.Random()`.

- [ ] **Step 1: Write failing tests** — `tests/test_quick.py`:

```python
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
])
def test_math(text, reply):
    assert at(text) == ("math", reply)


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
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_quick.py` → FAIL (no module).

- [ ] **Step 3: Implement** `veronica/brain/quick.py`:

```python
"""Local fast-path replies: trivial questions answered without the brain.

Everything here matches the WHOLE normalized utterance (no clause
splitting) so a longer request that merely contains "hello" or "time"
still goes to the brain."""
import datetime as dt
import random
import re
from collections.abc import Callable

from veronica.brain.intents import normalize

QuickReply = tuple[str, str]
_rng = random.Random()

# -- copy ---------------------------------------------------------------------
_TIME = {"what time is it", "whats the time", "time", "current time", "tell me the time", "what is the time"}
_TIME_HI = {"samay kya hai", "kitne baje hain", "kitne baje hai", "time kya hai", "time kya hua hai"}
_DATE = {"whats the date", "what is the date", "whats todays date", "what date is it", "todays date", "what is todays date"}
_DATE_HI = {"aaj kya tareekh hai", "aaj ki tareekh kya hai", "date kya hai", "aaj date kya hai"}
_DAY = {"what day is it", "what day is it today", "what day is today", "which day is it"}
_DAY_HI = {"aaj kya din hai", "aaj kaun sa din hai", "aaj konsa din hai"}
_BATTERY = {"battery", "battery level", "whats the battery", "whats my battery", "how much battery",
            "how much battery do i have", "battery percentage", "whats the battery level"}
_BATTERY_HI = {"battery kitni hai", "battery kitna hai"}
_VOLUME = {"whats the volume", "volume", "volume level", "how loud is it", "what volume is it"}
_VOLUME_HI = {"volume kitna hai", "volume kitni hai"}

_SOCIAL: list[tuple[set[str], set[str], list[str], list[str]]] = [
    # (en phrases, hi phrases, en replies, hi replies)
    ({"hello", "hi", "hey", "hi veronica", "hello veronica"}, {"namaste", "namaskar"},
     ["Hi Manik.", "Hello. What can I do for you?", "Hey there."], ["Namaste Manik.", "Haan, boliye."]),
    ({"thanks", "thank you", "thanks veronica", "thank you veronica", "thanks a lot", "cheers", "thank you so much"},
     {"shukriya", "dhanyavaad", "dhanyavad"},
     ["You're welcome.", "Anytime.", "Happy to help."], ["Koi baat nahi.", "Hamesha."]),
    ({"bye", "goodbye", "see you", "see you later"}, {"alvida", "phir milenge"},
     ["Bye, Manik.", "See you."], ["Alvida.", "Phir milenge."]),
    ({"how are you", "how are you doing", "hows it going", "how are you veronica"},
     {"kaise ho", "kaisi ho", "kya haal hai", "kya haal hain"},
     ["I'm doing well, thanks. How can I help?"], ["Main theek hoon. Aap batao, kya karna hai?"]),
    ({"who are you", "whats your name", "what is your name"}, {"tum kaun ho", "aap kaun ho", "tumhara naam kya hai"},
     ["I'm Veronica, your voice assistant on this Mac."], ["Main Veronica hoon, is Mac par aapki voice assistant."]),
    ({"what can you do", "what do you do", "help", "what can i ask you"}, {"tum kya kar sakti ho", "kya kar sakti ho"},
     ["I can answer questions, control this Mac, read your calendar and mail, play music, take notes, control your browser, and remember things for you."],
     ["Main sawaal jawab, Mac control, calendar aur mail, music, notes, browser aur yaad rakhne mein madad kar sakti hoon."]),
]
_GREETINGS = {"good morning": "Good morning, Manik.", "good afternoon": "Good afternoon, Manik.", "good evening": "Good evening, Manik."}
_GOOD_NIGHT = {"good night": "Good night.", "shubh ratri": "Shubh ratri."}


def ordinal(n: int) -> str:
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


def _clock(now: dt.datetime) -> str:
    h = now.hour % 12 or 12
    return f"{h}:{now.minute:02d} {'am' if now.hour < 12 else 'pm'}"


def format_number(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.4f}".rstrip("0").rstrip(".")
    return s


# -- math ---------------------------------------------------------------------
_MATH_LEAD = re.compile(r"^(?:whats|what is|calculate|compute|how much is|tell me)\s+")
_MATH_TRAIL_HI = re.compile(r"\s+(?:kitna hota hai|kya hota hai)$")
_OPS = [  # (normalized phrase, token, spoken en, spoken hi)
    ("to the power of", "^", "to the power of", "ki power"),
    ("multiplied by", "*", "times", "guna"),
    ("divided by", "/", "divided by", "bhaag"),
    ("percent of", "%", "percent of", "percent of"),
    ("square root of", "sqrt", "square root of", "square root of"),
    ("plus", "+", "plus", "jama"), ("jama", "+", "plus", "jama"),
    ("minus", "-", "minus", "ghata"), ("ghata", "-", "minus", "ghata"),
    ("times", "*", "times", "guna"), ("guna", "*", "times", "guna"), ("x", "*", "times", "guna"),
    ("over", "/", "divided by", "bhaag"), ("bhaag", "/", "divided by", "bhaag"), ("bhag", "/", "divided by", "bhaag"),
    ("squared", "sq", "squared", "ka square"),
    ("+", "+", "plus", "jama"), ("-", "-", "minus", "ghata"), ("*", "*", "times", "guna"), ("/", "/", "divided by", "bhaag"),
]
_TOKEN_RE = re.compile(r"\d{1,12}|[-+*/^%()]|sqrt|sq|\S+")


def _tokenize(expr: str) -> list[str] | None:
    s = f" {expr} "
    for phrase, tok, _, _ in _OPS:
        s = s.replace(f" {phrase} ", f" {tok} ")
    toks = _TOKEN_RE.findall(s)
    for t in toks:
        if not (t.isdigit() or t in "+-*/^%()" or t in ("sqrt", "sq")):
            return None
    if sum(1 for t in toks if not t.isdigit() and t not in "()") > 6:
        return None
    return toks


class _Parser:
    def __init__(self, toks: list[str]) -> None:
        self.t, self.i = toks, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self):
        tok = self.peek(); self.i += 1; return tok

    def expr(self) -> float:          # + -
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.take(); r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self) -> float:          # * / %
        v = self.power()
        while self.peek() in ("*", "/", "%"):
            op = self.take(); r = self.power()
            if op == "*": v *= r
            elif op == "/":
                if r == 0: raise ZeroDivisionError
                v /= r
            else: v = r * v / 100      # "p percent of x"
        return v

    def power(self) -> float:         # ^ (right-assoc), postfix sq
        v = self.unary()
        if self.peek() == "^":
            self.take(); v = v ** self.power()
        while self.peek() == "sq":
            self.take(); v = v * v
        return v

    def unary(self) -> float:
        tok = self.peek()
        if tok == "-":
            self.take(); return -self.unary()
        if tok == "sqrt":
            self.take(); v = self.unary()
            if v < 0: raise ValueError
            return v ** 0.5
        if tok == "(":
            self.take(); v = self.expr()
            if self.take() != ")": raise ValueError
            return v
        if tok is not None and tok.isdigit():
            return float(self.take())
        raise ValueError


def evaluate(expr: str) -> float | None:
    toks = _tokenize(expr.lower())
    if not toks:
        return None
    try:
        p = _Parser(toks); v = p.expr()
        if p.peek() is not None or abs(v) > 1e15 or v != v:
            return None
        return v
    except (ValueError, ZeroDivisionError, OverflowError, IndexError):
        return None


def _spoken(expr: str, hi: bool) -> str:
    s = f" {expr} "
    for phrase, _tok, en, hi_word in _OPS:
        s = s.replace(f" {phrase} ", f" {hi_word if hi else en} ")
    return " ".join(s.split())


def _match_math(norm: str) -> QuickReply | None:
    hi = bool(_MATH_TRAIL_HI.search(norm))
    body = _MATH_TRAIL_HI.sub("", norm) if hi else _MATH_LEAD.sub("", norm)
    if not re.search(r"\d", body) or not re.search(r"\d|\)|squared", body.split()[-1] if body.split() else ""):
        return None
    value = evaluate(body)
    if value is None:
        return None
    spoken, result = _spoken(body, hi), format_number(value)
    return ("math", f"{spoken}, {result} hota hai." if hi else f"{spoken} is {result}.")


# -- public -------------------------------------------------------------------
def reply_for(kind: str, lang: str, **kw) -> str:
    hi = lang == "hi"
    if kind == "battery":
        p, st = kw.get("percent"), kw.get("state")
        if p is None:
            return "Battery level nahi mil paaya." if hi else "I couldn't read the battery level."
        tail = {"charging": ("aur charge ho rahi hai", "and charging"),
                "discharging": ("aur charge nahi ho rahi", "and not charging"),
                "charged": ("aur full charge hai", "and fully charged")}.get(st, ("", ""))
        return (f"Battery {p} percent hai {tail[0]}." if hi else f"Battery is at {p} percent {tail[1]}.").replace(" .", ".")
    if kind == "volume":
        p = kw.get("percent")
        if p is None:
            return "Volume nahi mil paaya." if hi else "I couldn't read the volume."
        return f"Volume {p} percent hai." if hi else f"Volume is at {p} percent."
    raise ValueError(kind)


HINGLISH_PHRASES: frozenset[str] = frozenset(
    _TIME_HI | _DATE_HI | _DAY_HI | _BATTERY_HI | _VOLUME_HI | {"shubh ratri"}
    | set().union(*(hi for _en, hi, _r1, _r2 in _SOCIAL))
)


def is_hinglish_phrase(text: str) -> bool:
    """True if the whole normalized utterance is one of our Hinglish phrases
    (quick replies or local intents) — used by the orchestrator to treat a
    romanized-Hindi utterance as Hindi even when whisper says "en"."""
    from veronica.brain.intents import HINGLISH_INTENT_PHRASES  # Task 4 adds it; empty set until then
    return normalize(text) in HINGLISH_PHRASES or normalize(text) in HINGLISH_INTENT_PHRASES


def match_quick(text: str, *, now: Callable[[], dt.datetime] = dt.datetime.now, lang: str = "en") -> QuickReply | None:
    norm = normalize(text)
    if not norm:
        return None
    hi = lang == "hi"

    def pick(en_set, hi_set):
        if norm in hi_set:
            return True
        if norm in en_set:
            return hi
        return None

    t = now()
    for en_set, hi_set, kind, en_fn, hi_fn in (
        (_TIME, _TIME_HI, "time", lambda: f"It's {_clock(t)}.", lambda: f"Abhi {_clock(t)} hain."),
        (_DATE, _DATE_HI, "date", lambda: f"It's {t.strftime('%A, %B')} {ordinal(t.day)}.", lambda: f"Aaj {t.strftime('%A')}, {t.day} {t.strftime('%B')} hai."),
        (_DAY, _DAY_HI, "day", lambda: f"It's {t.strftime('%A')}.", lambda: f"Aaj {t.strftime('%A')} hai."),
        (_BATTERY, _BATTERY_HI, "battery", lambda: "en", lambda: "hi"),
        (_VOLUME, _VOLUME_HI, "volume", lambda: "en", lambda: "hi"),
    ):
        use_hi = pick(en_set, hi_set)
        if use_hi is not None:
            return (kind, hi_fn() if use_hi else en_fn())
    if norm in _GREETINGS:
        return ("social", _GREETINGS[norm])
    if norm in _GOOD_NIGHT:
        return ("social", _GOOD_NIGHT[norm])
    for en_set, hi_set, en_replies, hi_replies in _SOCIAL:
        use_hi = pick(en_set, hi_set)
        if use_hi is not None:
            return ("social", _rng.choice(hi_replies if use_hi else en_replies))
    return _match_math(norm)
```

Check the tricky tests by hand: `"what's 12 times 8"` → normalize → `"whats 12 times 8"` → lead stripped → `"12 times 8"` → tokens `12 * 8` → 96 → `"12 times 8 is 96."`. `"12 guna 8 kitna hota hai"` → hi trail → body `"12 guna 8"` → `_spoken(hi)` → `"12 guna 8"` → `"12 guna 8, 96 hota hai."`. `"what's 5 plus 5 in binary"` → body `"5 plus 5 in binary"` → tokens include `in` → `None`. `"time to go"` → not in any set, no digit → None. `"battery is low"` → None. `"what's 10 divided by 0"` → ZeroDivisionError → None. 8 operators → `> 6` → None. `"99999999999999 times 99999999999999"` → 14 digits each > 12 → `\d{1,12}` splits into `999999999999` and `99` → tokens `999999999999 99 * …` → parser leaves a leftover token → `peek() is not None` → None. `"7 squared"` → `7 sq` → 49. `"3 plus 4 times 2"` → 11. `"hello can you open safari"` → not exact → None. If `normalize()` maps `"whats"` differently (e.g. keeps the apostrophe form), adapt `_MATH_LEAD` and the sets to what `normalize` actually produces — read `normalize` first; the tests are the contract.

Also add to `veronica/brain/intents.py` (Task 4 fills it): `HINGLISH_INTENT_PHRASES: frozenset[str] = frozenset()`.

- [ ] **Step 4: Run** `uv run pytest -q tests/test_quick.py` → PASS. Commit:

```bash
git add veronica/brain/quick.py veronica/brain/intents.py tests/test_quick.py
git commit -m "feat(quick): local fast-path matcher, math evaluator and reply copy"
```

---

### Task 2: Orchestrator `_quick_turn` (battery/volume, memory, HUD)

**Files:**
- Modify: `veronica/orchestrator.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `quick.match_quick`, `quick.reply_for`; `mac_tools.volume_get.handler`.
- Produces: `Orchestrator._utterance_lang: str` (attribute, default `"en"`; Task 5 sets it per utterance), `async _quick_turn(self, quick: QuickReply, heard: str) -> None`, `_read_battery() -> tuple[int | None, str | None]` (module-level helper `read_battery(run=subprocess.run)` in `veronica/tools/mac.py` is fine too — choose one; the plan uses a module function `read_battery` in `veronica/tools/mac.py` so it's testable: `def read_battery(run=subprocess.run) -> tuple[int | None, str | None]` parsing `pmset -g batt` output).

- [ ] **Step 1: Failing tests** — append to `tests/test_orchestrator.py`:

```python
# -- batch C: quick replies ----------------------------------------------------

from veronica.tools import mac as mac_tools_mod


async def test_quick_time_reply_skips_brain_and_logs(monkeypatch):
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what day is it"])
    o.store = FakeStore()
    o.s = Settings(followup_window_s=0, confirm_listen_s=0, memory_enabled=True)
    await o.one_turn()
    assert o.tts.said == [f"It's {__import__('datetime').datetime.now().strftime('%A')}."]
    assert o.brain.asked == []
    assert ("tool", {"summary": "Quick reply", "decision": "auto"}) in ev
    assert o.store.turns[-1][0] == "what day is it"


async def test_quick_battery_reads_pmset(monkeypatch):
    monkeypatch.setattr(mac_tools_mod, "read_battery", lambda: (72, "charging"))
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["battery level"])
    await o.one_turn()
    assert o.tts.said == ["Battery is at 72 percent and charging."]


async def test_quick_battery_failure_copy(monkeypatch):
    monkeypatch.setattr(mac_tools_mod, "read_battery", lambda: (None, None))
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["battery kitni hai"])
    await o.one_turn()
    assert o.tts.said == ["Battery level nahi mil paaya."]


async def test_quick_volume_uses_mac_tool(monkeypatch):
    async def fake_get(args):
        return {"content": [{"type": "text", "text": "40"}]}
    monkeypatch.setattr(mac_tools_mod.volume_get, "handler", fake_get)
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what's the volume"])
    await o.one_turn()
    assert o.tts.said == ["Volume is at 40 percent."]


async def test_quick_does_not_shadow_local_intents_or_brain():
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["hello can you open safari"])
    await o.one_turn()
    assert o.brain.asked == ["hello can you open safari"]


def test_read_battery_parses_pmset():
    class R:
        def __init__(self, out): self.stdout = out; self.returncode = 0
    run = lambda *a, **k: R("Now drawing from 'AC Power'\n -InternalBattery-0 (id=123)\t72%; charging; 0:45 remaining present: true\n")
    assert mac_tools_mod.read_battery(run=run) == (72, "charging")
    run = lambda *a, **k: R(" -InternalBattery-0\t100%; charged; 0:00 remaining\n")
    assert mac_tools_mod.read_battery(run=run) == (100, "charged")
    run = lambda *a, **k: R(" -InternalBattery-0\t35%; discharging; 3:10 remaining\n")
    assert mac_tools_mod.read_battery(run=run) == (35, "discharging")
    run = lambda *a, **k: (_ for _ in ()).throw(OSError("no pmset"))
    assert mac_tools_mod.read_battery(run=run) == (None, None)
```

Check `FakeStore` in the file records `turns` as `(heard, reply)` — adapt the attribute name to what it actually stores.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement:**

`veronica/tools/mac.py` (plain function, not a tool):
```python
_BATT_RE = re.compile(r"(\d{1,3})%;\s*(charging|discharging|charged|finishing charge|AC attached)", re.IGNORECASE)


def read_battery(run=subprocess.run) -> tuple[int | None, str | None]:
    """(percent, state) from `pmset -g batt`; (None, None) if unavailable."""
    try:
        p = run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5)
        m = _BATT_RE.search(p.stdout or "")
    except Exception:
        return None, None
    if not m:
        return None, None
    state = m.group(2).lower()
    if state in ("finishing charge", "ac attached"):
        state = "charging"
    return int(m.group(1)), state
```

`veronica/orchestrator.py`:
```python
from veronica.brain import quick
...
        self._utterance_lang = "en"   # set per utterance (Task 5); "en" or "hi"
...
    # -- quick replies (C1) --------------------------------------------------------
    async def _quick_turn(self, hit: tuple[str, str], heard: str) -> None:
        kind, reply = hit
        lang = reply if kind in ("battery", "volume") else self._utterance_lang
        if kind == "battery":
            percent, state = await asyncio.to_thread(mac_tools.read_battery)
            reply = quick.reply_for("battery", lang, percent=percent, state=state)
        elif kind == "volume":
            res = await mac_tools.volume_get.handler({})
            try:
                percent = None if res.get("is_error") else int(float(res["content"][0]["text"].strip()))
            except (ValueError, KeyError, IndexError):
                percent = None
            reply = quick.reply_for("volume", lang, percent=percent)
        self._emit("tool", {"summary": "Quick reply", "decision": "auto"})
        await self.say(reply)
        if self.store is not None and self.s.memory_enabled:
            self.store.add_turn(heard, reply)
```
(`match_quick` returns `("battery", "en"|"hi")` / `("volume", "en"|"hi")` — the second element is the reply language for these two kinds; Task 1 already does this.)

Dispatch in `one_turn`: after the proactive branch and before dictation, compute `quick_hit = None if (any earlier local match) else quick.match_quick(text, lang=self._utterance_lang)`; add `or quick_hit is not None` to the dictation guard; branch `elif quick_hit is not None: self.player.reset(); await self._quick_turn(quick_hit, text)`.

- [ ] **Step 4:** `uv run pytest -q` → PASS. Commit `feat: quick local replies for time, date, math, battery, volume and small talk`.

---

### Task 3: STT detailed transcription, TTS Hindi voice, sentence splitter, Hindi voices table

**Files:**
- Modify: `veronica/speech/stt.py`, `veronica/speech/tts.py`, `veronica/speech/voices.py`, `veronica/brain/sentences.py`, `veronica/config.py`
- Test: `tests/test_stt.py`, `tests/test_tts.py`, `tests/test_voices.py`, `tests/test_sentences.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `Transcriber(model_name: str, language: str | None = "en")`, `.language` attr, `set_language(lang)`, `transcribe_detailed(pcm) -> tuple[str, str]`, `atranscribe_detailed(pcm)`; `Synthesizer.hindi_voice: str = "hf_alpha"`, `synth(text, lang: str | None = None)`, `asynth(text, lang=None)`, `has_devanagari(text) -> bool` (module-level in `veronica/speech/tts.py`); `voices.HINDI_VOICES = {"alpha": "hf_alpha", "beta": "hf_beta", "omega": "hm_omega", "psi": "hm_psi"}`, `voices.HINDI_VOICE_IDS`, `voices.DEFAULT_HINDI_VOICE = "hf_alpha"`, `resolve_voice("hindi") == "hf_alpha"`, `resolve_voice("hindi male") == "hm_omega"`, `resolve_voice("omega") == "hm_omega"`, `is_hindi_voice(vid) -> bool`; `Settings.language: str = "en"`, `Settings.whisper_multilingual_model: str = "small"`, `Settings.partial_stt_multilingual_model: str = "tiny"`.

- [ ] **Step 1: Failing tests**

`tests/test_stt.py` (read its fake model pattern first):
```python
def test_transcribe_detailed_reports_language(monkeypatch):
    calls = []
    class Info:  language = "hi"
    class Seg:
        def __init__(self, t): self.text = t
    class M:
        def __init__(self, *a, **k): pass
        def transcribe(self, audio, **kw):
            calls.append(kw)
            return iter([Seg(" नमस्ते ")]), Info()
    monkeypatch.setattr(Transcriber, "_model_cls", M)
    t = Transcriber("small", language=None)
    assert t.transcribe_detailed(np.zeros(16000, dtype=np.int16)) == ("नमस्ते", "hi")
    assert calls[-1]["language"] is None
    t.set_language("hi")
    t.transcribe(np.zeros(16000, dtype=np.int16))
    assert calls[-1]["language"] == "hi"
    t2 = Transcriber("small.en")
    t2.transcribe(np.zeros(16000, dtype=np.int16))
    assert calls[-1]["language"] == "en"
```

`tests/test_tts.py`:
```python
def test_synth_picks_hindi_voice_for_devanagari_or_lang(monkeypatch, tmp_path):
    calls = []
    class FakeKokoro:
        def __init__(self, *a): pass
        def create(self, text, voice, speed, lang):
            calls.append((voice, lang)); return [0.0], 24000
    monkeypatch.setattr(Synthesizer, "_kokoro_cls", FakeKokoro)
    s = Synthesizer("af_sarah", tmp_path)
    assert s.hindi_voice == "hf_alpha"
    s.synth("नमस्ते"); assert calls[-1] == ("hf_alpha", "hi")
    s.synth("kal teen baje", lang="hi"); assert calls[-1] == ("hf_alpha", "hi")
    s.synth("hello", lang="en"); assert calls[-1] == ("af_sarah", "en-us")
    s.synth("hello"); assert calls[-1] == ("af_sarah", "en-us")
    s.hindi_voice = "hm_omega"
    s.synth("ठीक है"); assert calls[-1] == ("hm_omega", "hi")
```

`tests/test_voices.py`:
```python
def test_hindi_voices():
    assert v.resolve_voice("hindi") == "hf_alpha"
    assert v.resolve_voice("indian") == "hf_alpha"
    assert v.resolve_voice("hindi male") == "hm_omega"
    assert v.resolve_voice("hindi female") == "hf_alpha"
    assert v.resolve_voice("omega") == "hm_omega"
    assert v.is_hindi_voice("hm_psi") and not v.is_hindi_voice("af_sarah")
    assert set(v.HINDI_VOICE_IDS) == {"hf_alpha", "hf_beta", "hm_omega", "hm_psi"}
    assert not set(v.HINDI_VOICE_IDS) & set(v.VOICE_IDS)      # cycling stays English
    assert v.display_name("hf_alpha") == "Alpha"
```

`tests/test_sentences.py`:
```python
def test_devanagari_danda_and_next_char():
    s = SentenceSplitter()
    out = s.feed("कल तीन बजे मीटिंग है। उसके बाद lunch है। ")
    assert out == ["कल तीन बजे मीटिंग है।", "उसके बाद lunch है।"]
    s = SentenceSplitter()
    assert s.feed("Kal 3 baje meeting hai. 4 baje free ho. ") == ["Kal 3 baje meeting hai.", "4 baje free ho."]
    s = SentenceSplitter()
    assert s.feed("ठीक है") == [] and s.flush() == ["ठीक है"]
```

`tests/test_config.py`: `Settings().language == "en"`, `whisper_multilingual_model == "small"`, `partial_stt_multilingual_model == "tiny"`.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

`stt.py`:
```python
class Transcriber:
    _model_cls = WhisperModel

    def __init__(self, model_name: str, language: str | None = "en") -> None:
        self.model_name = model_name
        self.language = language          # None = let whisper detect
        self._model = self._model_cls(model_name, device="cpu", compute_type="int8")

    def set_language(self, language: str | None) -> None:
        self.language = language

    def transcribe_detailed(self, pcm16: np.ndarray) -> tuple[str, str]:
        audio = pcm16.astype(np.float32) / 32768.0
        segments, info = self._model.transcribe(audio, beam_size=1, language=self.language, vad_filter=False)
        text = " ".join(s.text.strip() for s in segments).strip()
        detected = self.language or getattr(info, "language", None) or "en"
        return text, detected

    def transcribe(self, pcm16: np.ndarray) -> str:
        return self.transcribe_detailed(pcm16)[0]

    async def atranscribe(self, pcm16): return await asyncio.to_thread(self.transcribe, pcm16)
    async def atranscribe_detailed(self, pcm16): return await asyncio.to_thread(self.transcribe_detailed, pcm16)
```

`tts.py`:
```python
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

def has_devanagari(text: str) -> bool:
    return bool(_DEVANAGARI.search(text))

class Synthesizer:
    def __init__(self, voice, models_dir, speed=1.0, hindi_voice="hf_alpha"): ... self.hindi_voice = hindi_voice
    def synth(self, text: str, lang: str | None = None):
        hindi = lang == "hi" or (lang is None and has_devanagari(text))
        voice, kl = (self.hindi_voice, "hi") if hindi else (self.voice, "en-us")
        samples, sr = self._engine.create(text, voice=voice, speed=self.speed, lang=kl)
        ...
    async def asynth(self, text, lang=None): return await asyncio.to_thread(self.synth, text, lang)
```

`voices.py`: add `HINDI_VOICES`, `HINDI_VOICE_IDS = list(HINDI_VOICES.values())`, `DEFAULT_HINDI_VOICE = "hf_alpha"`, `ALL_VOICE_IDS = VOICE_IDS + HINDI_VOICE_IDS`, `is_hindi_voice(vid) = vid.startswith("h")`; in `resolve_voice`: single word in `HINDI_VOICES` → id; `_ACCENT` gains `"hindi": "h"`, `"indian": "h"`; the prefix loop iterates `ALL_VOICE_IDS`. `next_voice` and `VOICE_IDS` unchanged (English cycling only). `display_name` already works.

`sentences.py`: `_END = re.compile(r"(?<=[.!?।])\s+(?=[A-Z0-9ऀ-ॿ])")`; the "buffer ends with terminator" check includes `"।"`.

`config.py`: the three fields.

- [ ] **Step 4:** `uv run pytest -q` → PASS. Commit `feat(speech): language-aware STT/TTS, Hindi voices, Devanagari sentence boundaries`.

---

### Task 4: Hinglish local intents, language intent, confirm words

**Files:**
- Modify: `veronica/brain/intents.py`, `veronica/orchestrator.py` (CONFIRM/DENY tables only)
- Test: `tests/test_intents.py`, `tests/test_orchestrator.py`

**Interfaces:**
- Produces: `match_language_intent(text) -> Literal["en","hi","auto"] | None`; existing `match_intent` returns the same `Intent` values for the Hinglish phrases below.

- [ ] **Step 1: Failing tests** — `tests/test_intents.py`:

```python
@pytest.mark.parametrize("text,expected", [
    ("bas", "end"), ("bas karo", "end"), ("theek hai bas", "end"), ("chup", "end"), ("chup raho", "end"),
    ("band karo", "end"), ("ruko", "end"), ("ruk jao", "end"),
    ("mute karo", "mute"), ("awaaz band karo", "mute"), ("unmute karo", "unmute"), ("awaaz chalu karo", "unmute"),
    ("chhoti ho jao", "hud_mini"), ("chota karo", "hud_mini"), ("badi ho jao", "hud_full"), ("bada karo", "hud_full"),
    ("quit karo", "quit"), ("band ho jao", "quit"),
    ("Veronica, bas karo please", "end"),
])
def test_hinglish_local_intents(text, expected):
    assert match_intent(text) == expected


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
```

`tests/test_orchestrator.py`:
```python
@pytest.mark.parametrize("heard,ok", [
    ("haan", True), ("haanji", True), ("ji haan", True), ("theek hai", True), ("karo", True), ("ha", True),
    ("nahi", False), ("nahin", False), ("mat karo", False), ("rehne do", False), ("haan nahi", False),
])
def test_is_confirmation_hinglish(heard, ok):
    assert Orchestrator.is_confirmation(heard) is ok
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** — extend the existing phrase sets in `intents.py` (find the frozensets used by `_match_candidate` for end/mute/unmute/hud_mini/hud_full/quit and add the Hinglish phrases). Add:

```python
LanguageMode = Literal["en", "hi", "auto"]
_LANG_PHRASES: dict[str, LanguageMode] = {
    "speak hindi": "hi", "talk in hindi": "hi", "switch to hindi": "hi", "hindi mein bolo": "hi",
    "hindi me bolo": "hi", "hindi mein baat karo": "hi", "hindi me baat karo": "hi", "speak in hindi": "hi",
    "speak english": "en", "talk in english": "en", "switch to english": "en", "english mein bolo": "en",
    "english me bolo": "en", "angrezi mein bolo": "en", "speak in english": "en",
    "understand both": "auto", "both languages": "auto", "hindi and english": "auto", "hindi aur english": "auto",
    "auto language": "auto", "dono bhasha": "auto",
}

def match_language_intent(text: str) -> LanguageMode | None:
    for candidate in _candidates_for(normalize(text)):
        if candidate in _LANG_PHRASES:
            return _LANG_PHRASES[candidate]
    return None
```
(Whole-utterance candidates only — no clause split, so "translate this to hindi" stays with the brain.)

`intents.py`: set `HINGLISH_INTENT_PHRASES` to the frozenset of every Hinglish phrase added above (end/mute/unmute/hud/quit) plus the Hinglish keys of `_LANG_PHRASES`.

`orchestrator.py`: `CONFIRM_WORDS |= {"haan", "ha", "haanji", "ji haan", "theek hai", "karo"}`, `DENY_WORDS |= {"nahi", "nahin", "mat", "rehne"}` (multi-word "mat karo"/"rehne do" are denied via the single words `mat`/`rehne`; `"haan nahi"` → deny wins, as today for "yes no"). Note `"karo"` alone confirms, while `"mat karo"` denies because `mat` is checked first.

- [ ] **Step 4:** `uv run pytest -q` → PASS. Commit `feat(intents): Hinglish phrases for local intents, language switch intent, Hinglish yes/no`.

---

### Task 5: Language mode end-to-end (orchestrator, prompt, main, menu, HUD font, download, README)

**Files:**
- Modify: `veronica/orchestrator.py`, `veronica/brain/prompts.py`, `veronica/__main__.py`, `veronica/ui/menubar.py`, `veronica/ui/hud/hud.css`, `scripts/download_models.py`, `README.md`
- Test: `tests/test_orchestrator.py`, `tests/test_agent.py`, `tests/test_main.py`, `tests/test_menubar.py`

**Interfaces:**
- Consumes: Task 3 (`Transcriber(model, language)`, `transcribe_detailed`, `Synthesizer.synth(text, lang)`, `hindi_voice`, `voices.HINDI_VOICE_IDS`), Task 4 (`match_language_intent`), Task 2 (`_utterance_lang`).
- Produces: `Orchestrator(..., stt_factory: Callable[[str, str | None], Any] | None = None, language: str = "en")` where `stt_factory(model_name, language)` returns a Transcriber-like; `Orchestrator.language` (mode), `async _language_turn(mode)`, `_stt_spec(mode) -> tuple[str, str | None, str]` = `(main_model, language_kwarg, partial_model)`; `say(text, *, lang=None)`; `handle_text(text, images=(), *, lang=None)`.

- [ ] **Step 1: Failing tests** — `tests/test_orchestrator.py`:

```python
# -- batch C: language mode -----------------------------------------------------

class STT2(STT):
    def __init__(self, texts, langs=None):
        super().__init__(texts); self.langs = list(langs or []); self.language = "en"; self.model_name = "small.en"
    async def atranscribe_detailed(self, pcm):
        t = await self.atranscribe(pcm)
        return t, (self.langs.pop(0) if self.langs else "en")
    def set_language(self, lang): self.language = lang


class TTS3(TTS):
    def __init__(self):
        super().__init__(); self.voice = "af_sarah"; self.speed = 1.0; self.hindi_voice = "hf_alpha"; self.langs = []
    async def asynth(self, text, lang=None):
        self.langs.append((text, lang)); return await super().asynth(text)


def build_lang(stt_texts, langs=(), mode="en", monkeypatch=None):
    saved = []
    if monkeypatch:
        monkeypatch.setattr(prefs_mod, "save", lambda d: saved.append(d))
    made = []
    def factory(model, language):
        s = STT2([], []); s.model_name = model; s.language = language; made.append((model, language)); return s
    states, events = [], []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT2(stt_texts, langs),
        brain=Brain(), tts=TTS3(), player=Player(), on_state=states.append,
        on_event=lambda k, p: events.append((k, p)), stt_factory=factory, language=mode,
    )
    return o, saved, made, events


async def test_utterance_lang_from_detection_and_script():
    o, *_ = build_lang(["kal meeting hai"], langs=["hi"], mode="auto")
    await o.one_turn()
    assert o._utterance_lang == "hi"
    assert o.tts.langs[-1][1] == "hi"                      # brain reply spoken with the Hindi voice
    o, *_ = build_lang(["कल मीटिंग है"], langs=["en"], mode="en")   # Devanagari wins even if detector says en
    await o.one_turn()
    assert o._utterance_lang == "hi"
    o, *_ = build_lang(["what time is it"], langs=["en"], mode="auto")
    await o.one_turn()
    assert o._utterance_lang == "en" and o.tts.langs[-1][1] == "en"


async def test_hinglish_phrase_in_auto_mode_counts_as_hindi():
    o, *_ = build_lang(["shukriya"], langs=["en"], mode="auto")
    await o.one_turn()
    assert o.tts.said[-1] in {"Koi baat nahi.", "Hamesha."} and o.tts.langs[-1][1] == "hi"


async def test_language_switch_turn_swaps_models_and_saves(monkeypatch):
    o, saved, made, ev = build_lang(["speak hindi"], mode="en", monkeypatch=monkeypatch)
    await o.one_turn()
    assert o.language == "hi"
    assert made[-2:] == [("small", "hi"), ("tiny", "hi")]     # main + partial
    assert o.stt.model_name == "small" and o.partial_stt.model_name == "tiny"
    assert {"language": "hi"} in saved
    assert o.tts.said[:2] == ["Ek minute, Hindi load kar rahi hoon.", "Ab Hindi mein baat karte hain."]
    assert o.brain.asked == []

    o.stt = STT2(["speak english"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.language == "en" and made[-2:] == [("small.en", "en"), ("tiny.en", "en")]
    assert o.tts.said[-2:] == ["One moment, switching to English.", "Okay, English it is."]

    o.stt = STT2(["dono bhasha"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.language == "auto" and made[-2:] == [("small", None), ("tiny", None)]
    assert o.tts.said[-1] == "Theek hai, dono chalega."


async def test_language_switch_same_model_only_sets_language(monkeypatch):
    o, saved, made, _ = build_lang(["speak hindi"], mode="auto", monkeypatch=monkeypatch)
    o.stt.model_name = "small"; o.partial_stt = STT2([]); o.partial_stt.model_name = "tiny"
    await o.one_turn()
    assert made == [] and o.stt.language == "hi" and o.partial_stt.language == "hi"
    assert o.tts.said == ["Ab Hindi mein baat karte hain."]


async def test_announcements_speak_without_forced_lang():
    o, *_ = build_lang([], mode="hi")
    await o._deliver_announcement(("Timer done.", None))
    assert o.tts.langs[-1] == ("Timer done.", None)
```

`tests/test_agent.py`: system prompt contains `"Reply in the same language they used"`.

`tests/test_main.py`: with prefs `{"language": "hi"}` and `audio=True` fakes, `build_orchestrator` constructs `Transcriber("small", language="hi")` and partial `Transcriber("tiny", language="hi")`, `orch.language == "hi"`; with no pref → `("small.en", "en")`/`("tiny.en", "en")`; `orch.stt_factory` is callable and returns a `Transcriber`; `saved["tts_hindi_voice"]` applied to `tts.hindi_voice` when valid.

`tests/test_menubar.py`: Voice submenu titles end with `["Alpha", "Beta", "Omega", "Psi"]` after a separator following "Normal speed"... — **decide**: place Hindi voices after the English ten (before the speed items) separated by `None`; assert the 14 voice names in order and that picking "Omega" schedules `_voice_turn(("voice", "omega"))` and that `_refresh_voice_menu` checkmarks both the current English voice and the current Hindi voice.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement**

`orchestrator.py`:
- ctor kwargs `stt_factory=None, language="en"`; store `self.stt_factory`, `self.language`.
- `_stt_spec(mode)`: `"en"` → `(self.s.whisper_model, "en", self.s.partial_stt_model)`; `"hi"` → `(self.s.whisper_multilingual_model, "hi", self.s.partial_stt_multilingual_model)`; `"auto"` → `(multilingual, None, partial multilingual)`.
- `_language_turn(mode)`: if `mode == self.language` → say `"Already."`-style? No: just re-confirm with the final copy. Determine `main_model, lang, partial_model = _stt_spec(mode)`. If `self.stt.model_name != main_model` (or partial's differs) and `self.stt_factory` is not None: say the loading line (`"Ek minute, Hindi load kar rahi hoon."` for `hi`, `"One moment, switching to English."` for `en`, `"Ek minute."` for `auto`), then `self.stt = await asyncio.to_thread(self.stt_factory, main_model, lang)`; if `self.partial_stt is not None`: `self.partial_stt = await asyncio.to_thread(self.stt_factory, partial_model, lang)`; else just `set_language(lang)` on both. Set `self.language = mode`, `prefs.save({"language": mode})`, `_emit("tool", {"summary": f"Language: {mode}", "decision": "auto"})`, say the confirmation (`"Ab Hindi mein baat karte hain."` / `"Okay, English it is."` / `"Theek hai, dono chalega."`) — the Hindi confirmations with `lang="hi"`.
- In `one_turn`: replace `text = await self.stt.atranscribe(pcm)` with `text, detected = await self._transcribe(pcm)` where `_transcribe` uses `atranscribe_detailed` if present else `(await atranscribe(pcm), "en")`; then `self._utterance_lang = self._lang_for(text, detected)`:
  ```python
  def _lang_for(self, text, detected):
      if has_devanagari(text) or detected == "hi":
          return "hi"
      if self.language in ("hi", "auto") and quick.is_hinglish_phrase(text):
          return "hi"
      return "en"
  ```
  Task 1 adds `quick.is_hinglish_phrase(text) -> bool` (normalized text ∈ any `_*_HI` set or a Hinglish social/intent phrase — export a `HINGLISH_PHRASES` frozenset from `quick.py` that also unions `intents`' Hinglish sets; simplest: `quick.py` builds it from its own `_HI` sets plus a `HINGLISH_INTENT_PHRASES` constant exported by `intents.py` in Task 4). Language intent dispatch: `lang_mode = match_language_intent(text)` goes right after `voice_action` in the chain (before proactive), guarded like the others; branch → `self.player.reset(); await self._language_turn(lang_mode)`.
- `say(text, *, lang=None)` → `_say_unlocked(text, kind, lang)` → `self.tts.asynth(text, lang)` (fake TTS classes in older tests accept only `text` — call `asynth(text)` when `lang is None` to keep them working, else `asynth(text, lang)`; or update the base fake `TTS.asynth` signature to `(self, text, lang=None)` — do the latter, it's one line). `handle_text(text, images=(), *, lang=None)` passes `lang` to the pipelined synth; `one_turn` passes `lang=self._utterance_lang` for brain replies and quick replies; announcements/confirm prompts pass nothing (auto by script).
- `_deliver_announcement` unchanged (uses `say(text)`).

`prompts.py`: append `" The user may speak Hindi or Hinglish. Reply in the same language they used: if they spoke Hinglish (Hindi with English words, or romanized Hindi), reply in romanized Hinglish using Latin letters; if they spoke pure Hindi, reply in Devanagari; if they spoke English, reply in English. Keep replies just as short."`

`__main__.py`: `def make_stt(model, language): return Transcriber(model, language=language)`; read `saved.get("language")` (validate ∈ en/hi/auto else `s.language`); build `stt`/`partial_stt` from `_stt_spec`-equivalent logic (put `stt_spec(settings, mode)` as a module function in `veronica/speech/stt.py` so both `__main__` and the orchestrator use it — **do this**, and have `Orchestrator._stt_spec` call it); `hindi_voice = saved.get("tts_hindi_voice") if in HINDI_VOICE_IDS else DEFAULT_HINDI_VOICE`; pass `stt_factory=make_stt, language=mode` to the orchestrator.

`_voice_turn` (Task 2 of Batch B) update: if the resolved id `is_hindi_voice` → set `self.tts.hindi_voice`, `prefs.save({"tts_hindi_voice": vid})`, say `"Theek hai, ab main aise bolungi."` with `lang="hi"`; unknown-voice copy lists the Hindi names too: `"… George, Lewis, and in Hindi Alpha, Beta, Omega and Psi."` (update the Batch B test expecting `endswith("George and Lewis.")` → `endswith("Alpha, Beta, Omega and Psi.")`).

`menubar.py`: after the ten English items add `None` then the four Hindi names (same `_pick_voice`); `_refresh_voice_menu` checks `item.state = 1` when `name == display_name(tts.voice)` or `name == display_name(tts.hindi_voice)`; popup mirrors.

`hud.css` line 1 font stack: `-apple-system,system-ui,"SF Pro Text","Noto Sans Devanagari","Devanagari Sangam MN",sans-serif`.

`scripts/download_models.py`: `--hindi` flag → `WhisperModel("small", …)` and `WhisperModel("tiny", …)` constructed once to trigger the download (mirror how the script prefetches whisper today; if it doesn't, add both `small.en`/`tiny.en` default prefetch plus the multilingual pair under `--hindi`).

`README.md`: "Quick replies" (list of what's answered locally) and "Hindi & Hinglish" (switch phrases, what to expect, first-time ~500 MB download, Hindi voices in the Voice menu, "use a hindi voice").

- [ ] **Step 4:** `uv run pytest -q` → PASS. Commit `feat: Hindi/Hinglish language mode end-to-end (STT swap, prompt, Hindi voice, menu, docs)`.

---

## Self-review
- Spec coverage: C1 → T1+T2 (tables, math, battery/volume, memory log, HUD card, precedence guard). C2 → T3 (STT/TTS/splitter/voices/config), T4 (Hinglish intents, language intent, confirm words), T5 (mode switch + factory, `_utterance_lang`, prompt, TTS lang plumbing, Hindi voice pref + menu, HUD font, download flag, README).
- Placeholders: T5 implementation is prose+signatures with exact copy and test contracts; the touched code paths (`say`, `handle_text`, `one_turn`) are large and must be read in place — the implementer is instructed to read them. T1/T2/T3/T4 carry code.
- Type consistency: `match_quick` returns `(kind, lang)` for battery/volume per the T2 decision — T1's tests must use `("battery", "en")`, `("battery", "hi")`, `("volume", "en")` (T1 implementer: apply that to the parametrize table). `stt_spec(settings, mode)` lives in `stt.py` and is used by `__main__` and the orchestrator. `Synthesizer.synth(text, lang)` matches `asynth(text, lang)` and the orchestrator's `_say_unlocked`.
