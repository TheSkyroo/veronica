"""Local fast-path replies: trivial questions answered without the brain.

Time, date, day, battery/volume level, greetings/thanks/goodbye and simple
spoken arithmetic, in English and Hinglish. Everything here matches the
WHOLE utterance (after normalize() plus the same wrapper/filler stripping
the local intents use; no clause splitting), so a longer request that merely
contains "hello" or "time" still goes to the brain.

Math is the one matcher that does NOT use normalize(): that strips ".",
"-", "," and the symbol operators, which turned "3.14 times 2" into
"314 times 2" and "10-3" into "103". It runs on the raw text instead (see
_math_prep) and is evaluated with a small recursive-descent parser over a
whitelisted token set -- never eval()."""
import datetime as dt
import random
import re
from collections.abc import Callable

from veronica.brain.intents import _FILLERS_BY_LEN, _candidates_for, normalize

QuickReply = tuple[str, str]
_rng = random.Random()

# -- phrase tables (written in normalize()+strip_wrapper form) -----------------
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
    return f"{x:.4f}".rstrip("0").rstrip(".")


# -- math ---------------------------------------------------------------------
_MATH_LEAD = re.compile(r"^(?:whats|what is|calculate|compute|how much is|tell me)\s+")
_MATH_TRAIL_HI = re.compile(r"\s+(?:kitna hota hai|kya hota hai)$")
# Raw-text stand-ins for normalize()+_strip_wrapper: the wake-word prefix,
# a trailing "please", trailing sentence punctuation.
_MATH_WAKE_RE = re.compile(r"^(?:hey\s+)?veronica[\s,]+")
_MATH_PLEASE_RE = re.compile(r"[\s,]*\bplease$")
_MATH_TRAIL_PUNCT_RE = re.compile(r"[\s.!?]+$")
# A digit touching . , : is a decimal, thousands separator or clock time --
# the parser is integers only, so those go to the brain.
_MATH_NON_INTEGER_RE = re.compile(r"\d[.,:]|[.,:]\d")
_MATH_FILLER_RE = re.compile(r"^(?:" + "|".join(re.escape(f) for f in _FILLERS_BY_LEN) + r")[\s,]+")
_MATH_SYMBOL_SPACING_RE = re.compile(r"([-+*/^%()])")
_MATH_UNICODE_OPS = str.maketrans({"×": "*", "÷": "/", "−": "-", "–": "-", "—": "-"})
# (phrase, token, spoken en, spoken hi). Multi-word phrases come first so
# "to the power of" isn't half-rewritten by a later single word. The
# symbol rows are identity for the tokenizer but give _spoken its words
# ("5 + 5" is read back as "5 plus 5").
_OPS = [
    ("to the power of", "^", "to the power of", "ki power"),
    ("multiplied by", "*", "times", "guna"),
    ("divided by", "/", "divided by", "bhaag"),
    ("percent of", "%", "percent of", "percent of"),
    ("% of", "%", "percent of", "percent of"),
    ("square root of", "sqrt", "square root of", "square root of"),
    ("plus", "+", "plus", "jama"), ("jama", "+", "plus", "jama"),
    ("minus", "-", "minus", "ghata"), ("ghata", "-", "minus", "ghata"),
    ("times", "*", "times", "guna"), ("guna", "*", "times", "guna"), ("x", "*", "times", "guna"),
    ("over", "/", "divided by", "bhaag"), ("bhaag", "/", "divided by", "bhaag"), ("bhag", "/", "divided by", "bhaag"),
    ("squared", "sq", "squared", "ka square"),
    ("+", "+", "plus", "jama"), ("-", "-", "minus", "ghata"), ("*", "*", "times", "guna"),
    ("/", "/", "divided by", "bhaag"), ("^", "^", "to the power of", "ki power"), ("%", "%", "percent of", "percent of"),
]
_TOKEN_RE = re.compile(r"\d{1,12}|[-+*/^%()]|sqrt|sq|\S+")
_SYMBOLS = frozenset("+-*/^%()")
_WORD_TOKENS = frozenset({"sqrt", "sq"})
_MAX_OPERATORS = 6


def _tokenize(expr: str) -> list[str] | None:
    """Spoken operators -> symbols, then split. None if any token isn't a
    number/operator (so "5 plus 5 in binary" is left to the brain), there is
    no operator at all (a bare "5" or "2024" is an answer, not a sum), or the
    expression has more than _MAX_OPERATORS operators."""
    s = f" {expr} "
    for phrase, tok, _, _ in _OPS:
        s = s.replace(f" {phrase} ", f" {tok} ")
    toks = _TOKEN_RE.findall(s)
    for t in toks:
        if not (t.isdigit() or t in _SYMBOLS or t in _WORD_TOKENS):
            return None
    n_ops = sum(1 for t in toks if not t.isdigit() and t not in "()")
    if n_ops == 0 or n_ops > _MAX_OPERATORS:
        return None
    return toks


class _Parser:
    """expr := term (('+'|'-') term)*
       term := power (('*'|'/'|'%') power)*     -- 'p % x' is "p percent of x"
       power := unary ('^' power)? ('sq')*       -- '^' right-assoc, 'sq' postfix
       unary := '-' unary | 'sqrt' unary | '(' expr ')' | number"""

    def __init__(self, toks: list[str]) -> None:
        self.t, self.i = toks, 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self) -> str | None:
        tok = self.peek()
        self.i += 1
        return tok

    def expr(self) -> float:
        v = self.term()
        while self.peek() in ("+", "-"):
            op = self.take()
            r = self.term()
            v = v + r if op == "+" else v - r
        return v

    def term(self) -> float:
        v = self.power()
        while self.peek() in ("*", "/", "%"):
            op = self.take()
            r = self.power()
            if op == "*":
                v *= r
            elif op == "/":
                if r == 0:
                    raise ZeroDivisionError
                v /= r
            else:
                v = r * v / 100
        return v

    def power(self) -> float:
        v = self.unary()
        if self.peek() == "^":
            self.take()
            v = v ** self.power()
        while self.peek() == "sq":
            self.take()
            v = v * v
        return v

    def unary(self) -> float:
        tok = self.peek()
        if tok == "-":
            self.take()
            return -self.unary()
        if tok == "sqrt":
            self.take()
            v = self.unary()
            if v < 0:
                raise ValueError("negative sqrt")
            return v ** 0.5
        if tok == "(":
            self.take()
            v = self.expr()
            if self.take() != ")":
                raise ValueError("unbalanced parens")
            return v
        if tok is not None and tok.isdigit():
            return float(self.take())
        raise ValueError(f"unexpected token {tok!r}")


def evaluate(expr: str) -> float | None:
    """Evaluate a spoken/symbolic arithmetic expression. None when it isn't
    plain arithmetic, is too long, divides by zero, or overflows."""
    toks = _tokenize(expr.lower())
    if not toks:
        return None
    try:
        p = _Parser(toks)
        v = p.expr()
        if p.peek() is not None or isinstance(v, complex) or abs(v) > 1e15 or v != v:
            return None
        return v
    except (ValueError, ZeroDivisionError, OverflowError, IndexError, TypeError):
        return None


def _spoken(expr: str, hi: bool) -> str:
    """Render the expression the way it should be read back ("12 x 8" ->
    "12 times 8" / "12 guna 8")."""
    s = f" {expr} "
    for phrase, _tok, en, hi_word in _OPS:
        s = s.replace(f" {phrase} ", f" {hi_word if hi else en} ")
    return " ".join(s.split())


def _math_prep(text: str) -> str:
    """The raw-text counterpart of normalize()+_strip_wrapper for math:
    lowercase, drop apostrophes ("what's" -> "whats"), strip a leading
    "veronica,"/"hey veronica", one leading filler ("okay", "can you"),
    a trailing "please" and trailing sentence punctuation, normalise the
    unicode operator glyphs, and put spaces around symbol operators so
    "10-3" tokenizes as 10 - 3. Digits, ".", "," and ":" are otherwise
    left alone so _match_math can see a decimal or clock time and bail."""
    s = (text or "").lower().replace("'", "").replace("’", "").translate(_MATH_UNICODE_OPS)
    s = " ".join(s.split())
    s = _MATH_WAKE_RE.sub("", s)
    s = _MATH_TRAIL_PUNCT_RE.sub("", s)
    s = _MATH_PLEASE_RE.sub("", s)
    s = _MATH_TRAIL_PUNCT_RE.sub("", s)
    s = _MATH_FILLER_RE.sub("", s)
    return " ".join(_MATH_SYMBOL_SPACING_RE.sub(r" \1 ", s).split())


def _match_math(text: str) -> QuickReply | None:
    """Arithmetic over the RAW utterance (see _math_prep), not normalize()d
    text, so symbols and decimals stay visible. Integers only."""
    s = _math_prep(text)
    if not s or _MATH_NON_INTEGER_RE.search(s):
        return None
    hi = bool(_MATH_TRAIL_HI.search(s))
    body = _MATH_TRAIL_HI.sub("", s) if hi else _MATH_LEAD.sub("", s)
    words = body.split()
    # Must contain a digit and end in an operand ("set a timer for 5 minutes"
    # ends in "minutes" -> not math).
    if not words or not re.search(r"\d", body) or not re.search(r"\d|\)|squared", words[-1]):
        return None
    value = evaluate(body)
    if value is None:
        return None
    spoken, result = _spoken(body, hi), format_number(value)
    return ("math", f"{spoken}, {result} hota hai." if hi else f"{spoken} is {result}.")


# -- public -------------------------------------------------------------------
def reply_for(kind: str, lang: str, **kw) -> str:
    """Copy for replies whose data the orchestrator reads itself:
    kind "battery" (percent, state in charging|discharging|charged|None) and
    "volume" (percent)."""
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
    """True if the whole utterance (wrapper/filler stripped) is one of our
    Hinglish phrases (quick replies or local intents) -- used by the
    orchestrator to treat a romanized-Hindi utterance as Hindi even when the
    transcriber says "en"."""
    from veronica.brain.intents import HINGLISH_INTENT_PHRASES  # filled by the Hinglish intents work
    return any(c in HINGLISH_PHRASES or c in HINGLISH_INTENT_PHRASES for c in _candidates_for(normalize(text)))


def match_quick(text: str, *, now: Callable[[], dt.datetime] = dt.datetime.now, lang: str = "en") -> QuickReply | None:
    """Match an utterance against the fast-path tables. Returns (kind, reply)
    where kind is time|date|day|social|math (reply is the spoken copy) or
    battery|volume (reply is the language, "en"/"hi", and the orchestrator
    builds the copy with reply_for once it has read the value). A Hinglish
    phrase always answers in Hindi; an English phrase answers in `lang`."""
    norm = normalize(text)
    if not norm:
        return None
    hi = lang == "hi"
    candidates = list(_candidates_for(norm))

    def pick(en_set, hi_set):
        for c in candidates:
            if c in hi_set:
                return True
            if c in en_set:
                return hi
        return None

    t = now()
    for en_set, hi_set, kind, en_fn, hi_fn in (
        (_TIME, _TIME_HI, "time", lambda: f"It's {_clock(t)}.", lambda: f"Abhi {_clock(t)} hain."),
        (_DATE, _DATE_HI, "date", lambda: f"It's {t.strftime('%A, %B')} {ordinal(t.day)}.",
         lambda: f"Aaj {t.strftime('%A')}, {t.day} {t.strftime('%B')} hai."),
        (_DAY, _DAY_HI, "day", lambda: f"It's {t.strftime('%A')}.", lambda: f"Aaj {t.strftime('%A')} hai."),
        (_BATTERY, _BATTERY_HI, "battery", lambda: "en", lambda: "hi"),
        (_VOLUME, _VOLUME_HI, "volume", lambda: "en", lambda: "hi"),
    ):
        use_hi = pick(en_set, hi_set)
        if use_hi is not None:
            return (kind, hi_fn() if use_hi else en_fn())
    for c in candidates:
        if c in _GREETINGS:
            return ("social", _GREETINGS[c])
        if c in _GOOD_NIGHT:
            return ("social", _GOOD_NIGHT[c])
    for en_set, hi_set, en_replies, hi_replies in _SOCIAL:
        use_hi = pick(en_set, hi_set)
        if use_hi is not None:
            return ("social", _rng.choice(hi_replies if use_hi else en_replies))
    return _match_math(text)
