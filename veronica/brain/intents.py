"""Local voice intents: phrases handled entirely in the orchestrator, without
ever going to the brain (end the conversation, resize/hide the HUD)."""
import re
import string
from typing import Literal

Intent = Literal["end", "hud_mini", "hud_full", "hud_hide"]

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    return (text or "").lower().replace("’", "").translate(_PUNCT_TABLE).strip()


END_PHRASES = frozenset({
    "thanks veronica", "thank you veronica", "that's all", "thats all",
    "that is all", "stop", "goodbye", "never mind", "nevermind",
    "go idle", "turn yourself off", "go to sleep", "sleep", "go away",
    "bye", "dismiss",
})

HUD_MINI_PHRASES = frozenset({
    "make yourself small", "make yourself smaller", "shrink", "shrink yourself",
    "minimize", "minimise", "mini mode", "small mode", "go small",
})

HUD_FULL_PHRASES = frozenset({
    "expand", "expand yourself", "make yourself big", "make yourself bigger",
    "full mode", "show details", "go big",
})

HUD_HIDE_PHRASES = frozenset({
    "hide", "hide yourself", "hide the hud", "hide the panel",
})

_LEAD_PREFIXES = ("hey veronica ", "veronica ")
_TRAIL_SUFFIX = " please"

# Leading filler words/phrases stripped before matching a clause — spoken
# hedging ("I think...") or politeness ("please...") that shouldn't stop an
# otherwise-exact phrase from matching. "okay"/"ok" are included so e.g.
# "Okay stop." still ends the conversation.
FILLERS = frozenset({"i think", "please", "can you", "could you", "just", "okay", "ok"})
_FILLERS_BY_LEN = tuple(sorted(FILLERS, key=len, reverse=True))

# Clauses are split on sentence/list punctuation and the word "and", so a
# single utterance carrying multiple thoughts ("Make yourself small. I can't
# see you.") can still match on its first clause.
_CLAUSE_SPLIT_RE = re.compile(r"[.,!?;]+|\s+and\s+", re.IGNORECASE)


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
    return None


def match_intent(text: str) -> Intent | None:
    """Match a heard utterance (as-spoken, not yet normalized) against the
    local intent phrase sets. Tries the whole normalized utterance first
    (with an optional leading "veronica"/"hey veronica", trailing "please",
    and leading filler stripped), then each clause of the raw utterance in
    order — split on '.', ',', '!', '?', ';' and " and " — so the first
    clause that exactly matches a phrase wins."""
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
