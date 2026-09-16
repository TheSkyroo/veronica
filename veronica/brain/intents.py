"""Local voice intents: phrases handled entirely in the orchestrator, without
ever going to the brain (end the conversation, resize/hide the HUD)."""
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


def match_intent(text: str) -> Intent | None:
    """Match a heard utterance (as-spoken, not yet normalized) against the
    local intent phrase sets. Matches the whole normalized utterance, with
    an optional leading "veronica"/"hey veronica" and trailing "please"."""
    norm = normalize(text)
    for candidate in {norm, _strip_wrapper(norm)}:
        if not candidate:
            continue
        if candidate in END_PHRASES:
            return "end"
        if candidate in HUD_MINI_PHRASES:
            return "hud_mini"
        if candidate in HUD_FULL_PHRASES:
            return "hud_full"
        if candidate in HUD_HIDE_PHRASES:
            return "hud_hide"
    return None
