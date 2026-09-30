import asyncio
import contextlib
import datetime as dt
import difflib
import logging
import math
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from veronica import prefs
from veronica import proactive as proactive_mod
from veronica import version
from veronica.audio import devices, input_level
from veronica.audio.chime import tone
from veronica.audio.speaker import ENROL_MIN_S, voiced_s
from veronica.brain import quick
from veronica.brain.agent import confirm_prompt
from veronica.brain.backends import BACKENDS, check_backend
from veronica.brain.backends.cli import LimitError
from veronica.brain.gate import GateServer
from veronica.brain.intents import (
    is_pause_phrase,
    is_resume_phrase,
    is_stop_dictation,
    match_brain_intent,
    match_dictation_intent,
    match_intent,
    match_language_intent,
    match_memory_intent,
    match_music_intent,
    match_note_intent,
    match_proactive_intent,
    match_screen_intent,
    match_settings_intent,
    match_speaker_intent,
    match_update_intent,
    match_version_intent,
    match_voice_intent,
    normalize,
)
from veronica.config import Settings
from veronica.speech import voices
from veronica.speech.stt import stt_spec
from veronica.brain.sentences import has_devanagari
from veronica.tools import mac as mac_tools
from veronica.tools import music as music_tools
from veronica.tools import pim as pim_tools
from veronica.tools import registry
from veronica.tools.screen import capture_screenshot
from veronica.ui.events import envelope
from veronica.updater import UpdateInProgress

log = logging.getLogger("veronica.orchestrator")

# Sentinel returned by the PTT-aware capture helpers in place of a PCM array
# when the push-to-talk key went down while the capture was waiting: the
# caller should switch to a hold-mode (push-to-talk) capture instead.
PTT = object()

# What confirm()'s listen returns for an answer in a voice that isn't the
# enrolled one: never an answer, only a reason to listen once more.
_NOT_THE_USER = object()

# Trailing "stop dictation" (etc.) spoken in the same breath as the last
# dictated sentence: stripped from what gets typed, and ends the dictation.
_TRAILING_STOP_DICTATION_RE = re.compile(
    r"[\s,.;!?]*\b(?:stop|end)\s+dictat(?:ion|ing)\b[\s.!?]*$", re.IGNORECASE
)

# Small counts read better spoken as words ("Forgot three things") than as
# digits; past ten the digits are fine.
_COUNT_WORDS = ("no", "one", "two", "three", "four", "five",
                "six", "seven", "eight", "nine", "ten")


def _count_word(n: int) -> str:
    return _COUNT_WORDS[n] if 0 <= n < len(_COUNT_WORDS) else str(n)


def _cancel_or_reap(fut: asyncio.Future) -> None:
    """Cancel a not-yet-done synth future; for one that already completed
    (possibly with an exception) before we got to it, retrieve the result
    instead so asyncio doesn't complain about an exception nobody read."""
    if not fut.cancel():
        with contextlib.suppress(BaseException):
            fut.exception()


@dataclass(frozen=True)
class ConfirmResult:
    """What confirm() heard: "approved" (a yes), "denied" (a no, or
    silence), or "other" — anything that is neither, which becomes the next
    request. Truthy only when approved, so `if await confirm(...)` callers
    keep working."""
    outcome: Literal["approved", "denied", "other"]
    heard: str = ""
    # "yes, and stop asking": an approval that also asks her to remember it.
    # The gate persists it (Settings.auto_allow_tools) for the tools that may
    # be remembered at all; a plain yes covers this one call.
    always: bool = False

    def __bool__(self) -> bool:
        return self.outcome == "approved"


class Orchestrator:
    # No bare "ha": whisper writes laughter as "ha ha", which must never
    # approve a tool. Devanagari forms are for pinned Hindi mode, where
    # whisper emits the script rather than romanized Hindi.
    #
    # Two classes (controller ruling, batch D): STRONG confirms count
    # wherever they are (last decisive wins); FILLER confirms ("okay",
    # "sure", "theek hai"…) are how people start a question too ("okay so
    # what will it delete"), so they only count when the whole utterance is
    # nothing but confirm phrases.
    STRONG_CONFIRMS = frozenset({
        "yes", "yeah", "yep", "yup", "do it", "go ahead", "confirm", "absolutely", "please do", "go for it",
        "haan", "ji", "haanji", "ji haan", "karo", "bilkul", "kar do",
        "हाँ", "हां", "जी", "जी हाँ", "करो", "बिल्कुल", "कर दो",
    })
    FILLER_CONFIRMS = frozenset({
        "ok", "okay", "alright", "fine", "correct", "of course", "sure",
        "theek hai", "ठीक है", "ठीक",
    })
    CONFIRM_WORDS = STRONG_CONFIRMS | FILLER_CONFIRMS
    DENY_WORDS = frozenset({
        "no", "nope", "nah", "not", "don't", "dont", "cancel", "stop", "never", "skip",
        "nahi", "nahin", "mat", "rehne",
        "नहीं", "नही", "मत", "रहने",
    })
    # A question or hesitation after the last confirm phrase ("yes what?",
    # "ok wait", "alright hold on") means the answer isn't a yes.
    QUESTION_WORDS = frozenset({
        "what", "which", "how", "why", "wait", "hold",
        # apostrophes are stripped before tokenising, so contractions arrive
        # as "whats"/"hows"/"whys"/"wheres"
        "whats", "hows", "whys", "wheres", "when", "where",
        "kya", "kaun", "kaunsa", "kaunsi", "kyun", "kab", "ruko", "रुको", "क्या", "कौन", "क्यों", "कब",
    })
    # A confirm phrase directly after one of these is negated ("not okay",
    # "don't do it", "mat karo") rather than counted as a yes.
    _NEGATORS = frozenset({"not", "dont", "never", "no", "nahi", "nahin", "mat", "नहीं", "नही", "मत"})
    _SPOKEN_END_PHRASES = frozenset({"thanks veronica", "thank you veronica"})
    # Word characters for is_confirmation: Latin letters plus the Devanagari
    # block (U+0900-U+097F, which includes the vowel signs and chandrabindu).
    # Latin letters and Devanagari letters/marks are word characters; the
    # danda/double danda (U+0964/0965 — Hindi full stops, which whisper glues
    # onto the last word) and Devanagari digits are NOT, or "नहीं।" would be
    # one unknown token and "हाँ, नहीं।" would approve.
    _CONFIRM_NON_WORD_RE = re.compile(r"[^a-z\u0900-\u0963\u0970-\u097f ]")

    @staticmethod
    def is_confirmation(heard: str) -> bool:
        """True only if a confirm phrase is the *last decisive* thing said.

        Scans left to right: a deny word anywhere after the last confirm
        phrase wins ("yes… actually no" → False), but a deny before a later
        confirm does not ("no no, I said yes, do it" → True). A confirm
        phrase immediately preceded by a negator ("not okay", "don't do
        it") is not a confirm. A question/hesitation word after the last
        confirm phrase ("yes what?", "ok wait") is decisive-negative.
        Filler confirms ("okay", "sure", "fine"…) count only when the
        utterance consists solely of confirm phrases ("okay", "okay do it",
        "alright yes" — not "okay so what will it delete", "is that
        correct"). With no confirm phrase at all the answer is always
        False — never default to yes."""
        no_apostrophes = heard.lower().replace("'", "").replace("’", "")
        words = Orchestrator._CONFIRM_NON_WORD_RE.sub(" ", no_apostrophes).split()
        if not words:
            return False
        last_deny = max((i for i, w in enumerate(words) if w in Orchestrator.DENY_WORDS), default=-1)

        def matches(phrases):
            """(start, end) of every non-negated occurrence of any phrase."""
            found = []
            for phrase in phrases:
                pw = phrase.split()
                n = len(pw)
                for i in range(len(words) - n + 1):
                    if words[i:i + n] == pw and not (i > 0 and words[i - 1] in Orchestrator._NEGATORS):
                        found.append((i, i + n - 1))
            return found

        strong = matches(Orchestrator.STRONG_CONFIRMS)
        filler = matches(Orchestrator.FILLER_CONFIRMS)
        if not strong and not filler:
            return False
        last_confirm = max(end for _, end in strong + filler)
        if any(w in Orchestrator.QUESTION_WORDS for w in words[last_confirm + 1:]):
            return False
        last_strong = max((end for _, end in strong), default=-1)
        if last_strong >= 0 and last_strong > last_deny:
            return True
        # Fillers only: every word must belong to some confirm phrase.
        covered = set()
        for start, end in strong + filler:
            covered.update(range(start, end + 1))
        return len(covered) == len(words)

    # Words that can pad a yes/no without turning it into a request ("yes
    # please", "no thanks", "haan ji", "not now"). Anything else left over
    # after the confirm/deny tokens is new content the brain should hear.
    # "yes, and stop asking me" — an approval with a standing request
    # attached. The words carry a negator ("do NOT confirm again"), which
    # would otherwise read as taking the yes back, so they are matched
    # first. (Persisting the "stop asking" part is the auto-allow setting;
    # this only makes sure the answer counts as the yes it plainly is.)
    ALWAYS_PHRASES = (
        "dont ask again", "do not ask again", "dont ask me again", "dont ask",
        "dont confirm again", "do not confirm again", "dont confirm", "no need to confirm",
        "no need to ask", "stop asking", "without asking", "never ask again",
        "mat pucho", "mat poocho", "puchna mat", "मत पूछो", "हमेशा",
        "always", "always allow",
    )

    # Words that turn a "yes" into a qualified answer: the user is steering
    # somewhere else rather than approving what was asked.
    REDIRECT_MARKERS = frozenset({
        "but", "instead", "except", "rather", "actually", "although", "though",
        "however", "wait", "hold", "lekin", "magar", "balki", "लेकिन", "मगर", "बल्कि",
    })

    ANSWER_FILLERS = frozenset({
        "please", "pls", "thanks", "thank", "you", "veronica", "now", "it", "that", "this",
        "then", "so", "and", "just", "already", "really", "ahead", "for", "me", "on",
        "right", "good", "great", "cool", "yes", "no",
        "um", "uh", "hmm", "hm", "oh", "ah", "well", "er",
        "ji", "na", "hai", "ha", "do", "kar", "kijiye", "zaroor", "abhi", "bhai", "yaar", "aap", "toh", "to",
        "जी", "ना", "है", "अभी", "अब", "ही", "तो",
    })

    @staticmethod
    def _answer_words(heard: str | None) -> list[str]:
        """A confirm reply reduced to bare lowercase words."""
        no_apostrophes = (heard or "").lower().replace("'", "").replace("’", "")
        return Orchestrator._CONFIRM_NON_WORD_RE.sub(" ", no_apostrophes).split()

    @staticmethod
    def says_always(heard: str | None) -> bool:
        """True when the answer asked her to stop asking (ALWAYS_PHRASES).
        Only meaningful alongside an approval: "no, dont ask again" is a no,
        and classify_answer says so."""
        flat = " ".join(Orchestrator._answer_words(heard))
        return any(phrase in flat for phrase in Orchestrator.ALWAYS_PHRASES)

    @staticmethod
    def classify_answer(heard: str | None) -> Literal["approved", "denied", "other"]:
        """Three-way reading of a confirm reply. Silence is a no. A yes or a
        no with at most a little padding ("yes please", "no thanks", "haan
        karo") is what it says. Anything carrying content beyond the answer
        — a question ("what will that do?"), a qualifier ("yes, but in
        Chrome"), an instruction ("no, open it in Safari instead"), six or
        more leftover words — is "other": not an answer, but the next
        request."""
        words = Orchestrator._answer_words(heard)
        if not words:
            return "denied"
        flat = " ".join(words)
        for phrase in Orchestrator.ALWAYS_PHRASES:
            if phrase not in flat:
                continue
            # "yeah, just don't confirm again": an approval, whatever the
            # negator inside the phrase itself would otherwise say. A "no"
            # OUTSIDE the phrase still wins ("no, and don't ask again").
            rest = flat.replace(phrase, " ").split()
            return "denied" if any(w in Orchestrator.DENY_WORDS for w in rest) else "approved"
        confirmed = Orchestrator.is_confirmation(heard)
        # Every word that is part of a confirm phrase, negated or not.
        covered = set()
        for phrase in Orchestrator.CONFIRM_WORDS:
            pw = phrase.split()
            for i in range(len(words) - len(pw) + 1):
                if words[i:i + len(pw)] == pw:
                    covered.update(range(i, i + len(pw)))
        if confirmed and len(covered) == len(words):
            return "approved"   # pure yes, e.g. "okay yes do it"
        answer_tokens = Orchestrator.DENY_WORDS | Orchestrator._NEGATORS
        remaining = [w for i, w in enumerate(words) if i not in covered and w not in answer_tokens]
        if any(w in Orchestrator.QUESTION_WORDS for w in words):
            return "other"
        if len(remaining) >= 6:
            return "other"          # a whole instruction rode along
        if confirmed:
            # A yes stays a yes however it is dressed up — "yes sir",
            # "yes yes yes do it man", "yeah go on then". Only a word that
            # takes it back or points somewhere else ("yes, but in Chrome")
            # makes it the next request instead; requiring every extra word
            # to be on a whitelist turned ordinary approvals into declines.
            return "other" if any(w in Orchestrator.REDIRECT_MARKERS for w in remaining) else "approved"
        if any(w not in Orchestrator.ANSWER_FILLERS for w in remaining):
            return "other"
        # No yes, no content: a no, or a mumble with nothing to redirect to.
        return "denied"

    # Phrases that, inside a request, mean "and don't ask me first"; with
    # STRONG_CONFIRMS ("copy this, just do it", "open chrome and go ahead",
    # "haan kar do abhi") they pre-approve the one confirm-class action that
    # request produces (see Brain.preapprove).
    PREAPPROVE_PHRASES = frozenset({
        "without asking", "no need to ask", "dont ask", "bina puche", "bina pooche",
        "बिना पूछे",
    })
    # A request that opens like a question ("should I do it?", "can you do
    # it without asking") is asking, not telling.
    _QUESTION_LEADS = frozenset({"should", "shall", "can", "could", "would", "will", "may", "kya", "क्या"})
    # Padding that keeps a yes an answer ("just do it", "yes please") —
    # ANSWER_FILLERS minus the "now" words: "do it now" / "kar do abhi" is
    # an order, and gets pre-approved.
    _ANSWER_PADDING = ANSWER_FILLERS - {"now", "abhi", "अभी", "अब"}
    PREAPPROVE_WINDOW_S = 20

    @staticmethod
    def detect_preapproval(text: str) -> bool:
        """True when the request's own wording says go ahead: a strong
        confirm / "without asking" phrase, not negated and not taken back
        by a later no, inside something that is NOT itself a pure confirm
        answer ("do it", "yes", "haan karo" — those answer a question) and
        not a question ("should I do it?"). Pure, no state."""
        raw = (text or "").strip()
        no_apostrophes = raw.lower().replace("'", "").replace("’", "")
        words = Orchestrator._CONFIRM_NON_WORD_RE.sub(" ", no_apostrophes).split()
        if not words or raw.endswith("?"):
            return False
        if words[0] in Orchestrator._QUESTION_LEADS or any(w in Orchestrator.QUESTION_WORDS for w in words):
            return False
        hits: list[tuple[int, int]] = []
        for phrase in Orchestrator.STRONG_CONFIRMS | Orchestrator.PREAPPROVE_PHRASES:
            pw = phrase.split()
            n = len(pw)
            for i in range(len(words) - n + 1):
                if words[i:i + n] == pw and not (i > 0 and words[i - 1] in Orchestrator._NEGATORS):
                    hits.append((i, i + n - 1))
        if not hits:
            return False
        covered = {j for start, end in hits for j in range(start, end + 1)}
        # Nothing but confirm phrases and padding ("just do it", "okay yes
        # do it"): an answer, not a request.
        for phrase in Orchestrator.FILLER_CONFIRMS:
            pw = phrase.split()
            for i in range(len(words) - len(pw) + 1):
                if words[i:i + len(pw)] == pw:
                    covered.update(range(i, i + len(pw)))
        if all(i in covered or w in Orchestrator._ANSWER_PADDING for i, w in enumerate(words)):
            return False
        last_hit = max(end for _, end in hits)
        # "do it... actually no": a deny after the last go-ahead takes it back
        # (a deny that is part of a phrase, "no need to ask", doesn't count).
        last_deny = max((i for i, w in enumerate(words) if w in Orchestrator.DENY_WORDS and i not in covered), default=-1)
        return last_deny < last_hit

    def __init__(self, settings: Settings, *, wake, recorder, stt, brain, tts, player,
                 partial_stt=None, store=None,
                 on_state: Callable[[str], None] | None = None,
                 on_event: Callable[[str, Any], None] | None = None,
                 on_quit: Callable[[], None] | None = None,
                 proactive=None,
                 input_guard=None,
                 stt_factory: Callable[[str, str | None], Any] | None = None,
                 language: str = "en",
                 updater_check: Callable[[], Any] | None = None,
                 updater_update: Callable[[Any], str] | None = None,
                 relaunch: Callable[[], bool] | None = None,
                 can_relaunch: Callable[[], bool] | None = None,
                 version_describe: Callable[[], str] | None = None,
                 switcher=None,
                 speaker=None) -> None:
        self.s = settings
        # Optional veronica.audio.speaker.SpeakerGate ("only my voice"): with
        # a voice profile, a capture in someone else's voice is dropped as if
        # nothing was said (see _speaker_ok). None = every voice is heard.
        self.speaker = speaker
        # Optional veronica.brain.switch.BrainSwitcher: owns the active
        # brain (`self.brain` reads through to it), manual switches and the
        # usage-limit failover. None (tests, older callers) pins `brain`.
        self.switcher = switcher
        self._gate_server: GateServer | None = None
        # Self-update (D3): `updater_check()` -> UpdateStatus, `updater_update(
        # status)` -> log text, `relaunch()` restarts the app (and quits this
        # process). All three are injected by the menu bar app; None (tests,
        # --text mode) means "update yourself" just says it can't here.
        # `can_relaunch()` says up front whether relaunch() will reopen a
        # bundle (vs. just quit a dev run) so the "restart me" hint can be
        # spoken BEFORE the quit is scheduled; None = unknown.
        self.updater_check = updater_check
        self.updater_update = updater_update
        self.relaunch = relaunch
        self.can_relaunch = can_relaunch
        # "What version are you": the menu bar passes a cached describe();
        # the default asks git, so it runs on a thread, off the loop.
        self.version_describe = version_describe
        # Optional veronica.proactive.Proactive: the briefing/nudge ticker.
        # Started once by run_forever; its schedule is what the "brief me"
        # / "turn on nudges" intents edit. None in --text mode.
        self.proactive = proactive
        # Optional veronica.audio.input_level.InputLevelGuard: raises the
        # Mac's input volume back to settings.input_volume_floor when a call
        # app / device switch lowers it. run_forever starts its periodic
        # loop once and re-checks after every PortAudio refresh. None in
        # --text mode.
        self.input_guard = input_guard
        self._input_guard_task: asyncio.Task | None = None
        self._input_guard_stop: asyncio.Event | None = None
        self._input_hint_shown = False
        self.wake, self.recorder, self.stt = wake, recorder, stt
        self.brain, self.tts, self.player = brain, tts, player
        self.partial_stt = partial_stt
        self.store = store
        # Language mode ("en" | "hi" | "auto"): which whisper models are
        # loaded and how the transcriber is hinted. stt_factory(model_name,
        # language) builds a fresh Transcriber when a "speak hindi"-style
        # switch needs different models; None (tests, --text mode) means a
        # switch only re-hints the transcribers already loaded.
        self.language = language
        self.stt_factory = stt_factory
        # Language of the current utterance ("en" or "hi"): set from the
        # transcriber's detection / the script / a known Hinglish phrase at
        # the top of each turn; read by the quick replies and used to pick
        # the voice the reply is spoken with.
        self._utterance_lang = "en"
        self._on_state = on_state or (lambda _: None)
        self._on_event = on_event
        self._on_quit = on_quit or (lambda: None)
        self.state = "idle"
        self._muted = False
        self._unmute_event = asyncio.Event()
        self.ready = False
        self._speech_lock = asyncio.Lock()
        self._speech_queue: asyncio.Queue | None = None
        # True while ANY recorder.capture() is in flight (every capture goes
        # through self._capture()), so a barge/PTT teardown knows it must
        # recorder.stop() to unblock the capture thread before cancelling
        # the turn — whether that capture belongs to confirm(), dictation,
        # the follow-up window, or anything else.
        self._capture_in_flight = False
        # Whether the last capture was a hold (push-to-talk) one: the key
        # press is the proof it's the user, so it skips the speaker check.
        self._last_capture_hold = False
        # The "voice check unavailable" card is shown once per session.
        self._speaker_failed_shown = False
        self._barged = False
        # True only while confirm() has the mic open for its yes/no. Inside
        # that window the user speaking IS the answer, so _run_with_barge
        # must route a wake barge / PTT press to the capture instead of
        # tearing the turn down (which used to turn "yes" into a decline).
        self._confirm_listening = False
        # monotonic seconds of the last "On it." — see ACK_MIN_GAP_S
        self._last_ack_at = -1e9
        # F2: the sentences a barge stopped her before she could say them,
        # as the same (text, synth future) pairs handle_text's queue held —
        # kept alive so "continue" can speak them without re-synthesising.
        # Every barge parks them; one_turn drops them the moment the user
        # says anything other than a pause/continue phrase, and a new brain
        # turn drops them too, so they can never leak into a later answer.
        self._paused_tail: list[tuple[str, asyncio.Future]] | None = None
        self._now_speaking = ""
        # Brain turns are numbered per handle_text call; the brain gets the
        # id (begin_turn) so a pre-approval can be pinned to one turn.
        self._turn_id = 0
        # F3: the current turn's tool calls as a checklist ({summary, state}),
        # rebuilt per brain turn. `_plan_turn` is True only inside one, so a
        # local intent's card can't append to a turn that is already over;
        # `_plan_shown` records whether the HUD is currently showing a card
        # (it only ever is once a turn made more than one call).
        self._plan: list[dict] = []
        self._plan_turn = False
        self._plan_shown = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._partial_task: asyncio.Task | None = None
        # Push-to-talk is a *signal* into run_forever / the in-flight turn,
        # never a concurrent turn of its own: ptt_start() (key down) sets
        # _ptt_event, which run_forever races alongside the wake listener
        # while idle and _run_with_barge races alongside the barge listener
        # during a turn; every other capture races it too. _ptt_held tracks
        # the physical key (down..up); _ptt_capturing is True only while the
        # hold-mode capture itself is in flight, so a release before the
        # capture started is seen (via _ptt_held) rather than lost, and a
        # second press during Veronica's answer can barge it (spec A2).
        self._ptt_event = asyncio.Event()
        self._ptt_held = False
        self._ptt_capturing = False
        # Bumped every time a partial-eligible capture() returns, before STT
        # runs on it: any partial transcription still in flight (or one that
        # races in from the recorder thread right at that boundary) belongs
        # to a capture that's already over and must be dropped.
        self._partial_gen = 0
        if self.recorder is not None and self.partial_stt is not None:
            self.recorder.on_audio = self._on_recorder_audio
        # Own-speech suppression for the whisper wake engine: the mic's rolling
        # analysis window can still hold the tail of a just-finished sentence
        # (e.g. "...I'm Veronica") for up to wake_window_s + wake_hop_s after
        # _now_speaking is cleared, so we keep offering the last-spoken text to
        # the suppress check for that long too. self._clock is overridable in
        # tests to fake time.
        self._last_spoken = ""
        self._last_spoken_until = 0.0
        # last few sentences she spoke, for echo rejection on follow-ups
        self._recent_spoken: deque[str] = deque(maxlen=6)
        self._clock = time.monotonic
        self._announce_queue: asyncio.Queue = asyncio.Queue()

    @property
    def muted(self) -> bool:
        return self._muted

    @muted.setter
    def muted(self, value: bool) -> None:
        was_muted = self._muted
        self._muted = bool(value)
        if was_muted and not self._muted:
            # Unmuting: wake run_forever's wait (if it's parked there) so any
            # announcement that queued up while muted is delivered right
            # away instead of waiting for the next wake word. `muted` can be
            # set from a different thread (e.g. the menu bar's AppKit
            # thread), so this must go through call_soon_threadsafe rather
            # than setting the asyncio.Event directly.
            loop = self._loop
            if loop is not None:
                with contextlib.suppress(RuntimeError):
                    loop.call_soon_threadsafe(self._unmute_event.set)
            else:
                self._unmute_event.set()

    async def warmup(self) -> None:
        """Load models before the first turn so the first answer isn't slow."""
        self._loop = asyncio.get_running_loop()
        self._emit("warm", {"ready": False})
        self._set("warming")
        t0 = time.monotonic()
        await self.tts.asynth("ok")
        if self.stt is not None:
            await self.stt.atranscribe(np.zeros(16000, dtype=np.int16))
        if self.partial_stt is not None:
            await self.partial_stt.atranscribe(np.zeros(16000, dtype=np.int16))
        log.info("warmup done in %.1fs", time.monotonic() - t0)
        self.ready = True
        self._set("idle")
        self._emit("warm", {"ready": True})

    @property
    def brain(self):
        return self.switcher.brain if self.switcher is not None else self._brain

    @brain.setter
    def brain(self, value) -> None:
        self._brain = value
        if self.switcher is not None:
            self.switcher.brain = value

    @property
    def gate(self):
        """The shared ToolGate (None with the bare test brains)."""
        if self.switcher is not None:
            return self.switcher.gate
        return getattr(self._brain, "gate", None)

    def _brain_label(self) -> str:
        if self.switcher is not None:
            return self.switcher.status_label()
        info = BACKENDS.get(getattr(self.brain, "name", "claude"))
        return info.label if info else "Claude"

    def backend_changed(self, label: str, standing_in: bool) -> None:
        """BrainSwitcher.on_backend: the HUD and menu bar show the label."""
        self._emit("hud", {"backend": label})

    def _set(self, state: str) -> None:
        self.state = state
        log.info("state=%s", state)
        self._on_state(state)
        self._emit("state", state)

    def _emit(self, kind: str, payload) -> None:
        if self._on_event is not None:
            try:
                self._on_event(kind, payload)
            except Exception:
                log.exception("on_event failed for %s", kind)
        # The plan card is a *view* of the tool cards, so fold every one into
        # the turn's checklist — after the card itself has gone out, so a
        # listener sees the action before the plan that contains it.
        if kind == "tool" and isinstance(payload, dict):
            self._plan_note(str(payload.get("summary") or ""), str(payload.get("decision") or ""))

    # -- plan card (F3) -------------------------------------------------------
    # A turn that makes more than one tool call shows them as a checklist
    # instead of one card at a time. Nothing here decides anything: the steps
    # ARE the decisions ToolGate already reported (through tool_card) plus the
    # ones confirm() already emitted, so the card can never show an action the
    # gate didn't allow, and the gate never learns the card exists.

    # A decision that starts a step, and the state it starts in: an auto or
    # pre-approved call is already executing, an asked one waits on the user.
    _PLAN_START = {"auto": "running", "preapproved": "running", "ask": "pending"}
    # ...and how confirm()'s answer settles the step its "ask" started. A
    # redirect is a decline of *this* action (whatever the user said instead
    # comes back as its own request).
    _PLAN_SETTLE = {"allowed": "running", "declined": "declined", "redirected": "declined"}

    def tool_card(self, summary: str, decision: str) -> None:
        """ToolGate's on_tool hook: the cards it reports (auto, trusted,
        pre-approved) go out through here rather than straight to the HUD, so
        they land in the plan alongside the ones confirm() emits."""
        self._emit("tool", {"summary": summary, "decision": decision})

    def _plan_reset(self) -> None:
        """Top of a brain turn: the checklist is per turn, never cumulative."""
        self._plan = []
        self._plan_turn = True
        if self._plan_shown:
            # A card from the previous turn can still be on screen (a turn
            # that never passed through 'heard' — an announcement, --text
            # mode — doesn't clear the HUD by itself), so empty it explicitly.
            self._plan_shown = False
            self._emit("plan", {"steps": []})

    def _plan_note(self, summary: str, decision: str) -> None:
        """Fold one tool decision into the turn's checklist."""
        if not self._plan_turn:
            return          # a local intent's card outside a brain turn isn't a step
        state = self._PLAN_START.get(decision)
        if state is not None:
            # A new call is proof the previous one finished: the gate only
            # sees the next tool once the brain has the last one's result.
            for step in self._plan:
                if step["state"] == "running":
                    step["state"] = "done"
            self._plan.append({"summary": summary, "state": state})
        elif decision in self._PLAN_SETTLE and self._plan:
            self._plan[-1]["state"] = self._PLAN_SETTLE[decision]
        else:
            return          # "limit" and the like: swapping brains is not a step
        self._plan_emit()

    def _plan_emit(self) -> None:
        """Only a multi-step turn gets a card: one tool call keeps the plain
        action card it has always had."""
        if len(self._plan) < 2:
            return
        self._plan_shown = True
        self._emit("plan", {"steps": [dict(step) for step in self._plan]})

    def _plan_finish(self) -> None:
        """End of the turn, however it ended: nothing is still running."""
        self._plan_turn = False
        if any(step["state"] == "running" for step in self._plan):
            for step in self._plan:
                if step["state"] == "running":
                    step["state"] = "done"
            self._plan_emit()

    # -- speaking -------------------------------------------------------------
    def _finished_speaking(self, text: str) -> None:
        """Called right after a play() of `text` returns: keep offering it to
        the suppress check for wake_window_s + wake_hop_s more, since the
        mic's rolling analysis window can still hold its audio tail."""
        self._last_spoken = text
        self._last_spoken_until = self._clock() + self.s.wake_window_s + self.s.wake_hop_s
        self._now_speaking = ""
        self._recent_spoken.append(text)

    _OWN_SPEECH_MIN_WORDS = 3
    _OWN_SPEECH_RATIO = 0.72

    def _is_own_speech(self, heard: str) -> bool:
        """True if a follow-up transcript is (mostly) Veronica's own last few
        sentences leaking back in through the mic — speakers + laptop mic,
        or a Bluetooth input whose audio arrives hundreds of ms late, after
        followup_skip_ms has already elapsed. Compared against the recent
        sentences with a fuzzy ratio so partial echoes still match."""
        norm = normalize(heard)
        words = norm.split()
        if len(words) < self._OWN_SPEECH_MIN_WORDS or not self._recent_spoken:
            return False
        recent = [normalize(t) for t in self._recent_spoken if t]
        joined = " ".join(recent)
        if norm in joined:
            return True
        for sent in recent:
            if difflib.SequenceMatcher(None, norm, sent).ratio() >= self._OWN_SPEECH_RATIO:
                return True
        # most of the heard words appear in what she just said (echo with a
        # few mis-heard tokens)
        pool = set(joined.split())
        hits = sum(1 for w in words if w in pool)
        return hits / len(words) >= 0.8

    def _suppress_text(self) -> str:
        if self._clock() < self._last_spoken_until:
            return f"{self._now_speaking} {self._last_spoken}"
        return self._now_speaking

    # -- live partial transcript -----------------------------------------------
    def _end_partial_window(self) -> None:
        """Called right after any partial-eligible recorder.capture() returns,
        before STT runs on the result: bumps the generation counter so a
        partial transcription still in flight (or one that races in from the
        recorder thread right at this boundary) is recognized as stale and
        dropped rather than emitted after — or worse, overwriting — this
        turn's real 'heard' text. Also cancels the in-flight task outright."""
        self._partial_gen += 1
        if self._partial_task is not None and not self._partial_task.done():
            self._partial_task.cancel()

    def _on_recorder_audio(self, pcm: np.ndarray) -> None:
        """Called from the Recorder's capture thread (not the event loop)
        every partial_hop_s of captured speech. Hands off to the loop
        thread-safely; coalescing (skip while a partial transcription is
        already running) happens there, not here."""
        loop = self._loop
        if loop is None:
            return
        gen = self._partial_gen
        try:
            loop.call_soon_threadsafe(self._schedule_partial, pcm, gen)
        except RuntimeError:
            pass  # loop closed/closing; drop this partial

    def _schedule_partial(self, pcm: np.ndarray, gen: int) -> None:
        if gen != self._partial_gen:
            return  # the capture this came from is already over
        if self._partial_task is not None and not self._partial_task.done():
            return  # a partial transcription is already in flight; coalesce
        self._partial_task = asyncio.ensure_future(self._run_partial(pcm, gen))

    async def _run_partial(self, pcm: np.ndarray, gen: int) -> None:
        try:
            text = await self.partial_stt.atranscribe(pcm)
        except Exception:
            log.exception("partial transcription failed")
            return
        if gen != self._partial_gen:
            return  # capture ended (or another one started) while transcribing
        if text:
            self._emit("heard_partial", text)

    async def _say_unlocked(self, text: str, kind: str = "sentence", lang: str | None = None) -> None:
        # Called only while _speech_lock is already held (by say()/confirm()).
        # Emit right before play so a listener never sees "sentence"/"voice"
        # (or "prompt"/"voice") for audio that hasn't actually started playing
        # yet. `lang` picks the voice ("hi" -> the Hindi voice); None lets
        # the synthesizer decide from the script.
        samples, sr = await self.tts.asynth(text, lang)
        self._emit("voice", {"step_ms": 50, "levels": envelope(samples, sr)})
        self._emit(kind, text)
        self._now_speaking = text
        try:
            await self.player.play(samples)
        finally:
            self._finished_speaking(text)

    async def say(self, text: str, *, lang: str | None = None) -> None:
        if self.muted:
            return
        async with self._speech_lock:
            await self._say_unlocked(text, lang=lang)

    async def chime(self, freq_hz: float, ms: int) -> None:
        if self.muted:
            return
        async with self._speech_lock:
            self.player.reset()
            try:
                await self.player.play(tone(freq_hz, ms))
            except Exception:
                # A chime is a courtesy, not the turn — never let an audio
                # device hiccup abort listening or a reply.
                log.warning("chime failed", exc_info=True)

    async def handle_text(self, text: str, images: list[bytes] = (), *, lang: str | None = None) -> list[str]:
        """Ask the brain and speak each sentence; synth N+1 overlaps playback of N.
        `lang` ("hi"/"en"/None) picks the voice the reply is spoken with."""
        self._set("thinking")
        self._barged = False   # fresh turn: any earlier barge no longer applies
        # A new answer supersedes whatever an earlier pause was holding, even
        # if one_turn never saw the utterance that started it (announcements,
        # --text mode, a brain-side re-ask).
        self._drop_paused_tail("a new turn started")
        self._turn_id += 1
        getattr(self.brain, "begin_turn", lambda _tid: None)(self._turn_id)
        t0 = time.monotonic()
        spoken: list[str] = []
        first = True
        # The item taken off the queue but not yet handed to the player: a
        # pause landing while its synth is still running must keep it, while
        # one landing mid-playback must not make her repeat what was heard.
        unspoken: tuple[str, asyncio.Future] | None = None
        self.player.reset()
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)

        async def producer():
            try:
                # Only pass `images=` when there actually are any, so test
                # doubles for Brain.ask(text) that don't accept the kwarg
                # keep working unchanged.
                ask_iter = self.brain.ask(text, images=images) if images else self.brain.ask(text)
                async for sent in ask_iter:
                    spoken.append(sent)
                    # Enqueue at yield time (synth kicked off but not
                    # necessarily finished) so: (a) maxsize=2 bounds how far
                    # ahead synthesis can run, and (b) a concurrent confirm()
                    # joining the queue sees this sentence as pending *before*
                    # it's been spoken, closing the production ordering race.
                    fut = asyncio.ensure_future(self.tts.asynth(sent, lang))
                    try:
                        await queue.put((sent, fut))
                    except BaseException:
                        # if put() itself is cancelled (e.g. the queue was
                        # full when handle_text was cancelled), fut was never
                        # handed to anything that would cancel it — it'd
                        # otherwise run to completion orphaned.
                        fut.cancel()
                        raise
            finally:
                await queue.put(None)

        def _drain(q: asyncio.Queue, keep: list | None = None) -> None:
            # `keep` is the pause tail: with one, queued sentences are moved
            # into it (futures left running) instead of being cancelled. A
            # future that is already dead is never parked — "continue" would
            # only trip over it.
            while True:
                try:
                    item = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is not None:
                    if keep is None or item[1].cancelled():
                        _cancel_or_reap(item[1])
                    else:
                        keep.append(item)
                q.task_done()

        # Exposed so confirm() (which the brain may await mid-stream, e.g. as
        # a tool-use confirmation gate) can wait for already-queued sentences
        # to finish playing before it speaks its own prompt — otherwise it
        # could win the _speech_lock race against the consumer below and
        # jump the queue.
        self._speech_queue = queue
        prod = asyncio.create_task(producer())
        # "On it." if the brain is still thinking ack_after_s in. Its own
        # task, so it can never sit between the answer and the speakers; it
        # checks `spoken` at fire time, so a turn that has already said
        # something stays silent.
        ack = (
            asyncio.create_task(self._ack_if_slow(lambda: bool(spoken), lang))
            if self.s.ack_after_s > 0 else None
        )
        try:
            self._plan_reset()
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        break
                    sent, fut = item
                    unspoken = item
                    # Shielded: a barge landing in here cancels this turn,
                    # and the synth has to survive that — it is exactly the
                    # sentence "continue" would speak. The finally below
                    # cancels it when nothing is going to.
                    samples, sr = await asyncio.shield(fut)
                    if first:
                        first = False
                        self._set("speaking")
                        log.info("latency first-sentence=%.2fs", time.monotonic() - t0)
                    if self.muted:
                        unspoken = None   # nothing was played; nothing to resume
                    else:
                        async with self._speech_lock:
                            # Emit right after acquiring the lock, immediately
                            # before play, so a listener never sees these
                            # events for audio that hasn't started yet.
                            self._emit("voice", {"step_ms": 50, "levels": envelope(samples, sr)})
                            self._emit("sentence", sent)
                            self._now_speaking = sent
                            # Committed to playing it: a pause from here on
                            # resumes at the sentence AFTER this one, however
                            # little of it the user actually heard.
                            unspoken = None
                            try:
                                await self.player.play(samples)
                            finally:
                                self._finished_speaking(sent)
                finally:
                    # Accounted for whether this item played cleanly, raised,
                    # or we were cancelled mid-item — unfinished_tasks must
                    # always balance to zero so nothing can join() forever.
                    queue.task_done()
            await prod
        finally:
            # Cancellation-safe teardown: whether we exit normally, via an
            # exception raised from the loop above, or because this task
            # itself was cancelled, the producer must be stopped and the
            # queue must never be left with unbalanced put()/task_done()
            # counts — an unbalanced queue would hang any confirm() blocked
            # in queue.join() forever.
            if ack is not None:
                # Cancelled first (and reaped last): a still-sleeping ack
                # must not speak into the silence after the turn is over, and
                # cancel() alone doesn't yield to the loop — awaiting it here
                # would let the producer take another step before it is
                # stopped below, which is a different turn's business.
                ack.cancel()
            # A barge may turn out to be "hold on": park what she hadn't said
            # yet (with its synth already running) instead of cancelling it,
            # so "continue" can pick up exactly there. Anything other than
            # "continue" drops the lot — see _drop_paused_tail.
            tail: list | None = [] if self._barged else None
            if unspoken is not None:
                if tail is not None and not unspoken[1].cancelled():
                    tail.append(unspoken)
                else:
                    # Nobody will await it: shield() kept it alive past the
                    # cancellation, so stop it here rather than orphan it.
                    _cancel_or_reap(unspoken[1])
            prod.cancel()
            # Drain BEFORE awaiting prod: if the queue was full, the
            # producer's own `finally: await queue.put(None)` would block
            # forever with nobody left to consume it. Freeing space here
            # lets that put() (and thus `await prod` below) complete.
            _drain(queue, tail)
            with contextlib.suppress(BaseException):
                await prod
            # The producer's finally may have just put its None sentinel
            # (normal exit already consumed it above, so this is a no-op
            # then); drain it too so unfinished_tasks balances to zero.
            _drain(queue, tail)
            self._speech_queue = None
            if tail:
                self._paused_tail = tail
                log.info("barge parked %d unspoken sentence(s)", len(tail))
            if ack is not None:
                with contextlib.suppress(BaseException):
                    await ack
            # The turn is over — including when it was cancelled out from
            # under us — so no step of it can still be running.
            self._plan_finish()
        if not spoken:
            # A brain that stopped right after a redirected confirm isn't
            # speechless — the redirect is about to be run as the next
            # request (see _brain_turn).
            if not getattr(self.brain, "pending_redirect", None):
                await self.say("I have nothing to say to that.")
        elif self.store is not None and self.s.memory_enabled:
            reply = " ".join(spoken)
            self.store.add_turn(text, reply)
            # The brain never says which facts it leaned on, so infer it from
            # the words that came out (MemoryStore.touch_facts_used) and let
            # that order the facts block. Done here, after the last sentence
            # is spoken, so the scan can never sit between a sentence and the
            # speaker.
            self.store.touch_facts_used(reply)
        return spoken

    # Deliberately tiny: the acknowledgement must read as "heard you, still
    # working", never as the answer itself.
    ACK_TEXT = {"en": "On it.", "hi": "एक सेकंड।"}
    # A brain that habitually takes a few seconds would otherwise earn an
    # "On it." on every single turn, which grates fast. It is for the
    # unusually long wait, so it stays quiet for a while after each one.
    ACK_MIN_GAP_S = 600.0

    async def _ack_if_slow(self, produced: Callable[[], bool], lang: str | None) -> None:
        """Say a short "still here" line ack_after_s into a turn — but only
        while the brain has produced nothing at all, and not if one was
        already said in the last ACK_MIN_GAP_S. Runs as its own task
        (handle_text cancels it when the turn ends), so it can only ever
        take the speech lock ahead of a sentence, never delay one that is
        already synthesised and waiting."""
        await asyncio.sleep(self.s.ack_after_s)
        if produced():
            return
        now = time.monotonic()
        if now - self._last_ack_at < self.ACK_MIN_GAP_S:
            return
        self._last_ack_at = now
        hindi = (lang or self._utterance_lang) == "hi"
        log.info("ack: brain silent for %.1fs", self.s.ack_after_s)
        await self.say(self.ACK_TEXT["hi" if hindi else "en"], lang="hi" if hindi else None)

    def _drop_paused_tail(self, why: str) -> None:
        """Throw away the sentences a pause parked, cancelling the synth
        futures nobody will await now. Anything but "continue" gets here:
        the user has moved on, and a stale remainder spoken into a later
        turn would be worse than saying nothing."""
        tail, self._paused_tail = self._paused_tail, None
        if not tail:
            return
        log.info("dropping %d paused sentence(s): %s", len(tail), why)
        for _sent, fut in tail:
            if not fut.cancel():
                # already finished: retrieve the result/exception so asyncio
                # doesn't complain that nobody looked at it.
                with contextlib.suppress(BaseException):
                    fut.exception()

    async def _resume_tail(self, tail: list[tuple[str, asyncio.Future]]) -> None:
        """Speak what a pause held back, in order, starting at the first
        sentence the user never heard. The synth futures were started by the
        turn that produced them, so nothing is re-synthesised. A barge in
        here parks the rest again — "hold on ... continue ... hold on" works
        as many times as the user likes."""
        rest = list(tail)
        self._barged = False   # like a fresh turn: the pause barge is spent
        self._set("speaking")
        self.player.reset()
        try:
            while rest:
                sent, fut = rest[0]
                try:
                    samples, sr = await asyncio.shield(fut)
                except asyncio.CancelledError:
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise     # a barge is tearing this resume down
                    # The parked synth died with the turn that started it.
                    # One lost sentence is not worth taking the listening
                    # loop down for, so skip it like any other failure.
                    log.warning("parked synthesis was cancelled for %r", sent)
                    rest.pop(0)
                    continue
                except Exception:
                    log.exception("parked synthesis failed for %r", sent)
                    rest.pop(0)
                    continue
                if self.muted:
                    break
                async with self._speech_lock:
                    self._emit("voice", {"step_ms": 50, "levels": envelope(samples, sr)})
                    self._emit("sentence", sent)
                    self._now_speaking = sent
                    rest.pop(0)   # committed: a pause now resumes after it
                    try:
                        await self.player.play(samples)
                    finally:
                        self._finished_speaking(sent)
        finally:
            if self._barged and rest:
                self._paused_tail = rest
                log.info("barge parked %d unspoken sentence(s)", len(rest))
            else:
                # Muted part way through, or the resume ended some other
                # way: these futures have no reader left.
                for _sent, fut in rest:
                    _cancel_or_reap(fut)

    async def _brain_turn(self, text: str, images: list[bytes] = (), *, lang: str | None = None) -> list[str]:
        """handle_text plus the confirm redirect: if a confirmation in this
        turn was answered with something other than yes/no, the brain got
        it in the deny message and has usually re-planned in its reply
        already; if it said nothing after the deny, run the answer as the
        next request in the same session.

        A brain that hits its usage limit raises LimitError out of ask();
        the switcher moves to the next ready brain (and says so) and the
        same request is re-run there once per hop, at most one hop per
        backend so a chain of limits can't loop. The redirect re-run goes
        through the same path — a limit there is a limit like any other."""
        if self.switcher is not None:
            await self.switcher.maybe_return()
            # ...and, the other way round, drop to the local model when the
            # brain that's up needs a vendor host and the wire is dead.
            await self.switcher.maybe_offline()
        spoken = await self._ask_with_failover(text, images, lang=lang)
        if spoken is None:
            return []
        heard = getattr(self.brain, "pending_redirect", None)
        if not heard:
            return spoken
        self.brain.pending_redirect = None
        self._emit("heard", heard)
        log.info("heard=%r (redirected from confirm)", heard)
        if not spoken:
            return await self._ask_with_failover(heard, lang=lang) or []
        # The brain already answered the redirect inside this turn (the deny
        # message carried it), and handle_text stored that reply — no
        # second memory row with the same reply.
        return spoken

    async def _ask_with_failover(self, text: str, images: list[bytes] = (), *,
                                 lang: str | None = None) -> list[str] | None:
        """handle_text, retried on the next ready brain each time one reports
        its usage limit. None = nothing left to say (no brain is ready)."""
        for _hop in range(len(BACKENDS)):
            try:
                return await self.handle_text(text, images, lang=lang)
            except LimitError as e:
                if self.switcher is None:
                    await self.say(f"{self._brain_label()} hit its usage limit.")
                    return None
                old = self._brain_label()
                new = await self.switcher.failover(str(e))
                if new is None:
                    return None     # the switcher said "no other brain is ready"
                self._emit("tool", {"summary": f"{old}: usage limit — on {BACKENDS[new].label}",
                                    "decision": "limit"})
        return None

    # -- brains -----------------------------------------------------------------
    @staticmethod
    def _spoken_list(items: list[str]) -> str:
        """"Codex, Antigravity or Claude" — spoken, so "or" rather than a
        trailing comma."""
        if len(items) == 1:
            return items[0]
        return ", ".join(items[:-1]) + " or " + items[-1]

    async def _brain_switch_turn(self, action: tuple[str, str | None]) -> None:
        """Local fast path for "switch to codex" / "go offline" / "which
        brain are you on": the switcher does the switch (or says why it
        can't — "go online" asks it which brain to go back to); a switch first
        interrupts whatever the current brain is doing and drops any
        screen-control trust. Also run by the menu bar's Brain submenu and
        the settings page."""
        kind, name = action
        sw = self.switcher
        if sw is None:
            await self.say("I can only use Claude right now.")
            return
        if kind == "online":
            name = sw.online_candidate()
            if name is None:
                await self.say("No online brain is ready.")
                return
            kind, name = "switch", name
        if kind == "which_to":
            # "switch it" with no target: name the ones that are ready rather
            # than handing a bare "switch it" to the brain, which can't act.
            ready = [BACKENDS[n].label for n in BACKENDS
                     if n != sw.brain.name and check_backend(n).ok]
            if not ready:
                await self.say(f"I'm on {self._brain_label()} — nothing else is set up.")
                return
            await self.say(f"Switch to which one — {self._spoken_list(ready)}?")
            return
        if kind == "which":
            label = self._brain_label()
            if sw.standing_in and sw.brain.name == "local" and sw._standin_reason == "offline":
                await self.say("I'm on the local model — there's no internet.")
                return
            if sw.standing_in and sw.brain.name in BACKENDS:
                until = sw.limited_until.get(sw.preferred, 0.0)
                mins = max(1, round((until - sw._clock()) / 60))
                pref = BACKENDS[sw.preferred].label
                if until > sw._clock():
                    await self.say(f"I'm on {BACKENDS[sw.brain.name].label} — {pref} hit its limit, "
                                   f"I'll try it again in {mins} minutes.")
                else:
                    await self.say(f"I'm on {BACKENDS[sw.brain.name].label} — {pref} isn't ready.")
            elif sw.brain.name in BACKENDS:
                await self.say(f"I'm on {label}.")
            else:
                await self.say("No brain is ready right now.")
            return
        if name not in BACKENDS:
            await self.say(f"I don't know a brain called {name}.")
            return
        if sw.brain.name == name and not sw.standing_in:
            await self.say(f"Already on {BACKENDS[name].label}.")
            return
        await self.brain.interrupt()
        getattr(self.brain, "clear_trust", lambda: None)()
        avail = await sw.switch(name)
        await self.say(f"Switched to {BACKENDS[name].label}." if avail.ok else avail.hint)

    def request_brain_switch(self, name: str) -> None:
        """Menu bar / settings: run the "switch to <name>" turn on the
        orchestrator loop. Safe to call from the AppKit thread."""
        async def _turn():
            self.player.reset()
            await self._brain_switch_turn(("switch", name))

        try:
            here = asyncio.get_running_loop()
        except RuntimeError:
            here = None
        loop = self._loop
        if here is not None:
            here.create_task(_turn())                    # called on the loop itself
        elif loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(_turn(), loop)   # the AppKit thread
        else:
            log.warning("brain switch to %s requested before the loop started; ignored", name)

    async def start_brain(self) -> None:
        """Activate the brain (the switcher picks the preferred one or a
        stand-in) and open the gate socket the external brains' processes
        ask for permission on. run_forever does this; --text mode calls it."""
        if self.switcher is not None:
            await self.switcher.start()
        self._emit("hud", {"backend": self._brain_label()})
        gate = self.gate
        if gate is not None and self._gate_server is None:
            # run_tool: an external brain's tools.serve child asks the gate
            # to run our tools here, in the app — the process macOS granted
            # Screen Recording, Accessibility and Apple Events to.
            self._gate_server = GateServer(gate, self.s.gate_socket, run_tool=registry.call_tool)
            await self._gate_server.start()

    async def stop_brain(self) -> None:
        server, self._gate_server = self._gate_server, None
        if server is not None:
            await server.stop()

    # -- screen awareness -------------------------------------------------------
    async def _screen_turn(self, text: str) -> list[str]:
        """Local fast path for "what's on my screen"-style utterances:
        capture the screen ourselves (no ambiguity about which tool to call,
        no extra round trip) and hand both the text and the image to the
        brain in one turn."""
        self._emit("tool", {"summary": "Look at screen", "decision": "auto"})
        await self.chime(self.s.chime_wake_hz, 80)
        result = await asyncio.to_thread(capture_screenshot, "screen")
        if isinstance(result, str):
            log.warning("screen capture failed: %s", result)
            return await self._brain_turn(text, lang=self._utterance_lang)
        data, _path, _mime = result
        return await self._brain_turn(text, [data], lang=self._utterance_lang)

    # -- music -----------------------------------------------------------------
    _MUSIC_SUMMARIES = {
        "play": "Play music", "pause": "Pause music", "next": "Next track",
        "prev": "Previous track", "now_playing": "What's playing",
    }

    async def _music_turn(self, action: str) -> None:
        """Local fast path for "pause"/"resume"/"next song"/"previous"/
        "what's playing": call the music tool directly (no brain round
        trip) and speak its one-line result text."""
        handlers = {
            "play": music_tools.music_play, "pause": music_tools.music_pause,
            "next": music_tools.music_next, "prev": music_tools.music_prev,
            "now_playing": music_tools.music_now_playing,
        }
        self._emit("tool", {"summary": self._MUSIC_SUMMARIES[action], "decision": "auto"})
        res = await handlers[action].handler({})
        text = res["content"][0]["text"]
        if res.get("is_error"):
            text = "Sorry, I couldn't do that."
        await self.say(text)

    # -- voice & speed (B1) ---------------------------------------------------------
    async def _voice_turn(self, action: tuple[str, str]) -> None:
        """Local fast path for "use a british voice" / "speak faster":
        mutate the running Synthesizer, persist to prefs.json, and confirm
        in the new voice/speed so the user hears the change immediately.
        Also called by the menu bar's Voice/Speed items."""
        kind, arg = action
        if kind == "voice":
            vid = voices.next_voice(self.tts.voice) if arg == "next" else voices.resolve_voice(arg)
            if vid is None:
                names = [voices.display_name(v) for v in voices.VOICE_IDS]
                hindi = [voices.display_name(v) for v in voices.HINDI_VOICE_IDS]
                await self.say(
                    "I don't have that voice. I have " + ", ".join(names)
                    + ", and in Hindi " + ", ".join(hindi[:-1]) + " and " + hindi[-1] + "."
                )
                return
            if voices.is_hindi_voice(vid):
                # A Hindi voice only ever speaks Hindi replies; the English
                # voice stays as it was.
                self.tts.hindi_voice = vid
                prefs.save({"tts_hindi_voice": vid})
                self._emit("tool", {"summary": f"Voice: {voices.display_name(vid)}", "decision": "auto"})
                await self.say("ठीक है, अब मैं ऐसे बोलूँगी।", lang="hi")
                if getattr(self, "language", "en") == "en":
                    # Picking a Hindi voice while pinned to English is a
                    # strong hint they want to *speak* Hindi too — the
                    # English-only whisper can't hear it, so open up to
                    # both languages.
                    await self._language_turn("auto")
                return
            self.tts.voice = vid
            prefs.save({"tts_voice": vid})
            self._emit("tool", {"summary": f"Voice: {voices.display_name(vid)}", "decision": "auto"})
            await self.say(f"Okay, this is {voices.display_name(vid)}.")
            return
        # speed
        cur = float(self.tts.speed)
        if arg == "faster":
            new = voices.clamp_speed(cur + voices.SPEED_STEP)
            if new <= cur:
                await self.say("That's as fast as I go.")
                return
        elif arg == "slower":
            new = voices.clamp_speed(cur - voices.SPEED_STEP)
            if new >= cur:
                await self.say("That's as slow as I go.")
                return
        else:
            new = voices.DEFAULT_SPEED
            if abs(cur - new) < 1e-9:
                await self.say("Already at normal speed.")
                return
        self.tts.speed = round(new, 2)
        prefs.save({"tts_speed": self.tts.speed})
        self._emit("tool", {"summary": f"Speed: {self.tts.speed:.2f}x", "decision": "auto"})
        await self.say("Like this?")

    # -- language mode (C2) ---------------------------------------------------------
    _LANG_LOADING = {
        "hi": "एक मिनट, हिंदी load कर रही हूँ।",
        "en": "One moment, switching to English.",
        "auto": "Ek minute.",
    }
    _LANG_CONFIRM = {
        "hi": "अब हिंदी में बात करते हैं।",
        "en": "Okay, English it is.",
        "auto": "ठीक है, दोनों चलेगा।",
    }

    def _stt_spec(self, mode: str) -> tuple[str, str | None, str]:
        """(main_model, language_kwarg, partial_model) for a language mode."""
        return stt_spec(self.s, mode)

    async def _language_turn(self, mode: str) -> None:
        """Local fast path for "speak hindi" / "switch to english" / "dono
        bhasha": swap the transcribers for the mode's models (a first-time
        Hindi switch downloads ~500 MB, so warn before it), persist the mode
        to prefs.json and confirm in the new language. When the loaded
        models already match, only the language hint changes — no reload,
        no loading line."""
        main_model, lang, partial_model = self._stt_spec(mode)
        want_partial = self.partial_stt is not None or bool(self.s.partial_stt)
        needs_load = self.stt_factory is not None and (
            getattr(self.stt, "model_name", None) != main_model
            or (want_partial and getattr(self.partial_stt, "model_name", None) != partial_model)
        )
        spoken_lang = "en" if mode == "en" else "hi"
        if needs_load:
            await self.say(self._LANG_LOADING[mode], lang=spoken_lang)
            self._set("thinking")
            # Load both into locals and swap only once both succeeded: a
            # failed download (offline, disk full) must never leave the main
            # transcriber pinned to Hindi with the partial still English, or
            # the mode/prefs out of step with the models actually loaded.
            try:
                new_stt = await asyncio.to_thread(self.stt_factory, main_model, lang)
                new_partial = (
                    await asyncio.to_thread(self.stt_factory, partial_model, lang) if want_partial else None
                )
            except Exception:
                log.exception("language switch failed")
                if mode == "hi":
                    await self.say("हिंदी load नहीं हो पाई, बाद में try करो।", lang="hi")
                else:
                    await self.say("Couldn't switch language, check the log.")
                return
            self.stt = new_stt
            if want_partial:
                self.partial_stt = new_partial
                if self.recorder is not None:
                    # first partial transcriber (none was loaded at startup):
                    # hook the recorder's audio hops up as __init__ would have.
                    self.recorder.on_audio = self._on_recorder_audio
        else:
            for stt in (self.stt, self.partial_stt):
                set_language = getattr(stt, "set_language", None)
                if set_language is not None:
                    set_language(lang)
        self.language = mode
        prefs.save({"language": mode})
        self._emit("tool", {"summary": f"Language: {mode}", "decision": "auto"})
        await self.say(self._LANG_CONFIRM[mode], lang=spoken_lang)

    async def _transcribe(self, pcm: np.ndarray) -> tuple[str, str]:
        """(text, detected language) for an utterance. Transcribers without
        the detailed API (older doubles) are treated as English."""
        detailed = getattr(self.stt, "atranscribe_detailed", None)
        if detailed is None:
            return await self.stt.atranscribe(pcm), "en"
        return await detailed(pcm)

    def _lang_for(self, text: str, detected: str) -> str:
        """The language an utterance is answered in: Devanagari script or a
        "hi" detection means Hindi; in the Hindi/auto modes a known Hinglish
        phrase counts as Hindi too (whisper labels romanized Hindi "en");
        everything else is English."""
        if not text:
            return "en"
        if has_devanagari(text) or detected == "hi":
            return "hi"
        if self.language in ("hi", "auto") and quick.is_hinglish_phrase(text):
            return "hi"
        return "en"

    # -- only my voice ------------------------------------------------------------
    async def _speaker_ok(self, pcm: np.ndarray | None, where: str) -> bool:
        """False when a voice profile is active and `pcm` (just captured) is
        someone else's voice: the caller then treats it as silence — never
        as a request, a follow-up, or a confirm answer. The score is logged
        either way (SpeakerGate.check). Push-to-talk captures pass: the key
        is the proof."""
        sp = self.speaker
        if sp is None or pcm is None or not sp.active or self._last_capture_hold:
            return True
        ok, _score = await asyncio.to_thread(sp.check, pcm, where)
        if getattr(sp, "failed", False) and not self._speaker_failed_shown:
            # Failing open is silent by nature; say so once, so "only my
            # voice" isn't quietly off.
            self._speaker_failed_shown = True
            self._emit("tool", {"summary": "Voice check unavailable, hearing everyone", "decision": "auto"})
        return ok

    def _ignored_voice(self, where: str) -> None:
        log.info("ignored another voice (%s)", where)
        self._emit("tool", {"summary": "Ignored another voice", "decision": "declined"})

    # Read aloud one at a time, the user repeats each: ~3 s of speech apiece
    # is what the embedding needs. Plain English on purpose (a speaker
    # embedding doesn't care about the language), and no wake word in them.
    ENROL_LINES = (
        "The quick brown fox jumps over the lazy dog.",
        "I like my coffee strong, with no sugar.",
        "Remind me to water the plants this evening.",
    )

    async def _speaker_turn(self, action: str) -> None:
        """"learn my voice" / "forget my voice" (and the Settings buttons)."""
        hi = self._utterance_lang == "hi"
        sp = self.speaker
        if sp is None:
            await self.say("I can't learn voices here.")
            return
        if action == "forget":
            had = await asyncio.to_thread(sp.forget)
            self._emit("tool", {"summary": "Forget my voice", "decision": "auto"})
            if hi:
                await self.say("ठीक है, अब मैं सबकी सुनूँगी।" if had else "मेरे पास आपकी आवाज़ नहीं थी।", lang="hi")
            else:
                await self.say("Done, I'll listen to anyone now." if had else "I didn't have your voice.")
            return
        await self._enrol_voice(hi)

    async def _enrol_voice(self, hi: bool) -> None:
        sp = self.speaker
        if not sp.model_ready():
            await self.say("One moment, getting the voice model.")
        self._set("thinking")
        if not await asyncio.to_thread(sp.prepare):
            await self.say("Couldn't get the voice model, check the log.")
            return
        if hi:
            await self.say("मैं एक-एक लाइन बोलूँगी, बीप के बाद आप दोहराइए।", lang="hi")
        else:
            await self.say("I'll say three lines. Repeat each after the beep.")
        clips: list[np.ndarray] = []
        for line in self.ENROL_LINES:
            clip = None
            her = await self._her_voice(line)
            for _attempt in range(2):
                await self.say(line)
                self._set("listening")
                await self.chime(self.s.chime_followup_hz, 100)
                pcm = await self._capture(max_s=6, skip_ms=self.s.followup_skip_ms)
                problem = await self._enrol_take_problem(pcm, line, her)
                if problem is None:
                    clip = pcm
                    break
                log.info("voice enrolment: take rejected (%s)", problem)
                await self.say("Once more, a little louder.")
            if clip is None:
                await self.say("I couldn't hear you. Let's try later.")
                return
            clips.append(clip)
        self._set("thinking")
        profile, _agreement = await asyncio.to_thread(sp.enrol, clips)
        self._emit("tool", {"summary": "Learn my voice", "decision": "auto" if profile else "declined"})
        if profile is None:
            await self.say("Those didn't sound like one voice. Let's try again somewhere quieter.")
        elif not self.s.speaker_verification:
            await self.say("Got your voice. Turn on the voice check in Settings to use it.")
        elif hi:
            await self.say("आपकी आवाज़ याद हो गई। अब मैं सिर्फ़ आपकी सुनूँगी।", lang="hi")
        else:
            await self.say("Got it. I'll only listen to you now.")

    async def _her_voice(self, line: str) -> np.ndarray | None:
        """`line` in her own voice at 16 kHz, to tell her echo from the user."""
        try:
            samples, sr = await self.tts.asynth(line)
            from scipy.signal import resample_poly

            g = math.gcd(int(sr), self.s.sample_rate)
            x = resample_poly(np.asarray(samples, dtype=np.float32), self.s.sample_rate // g, int(sr) // g)
            return np.clip(x * 32767, -32768, 32767).astype(np.int16)
        except Exception:
            log.exception("couldn't synthesise the enrolment line for the echo check")
            return None

    # A take whose voice is this close to hers is her own line coming back.
    _HER_VOICE_MAX = 0.5

    async def _enrol_take_problem(self, pcm: np.ndarray | None, line: str, her: np.ndarray | None) -> str | None:
        """Why this enrolment take can't be used, or None. Three takes of
        the same wrong source (her tail, the TV, a fan) would agree with each
        other and make a profile that ignores the user — so each take must be
        at least ENROL_MIN_S of voice, must say (mostly) the line, and must
        not be her."""
        if pcm is None:
            return "nothing heard"
        if voiced_s(pcm) < ENROL_MIN_S:
            return "too little voice"
        heard = await self.stt.atranscribe(pcm) or ""
        want = set(normalize(line).split())
        got = set(normalize(heard).split())
        if len(want & got) < len(want) / 2:
            return f"heard {heard!r}"
        if her is not None and her.size:
            same = await asyncio.to_thread(self.speaker.similarity, pcm, her)
            if same >= self._HER_VOICE_MAX:
                return f"her own voice ({same:.2f})"
        return None

    async def queue_turn(self, turn: Callable[[], Any]) -> None:
        """Run `turn()` (a coroutine factory) as the next idle turn — for the
        Settings window's voice buttons, which must not run beside a turn
        that already has the mic. It rides the announcement queue."""
        await self._announce_queue.put((turn, None))

    # -- push-to-talk (A2) --------------------------------------------------------
    def ptt_start(self) -> None:
        """Called (on the event-loop thread, via the hotkey monitor's
        call_soon_threadsafe) when the push-to-talk key goes down. Only
        records the key state and raises the PTT signal; the actual work
        happens inside run_forever (idle) or the in-flight turn's
        _run_with_barge / PTT-aware capture, which race on _ptt_event —
        so there is never a second turn or a second capture running
        beside the main loop. Ignored until warmup is done, while muted,
        and while the key is already down (key-repeat flags-changed
        events, or a stray double dispatch)."""
        if not self.ready:
            log.info("ptt ignored: not ready")
            return
        if self.muted:
            log.info("ptt ignored: muted")
            return
        if self._ptt_held:
            return
        self._ptt_held = True
        self._ptt_event.set()

    def ptt_end(self) -> None:
        """Called when the push-to-talk key is released. If the hold-mode
        capture is in flight, end it gracefully — Recorder.finish()
        returns whatever was recorded so far, unlike stop() which
        discards it. If the capture hasn't started yet (quick tap: the
        key came back up while the chime was still playing or before the
        turn got to it), just drop the held flag so _listen_after_ptt
        sees the key is already up and doesn't open the mic at all."""
        if not self._ptt_held:
            return
        self._ptt_held = False
        if self._ptt_capturing:
            self.recorder.finish()

    async def _listen_after_ptt(self) -> np.ndarray | None:
        """The push-to-talk capture: consume the PTT signal, then (unless
        the key is already back up) chime *concurrently* with a hold-mode
        capture that ends on key release (Recorder.finish()) or after
        ptt_max_s at most. The capture is started before the chime is
        awaited so a release during the chime lands on a real in-flight
        capture instead of leaking."""
        self._ptt_event.clear()
        self._set("listening")
        if not self._ptt_held:
            log.info("ptt: key released before capture started; ignoring tap")
            return None
        chime_task = asyncio.ensure_future(self.chime(self.s.chime_wake_hz, 120))
        self._ptt_capturing = True
        try:
            pcm = await self._capture(max_s=self.s.ptt_max_s, hold=True, partial=True)
        finally:
            self._ptt_capturing = False
            # The key may still be physically down (capture ended on the
            # ptt_max_s cap, or an error); treat it as released so the
            # eventual key-up is a no-op and the next press is a fresh one.
            self._ptt_held = False
            self._end_partial_window()
            with contextlib.suppress(BaseException):
                await chime_task
        return pcm

    # -- captures ------------------------------------------------------------------
    async def _capture(self, **kw) -> np.ndarray | None:
        """Every recorder.capture() in the orchestrator goes through here so
        _capture_in_flight is accurate for the barge/PTT teardown, and so
        that cancelling the task that's capturing (a barged turn) never
        abandons the recorder's worker thread mid-capture: the recorder
        call is shielded, and on cancellation we ask it to stop and wait
        for it to actually return before re-raising. Without this, the
        next capture could start while the previous thread was still
        draining the mic — and a stop() meant for the old capture could
        be consumed by the new one."""
        self._capture_in_flight = True
        self._last_capture_hold = bool(kw.get("hold", False))
        # Arm the recorder's flags synchronously: capture()'s own body only
        # runs on the task's first step, and a stop()/finish() landing in
        # that gap (a PTT quick tap, a barge from the SDK's confirm task)
        # would otherwise be a silent no-op.
        arm = getattr(self.recorder, "arm", None)
        try:
            if arm is not None:
                arm(hold=kw.get("hold", False))
            fut = asyncio.ensure_future(self.recorder.capture(**kw))
        except BaseException:
            # Nothing is capturing, so don't leave the armed flags set for
            # the *next* capture to inherit (its stop() would be honored
            # against a thread that never started).
            if arm is not None:
                self.recorder.disarm()
            self._capture_in_flight = False
            raise
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            if not fut.done():
                self.recorder.stop()
                with contextlib.suppress(BaseException):
                    await fut
            else:
                # already finished (possibly with an error nobody will
                # look at now): retrieve it so asyncio doesn't log
                # "exception was never retrieved".
                with contextlib.suppress(BaseException):
                    fut.exception()
            raise
        finally:
            self._capture_in_flight = False

    async def _capture_or_ptt(self, **kw):
        """A capture that the push-to-talk key can pre-empt: returns the
        PCM (or None) as usual, or the PTT sentinel if the key went down
        first — in which case the capture has already been stopped and
        its thread has returned, so the caller can go straight to a
        hold-mode capture with nothing else touching the mic."""
        if self._ptt_event.is_set():
            return PTT
        cap = asyncio.ensure_future(self._capture(**kw))
        ptt = asyncio.ensure_future(self._ptt_event.wait())
        try:
            done, _ = await asyncio.wait({cap, ptt}, return_when=asyncio.FIRST_COMPLETED)
            if cap in done:
                return cap.result()
            log.info("ptt during capture")
            self.recorder.stop()
            with contextlib.suppress(BaseException):
                await cap
            return PTT
        finally:
            if not ptt.done():
                ptt.cancel()
                with contextlib.suppress(BaseException):
                    await ptt
            if not cap.done():
                # only reachable if we were cancelled mid-wait; don't leave
                # the capture thread orphaned.
                self.recorder.stop()
                with contextlib.suppress(BaseException):
                    await cap

    # -- proactive briefings & nudges (B2) ---------------------------------------
    @staticmethod
    def _time_spoken(hhmm: str) -> str:
        """"07:30" -> "7:30 am", "18:00" -> "6 pm", "00:15" -> "12:15 am"
        (how the confirmation reads a 24h HH:MM back)."""
        hour, minute = int(hhmm[:2]), int(hhmm[3:5])
        suffix = "am" if hour < 12 else "pm"
        h12 = hour % 12 or 12
        return f"{h12} {suffix}" if minute == 0 else f"{h12}:{minute:02d} {suffix}"

    async def _proactive_turn(self, action: tuple[str, object]) -> None:
        """Local fast path for the briefing/nudge intents: "brief me" speaks
        a briefing right now; the on/off phrases edit the Proactive
        schedule, persist it to prefs.json and read the result back."""
        kind, arg = action
        if self.proactive is None:
            await self.say("Briefings aren't available right now.")
            return
        sched = self.proactive.schedule
        if kind == "brief_now":
            # Dispatched through _run_with_barge by one_turn (see
            # _brief_now_turn), so a long briefing can be cut off.
            await self._brief_now_turn()
            return
        if kind in ("snooze", "resume"):
            # A snooze only holds announcements (the ticker keeps running and
            # queues them), so it never touches the saved schedule.
            if kind == "resume":
                self.proactive.hold_until = None
                reply = "Okay, notifications back on."
            else:
                until = proactive_mod.resolve_hold_until(dt.datetime.now(), 60 if arg is None else arg)
                self.proactive.hold_until = until
                reply = f"Okay, quiet until {self._time_spoken(until.strftime('%H:%M'))}."
            self._emit("tool", {"summary": f"{kind.capitalize()} notifications", "decision": "auto"})
            await self.say(reply)
            return
        if kind == "briefing_on":
            sched.briefing_enabled = True
            if isinstance(arg, str):
                sched.briefing_time = arg
            reply = f"Okay, I'll brief you every day at {self._time_spoken(sched.briefing_time)}."
        elif kind == "briefing_off":
            sched.briefing_enabled = False
            reply = "Okay, no more morning briefings."
        elif kind == "nudges_on":
            sched.nudges_enabled = True
            if isinstance(arg, int) and 1 <= arg <= 60:
                sched.nudge_minutes = arg
            reply = f"Okay, I'll warn you {sched.nudge_minutes} minutes before each event."
        else:
            sched.nudges_enabled = False
            reply = "Okay, no more meeting nudges."
        proactive_mod.save_schedule(sched)
        self._emit("tool", {"summary": "Update briefing schedule", "decision": "auto"})
        await self.say(reply)

    async def _brief_now_turn(self) -> None:
        """"brief me": fetch (Calendar/Mail/Reminders can take a few seconds,
        so show "thinking") and speak the briefing."""
        self._emit("tool", {"summary": "Briefing", "decision": "auto"})
        self._set("thinking")
        text = await self.proactive.build_briefing()
        self._set("speaking")
        await self.say(text)

    # -- quick replies (C1) ------------------------------------------------------
    async def _quick_turn(self, hit: quick.QuickReply, heard: str) -> None:
        """Answer a trivial question (time, date, battery, volume, small talk,
        arithmetic) locally, without the brain. For battery/volume the match
        only carries the language; the value is read here."""
        kind, reply = hit
        # A Hinglish/Devanagari phrase gets a Hindi reply even in English
        # mode ("shukriya" -> "कोई बात नहीं।"), so voice it in Hindi too.
        lang = reply if kind in ("battery", "volume") else quick.reply_lang(heard, self._utterance_lang)
        if kind == "battery":
            percent, state = await asyncio.to_thread(mac_tools.read_battery)
            reply = quick.reply_for("battery", lang, percent=percent, state=state)
        elif kind == "volume":
            res = await mac_tools.volume_get.handler({})
            try:
                percent = None if res.get("is_error") else int(float(res["content"][0]["text"].strip()))
            except (ValueError, KeyError, IndexError, TypeError):
                percent = None
            reply = quick.reply_for("volume", lang, percent=percent)
        self._emit("tool", {"summary": "Quick reply", "decision": "auto"})
        await self.say(reply, lang=lang)
        if self.store is not None and self.s.memory_enabled:
            self.store.add_turn(heard, reply)

    # -- settings window / self-update (Batch D) ---------------------------------
    async def _settings_turn(self, tab: str, heard: str) -> None:
        """Local fast path for "open settings" / "show history": ask the
        menu bar (which owns the window) to show it on the given tab, and
        confirm. A Hindi/Hinglish utterance gets the Hindi confirmation."""
        self._emit("settings", {"open": True, "tab": tab})
        hindi = self._utterance_lang == "hi" or has_devanagari(heard) or quick.is_hinglish_phrase(heard)
        if hindi:
            await self.say("यह लीजिए।", lang="hi")
        else:
            await self.say("Here you go.")

    async def _describe_version(self) -> str:
        if self.version_describe is not None:
            return self.version_describe()
        return await asyncio.to_thread(version.describe)

    async def _update_turn(self) -> None:
        """Local fast path for "update yourself" / "check for updates":
        check on a thread, and if something newer exists, pull/build (also
        on a thread) and relaunch — this IS the in-progress turn, so the
        settings bridge's idle rule doesn't apply. It runs outside the
        barge race so the build can't be orphaned mid-way; the app's
        `updater_update` hook holds the bridge's single update slot and
        raises UpdateInProgress when one is already running. Nothing is
        injected in --text mode/tests."""
        if self.updater_check is None or self.updater_update is None:
            await self.say("Updates aren't available in this mode.")
            return
        self._set("thinking")
        try:
            status = await asyncio.to_thread(self.updater_check)
        except Exception:
            log.warning("update check failed", exc_info=True)
            await self.say("Couldn't check for updates, check the log.")
            return
        if not status.available:
            await self.say("You're already on the latest.")
            return
        # Start the work first: the hook claims the bridge's update slot
        # (or refuses at once), so give it a beat before announcing.
        work = asyncio.ensure_future(asyncio.to_thread(self.updater_update, status))
        await asyncio.wait({work}, timeout=0.05)
        if work.done() and isinstance(work.exception(), UpdateInProgress):
            await self.say("An update is already running.")
            return
        await self.say("Updating, back in a moment.")
        self._emit("tool", {"summary": "Update Veronica", "decision": "auto"})
        self._set("thinking")
        try:
            log.info("update: %s", await work)
        except UpdateInProgress:
            await self.say("An update is already running.")
            return
        except Exception:
            log.exception("update failed")
            await self.say("The update failed, check the log.")
            return
        hint = "Update installed. Restart me from the terminal."
        if self.relaunch is None:
            await self.say(hint)
            return
        if self.can_relaunch is not None and not self.can_relaunch():
            # Dev run: relaunch() will only quit — say so first, so the
            # speech isn't torn down by the quit it schedules.
            await self.say(hint)
            self.relaunch()
            return
        if not self.relaunch():
            # relaunch() quits when it could schedule the reopen (or has no
            # bundle to reopen); still here means neither happened.
            await self.say(hint)

    # -- notes & dictation (A4) --------------------------------------------------
    async def _note_turn(self, body: str) -> None:
        """Local fast path for "take a note: X" / "note that X": create the
        note directly (no brain round trip) and confirm with "Noted."."""
        ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        title = f"{body[:40]} — {ts}"
        self._emit("tool", {"summary": f"Create note {title}", "decision": "auto"})
        res = await pim_tools.notes_create.handler({"title": title, "body": body})
        await self.say("Sorry, I couldn't save that note." if res.get("is_error") else "Noted.")

    _STOP_DICTATION_WAIT_S = 3

    @staticmethod
    def _strip_trailing_stop_dictation(text: str) -> tuple[str, bool]:
        """"hello there, stop dictation" -> ("hello there", True)."""
        stripped = _TRAILING_STOP_DICTATION_RE.sub("", text)
        return stripped.strip(), stripped != text

    async def _dictation_turn(self) -> None:
        """Local fast path for "dictate"/"start dictation": listen (each
        utterance endpointed normally by the VAD, looped like a follow-up
        window) until "stop dictation" is heard or STOP_DICTATION_WAIT_S of
        silence passes with nothing said. Each utterance is typed into the
        focused app as soon as it's transcribed (so a barge, error or
        timeout mid-dictation keeps what was already said, rather than
        losing everything); a trailing "stop dictation" spoken in the same
        breath as the last sentence is stripped from what gets typed."""
        self.player.reset()
        await self.say("Go ahead.")
        self._set("listening")
        typed_any = False
        failed = False
        deadline = self._clock() + self.s.dictation_max_s
        first = True
        while self._clock() < deadline:
            # skip_ms on the first capture only: drop the tail/echo of the
            # "Go ahead." prompt so it isn't endpointed as a false onset.
            pcm = await self._capture(
                max_s=self._STOP_DICTATION_WAIT_S, partial=True,
                skip_ms=self.s.followup_skip_ms if first else 0,
            )
            first = False
            self._end_partial_window()
            if pcm is None:
                break
            if not await self._speaker_ok(pcm, "dictation"):
                self._ignored_voice("dictation")
                continue   # someone else talking: not typed, keep listening
            text = await self.stt.atranscribe(pcm)
            self._emit("heard", text)
            if not text:
                break
            if is_stop_dictation(text):
                break
            text, stop = self._strip_trailing_stop_dictation(text)
            if text:
                if not typed_any:
                    self._emit("tool", {"summary": "Dictate text", "decision": "auto"})
                chunk = text if not typed_any else f" {text}"
                res = await asyncio.to_thread(mac_tools.dictate_type, chunk)
                if res.get("is_error"):
                    failed = True
                    break
                typed_any = True
            if stop:
                break
        self._set("thinking")
        if failed:
            await self.say("Sorry, I couldn't type that.")
        elif not typed_any:
            await self.say("I didn't catch anything.")
        else:
            await self.say("Done.")
        self._set("idle")

    # -- confirmation gate ----------------------------------------------------
    # The wording lives in brain.agent, next to the summarize_* that produce
    # these summaries: the gate builds the same question when it adds the
    # "say always" hint.
    @staticmethod
    def confirm_prompt(summary: str) -> str:
        """The spoken question for a tool `summary`: "Click 'Save'?" for a
        screen action, "Run Bash: ls?" for everything else."""
        return confirm_prompt(summary)

    async def confirm(self, summary: str, detail: str = "", *, question: str | None = None) -> ConfirmResult:
        """Ask `summary` (or `question`) aloud and listen for the answer.
        Returns a ConfirmResult: truthy for a yes; a no or silence is
        "denied"; anything else said is "other" and carried in `.heard` so
        the brain can take it as the next request instead."""
        if self.muted:
            log.info("confirm skipped (muted): %s", summary)
            return ConfirmResult("denied")
        prev = self.state
        self._set("confirming")
        result = ConfirmResult("denied")
        try:
            queue = self._speech_queue
            if queue is not None:
                # don't jump ahead of sentences already queued for playback by
                # an in-flight handle_text pipeline. Bounded by brain_timeout_s
                # as a belt-and-braces guard against ever hanging here.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(queue.join(), timeout=self.s.brain_timeout_s)
            if self._barged:
                # a barge landed while we were waiting for the queue to drain;
                # the turn this confirmation belongs to is already being torn
                # down, so don't speak the prompt or eat the follow-up capture.
                log.info("confirm aborted by barge")
                return result
            async with self._speech_lock:
                if self._barged:
                    # a barge landed while we were waiting to acquire the
                    # speech lock (e.g. a concurrent say()/chime() was still
                    # holding it); don't speak the prompt for a turn that's
                    # already being torn down.
                    log.info("confirm aborted by barge")
                    return result
                self.player.reset()
                prompt = question if question is not None else self.confirm_prompt(summary)
                await self._say_unlocked(prompt, kind="prompt")
                if self._barged:
                    # barged while the prompt was being spoken.
                    log.info("confirm aborted by barge")
                    return result
                # Emitted only now (after the question has actually been
                # spoken, immediately before we start listening) so the
                # listening window's countdown starts from when the user
                # could first respond, not from confirm()'s entry.
                self._emit("tool", {
                    "summary": summary, "detail": detail, "decision": "ask",
                    "timeout_ms": self.s.confirm_listen_s * 1000,
                })
                # From here until the capture returns, whatever the user says
                # belongs to this question: _run_with_barge stops treating
                # speech (or a PTT press) as a barge and lets it land in the
                # capture below, so classify_answer — still the only thing
                # that decides — sees it. Cleared in finally so an error
                # here can't leave barge-in disabled for the rest of the turn.
                self._confirm_listening = True
                try:
                    heard = await self._confirm_listen()
                    if heard is _NOT_THE_USER or (heard and self._is_own_speech(heard)):
                        # The question itself leaking back in through the mic.
                        # Read as an answer it was "other": the step declined
                        # and her own words run as the next request. It's no
                        # answer at all, so listen once more. The same goes for
                        # someone else's voice (the TV, a person in the room):
                        # it is never a yes, never a redirect.
                        log.info("confirm heard %s; listening again",
                                 "another voice" if heard is _NOT_THE_USER else f"its own question ({heard!r})")
                        heard = await self._confirm_listen()
                        if heard is _NOT_THE_USER or (heard and self._is_own_speech(heard)):
                            heard = None
                finally:
                    self._confirm_listening = False
                if self._barged:
                    # barged while we were listening for / transcribing the
                    # reply (a barge unblocks the capture, so pcm is None).
                    log.info("confirm aborted by barge")
                    return result
                if not heard:
                    # Silence: say so, so the user knows the window closed
                    # (an explicit "no" gets no such line).
                    log.info("confirm heard nothing -> denied")
                    await self._say_unlocked("Okay, skipping that.")
                    return result
                outcome = self.classify_answer(heard)
                result = ConfirmResult(outcome, heard,
                                       always=outcome == "approved" and self.says_always(heard))
                log.info("confirm heard=%r -> %s", heard, result.outcome)
        finally:
            self._set(prev)
            self._emit("tool", {"summary": summary, "decision": self._DECISION_EVENT[result.outcome]})
        return result

    async def _confirm_listen(self):
        """The answer's text, None for silence, or _NOT_THE_USER for a voice
        the speaker check rejected (never transcribed, so never classified)."""
        pcm = await self._capture(max_s=max(1, self.s.confirm_listen_s))
        if pcm is None or self._barged:
            return None
        if not await self._speaker_ok(pcm, "confirm"):
            return _NOT_THE_USER
        return await self.stt.atranscribe(pcm)

    # `allowed` stays the wire value for an approved confirm (the HUD and
    # older tests know it); "other" shows as a redirect.
    _DECISION_EVENT = {"approved": "allowed", "denied": "declined", "other": "redirected"}

    # -- barge-in ---------------------------------------------------------------
    async def _barge_teardown(self, turn: asyncio.Future) -> None:
        """Common teardown for a wake-word barge and a push-to-talk press
        landing mid-turn: stop playback, unblock ANY capture the turn has
        in flight (confirm()'s yes/no, dictation, ...) so its thread
        returns instead of being orphaned, cancel the turn, wait for it,
        then interrupt the brain."""
        self.player.stop()
        if self._capture_in_flight:
            # Unblock the capture thread (confirm()'s yes/no, dictation, a
            # brain-side capture — whatever it is) so it returns promptly
            # instead of being orphaned. _capture() additionally waits for
            # that thread to actually return before letting the
            # cancellation below propagate, so the mic is guaranteed free
            # for the re-listen.
            self.recorder.stop()
        self._barged = True
        turn.cancel()
        try:
            await turn
        except asyncio.CancelledError:
            pass
        except Exception:
            log.exception("barged turn raised while being torn down")
        await self.brain.interrupt()
        # A barge ends whatever screen-control sequence was running; the next
        # click/type must ask again (fake brains in tests may lack the method).
        getattr(self.brain, "clear_trust", lambda: None)()

    async def _run_with_barge(self, coro) -> str | None:
        """Run a turn coroutine racing the barge listener and the push-to-
        talk signal. Returns None if the turn ran to completion, "wake" if
        the wake word interrupted it, or "ptt" if the push-to-talk key did
        (in which case the caller should do a hold-mode capture instead of
        listening for a follow-up).

        One exception to all of that: while confirm() has the mic open for
        its yes/no (_confirm_listening), neither trigger is a barge — the
        user is answering "Run X?", and the answer is already going into
        that capture. Such a trigger is dropped, a fresh listener/waiter is
        armed in its place, and the turn keeps running."""
        turn = asyncio.ensure_future(coro)
        listener = asyncio.create_task(
            self.wake.wait(threshold=self.s.barge_threshold, suppress=self._suppress_text)
        )
        ptt = asyncio.ensure_future(self._ptt_event.wait())
        pending = {turn, listener, ptt}
        try:
            while True:
                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                if ptt in done:
                    if self._confirm_listening:
                        # The key went down to answer "Run X?", not to start a
                        # new turn: tearing down here would cancel the turn and
                        # throw the spoken answer away. Consume the press (clear
                        # the event, so it can't leak into the follow-up window
                        # or the next turn's _capture_or_ptt) and let the confirm
                        # capture — already open and recording — take the words
                        # that follow it. Then re-arm the waiter for the rest of
                        # the turn.
                        log.info("barge during confirm: routed to the answer")
                        self._ptt_event.clear()
                        pending.discard(ptt)
                        ptt = asyncio.ensure_future(self._ptt_event.wait())
                        pending.add(ptt)
                    else:
                        log.info("turn ended early: reason=barge_ptt detail=state=%s", self.state)
                        await self._barge_teardown(turn)
                        return "ptt"
                if listener in done:
                    pending.discard(listener)
                    if listener.exception() is not None:
                        # mic hiccup or similar in the barge listener; the
                        # turn is still good, so don't cancel it — just log
                        # and let it finish normally (PTT can still barge),
                        # as if no barge listener were running at all.
                        try:
                            listener.result()
                        except Exception:
                            log.exception("barge listener failed")
                    elif listener.result():
                        if self._confirm_listening:
                            # The wake engine heard the user answering the
                            # confirmation. The very same speech is going into
                            # confirm()'s capture, so drop this trigger and let
                            # classify_answer decide instead of cancelling the
                            # turn. Re-arm a fresh listener so a barge *after*
                            # the answer window still works — deliberately
                            # without stop()ing the resolved-True one, which
                            # would poison the next wait() with a spurious
                            # immediate False.
                            log.info("barge during confirm: routed to the answer")
                            listener = asyncio.create_task(
                                self.wake.wait(threshold=self.s.barge_threshold, suppress=self._suppress_text)
                            )
                            pending.add(listener)
                        else:
                            # Whether it was her own voice is the first thing to
                            # rule out when a task "just stopped": say what she
                            # was saying.
                            log.info("turn ended early: reason=barge_wake detail=state=%s speaking=%r",
                                     self.state, self._suppress_text()[:80])
                            await self._barge_teardown(turn)
                            return "wake"
                    # listener resolved False (a stop() consumed), or its True
                    # was routed to a pending confirm — keep waiting on the
                    # turn (and PTT).
                    if turn not in done:
                        continue
                if turn in done:
                    if not listener.done():
                        # turn finished first; the listener is still
                        # running, ask its thread to exit.
                        self.wake.stop()
                        await listener
                    turn.result()  # re-raise turn errors
                    return None
        finally:
            # Never leave any task pending, however we got here (normal
            # return, a re-raised turn error, or an exception out of the
            # listener itself, e.g. listener.result() raising because wait()
            # raised). Only stop() a listener that's still running here (one
            # already resolved True above and stopping it again would poison
            # the next wait() with a spurious immediate False).
            if not ptt.done():
                ptt.cancel()
                with contextlib.suppress(BaseException):
                    await ptt
            if not listener.done():
                self.wake.stop()
                with contextlib.suppress(BaseException):
                    await listener
            if not turn.done():
                turn.cancel()
                with contextlib.suppress(BaseException):
                    await turn

    # -- one interaction ------------------------------------------------------
    async def _listen_after_wake(self) -> np.ndarray | None:
        """Chime (unless the wake engine's pre-roll already contains speech,
        i.e. the user spoke the command in the same breath as the wake
        word) and capture, handing that pre-roll to the recorder so it
        isn't lost."""
        self._set("listening")
        pre = self.wake.take_preroll()
        if not self.recorder.has_speech(pre):
            await self.chime(self.s.chime_wake_hz, 120)
        pcm = await self._capture_or_ptt(max_s=self.s.listen_wait_s, preroll=pre, partial=True)
        self._end_partial_window()
        return pcm

    async def _relisten(self, how: str | None):
        """The capture that follows a barge: a hold-mode capture if the
        push-to-talk key caused it, else the usual post-wake listen — and
        if PTT lands during that listen, a hold capture after all."""
        pcm = await self._listen_after_ptt() if how == "ptt" else await self._listen_after_wake()
        if pcm is PTT:
            pcm = await self._listen_after_ptt()
        return pcm

    async def one_turn(self, ptt: bool = False) -> None:
        """Called after the wake word (or, with ptt=True, a push-to-talk
        press while idle): listen, answer, then follow-up window."""
        self._loop = asyncio.get_running_loop()
        pcm = await self._relisten("ptt" if ptt else None)
        if pcm is None:
            self._set("idle")
            return
        is_followup = False
        while True:
            other_voice = not await self._speaker_ok(pcm, "follow-up" if is_followup else "request")
            text, detected = await self._transcribe(pcm)
            if other_voice and match_speaker_intent(text) != "forget":
                # Someone else's voice: as if nothing was said, so the turn
                # ends here (no "didn't catch that", no brain turn); a card
                # says why. "Forget my voice" alone gets through, so a
                # profile that stopped matching the user can't lock them out.
                self._ignored_voice("follow-up" if is_followup else "request")
                break
            self._utterance_lang = self._lang_for(text, detected)
            if is_followup and text and self._is_own_speech(text):
                log.info("ignoring own speech echo on follow-up: %r", text)
                text = ""
            self._emit("heard", text)
            log.info("heard=%r lang=%s", text, self._utterance_lang)
            # F2 pause/continue. Checked before everything else, because a
            # phrase only means this while a barge is holding the rest of an
            # answer: outside that window "ruko" is still an end phrase and
            # "wait" is still a request for the brain, and the ladder below
            # keeps deciding those.
            resume_hit = False
            if self._paused_tail is not None:
                if is_pause_phrase(text):
                    # The barge already stopped playback and parked the rest,
                    # so there is nothing to do but stay quiet and listen for
                    # the word that starts her again. The post-wake window is
                    # used rather than the short follow-up one: the user
                    # interrupted on purpose and may need a moment.
                    log.info("paused with %d sentence(s) held", len(self._paused_tail))
                    self._set("paused")
                    pcm = await self._capture_or_ptt(max_s=self.s.listen_wait_s, partial=True)
                    self._end_partial_window()
                    if pcm is PTT:
                        pcm = await self._listen_after_ptt()
                    is_followup = False
                    if pcm is None:
                        # Nothing more said. The remainder stays parked (a
                        # later "continue" still works); the next request
                        # drops it.
                        break
                    continue
                if is_resume_phrase(text):
                    resume_hit = True
                else:
                    self._drop_paused_tail(f"superseded by {text!r}")
            # `not resume_hit`: "go on" is a continue, not a go-ahead for
            # whatever the next turn's first tool call happens to be.
            if not resume_hit and self.s.preapprove_by_wording and self.detect_preapproval(text):
                # The request itself said go ahead: the brain turn this is
                # about to become (the next handle_text) gets its first
                # confirm-class action without the yes/no.
                log.info("pre-approval by wording: %r", text)
                getattr(self.brain, "preapprove", lambda _tid, until: None)(
                    self._turn_id + 1, until=time.monotonic() + self.PREAPPROVE_WINDOW_S,
                )
            intent = match_intent(text)
            if intent == "end":
                getattr(self.brain, "clear_trust", lambda: None)()
                if normalize(text) in self._SPOKEN_END_PHRASES:
                    self.player.reset()
                    await self.say("Okay.")
                self._emit("hud", {"mode": "hide"})
                self._set("idle")
                return
            if intent == "hud_hide":
                self._emit("hud", {"mode": "hide"})
                self._set("idle")
                return
            if intent == "mute":
                self.player.reset()
                await self.say("Muted.")
                self.muted = True
                self._emit("hud", {"mode": "hide"})
                self._set("idle")
                return
            speaker_action = None if intent is not None else match_speaker_intent(text)
            mem = None if (intent is not None or speaker_action is not None) else match_memory_intent(text)
            screen_intent = intent is None and mem is None and match_screen_intent(text)
            music_action = (
                None if (intent is not None or mem is not None or screen_intent)
                else match_music_intent(text)
            )
            note_body = (
                None if (intent is not None or mem is not None or screen_intent or music_action)
                else match_note_intent(text)
            )
            voice_action = (
                None
                if (intent is not None or mem is not None or screen_intent or music_action or note_body is not None)
                else match_voice_intent(text)
            )
            lang_mode = (
                None
                if (intent is not None or mem is not None or screen_intent or music_action or note_body is not None or voice_action)
                else match_language_intent(text)
            )
            settings_tab = (
                None
                if (intent is not None or mem is not None or screen_intent or music_action or note_body is not None or voice_action or lang_mode is not None)
                else match_settings_intent(text)
            )
            version_intent = (
                False
                if (intent is not None or mem is not None or screen_intent or music_action or note_body is not None or voice_action or lang_mode is not None or settings_tab is not None)
                else match_version_intent(text)
            )
            update_intent = (
                False
                if (intent is not None or mem is not None or screen_intent or music_action or note_body is not None or voice_action or lang_mode is not None or settings_tab is not None or version_intent)
                else match_update_intent(text)
            )
            local_hit = settings_tab is not None or version_intent or update_intent
            proactive_action = (
                None
                if (intent is not None or mem is not None or screen_intent or music_action or note_body is not None or voice_action or lang_mode is not None or local_hit)
                else match_proactive_intent(text)
            )
            brain_action = (
                None
                if (intent is not None or mem is not None or screen_intent or music_action or note_body is not None or voice_action or lang_mode is not None or local_hit or proactive_action is not None)
                else match_brain_intent(text)
            )
            quick_hit = (
                None
                if (intent is not None or mem is not None or screen_intent or music_action or note_body is not None or voice_action or lang_mode is not None or local_hit or proactive_action is not None or brain_action is not None)
                else quick.match_quick(text, lang=self._utterance_lang)
            )
            dictation_intent = (
                False
                if (intent is not None or mem is not None or screen_intent or music_action or note_body is not None or voice_action or lang_mode is not None or local_hit or proactive_action is not None or brain_action is not None or quick_hit is not None)
                else match_dictation_intent(text)
            )
            if resume_hit:
                # "Continue": speak the parked remainder (already synthesised)
                # under the usual barge race, so she can be stopped — or
                # paused again — part way through it.
                tail, self._paused_tail = self._paused_tail, None
                barged = await self._run_with_barge(self._resume_tail(tail))
                if barged:
                    pcm = await self._relisten(barged)
                    is_followup = False
                    if pcm is None:
                        break
                    continue
            elif intent in ("hud_mini", "hud_full"):
                self._emit("hud", {"mode": "mini" if intent == "hud_mini" else "full"})
                self.player.reset()
                await self.say("Okay.")
            elif intent == "hud_reset":
                # "Where are you?": the menu bar forgets the saved HUD
                # position and shows the panel at its default spot.
                self._emit("hud", {"mode": "reset"})
                self.player.reset()
                if self._utterance_lang == "hi":
                    await self.say("मैं यहाँ हूँ।", lang="hi")
                else:
                    await self.say("Here I am.")
            elif intent == "quit":
                self.player.reset()
                if await self.confirm("Quit Veronica", question="Quit Veronica?"):
                    await self.say("Goodbye.")
                    self._on_quit()
                    self._set("idle")
                    return
            elif speaker_action is not None:
                self.player.reset()
                await self._speaker_turn(speaker_action)
            elif mem is not None:
                kind, arg = mem
                self.player.reset()
                if kind == "remember":
                    # A rewording of something she already has replaces it
                    # (MemoryStore.remember). The match is fuzzy, so say
                    # what went: "March 8" quietly eating "March 3" is the
                    # one failure the user can't otherwise hear.
                    replaced = ""
                    if self.store is not None:
                        _id, replaced = self.store.remember(arg)
                    await self.say(
                        f"Updated — that replaces '{replaced}'." if replaced else "Got it.")
                elif kind == "forget_topic":
                    n = self.store.delete_facts_about(arg) if self.store is not None else 0
                    await self.say(
                        f"Forgot {_count_word(n)} {'thing' if n == 1 else 'things'} about {arg}."
                        if n else f"I didn't have anything about {arg}."
                    )
                else:
                    n = self.store.delete_fact_matching(arg) if self.store is not None else 0
                    await self.say("Forgotten." if n else "I didn't have that.")
            elif not text:
                if is_followup:
                    # A follow-up capture (not the first listen after wake,
                    # nor the re-listen after a barge) that came back empty
                    # just means the user didn't say anything more — go
                    # quiet rather than nag with "Sorry, didn't catch that."
                    # and reopen yet another follow-up window.
                    self._set("idle")
                    return
                self.player.reset()
                await self.say("Sorry, didn't catch that.")
            elif music_action:
                self.player.reset()
                await self._music_turn(music_action)
            elif note_body is not None:
                self.player.reset()
                await self._note_turn(note_body)
            elif voice_action is not None:
                self.player.reset()
                await self._voice_turn(voice_action)
            elif lang_mode is not None:
                self.player.reset()
                await self._language_turn(lang_mode)
            elif settings_tab is not None:
                self.player.reset()
                await self._settings_turn(settings_tab, text)
            elif version_intent:
                self.player.reset()
                await self.say(await self._describe_version())
            elif update_intent:
                # Deliberately NOT under _run_with_barge: a barge would
                # cancel the turn and orphan a half-done pull/build.
                self.player.reset()
                await self._update_turn()
            elif proactive_action is not None and proactive_action[0] == "brief_now" and self.proactive is not None:
                self.player.reset()
                barged = await self._run_with_barge(self._brief_now_turn())
                if barged:
                    pcm = await self._relisten(barged)
                    is_followup = False
                    if pcm is None:
                        break
                    continue
            elif proactive_action is not None:
                self.player.reset()
                await self._proactive_turn(proactive_action)
            elif brain_action is not None:
                self.player.reset()
                await self._brain_switch_turn(brain_action)
            elif quick_hit is not None:
                self.player.reset()
                await self._quick_turn(quick_hit, text)
            elif dictation_intent:
                barged = await self._run_with_barge(self._dictation_turn())
                if barged:
                    pcm = await self._relisten(barged)
                    is_followup = False
                    if pcm is None:
                        break
                    continue
            elif screen_intent:
                barged = await self._run_with_barge(self._screen_turn(text))
                if barged:
                    pcm = await self._relisten(barged)
                    is_followup = False
                    if pcm is None:
                        break
                    continue
            else:
                barged = await self._run_with_barge(self._brain_turn(text, lang=self._utterance_lang))
                if barged:
                    pcm = await self._relisten(barged)
                    is_followup = False
                    if pcm is None:
                        break
                    continue
            if self._ptt_event.is_set():
                # PTT pressed in the same tick the answer finished: skip
                # straight to the hold capture, no follow-up chime.
                pcm = await self._listen_after_ptt()
                is_followup = False
                if pcm is None:
                    break
                continue
            self._set("followup")
            await self.chime(self.s.chime_followup_hz, 100)
            pcm = await self._capture_or_ptt(
                max_s=max(1, self.s.followup_window_s), partial=True, skip_ms=self.s.followup_skip_ms
            )
            is_followup = True
            self._end_partial_window()
            if pcm is PTT:
                pcm = await self._listen_after_ptt()
                is_followup = False
            if pcm is None:
                break
        self._set("idle")

    async def _muted_capture(self) -> None:
        """Called after a wake word fires while muted: capture exactly one
        utterance (no chime, no HUD show — state/HUD stay untouched) and
        check only whether it's the unmute phrase. Anything else (including
        silence) is ignored silently; run_forever goes straight back to
        idle either way."""
        self._loop = asyncio.get_running_loop()
        pcm = await self._capture(max_s=self.s.listen_wait_s)
        if pcm is None or not await self._speaker_ok(pcm, "unmute"):
            return
        text = await self.stt.atranscribe(pcm)
        if match_intent(text) == "unmute":
            self.muted = False
            await self.say("I'm back.")

    # -- announcements ----------------------------------------------------------
    async def announce(self, text: str, *, expires_at: dt.datetime | None = None) -> None:
        """Queue `text` to be spoken next time we're idle (e.g. a timer
        firing): never interrupts an in-flight turn. Safe to call from any
        task (e.g. TimerService's on_fire callback). An announcement with
        `expires_at` in the past by the time it's delivered (a "starts in 5
        minutes" nudge that sat behind a long turn) is dropped, not spoken."""
        await self._announce_queue.put((text, expires_at))

    async def _deliver_announcement(self, item: tuple[Any, dt.datetime | None]) -> None:
        text, expires_at = item
        if callable(text):
            # a queued local turn (queue_turn), not something to read out
            try:
                await text()
            except Exception:
                log.exception("queued turn failed")
            finally:
                self._set("idle")
            return
        if expires_at is not None and expires_at < dt.datetime.now():
            log.info("announcement expired: %r", text)
            return
        self._set("speaking")
        self.player.reset()
        await self.chime(self.s.chime_wake_hz, 120)
        await self.say(text)
        self._set("idle")

    async def _guarded_turn(self, *, ptt: bool = False) -> None:
        """one_turn() with the "Something went wrong" error reporting; used
        for wake-word and push-to-talk turns alike, so a failing PTT turn
        is reported to the user just like a failing wake-word turn."""
        try:
            await self.one_turn(ptt=ptt)
        except Exception as exc:
            log.exception("turn ended early: reason=error detail=%s", type(exc).__name__)
            try:
                self.player.reset()
                detail = f"{type(exc).__name__} {exc}".lower()
                if any(k in detail for k in ("login", "logged in", "authenticat")):
                    message = f"{self._brain_label()} isn't logged in."
                else:
                    message = "Something went wrong, check the log."
                await self.say(message)
            except Exception:
                log.exception("failed to report error")
            finally:
                self._set("idle")

    # -- input-volume floor guard --------------------------------------------
    def _start_input_guard(self) -> None:
        if self.input_guard is None or self._input_guard_task is not None:
            return
        self._input_guard_stop = asyncio.Event()
        self._input_guard_task = asyncio.ensure_future(
            input_level.run_periodic(self.input_guard, self._input_guard_stop)
        )
        # A device switch is exactly when macOS resets the input level:
        # re-check right after the mic has followed it, ignoring the throttle.
        devices.subscribe_change(lambda: self.input_guard.check(force=True))

    def stop_input_guard(self) -> None:
        if self._input_guard_stop is not None:
            self._input_guard_stop.set()
        if self._input_guard_task is not None:
            self._input_guard_task.cancel()
        self._input_guard_task = None
        self._input_guard_stop = None

    def input_volume_corrected(self, old: int, new: int, device: str) -> None:
        """Guard callback: surface the first correction of the session on the
        HUD so the user learns why the level keeps coming back; later ones
        only log (the guard already does)."""
        if self._input_hint_shown:
            return
        self._input_hint_shown = True
        self._emit("tool", {"summary": f"Input volume {old} → {new} ({device})", "decision": "auto"})

    async def run_forever(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._set("idle")
        await self.start_brain()
        try:
            await self._run_forever()
        finally:
            await self.stop_brain()

    async def _run_forever(self) -> None:
        if self.proactive is not None:
            await self.proactive.start()
        self._start_input_guard()
        while True:
            # Deliver anything queued while we were away (e.g. a timer that
            # fired mid-turn), unless muted — while muted, announcements just
            # sit in the queue (no state flicker) until unmuted.
            while not self.muted and not self._announce_queue.empty():
                await self._deliver_announcement(self._announce_queue.get_nowait())

            wake_task = asyncio.ensure_future(self.wake.wait())
            ptt_task = asyncio.ensure_future(self._ptt_event.wait())
            waiting_on_unmute = self.muted
            if waiting_on_unmute:
                # Don't consume the queue while muted: race the wake
                # listener against the unmute signal instead, so a queued
                # announcement is neither delivered (flicker) nor lost.
                self._unmute_event.clear()
                signal_task = asyncio.ensure_future(self._unmute_event.wait())
            else:
                signal_task = asyncio.ensure_future(self._announce_queue.get())
            done, _ = await asyncio.wait(
                {wake_task, signal_task, ptt_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if wake_task not in done:
                # Only a signal resolved (announcement / unmute / PTT): we're
                # idle right now (run_forever only waits here between
                # turns), so stop the wake listener, then loop back around
                # to start a fresh one — "resuming" it. An unmute signal
                # just loops back to the top, which delivers the now-unmuted
                # queue; a real announcement is spoken directly; a PTT press
                # runs a push-to-talk turn right here, in this task, so the
                # only listener alive during her answer is the barge
                # listener (with own-speech suppression).
                self.wake.stop()
                with contextlib.suppress(BaseException):
                    await wake_task
                for t in (signal_task, ptt_task):
                    if not t.done():
                        t.cancel()
                        with contextlib.suppress(BaseException):
                            await t
                if ptt_task in done:
                    if signal_task in done and not waiting_on_unmute:
                        # keep the same-tick announcement for after the turn
                        self._announce_queue.put_nowait(signal_task.result())
                    if self.muted:
                        # ptt_start() ignores presses while muted, but a
                        # press can land right before a mute; drop it.
                        self._ptt_event.clear()
                        continue
                    await self._guarded_turn(ptt=True)
                elif not waiting_on_unmute:
                    await self._deliver_announcement(signal_task.result())
                continue

            if signal_task in done:
                if not waiting_on_unmute:
                    # Both resolved in the same tick: a genuine wake-word
                    # detection must not be swallowed by a same-tick
                    # announcement, so prefer the wake path and put the
                    # announcement back — it'll be delivered after this turn
                    # ends (top of the next iteration finds the queue
                    # non-empty).
                    self._announce_queue.put_nowait(signal_task.result())
            else:
                signal_task.cancel()
                with contextlib.suppress(BaseException):
                    await signal_task
            if not ptt_task.done():
                ptt_task.cancel()
                with contextlib.suppress(BaseException):
                    await ptt_task
            # (a PTT press that resolved in the same tick as the wake word
            # stays set; the turn's first capture picks it up as a hold
            # capture.)

            try:
                detected = wake_task.result()
            except Exception:
                log.exception("wake listener failed; retrying in %s s", self.s.wake_retry_s)
                self._set("error")
                await asyncio.sleep(self.s.wake_retry_s)
                self._set("idle")
                continue
            if not detected:
                # a stale one-shot stop() (e.g. consumed in the same frame a
                # prior barge listener ended) must not start a spurious turn.
                continue
            if self.muted:
                log.info("muted; capturing one utterance to check for unmute")
                self._ptt_event.clear()
                await self._muted_capture()
                continue
            await self._guarded_turn()
