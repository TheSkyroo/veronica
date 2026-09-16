# Veronica Batch B Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user change Veronica's voice and speaking speed by voice, get scheduled morning briefings and pre-meeting nudges, and drive Chrome/Safari by voice through the brain.

**Architecture:** Three independent slices on branch `batch-b`. B1 adds a `voices` table + local intents that mutate the running `Synthesizer` and persist to `prefs.json`. B2 adds a `Proactive` ticker that composes announcements from the existing `pim` tool handlers and pushes them through `Orchestrator.announce()` (idle-only, mute-aware). B3 adds a `browser` MCP server that injects JavaScript into Chrome/Safari over AppleScript, risk-classified in `policy.py`.

**Tech Stack:** Python 3.12, uv, pytest (asyncio auto), claude-agent-sdk in-process MCP (`@tool` + `create_sdk_mcp_server`), Kokoro ONNX TTS, AppleScript via `osascript`, rumps/PyObjC menus.

**Spec:** `docs/superpowers/specs/2026-09-16-veronica-batch-b-design.md`

## Global Constraints

- Brain stays on the user's Claude Code subscription login via `claude-agent-sdk`. **No API key**, ever.
- Confirm-gate stays strict: `veronica/brain/policy.py` `classify()` is the only thing that may auto-allow a tool. Never set `allowed_tools`.
- Speech stays local/zero-key (faster-whisper STT, Kokoro TTS).
- Never add `Co-Authored-By` trailers or "Generated with Claude Code" to commits.
- Tests: `uv run pytest -q` (asyncio auto, `live` marker deselected). No test may touch the real mic, speakers, network, AppleScript or a real browser — `osascript`/`subprocess` are monkeypatched.
- Python 3.12, `uv`. Follow existing module patterns (`veronica/tools/pim.py` for MCP tools, `veronica/brain/intents.py` for local intents).
- Voice UX copy: short, friendly, spoken — one sentence, no markdown.
- Work on branch `batch-b` (already created off master, spec committed).

## File map

| File | Responsibility |
|---|---|
| `veronica/speech/voices.py` (new) | Voice table, descriptor resolution, display names, speed constants |
| `veronica/speech/tts.py` | `Synthesizer.voice`/`.speed` mutable, `speed` passed to Kokoro |
| `veronica/brain/intents.py` | `match_voice_intent`, `match_proactive_intent`, `parse_clock_time` |
| `veronica/orchestrator.py` | `_voice_turn`, `_proactive_turn`, dispatch in `one_turn`, `proactive` attr started in `run_forever` |
| `veronica/__main__.py` | read voice/speed prefs; build `Proactive` with pim adapters |
| `veronica/ui/menubar.py` | Voice submenu (menu bar + orb popup) |
| `veronica/proactive.py` (new) | `Schedule`, `Proactive` ticker, briefing composition, event/reminder parsers |
| `veronica/tools/browser.py` (new) | `browser` MCP server (Chrome/Safari via AppleScript+JS) |
| `veronica/brain/policy.py` | `MCP_TOOL_RISK["browser"]` |
| `veronica/brain/agent.py` | register `browser_server`, `summarize_detail` entries |
| `veronica/brain/prompts.py` | one sentence about browser tools |
| `README.md` | voice/speed, briefings, browser setup (Allow JavaScript from Apple Events) |

---

### Task 1: Voice table + Synthesizer speed

**Files:**
- Create: `veronica/speech/voices.py`
- Modify: `veronica/speech/tts.py`
- Test: `tests/test_voices.py`, `tests/test_tts.py`

**Interfaces:**
- Produces: `VOICES: dict[str, str]`, `VOICE_IDS: list[str]` (values of `VOICES` in order), `DEFAULT_VOICE="af_sarah"`, `DEFAULT_SPEED=1.0`, `SPEED_STEP=0.15`, `SPEED_MIN=0.7`, `SPEED_MAX=1.5`, `resolve_voice(request: str) -> str | None`, `display_name(voice_id: str) -> str`, `next_voice(current: str) -> str`, `clamp_speed(x: float) -> float`; `Synthesizer(voice, models_dir, speed=1.0)` with mutable `.voice`, `.speed`.

- [ ] **Step 1: Write failing tests**

`tests/test_voices.py`:
```python
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
```

Append to `tests/test_tts.py` (read the file first; it already has a fake Kokoro class — reuse its pattern):
```python
def test_synth_passes_current_voice_and_speed(monkeypatch, tmp_path):
    calls = []

    class FakeKokoro:
        def __init__(self, *a): pass
        def create(self, text, voice, speed, lang):
            calls.append((text, voice, speed, lang))
            return [0.0, 0.0], 24000

    monkeypatch.setattr(Synthesizer, "_kokoro_cls", FakeKokoro)
    s = Synthesizer("af_sarah", tmp_path)
    assert s.speed == 1.0
    s.voice = "am_adam"
    s.speed = 1.3
    s.synth("hi")
    assert calls == [("hi", "am_adam", 1.3, "en-us")]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest -q tests/test_voices.py tests/test_tts.py`
Expected: FAIL (`ModuleNotFoundError: veronica.speech.voices`, and speed assertion).

- [ ] **Step 3: Implement**

`veronica/speech/voices.py`:
```python
"""Kokoro voice table and the spoken-request → voice-id resolver behind
"use a British male voice" / "switch to Adam"."""

# spoken name -> Kokoro voice id (all shipped in voices-v1.0.bin).
# Prefix: a=American, b=British; f=female, m=male.
VOICES: dict[str, str] = {
    "sarah": "af_sarah",
    "bella": "af_bella",
    "nicole": "af_nicole",
    "sky": "af_sky",
    "adam": "am_adam",
    "michael": "am_michael",
    "emma": "bf_emma",
    "isabella": "bf_isabella",
    "george": "bm_george",
    "lewis": "bm_lewis",
}
VOICE_IDS: list[str] = list(VOICES.values())

DEFAULT_VOICE = "af_sarah"
DEFAULT_SPEED = 1.0
SPEED_STEP = 0.15
SPEED_MIN = 0.7
SPEED_MAX = 1.5

_GENDER = {"male": "m", "man": "m", "guy": "m", "female": "f", "woman": "f", "lady": "f"}
_ACCENT = {"british": "b", "english": "b", "uk": "b", "american": "a", "us": "a"}
_DEFAULT_WORDS = {"default", "normal", "usual", "original"}


def resolve_voice(request: str) -> str | None:
    """Map a spoken request to a voice id: an exact name ("adam"), a
    descriptor combo ("british male", "female"), or "default". Descriptors
    pick the first table entry whose id prefix matches; unknown → None."""
    words = request.lower().replace("-", " ").split()
    if not words:
        return None
    if len(words) == 1 and words[0] in VOICES:
        return VOICES[words[0]]
    if any(w in _DEFAULT_WORDS for w in words):
        return DEFAULT_VOICE
    accent = next((_ACCENT[w] for w in words if w in _ACCENT), None)
    gender = next((_GENDER[w] for w in words if w in _GENDER), None)
    if accent is None and gender is None:
        return None
    for vid in VOICE_IDS:
        if (accent is None or vid[0] == accent) and (gender is None or vid[1] == gender):
            return vid
    return None


def display_name(voice_id: str) -> str:
    return voice_id.split("_", 1)[-1].capitalize()


def next_voice(current: str) -> str:
    if current not in VOICE_IDS:
        return VOICE_IDS[0]
    return VOICE_IDS[(VOICE_IDS.index(current) + 1) % len(VOICE_IDS)]


def clamp_speed(x: float) -> float:
    return max(SPEED_MIN, min(SPEED_MAX, float(x)))
```

`veronica/speech/tts.py` — change constructor and `synth`:
```python
    def __init__(self, voice: str, models_dir: Path, speed: float = 1.0) -> None:
        self.voice = voice
        self.speed = speed
        self._engine = self._kokoro_cls(
            str(models_dir / "kokoro-v1.0.onnx"),
            str(models_dir / "voices-v1.0.bin"),
        )

    def synth(self, text: str) -> tuple[np.ndarray, int]:
        samples, sr = self._engine.create(text, voice=self.voice, speed=self.speed, lang="en-us")
        return np.asarray(samples, dtype=np.float32), sr
```

Note: "american woman" → `af_sarah` because the first `af_` id in table order is Sarah; "a man's" → None because `"a"`, `"man's"` aren't in any table (apostrophe keeps it from matching `man`). Keep the parametrized expectations as written.

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q tests/test_voices.py tests/test_tts.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add veronica/speech/voices.py veronica/speech/tts.py tests/test_voices.py tests/test_tts.py
git commit -m "feat(tts): voice table and mutable voice/speed on Synthesizer"
```

---

### Task 2: Voice/speed intents + orchestrator `_voice_turn` + prefs

**Files:**
- Modify: `veronica/brain/intents.py`, `veronica/orchestrator.py`, `veronica/__main__.py`
- Test: `tests/test_intents.py`, `tests/test_orchestrator.py`, `tests/test_main.py`

**Interfaces:**
- Consumes: Task 1 (`voices` module, `Synthesizer.voice/.speed`).
- Produces: `VoiceAction = tuple[Literal["voice","speed"], str]`, `match_voice_intent(text: str) -> VoiceAction | None`, `Orchestrator._voice_turn(action: VoiceAction) -> None` (also used by Task 3's menu), `Orchestrator.set_voice(voice_id: str)` and `Orchestrator.set_speed(kind: Literal["faster","slower","normal"])` — thin sync wrappers are NOT needed; the menu will schedule `_voice_turn` as a coroutine.

- [ ] **Step 1: Failing intent tests** — append to `tests/test_intents.py`:

```python
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
```

(Note: `normalize()` strips punctuation, so "Adam's" becomes "adams"; `resolve_voice("adams")` is None → Veronica says she doesn't have that voice. Acceptable; "switch to adam voice" works.)

- [ ] **Step 2: Run** `uv run pytest -q tests/test_intents.py -k voice` → FAIL (ImportError).

- [ ] **Step 3: Implement intents** — add to `veronica/brain/intents.py` (near the music intents; reuse `normalize`, `_candidates_for`):

```python
VoiceAction = tuple[Literal["voice", "speed"], str]

_VOICE_PICK_RE = re.compile(
    r"^(?:use|switch to|change to|speak (?:in|with))\s+(?:a |an |the )?(.+?)\s+voice$"
)
_VOICE_NEXT_PHRASES = frozenset({
    "change your voice", "different voice", "use a different voice",
    "change voice", "another voice", "use another voice",
})
_SPEED_PHRASES: dict[str, str] = {
    "speak faster": "faster", "talk faster": "faster", "faster please": "faster",
    "faster": "faster", "speed up": "faster", "speak quicker": "faster",
    "speak slower": "slower", "talk slower": "slower", "slower please": "slower",
    "slower": "slower", "slow down": "slower",
    "normal speed": "normal", "default speed": "normal", "reset speed": "normal",
    "reset your speed": "normal", "speak normally": "normal",
}


def _match_voice_candidate(candidate: str) -> VoiceAction | None:
    if candidate in _VOICE_NEXT_PHRASES:
        return ("voice", "next")
    if candidate in _SPEED_PHRASES:
        return ("speed", _SPEED_PHRASES[candidate])
    m = _VOICE_PICK_RE.match(candidate)
    if m:
        req = m.group(1).strip()
        if req and req not in ("different", "another"):
            return ("voice", req)
        return ("voice", "next")
    return None


def match_voice_intent(text: str) -> VoiceAction | None:
    """"use a british voice" / "switch to adam voice" / "speak faster" …
    Same candidate strategy as match_intent: whole normalized utterance,
    then each clause."""
    for candidate in _candidates_for(normalize(text)):
        result = _match_voice_candidate(candidate)
        if result is not None:
            return result
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        clause_norm = normalize(clause)
        if not clause_norm:
            continue
        for candidate in _candidates_for(clause_norm):
            result = _match_voice_candidate(candidate)
            if result is not None:
                return result
    return None
```

Check "faster please": `normalize` strips a trailing " please" (`_TRAIL_SUFFIX`), so the candidate is `"faster"` — that's why bare `"faster"`/`"slower"` are in the table. "faster internet please" → `"faster internet"` → no match. Good.

- [ ] **Step 4: Run** `uv run pytest -q tests/test_intents.py` → PASS. Commit:

```bash
git add veronica/brain/intents.py tests/test_intents.py
git commit -m "feat(intents): voice and speed phrases"
```

- [ ] **Step 5: Failing orchestrator tests** — append to `tests/test_orchestrator.py`:

```python
# -- batch B: voice / speed --------------------------------------------------

from veronica import prefs as prefs_mod


class TTS2(TTS):
    def __init__(self):
        super().__init__()
        self.voice = "af_sarah"
        self.speed = 1.0
        self.spoken_with = []   # (text, voice, speed) at synth time

    async def asynth(self, text):
        self.spoken_with.append((text, self.voice, self.speed))
        return await super().asynth(text)


def build_voice(stt_texts, monkeypatch):
    saved = []
    monkeypatch.setattr(prefs_mod, "save", lambda d: saved.append(d))
    states, events = [], []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT(stt_texts),
        brain=Brain(), tts=TTS2(), player=Player(), on_state=states.append,
        on_event=lambda k, p: events.append((k, p)),
    )
    return o, saved, events


async def test_voice_intent_switches_voice_and_saves(monkeypatch):
    o, saved, ev = build_voice(["use a british male voice"], monkeypatch)
    await o.one_turn()
    assert o.tts.voice == "bm_george"
    assert o.tts.spoken_with[-1] == ("Okay, this is George.", "bm_george", 1.0)
    assert {"tts_voice": "bm_george"} in saved
    assert ("tool", {"summary": "Voice: George", "decision": "auto"}) in ev
    assert o.brain.asked == []


async def test_voice_intent_unknown_lists_voices(monkeypatch):
    o, saved, _ = build_voice(["use a robot voice"], monkeypatch)
    await o.one_turn()
    assert o.tts.voice == "af_sarah"
    assert saved == []
    assert o.tts.said[-1].startswith("I don't have that voice. I have Sarah, Bella")
    assert o.tts.said[-1].endswith("George and Lewis.")


async def test_voice_intent_next_cycles(monkeypatch):
    o, saved, _ = build_voice(["change your voice"], monkeypatch)
    await o.one_turn()
    assert o.tts.voice == "af_bella"
    assert {"tts_voice": "af_bella"} in saved


async def test_speed_faster_and_clamp(monkeypatch):
    o, saved, _ = build_voice(["speak faster"], monkeypatch)
    await o.one_turn()
    assert o.tts.speed == pytest.approx(1.15)
    assert o.tts.spoken_with[-1][0] == "Like this?"
    assert {"tts_speed": pytest.approx(1.15)} in saved or any(abs(d.get("tts_speed", 0) - 1.15) < 1e-9 for d in saved)

    o.tts.speed = 1.5
    o.stt = STT(["speak faster"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.tts.speed == 1.5
    assert o.tts.said[-1] == "That's as fast as I go."


async def test_speed_slower_normal(monkeypatch):
    o, saved, _ = build_voice(["slow down"], monkeypatch)
    await o.one_turn()
    assert o.tts.speed == pytest.approx(0.85)
    o.stt = STT(["normal speed"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.tts.speed == 1.0
    assert o.tts.said[-1] == "Like this?"
    o.stt = STT(["normal speed"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert o.tts.said[-1] == "Already at normal speed."
```

- [ ] **Step 6: Run** `uv run pytest -q tests/test_orchestrator.py -k "voice or speed"` → FAIL.

- [ ] **Step 7: Implement `_voice_turn` and dispatch** in `veronica/orchestrator.py`:

Imports: `from veronica import prefs` (check it isn't already imported), `from veronica.speech import voices`, add `match_voice_intent` to the intents import.

Add method next to `_music_turn`:
```python
    # -- voice & speed (B1) ---------------------------------------------------------
    async def _voice_turn(self, action: tuple[str, str]) -> None:
        """Local fast path for "use a british voice" / "speak faster":
        mutate the running Synthesizer, persist to prefs.json, and confirm
        in the new voice/speed so the user hears the change immediately."""
        kind, arg = action
        if kind == "voice":
            vid = voices.next_voice(self.tts.voice) if arg == "next" else voices.resolve_voice(arg)
            if vid is None:
                names = [voices.display_name(v) for v in voices.VOICE_IDS]
                await self.say(
                    "I don't have that voice. I have " + ", ".join(names[:-1]) + " and " + names[-1] + "."
                )
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
```

Dispatch in `one_turn`: extend the local-intent chain right after `note_body` is computed:
```python
            voice_action = (
                None
                if (intent is not None or mem is not None or screen_intent or music_action or note_body is not None)
                else match_voice_intent(text)
            )
```
and make `dictation_intent`'s guard also include `or voice_action`. Add the branch right after the `elif note_body is not None:` branch:
```python
            elif voice_action is not None:
                self.player.reset()
                await self._voice_turn(voice_action)
```

- [ ] **Step 8: Prefs at startup** — in `veronica/__main__.py` `build_orchestrator`, replace `tts=Synthesizer(s.kokoro_voice, s.models_dir),` with:
```python
        tts=Synthesizer(saved_voice, s.models_dir, speed=saved_speed),
```
and before constructing the orchestrator:
```python
    saved = prefs.load()
    saved_voice = saved.get("tts_voice") or s.kokoro_voice
    saved_speed = voices.clamp_speed(saved.get("tts_speed", voices.DEFAULT_SPEED))
```
with imports `from veronica import prefs` and `from veronica.speech import voices`. Look at `tests/test_main.py` for how `build_orchestrator` is tested (it fakes `Synthesizer`); add:
```python
def test_build_orchestrator_applies_saved_voice_prefs(monkeypatch, ...):
    # monkeypatch veronica.__main__.prefs.load to return {"tts_voice": "am_adam", "tts_speed": 9}
    # build; assert orch.tts.voice == "am_adam" and orch.tts.speed == 1.5 (clamped)
```
Write it in the same style as the neighbouring `build_orchestrator` tests (copy their fixture/monkeypatching of model classes). If prefs.load is imported as a module attribute, patch `veronica.__main__.prefs.load`.

- [ ] **Step 9: Run full suite** `uv run pytest -q` → PASS. Commit:

```bash
git add veronica/orchestrator.py veronica/__main__.py tests/test_orchestrator.py tests/test_main.py
git commit -m "feat: voice and speed local intents, persisted in prefs"
```

---

### Task 3: Voice submenu in menu bar + orb popup

**Files:**
- Modify: `veronica/ui/menubar.py`
- Test: `tests/test_menubar.py`

**Interfaces:**
- Consumes: `Orchestrator._voice_turn(action)`, `voices.VOICES`, `voices.display_name`, orchestrator loop `self._loop`.
- Produces: `VeronicaApp._voice_menu: rumps.MenuItem` (submenu titled "Voice") with children for each display name + separator + "Faster", "Slower", "Normal speed"; `VeronicaApp._refresh_voice_menu()` sets `state` (checkmark) on the current voice.

- [ ] **Step 1: Read** `veronica/ui/menubar.py` fully (menu construction ~lines 95–130, `_build_popup_menu`/`_make_popup_handler_class` ~lines 20–65 and 273–330, `toggle_mute` for the orchestrator-thread pattern) and `tests/test_menubar.py` (how `VeronicaApp` is built with a fake rumps / fake orchestrator).

- [ ] **Step 2: Failing tests** — append to `tests/test_menubar.py` (adapt to the file's existing fixtures; below assumes the existing `make_app()`-style helper and fake orch pattern used by the mute tests):

```python
def test_voice_submenu_lists_voices_and_speed(make_app):
    app = make_app()
    sub = app._voice_menu
    titles = [getattr(i, "title", i) for i in sub.values()] if hasattr(sub, "values") else [i.title for i in sub]
    assert titles[:10] == ["Sarah", "Bella", "Nicole", "Sky", "Adam", "Michael", "Emma", "Isabella", "George", "Lewis"]
    assert titles[-3:] == ["Faster", "Slower", "Normal speed"]


def test_voice_menu_click_schedules_voice_turn(make_app):
    app = make_app()
    calls = []
    class Orch:
        tts = type("T", (), {"voice": "af_sarah"})()
        async def _voice_turn(self, action): calls.append(action)
    app._orch = Orch()
    app._loop = asyncio.new_event_loop()
    app._pick_voice(app._voice_items["Adam"])
    app._loop.run_until_complete(asyncio.sleep(0))
    assert calls == [("voice", "adam")]
    app._speed(app._speed_items["Faster"])
    app._loop.run_until_complete(asyncio.sleep(0))
    assert calls[-1] == ("speed", "faster")


def test_refresh_voice_menu_checks_current(make_app):
    app = make_app()
    app._orch = type("O", (), {"tts": type("T", (), {"voice": "bm_george"})()})()
    app._refresh_voice_menu()
    assert app._voice_items["George"].state == 1
    assert app._voice_items["Sarah"].state == 0
```

If `tests/test_menubar.py` has no `make_app` fixture, build the app the same way its existing tests do and adjust these three tests accordingly — the assertions are the contract.

- [ ] **Step 3: Implement** in `menubar.py`:

```python
from veronica.speech import voices
```
In `__init__` where `menu_items` is assembled, before `hud_mode_item` is appended:
```python
        self._voice_items: dict[str, rumps.MenuItem] = {}
        self._speed_items: dict[str, rumps.MenuItem] = {}
        voice_menu = rumps.MenuItem("Voice")
        for vid in voices.VOICE_IDS:
            name = voices.display_name(vid)
            item = rumps.MenuItem(name, callback=self._pick_voice)
            self._voice_items[name] = item
            voice_menu.add(item)
        voice_menu.add(None)
        for title in ("Faster", "Slower", "Normal speed"):
            item = rumps.MenuItem(title, callback=self._speed)
            self._speed_items[title] = item
            voice_menu.add(item)
        self._voice_menu = voice_menu
```
and insert `voice_menu` into `menu_items` after the HUD item. Methods:
```python
    def _schedule(self, coro) -> None:
        loop = getattr(self, "_loop", None)
        if loop is None:
            coro.close()
            return
        asyncio.run_coroutine_threadsafe(coro, loop) if loop.is_running() else loop.create_task(coro)

    def _pick_voice(self, item: rumps.MenuItem) -> None:
        orch = getattr(self, "_orch", None)
        if orch is None:
            return
        self._schedule(orch._voice_turn(("voice", item.title.lower())))
        self._refresh_voice_menu()

    def _speed(self, item: rumps.MenuItem) -> None:
        orch = getattr(self, "_orch", None)
        if orch is None:
            return
        kind = {"Faster": "faster", "Slower": "slower", "Normal speed": "normal"}[item.title]
        self._schedule(orch._voice_turn(("speed", kind)))

    def _refresh_voice_menu(self) -> None:
        orch = getattr(self, "_orch", None)
        current = getattr(getattr(orch, "tts", None), "voice", None)
        for name, item in self._voice_items.items():
            item.state = 1 if current and voices.display_name(current) == name else 0
```
Call `self._refresh_voice_menu()` from `_refresh` (the 0.25 s rumps timer) — cheap, keeps the checkmark right after a voice change by voice. Extend the orb popup (`_build_popup_menu`) with a "Voice" submenu mirroring the same items: build an `NSMenu` titled "Voice", add one `NSMenuItem` per voice name with action `onPickVoice_` and `representedObject`/tag = index, plus Faster/Slower/Normal speed with `onSpeed_`; add the corresponding handler methods to the popup handler class that call `self._app._pick_voice(self._app._voice_items[name])` / `self._app._speed(...)`. Follow exactly how the existing popup builds `Mute`/`HUD` items (selector + target pattern).

Note on `test_voice_menu_click_schedules_voice_turn`: with a non-running loop `_schedule` uses `loop.create_task`, so `run_until_complete(asyncio.sleep(0))` drives it.

- [ ] **Step 4: Run** `uv run pytest -q tests/test_menubar.py` then the full suite → PASS. Commit:

```bash
git add veronica/ui/menubar.py tests/test_menubar.py
git commit -m "feat(menubar): Voice submenu (voices, faster/slower/normal) in menu bar and orb popup"
```

---

### Task 4: `proactive.py` — Schedule, parsers, briefing composer, ticker

**Files:**
- Create: `veronica/proactive.py`
- Test: `tests/test_proactive.py`

**Interfaces:**
- Produces:
  - `@dataclass Schedule(briefing_enabled: bool=False, briefing_time: str="08:00", nudges_enabled: bool=False, nudge_minutes: int=5)`; `Schedule.from_prefs(d: dict) -> Schedule`; `Schedule.to_prefs() -> dict`; `load_schedule(load=prefs.load) -> Schedule`; `save_schedule(s: Schedule, save=prefs.save) -> None` (writes `{"proactive": {...}}`).
  - `EVENT_LINE_RE`, `parse_events(text: str, today: date) -> list[Event]` where `@dataclass Event(title: str, start: datetime | None, end: datetime | None, all_day: bool)`.
  - `parse_reminders(text: str) -> list[str]` (titles), `count_mail(text: str) -> int`.
  - `class Proactive(schedule, announce, calendar_events, mail_unread_count, reminders_due, now=datetime.now)` with `TICK_S = 60`, `EVENTS_CACHE_S = 300`, `async start()`, `stop()`, `async tick()`, `async build_briefing() -> str`.

- [ ] **Step 1: Failing tests** — `tests/test_proactive.py`:

```python
import asyncio
import datetime as dt

import pytest

from veronica import proactive as pr


TODAY = dt.date(2026, 9, 16)


def test_schedule_roundtrip():
    s = pr.Schedule(briefing_enabled=True, briefing_time="07:30", nudges_enabled=True, nudge_minutes=10)
    assert pr.Schedule.from_prefs(s.to_prefs()) == s
    assert pr.Schedule.from_prefs({}) == pr.Schedule()
    assert pr.Schedule.from_prefs({"briefing_time": "bogus", "nudge_minutes": "x"}) == pr.Schedule()


def test_load_save_schedule():
    store = {}
    s = pr.Schedule(briefing_enabled=True)
    pr.save_schedule(s, save=lambda d: store.update(d))
    assert store == {"proactive": s.to_prefs()}
    assert pr.load_schedule(load=lambda: store) == s


def test_parse_events():
    text = "09:30–10:00  Standup (Work) @ Zoom\n00:00–00:00  Holiday (Home)\n13:00–14:00  Lunch with Sam (Personal)"
    evs = pr.parse_events(text, TODAY)
    assert [e.title for e in evs] == ["Standup", "Holiday", "Lunch with Sam"]
    assert evs[0].start == dt.datetime(2026, 9, 16, 9, 30) and evs[0].end == dt.datetime(2026, 9, 16, 10, 0)
    assert evs[1].all_day and evs[1].start is None
    assert pr.parse_events("No events.", TODAY) == []
    assert pr.parse_events("", TODAY) == []


def test_parse_reminders_and_mail():
    assert pr.parse_reminders("2026-09-16 09:00  Pay rent (Bills)\n2026-09-16 18:00  Call mum") == ["Pay rent", "Call mum"]
    assert pr.parse_reminders("No reminders due.") == []
    assert pr.count_mail("10:02  Alice — Hi\n  preview\n09:00  Bob — Yo\n  preview") == 2
    assert pr.count_mail("No messages.") == 0


class Clock:
    def __init__(self, t): self.t = t
    def __call__(self): return self.t


def make(schedule, events="No events.", mail=0, reminders="No reminders due.", now=None):
    said = []
    calls = {"events": 0}

    async def announce(t): said.append(t)
    async def cal(day, days):
        calls["events"] += 1
        return events
    async def mail_count(): return mail
    async def rem(days): return reminders

    clock = Clock(now or dt.datetime(2026, 9, 16, 8, 0))
    p = pr.Proactive(schedule, announce, cal, mail_count, rem, now=clock)
    return p, said, clock, calls


async def test_build_briefing_composition():
    p, _, clock, _ = make(pr.Schedule(), events="09:30–10:00  Standup (Work)\n13:00–14:00  Lunch (P)\n15:00–15:30  A (P)\n16:00–16:30  B (P)\n17:00–17:30  C (P)\n18:00–18:30  D (P)", mail=3, reminders="2026-09-16 09:00  Pay rent (Bills)")
    text = await p.build_briefing()
    assert text == ("Good morning, Manik. You have 6 events today: Standup at 9:30, Lunch at 13:00, A at 15:00, "
                    "B at 16:00 and 2 more. You have 3 unread emails. Reminders due: Pay rent.")


async def test_build_briefing_empty_and_greetings():
    p, _, clock, _ = make(pr.Schedule())
    assert await p.build_briefing() == "Good morning, Manik. Nothing on your calendar today."
    clock.t = dt.datetime(2026, 9, 16, 13, 0)
    assert (await p.build_briefing()).startswith("Good afternoon, Manik.")
    clock.t = dt.datetime(2026, 9, 16, 19, 0)
    assert (await p.build_briefing()).startswith("Good evening, Manik.")


async def test_build_briefing_one_email_and_reminder_join():
    p, _, _, _ = make(pr.Schedule(), mail=1, reminders="2026-09-16 09:00  A\n2026-09-16 09:00  B\n2026-09-16 09:00  C\n2026-09-16 09:00  D")
    text = await p.build_briefing()
    assert "You have 1 unread email." in text
    assert text.endswith("Reminders due: A, B and C.")


async def test_build_briefing_tolerates_failing_fetcher():
    p, _, _, _ = make(pr.Schedule(), mail=2)
    async def boom(day, days): raise RuntimeError("no calendar")
    p._calendar_events = boom
    assert await p.build_briefing() == "Good morning, Manik. You have 2 unread emails."


async def test_briefing_fires_once_per_day_at_time():
    p, said, clock, _ = make(pr.Schedule(briefing_enabled=True, briefing_time="08:00"))
    clock.t = dt.datetime(2026, 9, 16, 7, 59)
    await p.tick(); assert said == []
    clock.t = dt.datetime(2026, 9, 16, 8, 0)
    await p.tick(); assert len(said) == 1 and said[0].startswith("Good morning")
    await p.tick(); assert len(said) == 1
    clock.t = dt.datetime(2026, 9, 17, 8, 0)
    await p.tick(); assert len(said) == 2


async def test_briefing_disabled_never_fires():
    p, said, clock, _ = make(pr.Schedule(briefing_enabled=False))
    await p.tick(); assert said == []


async def test_nudge_fires_once_within_window_and_skips_all_day():
    ev = "09:30–10:00  Standup (Work)\n00:00–00:00  Holiday (Home)\n11:00–12:00  Review (Work)"
    p, said, clock, calls = make(pr.Schedule(nudges_enabled=True, nudge_minutes=5), events=ev)
    clock.t = dt.datetime(2026, 9, 16, 9, 20)
    await p.tick(); assert said == []
    clock.t = dt.datetime(2026, 9, 16, 9, 25)
    await p.tick(); assert said == ["Heads up, Standup starts in 5 minutes."]
    clock.t = dt.datetime(2026, 9, 16, 9, 26)
    await p.tick(); assert len(said) == 1          # not repeated
    clock.t = dt.datetime(2026, 9, 16, 10, 59)
    await p.tick(); assert said[-1] == "Heads up, Review starts in a minute."
    assert calls["events"] == 1                      # cached within EVENTS_CACHE_S


async def test_nudge_refetches_after_cache_expiry_and_ignores_past():
    p, said, clock, calls = make(pr.Schedule(nudges_enabled=True), events="09:00–09:30  Old (W)")
    clock.t = dt.datetime(2026, 9, 16, 9, 10)
    await p.tick(); assert said == []              # already started: no nudge
    clock.t = dt.datetime(2026, 9, 16, 9, 16)
    await p.tick(); assert calls["events"] == 2


async def test_start_stop_runs_tick_loop():
    p, said, clock, _ = make(pr.Schedule(briefing_enabled=True, briefing_time="08:00"))
    p.TICK_S = 0.01
    await p.start()
    await asyncio.sleep(0.05)
    p.stop()
    assert len(said) == 1
```

- [ ] **Step 2: Run** `uv run pytest -q tests/test_proactive.py` → FAIL (no module).

- [ ] **Step 3: Implement** `veronica/proactive.py`:

```python
"""Proactive announcements: a scheduled daily briefing and "heads up"
nudges before calendar events. Composes text from the pim tool outputs
and hands it to Orchestrator.announce(), which only speaks when idle and
not muted — this module never touches audio itself."""
import asyncio
import contextlib
import datetime as dt
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass

from veronica import prefs

log = logging.getLogger("veronica.proactive")

_TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


@dataclass
class Schedule:
    briefing_enabled: bool = False
    briefing_time: str = "08:00"   # HH:MM, 24h, local time
    nudges_enabled: bool = False
    nudge_minutes: int = 5

    @classmethod
    def from_prefs(cls, d: dict) -> "Schedule":
        s = cls()
        if not isinstance(d, dict):
            return s
        if isinstance(d.get("briefing_enabled"), bool):
            s.briefing_enabled = d["briefing_enabled"]
        if isinstance(d.get("briefing_time"), str) and _TIME_RE.match(d["briefing_time"]):
            s.briefing_time = d["briefing_time"]
        if isinstance(d.get("nudges_enabled"), bool):
            s.nudges_enabled = d["nudges_enabled"]
        nm = d.get("nudge_minutes")
        if isinstance(nm, int) and not isinstance(nm, bool) and 1 <= nm <= 60:
            s.nudge_minutes = nm
        return s

    def to_prefs(self) -> dict:
        return asdict(self)


def load_schedule(load: Callable[[], dict] = prefs.load) -> Schedule:
    return Schedule.from_prefs((load() or {}).get("proactive", {}))


def save_schedule(s: Schedule, save: Callable[[dict], None] = prefs.save) -> None:
    save({"proactive": s.to_prefs()})


# -- parsing the pim tools' text ---------------------------------------------
# pim._format_events: "HH:MM–HH:MM  title (calendar)" [" @ location"], en dash.
EVENT_LINE_RE = re.compile(r"^(\d{2}):(\d{2})–(\d{2}):(\d{2})  (.+?) \([^()]*\)(?: @ .*)?$")
# pim._format_reminders: "YYYY-MM-DD HH:MM  name" [" (list)"]
REMINDER_LINE_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}  (.+?)(?: \([^()]*\))?$")


@dataclass
class Event:
    title: str
    start: dt.datetime | None
    end: dt.datetime | None
    all_day: bool


def parse_events(text: str, today: dt.date) -> list[Event]:
    out: list[Event] = []
    for line in (text or "").splitlines():
        m = EVENT_LINE_RE.match(line.strip())
        if not m:
            continue
        sh, sm, eh, em, title = int(m[1]), int(m[2]), int(m[3]), int(m[4]), m[5].strip()
        if (sh, sm, eh, em) == (0, 0, 0, 0):
            out.append(Event(title, None, None, True))
            continue
        out.append(Event(
            title,
            dt.datetime.combine(today, dt.time(sh, sm)),
            dt.datetime.combine(today, dt.time(eh, em)),
            False,
        ))
    return out


def parse_reminders(text: str) -> list[str]:
    out = []
    for line in (text or "").splitlines():
        m = REMINDER_LINE_RE.match(line.strip())
        if m:
            out.append(m[1].strip())
    return out


def count_mail(text: str) -> int:
    # pim._format_mail: header line per message, preview line indented by 2.
    if not text or text.startswith("No messages"):
        return 0
    return sum(1 for ln in text.splitlines() if ln and not ln.startswith("  "))


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _clock(t: dt.datetime) -> str:
    return f"{t.hour}:{t.minute:02d}"


# -- the ticker --------------------------------------------------------------
class Proactive:
    TICK_S = 60
    EVENTS_CACHE_S = 300
    USER_NAME = "Manik"
    BRIEFING_MAX_TITLES = 4
    REMINDERS_MAX = 3

    def __init__(
        self,
        schedule: Schedule,
        announce: Callable[[str], Awaitable[None]],
        calendar_events: Callable[[str, int], Awaitable[str]],
        mail_unread_count: Callable[[], Awaitable[int]],
        reminders_due: Callable[[int], Awaitable[str]],
        now: Callable[[], dt.datetime] = dt.datetime.now,
    ) -> None:
        self.schedule = schedule
        self._announce = announce
        self._calendar_events = calendar_events
        self._mail_unread_count = mail_unread_count
        self._reminders_due = reminders_due
        self._now = now
        self._task: asyncio.Task | None = None
        self._last_briefing_date: dt.date | None = None
        self._nudged: set[tuple[str, dt.datetime]] = set()
        self._events_cache: tuple[dt.datetime, list[Event]] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self._loop())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("proactive tick failed")
            await asyncio.sleep(self.TICK_S)

    async def tick(self) -> None:
        now = self._now()
        if self.schedule.briefing_enabled and now.strftime("%H:%M") == self.schedule.briefing_time \
                and self._last_briefing_date != now.date():
            self._last_briefing_date = now.date()
            await self._announce(await self.build_briefing())
        if self.schedule.nudges_enabled:
            await self._check_nudges(now)

    # -- briefing --------------------------------------------------------------
    async def _events_today(self, now: dt.datetime) -> list[Event]:
        if self._events_cache is not None:
            fetched_at, events = self._events_cache
            if (now - fetched_at).total_seconds() < self.EVENTS_CACHE_S and fetched_at.date() == now.date():
                return events
        events = parse_events(await self._calendar_events("today", 1), now.date())
        self._events_cache = (now, events)
        return events

    async def build_briefing(self) -> str:
        now = self._now()
        hour = now.hour
        greeting = "Good morning" if hour < 12 else ("Good afternoon" if hour < 17 else "Good evening")
        parts = [f"{greeting}, {self.USER_NAME}."]

        try:
            events = parse_events(await self._calendar_events("today", 1), now.date())
        except Exception:
            log.exception("briefing: calendar fetch failed")
            events = None
        if events is not None:
            if not events:
                parts.append("Nothing on your calendar today.")
            else:
                names = [f"{e.title} at {_clock(e.start)}" if e.start else e.title for e in events]
                shown = names[: self.BRIEFING_MAX_TITLES]
                rest = len(names) - len(shown)
                listed = _join(shown) if rest == 0 else ", ".join(shown) + f" and {rest} more"
                plural = "event" if len(events) == 1 else "events"
                parts.append(f"You have {len(events)} {plural} today: {listed}.")

        try:
            n = int(await self._mail_unread_count())
        except Exception:
            log.exception("briefing: mail fetch failed")
            n = 0
        if n:
            parts.append(f"You have {n} unread email{'s' if n != 1 else ''}.")

        try:
            reminders = parse_reminders(await self._reminders_due(1))
        except Exception:
            log.exception("briefing: reminders fetch failed")
            reminders = []
        if reminders:
            parts.append(f"Reminders due: {_join(reminders[: self.REMINDERS_MAX])}.")
        return " ".join(parts)

    # -- nudges ----------------------------------------------------------------
    async def _check_nudges(self, now: dt.datetime) -> None:
        events = await self._events_today(now)
        window = dt.timedelta(minutes=self.schedule.nudge_minutes)
        self._nudged = {k for k in self._nudged if k[1].date() == now.date()}
        for e in events:
            if e.all_day or e.start is None:
                continue
            delta = e.start - now
            if delta < dt.timedelta(0) or delta > window:
                continue
            key = (e.title, e.start)
            if key in self._nudged:
                continue
            self._nudged.add(key)
            mins = max(1, int(round(delta.total_seconds() / 60)))
            when = "in a minute" if mins == 1 else f"in {mins} minutes"
            await self._announce(f"Heads up, {e.title} starts {when}.")
```

Check the composition test by hand: 6 events → first 4 shown joined with ", " then " and 2 more" — matches `"Standup at 9:30, Lunch at 13:00, A at 15:00, B at 16:00 and 2 more"`. Reminders 4 → first 3 → `"A, B and C"`. One email → `"1 unread email"`.

`test_nudge_refetches_after_cache_expiry_and_ignores_past`: at 9:10 the fetch happens (cache set), the event at 9:00 is in the past → no nudge; at 9:16 (6 min later, > 300 s) it refetches → `calls == 2`.

- [ ] **Step 4: Run** `uv run pytest -q tests/test_proactive.py` → PASS. Commit:

```bash
git add veronica/proactive.py tests/test_proactive.py
git commit -m "feat(proactive): schedule, briefing composer and nudge ticker"
```

---

### Task 5: Proactive intents + orchestrator wiring + `__main__` adapters

**Files:**
- Modify: `veronica/brain/intents.py`, `veronica/orchestrator.py`, `veronica/__main__.py`, `README.md`
- Test: `tests/test_intents.py`, `tests/test_orchestrator.py`, `tests/test_main.py`

**Interfaces:**
- Consumes: Task 4 (`Proactive`, `Schedule`, `load_schedule`, `save_schedule`), `Orchestrator.announce`, pim handlers `calendar_events.handler`, `mail_unread.handler`, `reminders_due.handler`.
- Produces: `ProactiveAction = tuple[Literal["brief_now","briefing_on","briefing_off","nudges_on","nudges_off"], str | int | None]`, `match_proactive_intent(text) -> ProactiveAction | None`, `parse_clock_time(s: str) -> str | None` (→ `"HH:MM"`), `Orchestrator.proactive: Proactive | None` (constructor kwarg `proactive=None`), `Orchestrator._proactive_turn(action)`.

- [ ] **Step 1: Failing intent tests** — append to `tests/test_intents.py`:

```python
from veronica.brain.intents import match_proactive_intent, parse_clock_time


@pytest.mark.parametrize("s,expected", [
    ("8", "08:00"), ("8 am", "08:00"), ("8am", "08:00"), ("8:30", "08:30"), ("8:30 am", "08:30"),
    ("7 30 am", "07:30"), ("6 pm", "18:00"), ("6:15 pm", "18:15"), ("12 pm", "12:00"), ("12 am", "00:00"),
    ("noon", "12:00"), ("midnight", "00:00"), ("18:45", "18:45"), ("25", None), ("8:75", None), ("", None),
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
])
def test_match_proactive_intent(text, expected):
    assert match_proactive_intent(text) == expected
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** in `intents.py`:

```python
ProactiveAction = tuple[
    Literal["brief_now", "briefing_on", "briefing_off", "nudges_on", "nudges_off"], str | int | None
]

_BRIEF_NOW_PHRASES = frozenset({
    "brief me", "give me a briefing", "give me my briefing", "morning briefing", "my briefing",
    "whats my day look like", "what does my day look like", "what does my day look like today",
    "how does my day look", "whats my day like",
})
_BRIEFING_ON_RE = re.compile(
    r"^(?:give me|start|turn on|enable|set up)\s+(?:a |the |my )?(?:morning |daily )?briefings?"
    r"(?:\s+(?:every day|every morning|daily|each morning))?(?:\s+at\s+(.+))?$"
)
_BRIEFING_OFF_RE = re.compile(
    r"^(?:stop|turn off|cancel|disable)\s+(?:the |my )?(?:morning |daily )?briefings?$"
)
_NUDGES_ON_RE = re.compile(
    r"^(?:remind me|warn me|nudge me|tell me|alert me|turn on nudges|enable nudges)"
    r"(?:\s+(\d{1,2})\s+minutes?)?(?:\s+before\s+(?:my |the )?(?:meetings?|events?|calendar events?))?$"
)
_NUDGES_OFF_RE = re.compile(
    r"^(?:stop|turn off|disable|cancel)\s+(?:the |my )?"
    r"(?:meeting nudges|nudges|meeting reminders|reminders before (?:my )?meetings)$"
)
_CLOCK_RE = re.compile(r"^(\d{1,2})(?:[:\s](\d{2}))?\s*(am|pm)?$")


def parse_clock_time(s: str) -> str | None:
    """"8", "8 am", "8:30", "7 30 am", "6 pm", "noon" → "HH:MM" (24h) or None."""
    s = (s or "").strip().lower().replace(".", "")
    if s == "noon":
        return "12:00"
    if s == "midnight":
        return "00:00"
    m = _CLOCK_RE.match(s)
    if not m:
        return None
    h, mm, ap = int(m[1]), int(m[2] or 0), m[3]
    if mm > 59:
        return None
    if ap:
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if ap == "pm" else 0)
    elif not 0 <= h <= 23:
        return None
    return f"{h:02d}:{mm:02d}"


def _match_proactive_candidate(candidate: str) -> ProactiveAction | None:
    if candidate in _BRIEF_NOW_PHRASES:
        return ("brief_now", None)
    m = _BRIEFING_OFF_RE.match(candidate)
    if m:
        return ("briefing_off", None)
    m = _BRIEFING_ON_RE.match(candidate)
    if m:
        if m.group(1):
            t = parse_clock_time(m.group(1))
            return ("briefing_on", t) if t else None
        return ("briefing_on", None)
    m = _NUDGES_OFF_RE.match(candidate)
    if m:
        return ("nudges_off", None)
    m = _NUDGES_ON_RE.match(candidate)
    if m:
        # "remind me" alone (no "before meetings") is a reminder request, not nudges
        if "before" not in candidate and "nudges" not in candidate:
            return None
        return ("nudges_on", int(m.group(1)) if m.group(1) else None)
    return None


def match_proactive_intent(text: str) -> ProactiveAction | None:
    for candidate in _candidates_for(normalize(text)):
        result = _match_proactive_candidate(candidate)
        if result is not None:
            return result
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        clause_norm = normalize(clause)
        if not clause_norm:
            continue
        for candidate in _candidates_for(clause_norm):
            result = _match_proactive_candidate(candidate)
            if result is not None:
                return result
    return None
```

Watch `normalize()`: it strips punctuation (so `"7:30"` becomes `"730"`, `"what's"` → `"whats"`). Read `normalize` before implementing. If colons are stripped, `_CLOCK_RE` must also accept `^(\d{1,2})(\d{2})\s*(am|pm)?$` for 3–4 digit forms (`"730"` → 7:30, `"1845"` → 18:45): add that as a second regex `_CLOCK_COMPACT_RE = re.compile(r"^(\d{1,2})(\d{2})\s*(am|pm)?$")` tried when the first fails, and make `parse_clock_time` handle both. `test_parse_clock_time` exercises the un-normalized strings directly, and `test_match_proactive_intent` the normalized path — both must pass.

- [ ] **Step 4: Run** `uv run pytest -q tests/test_intents.py` → PASS. Commit `feat(intents): briefing and nudge phrases, clock-time parser`.

- [ ] **Step 5: Failing orchestrator tests** — append to `tests/test_orchestrator.py`:

```python
# -- batch B: proactive -------------------------------------------------------

from veronica import proactive as pr_mod


class FakeProactive:
    def __init__(self):
        self.schedule = pr_mod.Schedule()
        self.started = 0
    async def start(self): self.started += 1
    def stop(self): pass
    async def build_briefing(self): return "Good morning, Manik. Nothing on your calendar today."


def build_pro(stt_texts, monkeypatch):
    saved = []
    monkeypatch.setattr(pr_mod, "save_schedule", lambda s, save=None: saved.append(s.to_prefs()))
    states, events = [], []
    p = FakeProactive()
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec([np.zeros(1, np.int16), None]), stt=STT(stt_texts),
        brain=Brain(), tts=TTS(), player=Player(), on_state=states.append,
        on_event=lambda k, p: events.append((k, p)), proactive=p,
    )
    return o, p, saved, events


async def test_brief_now_speaks_briefing(monkeypatch):
    o, p, saved, ev = build_pro(["brief me"], monkeypatch)
    await o.one_turn()
    assert o.tts.said == ["Good morning, Manik. Nothing on your calendar today."]
    assert ("tool", {"summary": "Briefing", "decision": "auto"}) in ev
    assert o.brain.asked == [] and saved == []


async def test_briefing_on_with_time_saves_and_confirms(monkeypatch):
    o, p, saved, _ = build_pro(["give me a briefing every morning at 7:30 am"], monkeypatch)
    await o.one_turn()
    assert p.schedule.briefing_enabled and p.schedule.briefing_time == "07:30"
    assert saved[-1]["briefing_time"] == "07:30"
    assert o.tts.said[-1] == "Okay, I'll brief you every day at 7:30."


async def test_briefing_on_without_time_keeps_stored(monkeypatch):
    o, p, saved, _ = build_pro(["turn on the morning briefing"], monkeypatch)
    p.schedule.briefing_time = "09:15"
    await o.one_turn()
    assert p.schedule.briefing_enabled and p.schedule.briefing_time == "09:15"
    assert o.tts.said[-1] == "Okay, I'll brief you every day at 9:15."


async def test_briefing_off_nudges_on_off(monkeypatch):
    o, p, saved, _ = build_pro(["stop the morning briefing"], monkeypatch)
    p.schedule.briefing_enabled = True
    await o.one_turn()
    assert not p.schedule.briefing_enabled and o.tts.said[-1] == "Okay, no more morning briefings."

    o.stt = STT(["warn me 10 minutes before my meetings"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert p.schedule.nudges_enabled and p.schedule.nudge_minutes == 10
    assert o.tts.said[-1] == "Okay, I'll warn you 10 minutes before each event."

    o.stt = STT(["turn off nudges"]); o.recorder = Rec([np.zeros(1, np.int16), None])
    await o.one_turn()
    assert not p.schedule.nudges_enabled and o.tts.said[-1] == "Okay, no more meeting nudges."


async def test_proactive_intent_without_proactive_says_unavailable(monkeypatch):
    o, _ = build(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["brief me"])
    await o.one_turn()
    assert o.tts.said[-1] == "Briefings aren't available right now."
    assert o.brain.asked == []


async def test_run_forever_starts_proactive(monkeypatch):
    o, p, _, _ = build_pro([], monkeypatch)
    # run_forever loops forever; drive one iteration by cancelling after start
    task = asyncio.ensure_future(o.run_forever())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert p.started == 1
```

If `run_forever` in the existing tests is driven differently (look for an existing `run_forever` test and copy its harness), adapt the last test to that harness — the assertion `p.started == 1` is the contract.

- [ ] **Step 6: Implement** in `orchestrator.py`:

- Constructor: add kwarg `proactive=None`, store `self.proactive = proactive`.
- `run_forever`: right after `self._set("idle")` at the top, add
  ```python
        if self.proactive is not None:
            await self.proactive.start()
  ```
- Imports: `match_proactive_intent`, `from veronica import proactive as proactive_mod`.
- Method:
```python
    # -- proactive briefings & nudges (B2) ---------------------------------------
    _TIME_SPOKEN = staticmethod(lambda hhmm: f"{int(hhmm[:2])}:{hhmm[3:]}")

    async def _proactive_turn(self, action: tuple[str, object]) -> None:
        kind, arg = action
        if self.proactive is None:
            await self.say("Briefings aren't available right now.")
            return
        sched = self.proactive.schedule
        if kind == "brief_now":
            self._emit("tool", {"summary": "Briefing", "decision": "auto"})
            await self.say(await self.proactive.build_briefing())
            return
        if kind == "briefing_on":
            sched.briefing_enabled = True
            if isinstance(arg, str):
                sched.briefing_time = arg
            reply = f"Okay, I'll brief you every day at {self._TIME_SPOKEN(sched.briefing_time)}."
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
```
- Dispatch in `one_turn`: compute `proactive_action` after `voice_action` (guarded by all earlier intents), add `or proactive_action is not None` to the dictation guard, and branch:
```python
            elif proactive_action is not None:
                self.player.reset()
                await self._proactive_turn(proactive_action)
```

- [ ] **Step 7: `__main__.py` adapters** — in `build_orchestrator`, before constructing the orchestrator:
```python
    from veronica.tools import pim as pim_tools

    async def _cal(day: str, days: int) -> str:
        res = await pim_tools.calendar_events.handler({"day": day, "days": days})
        return res["content"][0]["text"]

    async def _mail_count() -> int:
        res = await pim_tools.mail_unread.handler({"limit": 50})
        return proactive.count_mail(res["content"][0]["text"]) if not res.get("is_error") else 0

    async def _rem(days: int) -> str:
        res = await pim_tools.reminders_due.handler({"days": days})
        return res["content"][0]["text"]

    pro = None
    if audio:
        pro = proactive.Proactive(
            proactive.load_schedule(),
            announce=lambda t: holder["orch"].announce(t),
            calendar_events=_cal, mail_unread_count=_mail_count, reminders_due=_rem,
        )
```
Pass `proactive=pro` to `Orchestrator(...)`. (`holder["orch"]` is set right after construction, and `announce` is only called from ticks that start in `run_forever`, so the lambda is safe.) Import `from veronica import proactive`.

Test in `tests/test_main.py` (same fixture style as existing `build_orchestrator` tests): `test_build_orchestrator_wires_proactive` — with `audio=True` fakes, assert `orch.proactive is not None` and `orch.proactive.schedule == Schedule.from_prefs(<whatever prefs.load fake returns>["proactive"])`; with `audio=False`, `orch.proactive is None`.

- [ ] **Step 8: README** — add a "Briefings & nudges" section under features: the phrases (`"brief me"`, `"give me a briefing every morning at 8"`, `"stop the morning briefing"`, `"warn me 10 minutes before my meetings"`, `"turn off nudges"`), that they're spoken only when idle and not muted, and stored in `~/.veronica/prefs.json`. Also a "Voice & speed" section with the B1 phrases and the menu.

- [ ] **Step 9: Run** `uv run pytest -q` → PASS. Commit:

```bash
git add veronica/brain/intents.py veronica/orchestrator.py veronica/__main__.py README.md tests/
git commit -m "feat: briefing/nudge voice control, proactive ticker wired into the app"
```

---

### Task 6: Browser MCP server

**Files:**
- Create: `veronica/tools/browser.py`
- Test: `tests/test_browser_tools.py`

**Interfaces:**
- Produces: `browser_server` (MCP server name `"browser"`), tools `browser_tabs`, `browser_open`, `browser_read`, `browser_find`, `browser_click`, `browser_type`, `browser_scroll`, `browser_back`; module-level `run(argv, stdin=None, ok_text=None) -> dict` and `_osascript(script) -> dict` (same shape as pim's: `{"content":[{"type":"text","text":...}], "is_error"?: True}`) — tests monkeypatch `browser._osascript`; `target_browser() -> str` returning `"Google Chrome"` or `"Safari"` or raising `BrowserUnavailable`.

- [ ] **Step 1: Failing tests** — `tests/test_browser_tools.py`:

```python
import json

import pytest

from veronica.tools import browser as b


def _ok(text):
    return {"content": [{"type": "text", "text": text}]}


def _err(text):
    return {"content": [{"type": "text", "text": text}], "is_error": True}


@pytest.fixture
def scripts(monkeypatch):
    """Capture every AppleScript sent; reply from a queue of canned results."""
    sent = []
    replies = []

    async def fake_osascript(script):
        sent.append(script)
        return replies.pop(0) if replies else _ok("")

    monkeypatch.setattr(b, "_osascript", fake_osascript)
    return sent, replies


def frontmost(replies, name):
    replies.append(_ok(name))


async def test_target_browser_prefers_frontmost(scripts):
    sent, replies = scripts
    frontmost(replies, "Safari")
    assert await b.target_browser() == "Safari"
    assert "frontmost" in sent[0]


async def test_target_browser_falls_back_to_running_chrome(scripts):
    sent, replies = scripts
    frontmost(replies, "Finder")
    replies.append(_ok("true"))          # Chrome running?
    assert await b.target_browser() == "Google Chrome"
    assert 'application "Google Chrome"' in sent[1] or "Google Chrome" in sent[1]


async def test_target_browser_none_running(scripts):
    sent, replies = scripts
    frontmost(replies, "Finder")
    replies.append(_ok("false"))         # Chrome
    replies.append(_ok("false"))         # Safari
    with pytest.raises(b.BrowserUnavailable):
        await b.target_browser()


async def test_tabs_lists_and_marks_current(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok("2\nGitHub\thttps://github.com\nDocs\thttps://docs.example\n"))
    res = await b.browser_tabs.handler({})
    assert res["content"][0]["text"] == "1. GitHub — https://github.com\n2. * Docs — https://docs.example"
    assert 'tell application "Google Chrome"' in sent[1]


async def test_open_rejects_non_http(scripts):
    res = await b.browser_open.handler({"url": "file:///etc/passwd"})
    assert res.get("is_error") and "http" in res["content"][0]["text"]


async def test_open_new_tab_chrome(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok("ok"))
    res = await b.browser_open.handler({"url": "https://example.com", "new_tab": True})
    assert not res.get("is_error")
    assert "make new tab" in sent[1] and "https://example.com" in sent[1]


async def test_read_caps_and_truncates(scripts):
    sent, replies = scripts
    frontmost(replies, "Safari")
    replies.append(_ok(json.dumps({"title": "T", "url": "https://x", "text": "a " * 5000})))
    res = await b.browser_read.handler({"max_chars": 100})
    text = res["content"][0]["text"]
    assert text.startswith("T\nhttps://x\n")
    assert text.endswith("…[truncated]")
    assert len(text) < 200
    assert "do JavaScript" in sent[1] and "innerText" in sent[1]


async def test_find_returns_numbered_lines(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok(json.dumps({"lines": [[3, "Pricing plans"], [9, "See pricing"]]})))
    res = await b.browser_find.handler({"text": "pricing"})
    assert res["content"][0]["text"] == "3: Pricing plans\n9: See pricing"


async def test_find_not_found(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok(json.dumps({"lines": []})))
    res = await b.browser_find.handler({"text": "zzz"})
    assert res["content"][0]["text"] == "not found"


async def test_click_reports_element(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok(json.dumps({"clicked": "BUTTON Log in"})))
    res = await b.browser_click.handler({"target": "Log in"})
    assert res["content"][0]["text"] == "Clicked BUTTON Log in"
    assert "execute active tab" in sent[1] and "aria-label" in sent[1]


async def test_click_no_match(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_ok(json.dumps({"clicked": None})))
    res = await b.browser_click.handler({"target": "Nope"})
    assert res.get("is_error") and "no element matching" in res["content"][0]["text"]


async def test_type_sets_value_and_submits(scripts):
    sent, replies = scripts
    frontmost(replies, "Safari")
    replies.append(_ok(json.dumps({"typed": "INPUT search"})))
    res = await b.browser_type.handler({"target": "search", "text": "hello", "submit": True})
    assert res["content"][0]["text"] == "Typed into INPUT search"
    js = sent[1]
    assert "placeholder" in js and "dispatchEvent" in js and "Enter" in js


async def test_scroll_and_back(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome"); replies.append(_ok("ok"))
    assert not (await b.browser_scroll.handler({"direction": "bottom"})).get("is_error")
    assert "scrollTo" in sent[1]
    frontmost(replies, "Google Chrome"); replies.append(_ok("ok"))
    assert not (await b.browser_back.handler({})).get("is_error")
    assert "history.back" in sent[3]
    res = await b.browser_scroll.handler({"direction": "sideways"})
    assert res.get("is_error")


async def test_js_error_mapping(scripts):
    sent, replies = scripts
    frontmost(replies, "Google Chrome")
    replies.append(_err("execution error: Google Chrome got an error: Executing JavaScript through AppleScript is turned off. To turn it on, from the menu bar, go to View > Developer > Allow JavaScript from Apple Events. (12)"))
    res = await b.browser_read.handler({})
    assert res.get("is_error")
    assert "Allow JavaScript from Apple Events" in res["content"][0]["text"] and "View" in res["content"][0]["text"]

    frontmost(replies, "Safari")
    replies.append(_err("execution error: Not authorized to send Apple events to Safari. (-1743)"))
    res = await b.browser_read.handler({})
    assert "Automation" in res["content"][0]["text"]


def test_js_string_is_escaped_for_applescript():
    s = b._wrap_js("Google Chrome", 'alert("x\\y")')
    assert '\\"' in s and "\\\\" in s
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** `veronica/tools/browser.py`:

```python
"""Browser control for Chrome and Safari via AppleScript + injected
JavaScript. One-time user setup: Chrome → View ▸ Developer ▸ Allow
JavaScript from Apple Events; Safari → Develop ▸ Allow JavaScript from
Apple Events. No remote debugging / CDP (Chrome ≥136 refuses it on the
default profile, which would lose the user's logins)."""
import asyncio
import json
import logging
import subprocess

from claude_agent_sdk import create_sdk_mcp_server, tool

log = logging.getLogger("veronica.tools.browser")

CHROME = "Google Chrome"
SAFARI = "Safari"
READ_DEFAULT = 6000
READ_MAX = 20000
FIND_MAX_LINES = 10


class BrowserUnavailable(RuntimeError):
    pass


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


def run(argv: list[str], stdin: str | None = None, ok_text: str | None = None) -> dict:
    try:
        p = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return _err(f"{argv[0]} timed out")
    except OSError as e:
        return _err(str(e))
    if p.returncode != 0:
        return _err(p.stderr.strip() or f"{argv[0]} failed")
    return _ok(ok_text if ok_text is not None else p.stdout.strip())


async def _osascript(script: str) -> dict:
    return await asyncio.to_thread(run, ["osascript", "-e", script])


def _q(s: str) -> str:
    """Escape for inclusion inside an AppleScript double-quoted string."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _wrap_js(browser: str, js: str) -> str:
    if browser == CHROME:
        return f'tell application "{CHROME}" to execute active tab of front window javascript "{_q(js)}"'
    return f'tell application "{SAFARI}" to do JavaScript "{_q(js)}" in current tab of front window'


async def target_browser() -> str:
    res = await _osascript(
        'tell application "System Events" to get name of first application process whose frontmost is true'
    )
    front = res["content"][0]["text"].strip() if not res.get("is_error") else ""
    if front in (CHROME, SAFARI):
        return front
    for name in (CHROME, SAFARI):
        r = await _osascript(f'tell application "System Events" to (exists process "{name}")')
        if not r.get("is_error") and r["content"][0]["text"].strip().lower() == "true":
            return name
    raise BrowserUnavailable("No supported browser is open (Chrome or Safari).")


def _map_error(browser: str, text: str) -> dict:
    low = text.lower()
    if "allow javascript from apple events" in low or "turned off" in low:
        menu = "View > Developer" if browser == CHROME else "Develop"
        return _err(
            f"JavaScript from Apple Events is off in {browser}. Turn it on under "
            f"{menu} > Allow JavaScript from Apple Events and try again."
        )
    if "-1743" in text or "not authorized" in low:
        return _err(
            f"Veronica isn't allowed to control {browser} yet; allow it in "
            "System Settings > Privacy & Security > Automation."
        )
    return _err(text)


async def _js(js: str) -> tuple[str, dict]:
    """Run `js` in the target browser; returns (browser, result)."""
    try:
        browser = await target_browser()
    except BrowserUnavailable as e:
        return "", _err(str(e))
    res = await _osascript(_wrap_js(browser, js))
    if res.get("is_error"):
        return browser, _map_error(browser, res["content"][0]["text"])
    return browser, res


def _json(res: dict) -> dict | None:
    try:
        return json.loads(res["content"][0]["text"])
    except (ValueError, KeyError, IndexError):
        return None


def _guard(fn):
    async def wrapped(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as e:
            log.exception("browser tool failed")
            return _err(f"{type(e).__name__}: {e}")
    wrapped.__name__ = fn.__name__
    return wrapped


# -- JS snippets (each an IIFE returning a JSON string) -------------------------
_JS_MATCH_HELPERS = """
function norm(s){return (s||'').replace(/\\s+/g,' ').trim().toLowerCase();}
function labelsOf(el){
  var out=[el.innerText, el.getAttribute('aria-label'), el.value, el.title, el.alt,
           el.placeholder, el.name, el.id];
  if(el.labels){for(var i=0;i<el.labels.length;i++){out.push(el.labels[i].innerText);}}
  return out.map(norm).filter(Boolean);
}
function visible(el){var r=el.getBoundingClientRect();return r.width>0&&r.height>0;}
function findEl(sel, target){
  var t=norm(target), els=Array.from(document.querySelectorAll(sel)).filter(visible);
  for(var i=0;i<els.length;i++){if(labelsOf(els[i]).indexOf(t)>=0)return els[i];}
  for(var j=0;j<els.length;j++){if(labelsOf(els[j]).some(function(l){return l.indexOf(t)>=0;}))return els[j];}
  return null;
}
function describe(el){return el.tagName+' '+norm(el.innerText||el.value||el.getAttribute('aria-label')||el.placeholder||'').slice(0,60);}
"""

_JS_READ = """(function(){
  var t=(document.body&&document.body.innerText||'').replace(/[ \\t]+/g,' ').replace(/\\n{3,}/g,'\\n\\n');
  return JSON.stringify({title:document.title,url:location.href,text:t});
})()"""

_JS_FIND = """(function(){%s
  var q=norm(%s), lines=(document.body&&document.body.innerText||'').split('\\n'), out=[];
  for(var i=0;i<lines.length&&out.length<%d;i++){var l=lines[i].trim(); if(l&&norm(l).indexOf(q)>=0)out.push([i+1,l.slice(0,160)]);}
  return JSON.stringify({lines:out});
})()"""

_JS_CLICK = """(function(){%s
  var el=findEl('a,button,input[type=submit],input[type=button],[role=button],[role=link],[onclick],summary,label', %s);
  if(!el)return JSON.stringify({clicked:null});
  el.scrollIntoView({block:'center'}); el.click();
  return JSON.stringify({clicked:describe(el)});
})()"""

_JS_TYPE = """(function(){%s
  var el=findEl('input:not([type=hidden]):not([type=submit]):not([type=button]),textarea,[contenteditable=true],[role=textbox]', %s);
  if(!el)return JSON.stringify({typed:null});
  el.scrollIntoView({block:'center'}); el.focus();
  var v=%s;
  if(el.isContentEditable){el.textContent=v;}else{
    var setter=Object.getOwnPropertyDescriptor(Object.getPrototypeOf(el),'value');
    if(setter&&setter.set){setter.set.call(el,v);}else{el.value=v;}
  }
  el.dispatchEvent(new Event('input',{bubbles:true})); el.dispatchEvent(new Event('change',{bubbles:true}));
  if(%s){
    var opts={key:'Enter',code:'Enter',keyCode:13,which:13,bubbles:true};
    el.dispatchEvent(new KeyboardEvent('keydown',opts)); el.dispatchEvent(new KeyboardEvent('keypress',opts));
    el.dispatchEvent(new KeyboardEvent('keyup',opts));
    if(el.form&&document.activeElement===el){ if(el.form.requestSubmit){el.form.requestSubmit();} else {el.form.submit();} }
  }
  return JSON.stringify({typed:describe(el)});
})()"""

_JS_SCROLL = {
    "down": "window.scrollBy(0, Math.round(window.innerHeight*0.8)); 'ok'",
    "up": "window.scrollBy(0, -Math.round(window.innerHeight*0.8)); 'ok'",
    "top": "window.scrollTo(0, 0); 'ok'",
    "bottom": "window.scrollTo(0, document.body.scrollHeight); 'ok'",
}
_JS_BACK = "history.back(); 'ok'"


# -- tools -----------------------------------------------------------------------
@tool("browser_tabs", "List the open tabs of the front window of Chrome or Safari (current tab marked *)", {})
@_guard
async def browser_tabs(args: dict) -> dict:
    try:
        browser = await target_browser()
    except BrowserUnavailable as e:
        return _err(str(e))
    if browser == CHROME:
        script = (
            f'tell application "{CHROME}"\n'
            'set w to front window\nset out to (active tab index of w as text) & linefeed\n'
            'repeat with t in tabs of w\nset out to out & (title of t) & tab & (URL of t) & linefeed\nend repeat\n'
            'return out\nend tell'
        )
    else:
        script = (
            f'tell application "{SAFARI}"\n'
            'set w to front window\nset out to (index of current tab of w as text) & linefeed\n'
            'repeat with t in tabs of w\nset out to out & (name of t) & tab & (URL of t) & linefeed\nend repeat\n'
            'return out\nend tell'
        )
    res = await _osascript(script)
    if res.get("is_error"):
        return _map_error(browser, res["content"][0]["text"])
    lines = [ln for ln in res["content"][0]["text"].split("\n") if ln.strip()]
    if not lines:
        return _ok("No tabs.")
    try:
        current = int(lines[0].strip())
    except ValueError:
        current = -1
    out = []
    for i, ln in enumerate(lines[1:], start=1):
        title, _, url = ln.partition("\t")
        mark = "* " if i == current else ""
        out.append(f"{i}. {mark}{title.strip()} — {url.strip()}")
    return _ok("\n".join(out) if out else "No tabs.")


@tool("browser_open", "Open an http(s) URL in the current browser (new tab by default)", {"url": str, "new_tab": bool})
@_guard
async def browser_open(args: dict) -> dict:
    url = str(args.get("url", "")).strip()
    if not url.startswith(("http://", "https://")):
        return _err("only http(s) URLs are allowed")
    new_tab = bool(args.get("new_tab", True))
    try:
        browser = await target_browser()
    except BrowserUnavailable as e:
        return _err(str(e))
    u = _q(url)
    if browser == CHROME:
        script = (
            f'tell application "{CHROME}"\nactivate\n'
            + (f'tell front window to make new tab with properties {{URL:"{u}"}}\n' if new_tab
               else f'set URL of active tab of front window to "{u}"\n')
            + 'end tell'
        )
    else:
        script = (
            f'tell application "{SAFARI}"\nactivate\n'
            + (f'tell front window to set current tab to (make new tab with properties {{URL:"{u}"}})\n' if new_tab
               else f'set URL of current tab of front window to "{u}"\n')
            + 'end tell'
        )
    res = await _osascript(script)
    if res.get("is_error"):
        return _map_error(browser, res["content"][0]["text"])
    return _ok(f"Opened {url}")


@tool("browser_read", "Read the current tab: title, URL and visible text (capped)", {"max_chars": int})
@_guard
async def browser_read(args: dict) -> dict:
    try:
        max_chars = int(args.get("max_chars") or READ_DEFAULT)
    except (TypeError, ValueError):
        max_chars = READ_DEFAULT
    max_chars = max(200, min(READ_MAX, max_chars))
    _, res = await _js(_JS_READ)
    if res.get("is_error"):
        return res
    data = _json(res) or {}
    text = " ".join(str(data.get("text", "")).split())
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "…[truncated]"
    return _ok(f"{data.get('title', '')}\n{data.get('url', '')}\n{text}")


@tool("browser_find", "Find lines on the current page containing text (case-insensitive)", {"text": str})
@_guard
async def browser_find(args: dict) -> dict:
    needle = str(args.get("text", "")).strip()
    if not needle:
        return _err("text is required")
    _, res = await _js(_JS_FIND % (_JS_MATCH_HELPERS, json.dumps(needle), FIND_MAX_LINES))
    if res.get("is_error"):
        return res
    lines = (_json(res) or {}).get("lines") or []
    if not lines:
        return _ok("not found")
    return _ok("\n".join(f"{n}: {t}" for n, t in lines))


@tool("browser_click", "Click a link/button on the current page by its visible text or label", {"target": str})
@_guard
async def browser_click(args: dict) -> dict:
    target = str(args.get("target", "")).strip()
    if not target:
        return _err("target is required")
    _, res = await _js(_JS_CLICK % (_JS_MATCH_HELPERS, json.dumps(target)))
    if res.get("is_error"):
        return res
    clicked = (_json(res) or {}).get("clicked")
    if not clicked:
        return _err(f"no element matching '{target}'")
    return _ok(f"Clicked {clicked}")


@tool("browser_type", "Type text into a field on the current page (by placeholder/label/name), optionally pressing Enter", {"target": str, "text": str, "submit": bool})
@_guard
async def browser_type(args: dict) -> dict:
    target = str(args.get("target", "")).strip()
    text = str(args.get("text", ""))
    if not target:
        return _err("target is required")
    submit = "true" if bool(args.get("submit", False)) else "false"
    _, res = await _js(_JS_TYPE % (_JS_MATCH_HELPERS, json.dumps(target), json.dumps(text), submit))
    if res.get("is_error"):
        return res
    typed = (_json(res) or {}).get("typed")
    if not typed:
        return _err(f"no field matching '{target}'")
    return _ok(f"Typed into {typed}")


@tool("browser_scroll", "Scroll the current page: up, down, top or bottom", {"direction": str})
@_guard
async def browser_scroll(args: dict) -> dict:
    direction = str(args.get("direction", "down")).strip().lower()
    if direction not in _JS_SCROLL:
        return _err("direction must be up, down, top or bottom")
    _, res = await _js(_JS_SCROLL[direction])
    return res if res.get("is_error") else _ok(f"Scrolled {direction}")


@tool("browser_back", "Go back one page in the current tab", {})
@_guard
async def browser_back(args: dict) -> dict:
    _, res = await _js(_JS_BACK)
    return res if res.get("is_error") else _ok("Went back")


browser_server = create_sdk_mcp_server(
    name="browser",
    version="1.0.0",
    tools=[browser_tabs, browser_open, browser_read, browser_find, browser_click, browser_type, browser_scroll, browser_back],
)
```

Check `test_js_string_is_escaped_for_applescript`: `_wrap_js` escapes `"` → `\"` and `\` → `\\`. Check `test_tabs_lists_and_marks_current`: current index `2` → second line marked `* `. Check `test_scroll_and_back`: `sent[3]` is the fourth script (frontmost, scroll, frontmost, back). Check the read test: `"a " * 5000` collapses to `"a a a …"`, cut to 100 chars + `"…[truncated]"`.

Look at how `pim.py` and `music.py` build their `create_sdk_mcp_server` / `_guard` — match their signature exactly (the `@tool(...)` decorator order relative to `_guard` must mirror pim's).

- [ ] **Step 4: Run** `uv run pytest -q tests/test_browser_tools.py` → PASS. Commit:

```bash
git add veronica/tools/browser.py tests/test_browser_tools.py
git commit -m "feat(tools): browser MCP server for Chrome/Safari via AppleScript + JS"
```

---

### Task 7: Register browser server, policy, summaries, prompt, README

**Files:**
- Modify: `veronica/brain/policy.py`, `veronica/brain/agent.py`, `veronica/brain/prompts.py`, `README.md`
- Test: `tests/test_policy.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: Task 6 `browser_server`.

- [ ] **Step 1: Failing tests** — append to `tests/test_policy.py`:

```python
@pytest.mark.parametrize("short,expected", [
    ("browser_tabs", "allow"), ("browser_open", "allow"), ("browser_read", "allow"),
    ("browser_find", "allow"), ("browser_scroll", "allow"), ("browser_back", "allow"),
    ("browser_click", "confirm"), ("browser_type", "confirm"), ("browser_unknown", "confirm"),
])
def test_browser_tool_risk(short, expected):
    assert classify(f"mcp__browser__{short}", {}) == expected
```

Append to `tests/test_agent.py` (use its existing `summarize_detail` import):

```python
@pytest.mark.parametrize("short,inp,expected", [
    ("browser_tabs", {}, "List tabs"),
    ("browser_open", {"url": "https://x.y"}, "Open https://x.y"),
    ("browser_read", {}, "Read the page"),
    ("browser_find", {"text": "pricing"}, "Find 'pricing' on the page"),
    ("browser_click", {"target": "Log in"}, "Click 'Log in'"),
    ("browser_type", {"target": "search", "text": "hi"}, "Type into 'search'"),
    ("browser_scroll", {"direction": "down"}, "Scroll down"),
    ("browser_back", {}, "Go back"),
])
def test_summarize_browser_tools(short, inp, expected):
    assert summarize_detail(f"mcp__browser__{short}", inp) == expected


def test_options_register_browser_server(...):
    # copy the existing test that asserts mcp_servers keys (search for '"screen"' in this file)
    # and assert "browser" is in options.mcp_servers
```

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement:**

`policy.py` — add to `MCP_TOOL_RISK`:
```python
    "browser": {
        # Read-only / navigation: same risk as mac.open_url.
        "browser_tabs": "allow",
        "browser_open": "allow",
        "browser_read": "allow",
        "browser_find": "allow",
        "browser_scroll": "allow",
        "browser_back": "allow",
        # Acts inside the user's logged-in session: confirm.
        "browser_click": "confirm",
        "browser_type": "confirm",
    },
```

`agent.py` — `from veronica.tools.browser import browser_server`; add `"browser": browser_server` to `mcp_servers`; add `BROWSER_PREFIX = "mcp__browser__"` and in `summarize_detail` before the generic fallbacks:
```python
    if tool_name.startswith(BROWSER_PREFIX):
        short = tool_name[len(BROWSER_PREFIX):]
        if short == "browser_tabs":
            return "List tabs"
        if short == "browser_open":
            return f"Open {input.get('url', '')}"
        if short == "browser_read":
            return "Read the page"
        if short == "browser_find":
            return f"Find '{input.get('text', '')}' on the page"
        if short == "browser_click":
            return f"Click '{input.get('target', '')}'"
        if short == "browser_type":
            return f"Type into '{input.get('target', '')}'"
        if short == "browser_scroll":
            return f"Scroll {input.get('direction', 'down')}"
        if short == "browser_back":
            return "Go back"
        return short
```

`prompts.py` — append to `base`:
```python
        " When the user refers to this page, this tab, the current article or site, or asks you "
        "to do something inside the browser, use the browser tools; summarise browser_read output "
        "in your own words rather than reading it aloud."
```
Update any prompt test that asserts the full base string (search `tests/` for `"what they're looking at"`).

`README.md` — "Browser control" section: supported browsers (Chrome, Safari), the one-time setting (Chrome: View ▸ Developer ▸ Allow JavaScript from Apple Events; Safari: enable the Develop menu in Settings ▸ Advanced, then Develop ▸ Allow JavaScript from Apple Events), Automation permission prompt on first use, example phrases ("read this page", "summarize this article", "find pricing on this page", "click the login button", "type hello in the search box and press enter", "open a new tab with github"), and that click/type ask for confirmation.

- [ ] **Step 4: Run full suite** `uv run pytest -q` → PASS. Commit:

```bash
git add veronica/brain/policy.py veronica/brain/agent.py veronica/brain/prompts.py README.md tests/test_policy.py tests/test_agent.py
git commit -m "feat(brain): register browser tools with risk classes, summaries and prompt guidance"
```

---

## Self-review

- Spec coverage: B1 → Tasks 1–3 (table, resolver, intents, `_voice_turn`, prefs at startup, menu + orb popup). B2 → Tasks 4–5 (Schedule/prefs, parsers, briefing composer, ticker with cache, intents + time parsing, orchestrator wiring, adapters, README). B3 → Tasks 6–7 (server, target browser, error mapping, policy, summaries, prompt, README). Text-mode: spec says unchanged — no task. HUD: local turns emit `tool` cards via `_emit` — covered.
- Placeholder scan: Task 2 step 8 and Task 5 step 7 and Task 7 step 1 point at *existing* test fixtures instead of inlining them, because those files' fixture shapes must be read first; the contracts to assert are stated. Everything else is concrete.
- Type consistency: `VoiceAction`/`ProactiveAction` tuples, `_voice_turn(action)` used by Task 3, `Proactive(schedule, announce, calendar_events, mail_unread_count, reminders_due, now)` used identically in Tasks 4 and 5, `browser_server` name matches Task 7 import, `_osascript(script)` single-arg in Task 6 matches the test fake.
