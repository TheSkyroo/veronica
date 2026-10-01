# Veronica Phase 2.5 — HUD Orb Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A floating, transparent HUD panel with an animated orb that appears on wake, reacts to mic/voice levels and state, shows transcript / reply / tool activity, and fades after idle.

**Architecture:** Orchestrator/Brain/Recorder emit typed events through one `on_event(kind, payload)` hook. The menubar puts events on a thread-safe queue and a 30 Hz `rumps.Timer` drains them into `HudWindow.push()`, which runs `window.hud.push(json)` in a `WKWebView` hosted by a non-activating floating `NSPanel` (PyObjC). All visuals are HTML/Canvas/JS.

**Tech Stack:** Python 3.12, PyObjC (`pyobjc-framework-WebKit` added), rumps, numpy, HTML5 Canvas/JS (no libraries), pytest, Playwright (live test only).

**Spec:** `docs/superpowers/specs/2026-09-16-veronica-hud-design.md`

## Global Constraints

- Event kinds and payloads exactly as the spec table: `state`, `heard`, `sentence`, `tool` (`{"summary", "decision": auto|ask|allowed|declined}`), `mic` (float 0–1), `voice` (`{"step_ms": 50, "levels": [...]}`), `warm` (`{"ready": bool}`).
- `Settings`: `hud_enabled=True`, `hud_hide_after_s=3.0`, `hud_width=380`, `hud_height=220`, `hud_margin=24`.
- HUD shows on any non-idle state; hides `hud_hide_after_s` after `idle` unless a non-idle state arrives first.
- Panel: borderless, non-activating, transparent, floating level, all Spaces, ignores mouse events; never steals focus.
- `push()` always executes on the AppKit main thread. HUD failures never break the assistant (log + continue).
- Existing behaviors/strings unchanged; all Phase 1/2 tests keep passing; `uv run pytest -q` pristine.
- Commit per task; conventional messages; NO `Co-Authored-By` trailer. Test fakes via `monkeypatch`.

---

## File structure

```
veronica/ui/events.py          # rms(), envelope(), EVENT_KINDS                       (new)
veronica/ui/hud.py             # HudWindow (NSPanel + WKWebView), visibility policy     (new)
veronica/ui/hud/index.html     # markup                                                 (new)
veronica/ui/hud/hud.css        # glass card, text rows                                  (new)
veronica/ui/hud/hud.js         # window.hud API, orb renderer, typewriter               (new)
veronica/ui/menubar.py         # queue + drain timer + HUD lifecycle
veronica/orchestrator.py       # on_event emission
veronica/brain/agent.py        # on_tool hook
veronica/audio/record.py       # on_level hook
veronica/__main__.py           # build_orchestrator(on_event=...)
veronica/config.py             # hud_* settings
pyproject.toml                 # pyobjc-framework-WebKit; [dev] playwright
tests/test_events.py, tests/test_hud_window.py, tests/test_hud_web.py (live) (new); test_orchestrator/test_agent/test_record/test_menubar/test_config/test_main extended
```

---

### Task 1: Settings, event helpers, Recorder `on_level`, Brain `on_tool`

**Files:**
- Create: `veronica/ui/events.py`, `tests/test_events.py`
- Modify: `veronica/config.py`, `veronica/audio/record.py`, `veronica/brain/agent.py`, `pyproject.toml`; tests `test_config.py`, `test_record.py`, `test_agent.py`

**Interfaces:**
- Produces: `events.rms(pcm16: np.ndarray) -> float` (0–1, int16 input; empty → 0.0); `events.envelope(samples_f32: np.ndarray, sample_rate: int, step_ms: int = 50) -> list[float]` (RMS per window, normalized so max→1.0 when any non-zero; length = ceil(n/step)); `EVENT_KINDS = frozenset({...})`.
- `Recorder(settings, frames=None, on_level: Callable[[float], None] | None = None)` — called per frame during `_capture` with `rms(frame)`, from the capture thread; exceptions in the callback are swallowed+logged once.
- `Brain(settings, confirm, on_tool: Callable[[str, str], None] | None = None)` — `on_tool(summary, "auto")` on auto-allow; the confirm-path decisions are emitted by the Orchestrator (Task 2), not here.
- `Settings.hud_enabled/hud_hide_after_s/hud_width/hud_height/hud_margin`.

- [ ] **Step 1: Failing tests**

`tests/test_events.py`:
```python
import math
import numpy as np

from veronica.ui.events import EVENT_KINDS, envelope, rms


def test_rms_range():
    assert rms(np.zeros(480, dtype=np.int16)) == 0.0
    full = np.full(480, 32767, dtype=np.int16)
    assert 0.99 <= rms(full) <= 1.0
    assert rms(np.zeros(0, dtype=np.int16)) == 0.0


def test_envelope_shape_and_normalization():
    sr = 24000
    x = np.zeros(sr, dtype=np.float32)          # 1 s
    x[sr // 2 : sr // 2 + 1200] = 0.5            # 50 ms burst in the middle
    env = envelope(x, sr, 50)
    assert len(env) == math.ceil(sr / (sr * 50 // 1000))
    assert max(env) == 1.0 and min(env) == 0.0
    assert envelope(np.zeros(10, dtype=np.float32), sr) == [0.0]


def test_event_kinds():
    assert EVENT_KINDS == {"state", "heard", "sentence", "tool", "mic", "voice", "warm"}
```

Append to `tests/test_config.py::test_defaults`:
```python
    assert s.hud_enabled is True and s.hud_hide_after_s == 3.0
    assert (s.hud_width, s.hud_height, s.hud_margin) == (380, 220, 24)
```

Append to `tests/test_record.py` (reuse `make`, `FRAME`, `frames` from that file):
```python
async def test_on_level_called_per_frame(monkeypatch):
    levels = []
    r = make("..sss..", monkeypatch)
    r._on_level = levels.append          # constructor kwarg is on_level=; set after make() for simplicity
    await r.capture()
    assert len(levels) >= 7 and all(0.0 <= v <= 1.0 for v in levels)
    assert max(levels) > 0.0            # speech frames are non-zero
```
(If `make()` cannot pass kwargs through, add `on_level` to its `**over` handling — the Recorder constructor must accept `on_level=`.)

Append to `tests/test_agent.py` (reuse `brain` fixture):
```python
async def test_on_tool_auto_allow(brain):
    seen = []
    brain._on_tool = seen.append_pair if False else (lambda s, d: seen.append((s, d)))
    await brain._can_use_tool("Read", {"file_path": "/x"}, None)
    assert seen == [("Read: /x", "auto")]


async def test_on_tool_not_called_on_confirm_path(brain):
    seen = []
    brain._on_tool = lambda s, d: seen.append((s, d))
    await brain._can_use_tool("Write", {"file_path": "/a"}, None)
    assert seen == []
```
(Replace the silly first line with `brain._on_tool = lambda s, d: seen.append((s, d))`.)

- [ ] **Step 2: Run → FAIL** (`uv run pytest tests/test_events.py tests/test_config.py tests/test_record.py tests/test_agent.py -q`).

- [ ] **Step 3: Implement**

`veronica/ui/events.py`:
```python
"""Helpers for HUD events: audio levels and envelopes."""
import math

import numpy as np

EVENT_KINDS = frozenset({"state", "heard", "sentence", "tool", "mic", "voice", "warm"})


def rms(pcm16: np.ndarray) -> float:
    """RMS of int16 audio scaled to 0..1."""
    if pcm16.size == 0:
        return 0.0
    x = pcm16.astype(np.float32) / 32768.0
    return float(min(1.0, math.sqrt(float(np.mean(x * x)))))


def envelope(samples: np.ndarray, sample_rate: int, step_ms: int = 50) -> list[float]:
    """RMS per step_ms window, normalized so the loudest window is 1.0."""
    step = max(1, sample_rate * step_ms // 1000)
    n = samples.size
    if n == 0:
        return [0.0]
    out = []
    for i in range(0, n, step):
        w = samples[i : i + step].astype(np.float32)
        out.append(float(math.sqrt(float(np.mean(w * w)))) if w.size else 0.0)
    peak = max(out)
    return [v / peak for v in out] if peak > 0 else [0.0 for _ in out]
```

`veronica/config.py` — add section:
```python
    # HUD
    hud_enabled: bool = True
    hud_hide_after_s: float = 3.0
    hud_width: int = 380
    hud_height: int = 220
    hud_margin: int = 24
```

`veronica/audio/record.py`: constructor `on_level: Callable[[float], None] | None = None` → `self._on_level = on_level`; in `_capture` loop right after `is_speech = ...`:
```python
                if self._on_level is not None:
                    try:
                        self._on_level(rms(np.frombuffer(frame, dtype=np.int16)))
                    except Exception:
                        if not self._level_error_logged:
                            log.exception("on_level callback failed")
                            self._level_error_logged = True
```
(import `rms` from `veronica.ui.events`; add `self._level_error_logged = False` in `__init__`; add a module logger `log = logging.getLogger("veronica.audio")` if missing.)

`veronica/brain/agent.py`: `__init__(..., on_tool: Callable[[str, str], None] | None = None)` → `self._on_tool = on_tool`; in `_can_use_tool` auto-allow branch, after the log line: `if self._on_tool: self._on_tool(summary, "auto")`.

`pyproject.toml`: add `"pyobjc-framework-WebKit>=10"` to dependencies; add `"playwright>=1.45"` to `[project.optional-dependencies] dev`. Run `uv pip install -e ".[dev]"` then `uv run playwright install chromium` (needed by Task 3's live test).

- [ ] **Step 4: Run → PASS**; `uv run pytest -q` pristine.
- [ ] **Step 5: Commit** — `feat: HUD settings, level/envelope helpers, recorder and brain event hooks`

---

### Task 2: Orchestrator `on_event` emission

**Files:**
- Modify: `veronica/orchestrator.py`, `veronica/__main__.py`, `tests/test_orchestrator.py`, `tests/test_main.py`

**Interfaces:**
- `Orchestrator(..., on_state=None, on_event: Callable[[str, Any], None] | None = None)`; `self._emit(kind, payload)` swallows+logs callback errors.
- Emits: `state` in `_set`; `heard` after each `atranscribe` in `one_turn` (both empty and non-empty); `sentence` right before `player.play` of each pipeline sentence and inside `_say_unlocked`; `voice` (`envelope(samples, 24000)`) right before each play (pipeline + `_say_unlocked`); `tool` `ask` at `confirm()` entry (after the muted check), `allowed`/`declined` with the final result (also `declined` on the early-return paths: barged, no speech); `warm` `{"ready": False}` at `warmup()` start and `{"ready": True}` at end.
- `build_orchestrator(s, on_state=None, on_event=None, *, audio=True)` passes `on_event` to the Orchestrator, `on_level=lambda v: on_event("mic", v)` to the Recorder (when audio and on_event), and `on_tool=lambda s, d: on_event("tool", {"summary": s, "decision": d})` to the Brain.

- [ ] **Step 1: Failing tests** (append to `tests/test_orchestrator.py`; reuse `build()` but add an `events` list: modify `build()` to accept `on_event=events.append` where `events` is created inside and returned as a third value — keep the existing two-tuple return for old tests by adding a new helper `build3()` that returns `(o, states, events)`).

```python
def build3(rec_pcms=(), stt_texts=()):
    states, events = [], []
    o = Orchestrator(
        Settings(followup_window_s=0, confirm_listen_s=0),
        wake=Wake(), recorder=Rec(rec_pcms), stt=STT(stt_texts),
        brain=Brain(), tts=TTS(), player=Player(), on_state=states.append,
        on_event=lambda k, p: events.append((k, p)),
    )
    return o, states, events


async def test_events_full_turn():
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16), None], stt_texts=["what time is it"])
    await o.one_turn()
    kinds = [k for k, _ in ev]
    assert ("heard", "what time is it") in ev
    assert [p for k, p in ev if k == "sentence"] == ["Sure.", "Done."]
    voices = [p for k, p in ev if k == "voice"]
    assert len(voices) == 2 and voices[0]["step_ms"] == 50 and isinstance(voices[0]["levels"], list)
    # every sentence is preceded by its voice envelope
    assert kinds.index("voice") < kinds.index("sentence")
    assert ("state", "listening") in ev and ("state", "idle") in ev


async def test_events_confirm_ask_then_allowed_and_declined():
    o, _, ev = build3(rec_pcms=[np.zeros(1, np.int16), np.zeros(1, np.int16)], stt_texts=["yes", "no"])
    assert await o.confirm("Bash: rm x") is True
    assert await o.confirm("Bash: rm y") is False
    tools = [p for k, p in ev if k == "tool"]
    assert tools == [
        {"summary": "Bash: rm x", "decision": "ask"},
        {"summary": "Bash: rm x", "decision": "allowed"},
        {"summary": "Bash: rm y", "decision": "ask"},
        {"summary": "Bash: rm y", "decision": "declined"},
    ]
    assert ("sentence", "Run Bash: rm x?") in ev


async def test_events_confirm_no_speech_declined():
    o, _, ev = build3(rec_pcms=[None])
    assert await o.confirm("Bash: rm x") is False
    assert [p["decision"] for k, p in ev if k == "tool"] == ["ask", "declined"]


async def test_events_warm():
    o, _, ev = build3()
    await o.warmup()
    assert [p for k, p in ev if k == "warm"] == [{"ready": False}, {"ready": True}]


async def test_on_event_errors_are_swallowed():
    def boom(k, p): raise RuntimeError("x")
    o = Orchestrator(Settings(), wake=Wake(), recorder=Rec([]), stt=STT([]), brain=Brain(), tts=TTS(), player=Player(), on_event=boom)
    await o.say("hi")   # must not raise
```

`tests/test_main.py`: extend the existing `test_build_orchestrator_text_mode`-style test with fakes for `Recorder`/`Brain` that record kwargs; call `build_orchestrator(Settings(), on_event=lambda k, p: seen.append((k, p)), audio=True)` with `WakeWord`/`Transcriber`/`Synthesizer` also faked; call the recorded `on_level(0.5)` and `on_tool("Read: /x", "auto")` and assert `seen == [("mic", 0.5), ("tool", {"summary": "Read: /x", "decision": "auto"})]`.

- [ ] **Step 2: Run → FAIL**.

- [ ] **Step 3: Implement** (orchestrator)

```python
from veronica.ui.events import envelope
...
    def __init__(..., on_state=None, on_event: Callable[[str, Any], None] | None = None):
        ...
        self._on_event = on_event

    def _emit(self, kind: str, payload) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, payload)
        except Exception:
            log.exception("on_event failed for %s", kind)

    def _set(self, state):  # add after existing on_state call
        self._emit("state", state)

    async def _say_unlocked(self, text):
        samples, sr = await self.tts.asynth(text)
        self._emit("voice", {"step_ms": 50, "levels": envelope(samples, sr)})
        self._emit("sentence", text)
        await self.player.play(samples)
```
In `handle_text`'s consumer, right before `await self.player.play(samples)` (inside the not-muted branch): `self._emit("voice", {...envelope(samples, 24000)...}); self._emit("sentence", sent)`. (The pipeline futures return `(samples, sr)` — use the returned `sr`.)
In `one_turn`, after each `text = await self.stt.atranscribe(pcm)`: `self._emit("heard", text)`.
In `confirm()`: after the muted early-return, `self._emit("tool", {"summary": summary, "decision": "ask"})`; wrap the remainder so every exit emits the final decision: simplest is to compute `ok` via an inner function and `self._emit("tool", {"summary": summary, "decision": "allowed" if ok else "declined"})` once in the outer `finally` — do that with a local `result = False` updated before each `return`, and emit in the existing `finally` after `_set(prev)`.
In `warmup()`: `self._emit("warm", {"ready": False})` first, `{"ready": True}` after `self.ready = True`.

`__main__.py::build_orchestrator`:
```python
def build_orchestrator(s: Settings, on_state=None, on_event=None, *, audio: bool = True) -> Orchestrator:
    holder: dict = {}

    async def confirm(summary: str) -> bool:
        return await holder["orch"].confirm(summary)

    on_level = (lambda v: on_event("mic", v)) if (on_event and audio) else None
    on_tool = (lambda su, d: on_event("tool", {"summary": su, "decision": d})) if on_event else None
    orch = Orchestrator(
        s,
        wake=WakeWord(s) if audio else None,
        recorder=Recorder(s, on_level=on_level) if audio else None,
        stt=Transcriber(s.whisper_model) if audio else None,
        brain=Brain(s, confirm=confirm, on_tool=on_tool),
        tts=Synthesizer(s.kokoro_voice, s.models_dir),
        player=Player(),
        on_state=on_state,
        on_event=on_event,
    )
    holder["orch"] = orch
    return orch
```

- [ ] **Step 4: Run → PASS**; full suite pristine (`-W error` on orchestrator tests too).
- [ ] **Step 5: Commit** — `feat: orchestrator emits HUD events`

---

### Task 3: HUD web assets + Playwright live test

**Files:**
- Create: `veronica/ui/hud/index.html`, `veronica/ui/hud/hud.css`, `veronica/ui/hud/hud.js`, `tests/test_hud_web.py`

**Interfaces:**
- `window.hud.push({kind, payload})`, `window.hud.state()` → `{state, heard, reply, tool, mic, ready}`.
- Package data: ensure `pyproject.toml` hatch config includes the `veronica/ui/hud/*` files (hatch includes package files by default; verify with `uv pip show -f veronica` after install or by importing `importlib.resources.files("veronica.ui.hud")`). Add `veronica/ui/hud/__init__.py` (empty) so `importlib.resources` can locate the directory.

- [ ] **Step 1: Write the assets**

`index.html`:
```html
<!doctype html>
<html><head><meta charset="utf-8"><title>Veronica HUD</title>
<link rel="stylesheet" href="hud.css"></head>
<body>
<div id="card">
  <canvas id="orb" width="150" height="150"></canvas>
  <div id="text">
    <div class="row" id="heard"><span class="who">You</span><span class="msg"></span></div>
    <div class="row" id="reply"><span class="who">Veronica</span><span class="msg"></span></div>
    <div class="row" id="tool"><span class="badge"></span><span class="msg"></span></div>
  </div>
</div>
<script src="hud.js"></script>
</body></html>
```

`hud.css`:
```css
html,body{margin:0;background:transparent;font-family:-apple-system,system-ui,"SF Pro Text",sans-serif;color:#e8ecf1;overflow:hidden}
#card{box-sizing:border-box;width:380px;height:220px;padding:20px;display:flex;gap:18px;align-items:center;
  background:rgba(8,10,16,.72);border:1px solid rgba(255,255,255,.10);border-radius:18px;backdrop-filter:blur(18px)}
#orb{width:150px;height:150px;flex:0 0 150px}
#text{flex:1;min-width:0;display:flex;flex-direction:column;gap:10px}
.row{display:flex;gap:8px;align-items:baseline;min-height:18px;font-size:13px;line-height:1.35}
.who{flex:0 0 auto;font-size:10px;letter-spacing:.08em;text-transform:uppercase;opacity:.55}
.msg{flex:1;min-width:0;white-space:pre-wrap;word-break:break-word}
#heard .msg{opacity:.7}
#reply .msg{font-weight:500}
#tool{font-family:"SF Mono",ui-monospace,Menlo,monospace;font-size:11.5px;opacity:.9}
.badge{flex:0 0 auto;display:inline-block;min-width:16px;text-align:center;border-radius:6px;padding:1px 5px;font-size:11px;background:rgba(255,255,255,.08)}
.badge.auto{color:#7ad0ff}.badge.ask{color:#ffb454}.badge.allowed{color:#7dffb0}.badge.declined{color:#ff7a7a}
.row.hidden{opacity:0}
```

`hud.js` (complete):
```javascript
(() => {
  const PALETTE = {
    idle:       {core:'#3a6df0', ring:'#4c7cff', glow:'rgba(76,124,255,.35)', speed:0.25},
    warming:    {core:'#6b7280', ring:'#9ca3af', glow:'rgba(156,163,175,.25)', speed:0.6},
    listening:  {core:'#22d3ee', ring:'#67e8f9', glow:'rgba(34,211,238,.45)', speed:0.6},
    thinking:   {core:'#f59e0b', ring:'#fbbf24', glow:'rgba(245,158,11,.40)', speed:1.6},
    speaking:   {core:'#8b5cf6', ring:'#a78bfa', glow:'rgba(139,92,246,.45)', speed:0.9},
    followup:   {core:'#22d3ee', ring:'#67e8f9', glow:'rgba(34,211,238,.30)', speed:0.4},
    confirming: {core:'#f97316', ring:'#fdba74', glow:'rgba(249,115,22,.45)', speed:0.8},
    error:      {core:'#ef4444', ring:'#f87171', glow:'rgba(239,68,68,.40)', speed:0.0},
  };
  const model = {state:'idle', heard:'', reply:'', tool:null, mic:0, ready:true,
                 voice:null, voiceStart:0, confirmStart:0};
  let micSmooth = 0, replyQueue = [], typing = false;

  const $ = id => document.getElementById(id);
  const heardEl = $('heard').querySelector('.msg');
  const replyEl = $('reply').querySelector('.msg');
  const toolEl = $('tool').querySelector('.msg');
  const badgeEl = $('tool').querySelector('.badge');

  function typeNext() {
    if (typing || replyQueue.length === 0) return;
    typing = true;
    const s = replyQueue.shift();
    const start = replyEl.textContent.length ? replyEl.textContent + ' ' : '';
    let i = 0;
    const step = () => {
      i = Math.min(s.length, i + 2);           // ~40 chars/s at 20 fps ticks
      replyEl.textContent = start + s.slice(0, i);
      if (i < s.length) setTimeout(step, 50); else { typing = false; typeNext(); }
    };
    step();
  }

  const hud = {
    push(ev) {
      const {kind, payload} = ev || {};
      switch (kind) {
        case 'state':
          if (payload === 'listening' && model.state !== 'followup') { model.heard=''; model.reply=''; replyQueue=[]; replyEl.textContent=''; heardEl.textContent=''; model.tool=null; badgeEl.className='badge'; badgeEl.textContent=''; toolEl.textContent=''; }
          if (payload === 'confirming') model.confirmStart = performance.now();
          model.state = payload; break;
        case 'heard': model.heard = payload || ''; heardEl.textContent = model.heard; break;
        case 'sentence': model.reply += (model.reply ? ' ' : '') + payload; replyQueue.push(payload); typeNext(); break;
        case 'tool': model.tool = payload; badgeEl.className = 'badge ' + payload.decision;
          badgeEl.textContent = {auto:'⚡', ask:'?', allowed:'✓', declined:'✕'}[payload.decision] || '';
          toolEl.textContent = (payload.summary || '').slice(0, 60); break;
        case 'mic': model.mic = Math.max(0, Math.min(1, +payload || 0)); break;
        case 'voice': model.voice = payload; model.voiceStart = performance.now(); break;
        case 'warm': model.ready = !!(payload && payload.ready); break;
      }
    },
    state() { return {state:model.state, heard:model.heard, reply:model.reply, tool:model.tool, mic:model.mic, ready:model.ready}; },
  };
  window.hud = hud;

  // ---- orb renderer --------------------------------------------------------
  const canvas = $('orb'), ctx = canvas.getContext('2d');
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  canvas.width = 150 * dpr; canvas.height = 150 * dpr; ctx.scale(dpr, dpr);
  const particles = Array.from({length: 28}, (_, i) => ({a: (i / 28) * Math.PI * 2, r: 52 + (i % 5) * 4, s: 0.2 + (i % 7) * 0.05}));
  let t0 = performance.now(), rot = 0;

  function voiceLevel(now) {
    const v = model.voice; if (!v || !v.levels || !v.levels.length) return 0;
    const idx = Math.floor((now - model.voiceStart) / (v.step_ms || 50));
    return idx < v.levels.length ? v.levels[idx] : 0;
  }

  function frame(now) {
    const dt = (now - t0) / 1000; t0 = now;
    const p = PALETTE[model.state] || PALETTE.idle;
    rot += dt * p.speed;
    micSmooth += (model.mic - micSmooth) * 0.25;
    const cx = 75, cy = 75;
    ctx.clearRect(0, 0, 150, 150);

    let pulse = 0;
    if (model.state === 'listening' || model.state === 'followup') pulse = micSmooth;
    else if (model.state === 'speaking') pulse = voiceLevel(now);
    else if (model.state === 'thinking') pulse = 0.5 + 0.5 * Math.sin(now / 250);

    // glow
    const g = ctx.createRadialGradient(cx, cy, 10, cx, cy, 70);
    g.addColorStop(0, p.glow); g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, 150, 150);

    // core
    const coreR = 26 + pulse * 6;
    const cg = ctx.createRadialGradient(cx - 8, cy - 8, 4, cx, cy, coreR);
    cg.addColorStop(0, '#ffffff'); cg.addColorStop(0.25, p.core); cg.addColorStop(1, 'rgba(0,0,0,0.85)');
    ctx.beginPath(); ctx.arc(cx, cy, coreR, 0, Math.PI * 2); ctx.fillStyle = cg; ctx.fill();

    // rings
    ctx.lineWidth = 2 + pulse * 3; ctx.strokeStyle = p.ring;
    ctx.beginPath(); ctx.ellipse(cx, cy, 44 + pulse * 8, 30, rot, 0, Math.PI * 2); ctx.stroke();
    ctx.globalAlpha = 0.6;
    ctx.beginPath(); ctx.ellipse(cx, cy, 30, 44 + pulse * 8, -rot * 1.3, 0, Math.PI * 2); ctx.stroke();
    ctx.globalAlpha = 1;

    // particles / thinking dots
    ctx.fillStyle = p.ring;
    for (const q of particles) {
      const a = q.a + rot * q.s * (model.state === 'thinking' ? 4 : 1);
      const r = q.r + pulse * 6;
      ctx.globalAlpha = model.state === 'thinking' ? 0.9 : 0.35;
      ctx.beginPath(); ctx.arc(cx + Math.cos(a) * r, cy + Math.sin(a) * r * 0.75, 1.4, 0, Math.PI * 2); ctx.fill();
    }
    ctx.globalAlpha = 1;

    // confirming: "?" + countdown arc (5 s)
    if (model.state === 'confirming') {
      const frac = Math.max(0, 1 - (now - model.confirmStart) / 5000);
      ctx.beginPath(); ctx.arc(cx, cy, 58, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
      ctx.lineWidth = 3; ctx.strokeStyle = p.ring; ctx.stroke();
      ctx.fillStyle = '#fff'; ctx.font = 'bold 26px system-ui'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText('?', cx, cy + 1);
    }
    // warming spinner
    if (model.state === 'warming') {
      ctx.beginPath(); ctx.arc(cx, cy, 58, rot * 2, rot * 2 + Math.PI * 0.6);
      ctx.lineWidth = 3; ctx.strokeStyle = p.ring; ctx.stroke();
    }
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
})();
```

- [ ] **Step 2: Playwright live test** — `tests/test_hud_web.py`:
```python
import pathlib

import pytest

HUD = pathlib.Path(__file__).resolve().parents[1] / "veronica" / "ui" / "hud" / "index.html"


@pytest.mark.live
def test_hud_dom_and_canvas_react():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 400, "height": 240})
        page.goto(HUD.as_uri())
        page.wait_for_function("window.hud !== undefined")
        idle_px = page.evaluate("document.getElementById('orb').toDataURL()")
        page.evaluate("window.hud.push({kind:'state', payload:'listening'})")
        page.evaluate("window.hud.push({kind:'heard', payload:'what time is it'})")
        page.evaluate("window.hud.push({kind:'state', payload:'speaking'})")
        page.evaluate("window.hud.push({kind:'voice', payload:{step_ms:50, levels:[1,1,1,1,1,1,1,1,1,1]}})")
        page.evaluate("window.hud.push({kind:'sentence', payload:'It is noon.'})")
        page.evaluate("window.hud.push({kind:'tool', payload:{summary:'Open Safari', decision:'auto'}})")
        page.wait_for_function("document.querySelector('#reply .msg').textContent === 'It is noon.'", timeout=3000)
        assert page.inner_text("#heard .msg") == "what time is it"
        assert page.inner_text("#tool .msg") == "Open Safari"
        assert "auto" in page.get_attribute("#tool .badge", "class")
        page.wait_for_timeout(120)
        speaking_px = page.evaluate("document.getElementById('orb').toDataURL()")
        assert speaking_px != idle_px
        st = page.evaluate("window.hud.state()")
        assert st["state"] == "speaking" and st["reply"] == "It is noon."
        browser.close()
```
Run: `uv run pytest tests/test_hud_web.py -m live -q` → PASS (requires `uv run playwright install chromium` from Task 1).

- [ ] **Step 3: Package data check** — `uv run python -c "from importlib.resources import files; print((files('veronica.ui.hud')/'index.html').read_text()[:30])"` prints `<!doctype html>`.
- [ ] **Step 4: Commit** — `feat: HUD web assets (orb, transcript, tool badge)`

---

### Task 4: `HudWindow` (PyObjC) with injectable web view

**Files:**
- Create: `veronica/ui/hud.py`, `tests/test_hud_window.py`

**Interfaces:**
- `HudWindow(settings, *, webview_factory=None, panel_factory=None, clock=time.monotonic, schedule=None)`. Real factories build `NSPanel`/`WKWebView`; tests inject fakes. `push(event: dict)`, `show()`, `hide()`, `on_state(state: str)` (visibility policy), `tick()` (called by the menubar timer; hides when the idle deadline passed), `close()`.
- `push` serializes with `json.dumps(event, ensure_ascii=False)` and calls `webview.evaluateJavaScript_completionHandler_(f"window.hud.push({js})", None)` — through `_main(fn)` which uses `AppHelper.callAfter` when not on the main thread (`NSThread.isMainThread()`), else calls directly. Tests pass `main=lambda fn: fn()`.

- [ ] **Step 1: Failing tests**

`tests/test_hud_window.py`:
```python
import json

from veronica.config import Settings
from veronica.ui.hud import HudWindow


class FakeWeb:
    def __init__(self): self.js = []
    def evaluateJavaScript_completionHandler_(self, js, cb): self.js.append(js)


class FakePanel:
    def __init__(self): self.alpha = 0.0; self.visible = False; self.orders = []
    def setAlphaValue_(self, a): self.alpha = a
    def orderFrontRegardless(self): self.visible = True; self.orders.append("front")
    def orderOut_(self, _): self.visible = False; self.orders.append("out")


def make(t=[0.0]):
    web, panel = FakeWeb(), FakePanel()
    h = HudWindow(Settings(hud_hide_after_s=3.0), webview_factory=lambda s: web, panel_factory=lambda s, w: panel,
                  clock=lambda: t[0], main=lambda fn: fn())
    return h, web, panel, t


def test_push_serializes_json():
    h, web, _, _ = make()
    h.push({"kind": "heard", "payload": "héllo \"q\""})
    assert web.js == ['window.hud.push(' + json.dumps({"kind": "heard", "payload": "héllo \"q\""}, ensure_ascii=False) + ')']


def test_show_on_non_idle_and_hide_after_delay():
    h, _, panel, t = make()
    h.on_state("listening")
    assert panel.visible and panel.alpha == 1.0
    h.on_state("idle"); h.tick()
    assert panel.visible                     # not yet
    t[0] = 3.1; h.tick()
    assert not panel.visible and panel.alpha == 0.0


def test_non_idle_cancels_pending_hide():
    h, _, panel, t = make()
    h.on_state("listening"); h.on_state("idle"); t[0] = 2.0
    h.on_state("thinking"); t[0] = 5.0; h.tick()
    assert panel.visible


def test_push_after_close_is_noop():
    h, web, _, _ = make()
    h.close(); h.push({"kind": "mic", "payload": 0.1})
    assert web.js == []


def test_factory_failure_is_soft(caplog):
    def bad(s): raise RuntimeError("no webkit")
    h = HudWindow(Settings(), webview_factory=bad, panel_factory=lambda s, w: FakePanel(), main=lambda fn: fn())
    h.push({"kind": "state", "payload": "idle"}); h.on_state("listening")   # no raise
    assert h.available is False
```

- [ ] **Step 2: Run → FAIL**.

- [ ] **Step 3: Implement** `veronica/ui/hud.py`:
```python
"""Floating HUD panel hosting the orb web view (PyObjC)."""
import json
import logging
import time
from collections.abc import Callable
from importlib.resources import files

from veronica.config import Settings

log = logging.getLogger("veronica.ui.hud")

NON_IDLE = frozenset({"listening", "thinking", "speaking", "followup", "confirming", "error", "warming"})


def _real_webview(s: Settings):
    import AppKit
    import Foundation
    import WebKit

    cfg = WebKit.WKWebViewConfiguration.alloc().init()
    web = WebKit.WKWebView.alloc().initWithFrame_configuration_(
        Foundation.NSMakeRect(0, 0, s.hud_width, s.hud_height), cfg)
    web.setValue_forKey_(False, "drawsBackground")
    html = files("veronica.ui.hud") / "index.html"
    url = Foundation.NSURL.fileURLWithPath_(str(html))
    web.loadFileURL_allowingReadAccessToURL_(url, url.URLByDeletingLastPathComponent())
    return web


def _real_panel(s: Settings, web):
    import AppKit
    import Foundation

    screen = AppKit.NSScreen.mainScreen().visibleFrame()
    x = screen.origin.x + screen.size.width - s.hud_width - s.hud_margin
    y = screen.origin.y + screen.size.height - s.hud_height - s.hud_margin
    style = AppKit.NSWindowStyleMaskBorderless | AppKit.NSWindowStyleMaskNonactivatingPanel
    panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
        Foundation.NSMakeRect(x, y, s.hud_width, s.hud_height), style, AppKit.NSBackingStoreBuffered, False)
    panel.setOpaque_(False)
    panel.setBackgroundColor_(AppKit.NSColor.clearColor())
    panel.setLevel_(AppKit.NSFloatingWindowLevel)
    panel.setCollectionBehavior_(AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces | AppKit.NSWindowCollectionBehaviorStationary)
    panel.setIgnoresMouseEvents_(True)
    panel.setHasShadow_(False)
    panel.setAlphaValue_(0.0)
    panel.setContentView_(web)
    return panel


def _main_thread(fn: Callable[[], None]) -> None:
    import Foundation
    from PyObjCTools import AppHelper

    if Foundation.NSThread.isMainThread():
        fn()
    else:
        AppHelper.callAfter(fn)


class HudWindow:
    def __init__(self, settings: Settings, *, webview_factory=None, panel_factory=None,
                 clock: Callable[[], float] = time.monotonic, main: Callable[[Callable[[], None]], None] = _main_thread) -> None:
        self.s = settings
        self._clock = clock
        self._main = main
        self._hide_at: float | None = None
        self._closed = False
        self.available = False
        try:
            self._web = (webview_factory or _real_webview)(settings)
            self._panel = (panel_factory or _real_panel)(settings, self._web)
            self.available = True
        except Exception:
            log.warning("HUD unavailable; continuing without it", exc_info=True)
            self._web = self._panel = None

    # -- events ---------------------------------------------------------------
    def push(self, event: dict) -> None:
        if not self.available or self._closed:
            return
        js = "window.hud.push(" + json.dumps(event, ensure_ascii=False) + ")"
        self._main(lambda: self._web.evaluateJavaScript_completionHandler_(js, None))

    def on_state(self, state: str) -> None:
        if not self.available or self._closed:
            return
        if state in NON_IDLE:
            self._hide_at = None
            self.show()
        elif state == "idle":
            self._hide_at = self._clock() + self.s.hud_hide_after_s

    def tick(self) -> None:
        if self._hide_at is not None and self._clock() >= self._hide_at:
            self._hide_at = None
            self.hide()

    # -- visibility -----------------------------------------------------------
    def show(self) -> None:
        def _do():
            self._panel.setAlphaValue_(1.0)
            self._panel.orderFrontRegardless()
        self._main(_do)

    def hide(self) -> None:
        def _do():
            self._panel.setAlphaValue_(0.0)
            self._panel.orderOut_(None)
        self._main(_do)

    def close(self) -> None:
        self._closed = True
        if self.available:
            self.hide()
```
(Fade animation: keep instant alpha in v1 — the `NSAnimationContext` fade is a Task 5 polish step if time permits; the spec's 150 ms fade is satisfied there.)

- [ ] **Step 4: Run → PASS**; full suite pristine. Live check (needs a display): `uv run python -c "
import AppKit; from PyObjCTools import AppHelper
from veronica.config import settings; from veronica.ui.hud import HudWindow
h = HudWindow(settings); h.on_state('speaking'); h.push({'kind':'state','payload':'speaking'}); h.push({'kind':'sentence','payload':'Hello from the HUD.'})
AppHelper.callLater(4, AppHelper.stopEventLoop); AppHelper.runEventLoop()"` → a glass card with a violet orb appears top-right for 4 s.
- [ ] **Step 5: Commit** — `feat: native HUD panel with injectable web view`

---

### Task 5: Menubar wiring, fade, README, manual check

**Files:**
- Modify: `veronica/ui/menubar.py`, `veronica/ui/hud.py` (fade), `tests/test_menubar.py`, `README.md`

**Interfaces:**
- `VeronicaApp.__init__`: if `settings.hud_enabled`, `self._hud = HudWindow(settings)` (main thread); `self._events = queue.Queue()`; `self._hud_timer = rumps.Timer(self._drain, 1/30)`; `build_orchestrator(settings, on_state=self._on_state, on_event=self._events.put_nowait_pair)` — implement `on_event` as `lambda k, p: self._events.put((k, p))`.
- `_drain(_)`: pop up to 64 events; coalesce `mic` (keep last); for each `("state", s)` call `self._hud.on_state(s)`; `self._hud.push({"kind": k, "payload": p})`; then `self._hud.tick()`. Overflow guard: if `qsize() > 1000`, drop `mic` events first.
- `quit`: `self._hud.close()` before stopping the loop.
- Fade: in `hud.py` `show/hide` use `NSAnimationContext` with duration 0.15 when the panel has `animator` (real), else instant (fakes).

- [ ] **Step 1: Failing tests** (append to `tests/test_menubar.py`, using the existing fake-rumps fixture; add a `FakeHud` with `pushed`, `states`, `ticks`, `closed` and monkeypatch `veronica.ui.menubar.HudWindow` to return it):
```python
def test_events_drained_to_hud(fake_env):
    app = VeronicaApp()
    app._events.put(("mic", 0.1)); app._events.put(("mic", 0.9)); app._events.put(("state", "listening")); app._events.put(("heard", "hi"))
    app._drain(None)
    kinds = [e["kind"] for e in app._hud.pushed]
    assert kinds.count("mic") == 1 and app._hud.pushed[[i for i, e in enumerate(app._hud.pushed) if e["kind"] == "mic"][0]]["payload"] == 0.9
    assert app._hud.states == ["listening"] and app._hud.ticks == 1
    _quit_and_join(app)


def test_quit_closes_hud(fake_env):
    app = VeronicaApp(); app.quit(None); assert app._hud.closed
```

- [ ] **Step 2: Run → FAIL**.
- [ ] **Step 3: Implement** per the interface block. Fade in `hud.py`:
```python
    def _fade(self, alpha: float, then=None):
        def _do():
            try:
                import AppKit
                AppKit.NSAnimationContext.beginGrouping()
                AppKit.NSAnimationContext.currentContext().setDuration_(0.15)
                self._panel.animator().setAlphaValue_(alpha)
                AppKit.NSAnimationContext.endGrouping()
            except Exception:
                self._panel.setAlphaValue_(alpha)
            if then: then()
        self._main(_do)
```
`show()` → `orderFrontRegardless()` then `_fade(1.0)`; `hide()` → `_fade(0.0, then=lambda: self._panel.orderOut_(None))`. (Fakes have no `animator` → the except path sets alpha instantly, so Task 4 tests still pass.)
- [ ] **Step 4: Run → PASS**; full suite pristine; menubar tests with `--log-cli-level=ERROR` clean.
- [ ] **Step 5: README** — add: `A floating HUD appears at the top-right when Veronica wakes (orb + transcript + tool activity) and fades after 3 s of idle. Disable with VERONICA_HUD_ENABLED=false.`
- [ ] **Step 6: Manual check (human):** `uv run python -m veronica` → say wake word → orb appears cyan, your words show, reply types in violet with the ring pulsing, tool badge shows on "open Safari"; fades 3 s after idle; front app keeps keyboard focus throughout.
- [ ] **Step 7: Commit** — `feat: wire HUD into the menu bar app with fade and event drain`

---

## Self-review
- Spec §1 window → T4 (+fade T5); §2 events → T1 (hooks/helpers) + T2 (emission) + T5 (queue/drain/coalesce/overflow); §3 visuals → T3; §4 wiring/settings → T1 (settings), T2 (`build_orchestrator`), T5 (menubar); §5 errors → T4 (`available`, closed no-op), T2 (`_emit` swallow), T5 overflow; §6 tests → each task; Playwright live → T3; manual → T5.
- Placeholders: none (T1 test snippet has one line to replace, stated inline).
- Types: `on_event(kind, payload)` consistent T2/T5; `HudWindow.push/on_state/tick/close` T4↔T5; `envelope(samples, sr)` T1↔T2; `Recorder(on_level=)`/`Brain(on_tool=)` T1↔T2.
