# Veronica Phase 2.5 — HUD Orb

Date: 2026-09-16
Status: approved
Builds on: Phase 1 + Phase 2 (master 7719df8)

## Goal

A JARVIS-style floating HUD: an animated orb that appears when Veronica wakes, reacts to your voice and hers, shows what she heard, what she is saying, and what tools she runs, then fades away.

## Decision: native panel, not pywebview

`rumps` owns the AppKit main loop; pywebview also needs it. The HUD is therefore a native `NSPanel` hosting a `WKWebView` created through PyObjC (`pyobjc-framework-WebKit`; rumps already pulls `pyobjc-framework-Cocoa`). All visuals live in HTML/Canvas/JS inside the web view. Python → JS only (`evaluateJavaScript:`); no JS → Python in v1.

## 1. Window — `veronica/ui/hud.py`

- `HudWindow(settings, webview_factory=None)` creates on the main thread: borderless, transparent, non-activating floating `NSPanel` (`NSWindowStyleMaskBorderless | NSWindowStyleMaskNonactivatingPanel`, `setOpaque_(False)`, clear background, `NSFloatingWindowLevel`, `NSWindowCollectionBehaviorCanJoinAllSpaces | Stationary`, `setIgnoresMouseEvents_(True)`), size `settings.hud_width × settings.hud_height` (380 × 220), positioned at the top-right of the main screen with a 24 px margin.
- Hosts a `WKWebView` filling the panel, transparent (`setValue_forKey_(False, "drawsBackground")`), loading `index.html` from the package directory `veronica/ui/hud/` via `loadFileURL:allowingReadAccessToURL:`.
- `push(event: dict)` → `evaluateJavaScript_completionHandler_(f"window.hud.push({json})", None)`; always dispatched to the main thread (`AppHelper.callAfter` / `performSelectorOnMainThread`).
- `show()` fades `alphaValue` 0→1 over 150 ms; `hide()` fades to 0 then `orderOut_`.
- Visibility policy (driven by state events): show on `listening`, `thinking`, `speaking`, `followup`, `confirming`, `error`, `warming`; on `idle` start a `settings.hud_hide_after_s` (3 s) timer, hide when it fires unless a non-idle state arrived meanwhile.
- `webview_factory` injection lets tests substitute a fake that records JS strings; the real factory is used only when `settings.hud_enabled` and not in `--text` mode.

## 2. Event bus — `veronica/ui/events.py`

`HudEvent = dict` with `kind` and `payload`. Orchestrator gains `on_event: Callable[[str, Any], None] | None` next to `on_state` and emits:

| kind | payload | when |
|---|---|---|
| `state` | state name | every `_set` (mirrors `on_state`) |
| `heard` | transcript string | after each STT result (empty string allowed) |
| `sentence` | sentence text | when the consumer starts playing it (also for canned messages spoken via `say`) |
| `tool` | `{"summary": str, "decision": "auto"|"ask"|"allowed"|"declined"}` | `auto` from the Brain's auto-allow; `ask` when `confirm()` starts; `allowed`/`declined` with the result |
| `mic` | float 0–1 RMS | every recorder frame while capturing (≈33 Hz) |
| `voice` | `{"step_ms": 50, "levels": [float…]}` | before each sentence plays: RMS per 50 ms window of the TTS samples |
| `warm` | `{"ready": bool}` | warm-up start/end |

Emission is synchronous from the asyncio thread; the menubar hands events to a thread-safe `queue.Queue`; a `rumps.Timer` at 30 Hz drains up to 64 events per tick into `HudWindow.push`. `mic` events are coalesced (only the latest per tick).

Brain gets an `on_tool: Callable[[str, str], None]` hook (`summary, decision`) that `build_orchestrator` wires to the same emitter. Recorder gets `on_level: Callable[[float], None] | None` called per frame from the capture thread (must be cheap and thread-safe — the emitter only `queue.put`s).

## 3. Visual — `veronica/ui/hud/index.html`, `hud.js`, `hud.css`

- Dark glass card (rounded 18 px, 70 % black, 1 px white/10 border), no chrome. Left: 150 × 150 canvas orb. Right: three text rows.
- Orb: radial-gradient core, two counter-rotating rings, faint particle halo, additive glow. State → palette/motion: `idle` dim blue slow rotation · `listening` cyan, outer ring radius follows `mic` (smoothed) · `thinking` amber, orbiting dots · `speaking` violet, ring thickness pulses with the `voice` envelope advanced by wall clock from the sentence start · `confirming` orange, "?" glyph in the core, 5 s countdown arc · `error` red, still · `warming` grey with a spinner arc.
- Text rows: "You:" transcript (muted, fades in), "Veronica:" reply typed at ~40 chars/s per sentence (queue of sentences, never truncates mid-word), tool row with badge ⚡ auto / ? asking / ✓ allowed / ✕ declined + summary (≤ 60 chars, ellipsis).
- API: `window.hud.push({kind, payload})`; `window.hud.state()` returns the current model (for tests). 60 fps `requestAnimationFrame`; canvas sized for devicePixelRatio.
- Fonts: system-ui / SF Mono fallback stack; no network requests (file:// only).

## 4. Wiring

- `Settings`: `hud_enabled: bool = True`, `hud_hide_after_s: float = 3.0`, `hud_width: int = 380`, `hud_height: int = 220`, `hud_margin: int = 24`.
- `build_orchestrator(settings, on_state=None, on_event=None, *, audio=True)`; `Brain(settings, confirm, on_tool=None)`; `Recorder(settings, frames=None, on_level=None)`.
- `menubar.py`: after `rumps.App.__init__`, if `settings.hud_enabled` create `HudWindow` (main thread), a `queue.Queue`, and a 30 Hz `rumps.Timer` drain; pass `on_event=queue.put` into `build_orchestrator`; state events also drive show/hide. Quit hides the panel.
- `--text` mode: no HUD, no events consumer (events dropped).

## 5. Error handling

- HUD construction failure (no WebKit, headless) → log warning, run without HUD (menu bar only).
- `push` on a closed/absent web view → no-op. JS exceptions never reach Python (completion handler ignored).
- Event queue overflow (> 1000) → drop oldest `mic` events first.

## 6. Testing

- Unit (fakes): orchestrator emits `state/heard/sentence/tool/mic/voice/warm` in the right order for a full turn, a confirm (ask → allowed/declined), a barge; envelope math (`voice` levels count = ceil(duration/50 ms), values 0–1); Brain `on_tool` on auto-allow and confirm paths; Recorder `on_level` per frame; `HudWindow` with a fake web view: `push` serializes JSON, show/hide timing with a fake clock, visibility policy; drain timer coalesces `mic`.
- Live-marked Playwright test (`tests/test_hud_web.py`): opens `index.html` in headless Chromium, pushes a scripted sequence, asserts the transcript/reply/tool DOM text and that canvas pixel data changes between `idle` and `speaking`.
- Manual: run the app, wake, ask, confirm a tool, barge — the orb must track each state; the panel must never steal focus from the front app.
