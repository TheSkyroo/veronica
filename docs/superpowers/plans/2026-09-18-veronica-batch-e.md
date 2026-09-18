# Veronica Batch E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Veronica click, type, scroll and press keys on the frontmost app based on the screenshot she already takes, with OCR-by-label, a per-app trust window, and hard safety rules.

**Architecture:** Screenshot gains a geometry sidecar (image px ↔ screen points). A pure-ish `computer_events` module wraps Quartz CGEvents / AX behind injectable accessors. `tools/computer.py` exposes eight MCP tools that convert coordinates, run OCR (`tools/ocr.py`, Vision), post events, and report the frontmost window. The brain's permission gate gets a trust window for `computer` confirm-class tools; the orchestrator clears it on barge/end.

**Tech Stack:** Python 3.12, uv, pytest; PyObjC Quartz (CGEvent, CGWindowList, CGDisplayBounds), Vision (VNRecognizeTextRequest), ApplicationServices (AX*), AppKit (NSWorkspace); claude-agent-sdk in-process MCP.

**Spec:** `docs/superpowers/specs/2026-09-18-veronica-batch-e-design.md`

## Global Constraints

- Brain stays on the user's Claude Code subscription login via `claude-agent-sdk`. **No API key**, ever.
- Confirm-gate stays strict: `policy.classify()` + the spec's trust window (E4) are the only auto-allow paths; never set `allowed_tools`.
- Never add `Co-Authored-By` trailers or "Generated with Claude Code" to commits.
- Tests never post real CGEvents, call real Vision/AX/CoreGraphics, or read the real screen; framework access goes through module accessors (`_quartz()`, `_vision()`, `_ax()`, `_appkit()`) that tests monkeypatch.
- Coordinates: tool args are **image pixels of the last screenshot** unless `space="screen"`; `Geometry.to_screen` does the conversion; geometry older than 120 s is stale.
- Copy strings exactly as in the spec. Branch `batch-e` (created off master; spec committed).

## File map

| File | Responsibility |
|---|---|
| `veronica/tools/screen.py` | `latest.json` geometry sidecar, `Geometry`, `load_geometry`, result text with sizes |
| `veronica/tools/ocr.py` (new) | Vision OCR → `Word` list in image px; `find_text` |
| `veronica/tools/computer_events.py` (new) | Quartz/AX primitives: move/click/drag/scroll/type/key, `frontmost`, `accessibility_trusted`, `focused_is_secure`, `is_system_dialog`, keycode table |
| `veronica/tools/computer.py` (new) | `computer` MCP server (8 tools) |
| `veronica/brain/policy.py`, `agent.py`, `prompts.py` | risk table, summaries, server registration, trust window, prompt |
| `veronica/orchestrator.py` | `clear_trust()` on barge/end |
| `veronica/config.py`, `ui/settings/bridge.py` | `computer_trust_s` |
| `README.md` | Computer use section |

---

### Task 1: E1 — screenshot geometry sidecar + OCR module

**Files:** modify `veronica/tools/screen.py`; create `veronica/tools/ocr.py`; tests `tests/test_screen.py`, `tests/test_ocr.py`.

**Interfaces (produces):**
```python
# screen.py
@dataclass
class Geometry:
    region: str; image_w: int; image_h: int
    origin_x: float; origin_y: float; width_pt: float; height_pt: float
    scale: float; captured_at: float; window: dict | None
    def to_screen(self, x_img: float, y_img: float) -> tuple[float, float]
    @property
    def age_s(self) -> float           # time.time() - captured_at (clock injectable via module attr _now)
GEOMETRY_PATH: Path                   # SCREENS_DIR / "latest.json"
GEOMETRY_MAX_AGE_S = 120
def _display_bounds(quartz=None) -> tuple[float, float, float, float]   # CGDisplayBounds(CGMainDisplayID()) → (x, y, w, h) points
def _window_bounds(window_id: int, quartz=None) -> dict | None           # {"id","app","title","x","y","w","h"} via CGWindowListCopyWindowInfo
def write_geometry(...) / load_geometry() -> Geometry | None            # None if missing/corrupt
# ocr.py
@dataclass
class Word: text: str; x: float; y: float; w: float; h: float; confidence: float   # image px, top-left origin
def recognize_text(png_path: Path, *, vision=None, languages=("en-US", "hi-IN")) -> list[Word]
def find_text(words: list[Word], query: str) -> list[Word]
```
`capture_screenshot` writes the sidecar after the (possibly downscaled) PNG exists: `image_w/h` from `sips -g pixelWidth -g pixelHeight` (already used? if not, parse `sips -g pixelHeight -g pixelWidth <png>` output — inject via the same `run`), display bounds via `_display_bounds()`, window bounds when `region == "window"`. `_image_result` text: `f"Screenshot of the {region}: {w}×{h} px (screen {width_pt:.0f}×{height_pt:.0f} pt). Coordinates you pass to computer_* tools are in these image pixels."`. Vision access: `_vision()` accessor returning the `Vision` module; in `recognize_text` build `VNImageRequestHandler.alloc().initWithURL_options_(NSURL.fileURLWithPath_(str(png_path)), None)`, `VNRecognizeTextRequest.alloc().init()`, `setRecognitionLevel_(VNRequestTextRecognitionLevelAccurate)`, `setUsesLanguageCorrection_(True)`, try `setRecognitionLanguages_(list(languages))` (ignore errors), `performRequests_error_([req], None)`; for each `obs in req.results()`: `cand = obs.topCandidates_(1)[0]`, `bb = obs.boundingBox()` → `Word(cand.string(), bb.origin.x * W, (1 - bb.origin.y - bb.size.height) * H, bb.size.width * W, bb.size.height * H, cand.confidence())`, with `W/H` from the PNG size (reuse the sidecar or `sips`; accept `image_size: tuple[int,int] | None` kwarg, default from `load_geometry()`). `find_text`: normalize (lower, collapse spaces); tiers: exact == → contains → `difflib.SequenceMatcher(...).ratio() >= 0.8`; stable sort by `(y, x)`.

- [ ] Tests first: geometry sidecar written with correct fields for `screen` (fake `run` incl. `sips -g` output, fake quartz bounds `(0,0,1470,956)`, image 1568×1019 → scale 1.0667), `to_screen((784, 509)) ≈ (735, 477)`, `age_s`, missing/corrupt → None; `window` region uses window bounds; result text format. OCR: fake Vision objects (observation with `boundingBox()` returning an object with `.origin.x/.y` `.size.width/.height`, `topCandidates_` → candidate with `.string()`/`.confidence()`) → y-flip verified numerically; `find_text` tiers/ordering; language set failure ignored.
- [ ] Implement; `uv run pytest -q`; commit `feat(screen): screenshot geometry sidecar and Vision OCR`.

---

### Task 2: E2 — `computer_events.py`

**Files:** create `veronica/tools/computer_events.py`; tests `tests/test_computer_events.py`.

**Interfaces (produces):** as spec E2: `accessibility_trusted(prompt=False) -> bool`, `PERMISSION_HINT`, `move`, `click`, `drag`, `scroll`, `type_text`, `key`, `frontmost() -> Front`, `focused_is_secure() -> bool`, `is_system_dialog(front) -> bool`, `KEYCODES: dict[str, int]` (US layout: letters a–z, digits, `enter 36, return 36, esc 53, escape 53, tab 48, space 49, backspace 51, delete 51, forwarddelete 117, up 126, down 125, left 123, right 124, home 115, end 119, pageup 116, pagedown 121, f1..f12, minus 27, equal 24, leftbracket 33, rightbracket 30, backslash 42, semicolon 41, quote 39, comma 43, period 47, slash 44, grave 50`), `MODIFIERS = {"cmd": kCGEventFlagMaskCommand, "command":…, "shift":…, "alt":…, "option":…, "ctrl":…, "control":…}`, `SYSTEM_DIALOG_BUNDLES`, `DISALLOWED_DIALOG_TARGETS`. Module accessors `_quartz()`, `_ax()`, `_appkit()`; `_sleep = time.sleep`; `post(event)` helper → `CGEventPost(kCGHIDEventTap, event)`.

Event construction (real Quartz): `CGEventCreateMouseEvent(None, type, (x, y), button)`; types `kCGEventMouseMoved`, `kCGEventLeftMouseDown/Up`, `kCGEventRightMouseDown/Up`, `kCGEventOtherMouseDown/Up`, `kCGEventLeftMouseDragged`; buttons `kCGMouseButtonLeft/Right/Center`; double: `CGEventSetIntegerValueField(ev, kCGMouseEventClickState, 2)` on both down/up of the second click; scroll: `CGEventCreateScrollWheelEvent(None, kCGScrollEventUnitPixel, 2, int(dy), int(dx))`; keys: `CGEventCreateKeyboardEvent(None, keycode, True/False)` + `CGEventSetFlags(ev, flags)`; unicode: `ev = CGEventCreateKeyboardEvent(None, 0, True); CGEventKeyboardSetUnicodeString(ev, len(chunk_utf16), chunk)` then the matching key-up. AX: `AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": prompt})` (the constant `kAXTrustedCheckOptionPrompt` is the string "AXTrustedCheckOptionPrompt"); focused element: `AXUIElementCreateSystemWide()` → `AXUIElementCopyAttributeValue(sys, "AXFocusedUIElement", None)` → `(err, elem)`; role via `AXUIElementCopyAttributeValue(elem, "AXRole", None)`.

- [ ] Tests first with a `FakeQuartz` recording `(type, x, y, button, flags, clickstate)` per posted event and a `FakeAX`: click posts move+down+up with correct types; right/middle buttons; double posts 2 pairs with clickstate 2; drag posts down, 8 dragged moves, up with interpolated coords; scroll posts move + wheel with (dy, dx); `type_text("hello world, नमस्ते")` chunks ≤20 UTF-16 units and posts down/up per chunk; `key("cmd+shift+s")` → keycode 1 with cmd|shift flags on down and up; `key("enter")` 36; unknown → ValueError; `accessibility_trusted` passes prompt option; `focused_is_secure` True for AXSecureTextField, False on error; `frontmost` from fake NSWorkspace + window list; `is_system_dialog` table incl. System Settings + "Privacy".
- [ ] Implement; full suite; commit `feat(computer): Quartz input primitives, AX checks, frontmost lookup`.

---

### Task 3: E3 — `computer` MCP server + policy + summaries + prompt

**Files:** create `veronica/tools/computer.py`; modify `veronica/brain/policy.py`, `agent.py` (register `computer_server`, summaries), `prompts.py`; tests `tests/test_computer_tools.py`, `tests/test_policy.py`, `tests/test_agent.py`.

**Interfaces:** tools exactly per the spec table (names, args, refusals, result text `"done — frontmost: {app} — {title}"`); helper `_coords(args, geometry) -> (x_pt, y_pt)` honoring `space`; `_fresh_geometry() -> Geometry | str`; `computer_find` result lines `"{i}. '{text}' at ({cx}, {cy}) size {w}×{h} (conf {c:.2f})"` with integer image-px centers; `computer_click_text` picks `matches[index]`; `TOOLS`, `COMPUTER_TOOL_NAMES`, `computer_server = create_sdk_mcp_server(name="computer", …)`. Policy table `MCP_TOOL_RISK["computer"]`. Summaries in `summarize_detail` (prefix `mcp__computer__`) exactly as spec. Prompt sentence appended verbatim.

- [ ] Tests first: each tool with monkeypatched `computer_events` (recording fakes), `screen.load_geometry` (fresh/stale/None), `ocr.recognize_text`; permission hint when not trusted (and that `prompt=True` was requested); stale geometry error; image→screen conversion using a Geometry with scale 2.0 and origin (100, 50); `space="screen"` passthrough; `click_text` no match / index; secure-field refusal; system-dialog refusal for "Allow"; policy parametrize; summaries parametrize; `_options` registers `"computer"`; prompt contains the new sentence.
- [ ] Implement; full suite; commit `feat(tools): computer MCP server (click/type/key/scroll/drag/find) with safety refusals`.

---

### Task 4: E4/E5 — trust window in the brain, orchestrator clears it, setting, README

**Files:** modify `veronica/brain/agent.py`, `veronica/orchestrator.py`, `veronica/config.py`, `veronica/ui/settings/bridge.py`, `README.md`; tests `tests/test_agent.py`, `tests/test_orchestrator.py`, `tests/test_config.py`, `tests/test_settings_bridge.py`.

**Interfaces:** `Settings.computer_trust_s: int = 90`; `EDITABLE_SETTINGS["computer_trust_s"] = EditableField("int", "Screen-control trust window (seconds)", "After you approve one click/type, further screen actions in the same app are allowed for this long. 0 = ask every time.", min=0, max=600, restart=False)` in `SETTING_SECTIONS["brain"]`; `Brain(..., frontmost: Callable[[], Front] | None = None, clock=time.monotonic)` reading `self.s.computer_trust_s` live; `Brain.clear_trust()`; `_can_use_tool` logic per spec E4 (log lines `"trusted: %s"` and `"trust window opened for %s (%ss)"`); orchestrator: `_barge_teardown` and the `end` intent branch call `getattr(self.brain, "clear_trust", lambda: None)()`.

- [ ] Tests first: agent — a confirm-class computer tool asks the first time; after a yes with trust 90 s, a second computer tool in the same app is allowed without asking (on_tool "auto"); different app asks; after 91 s asks; trust 0 never auto-allows; system dialog (`is_system_dialog` True) always asks even inside the window; non-computer tools unaffected; `clear_trust` resets. Orchestrator — barge teardown and "that's all" end intent call `clear_trust` (fake brain with a counter). Config/bridge — field present under brain, clamp.
- [ ] README "Computer use" section per E5.
- [ ] Implement; full suite; commit `feat(brain): per-app trust window for screen actions; settings + docs`.

## Self-review
- Coverage: E1→T1, E2→T2, E3→T3, E4/E5→T4. Playwright not needed. Copy strings carried verbatim in the spec table.
- Types: `Geometry.to_screen` used by T3; `Word` used by T3 `computer_find`; `Front` used by T3 result text and T4 trust; `is_system_dialog(front)` shared T3/T4.
