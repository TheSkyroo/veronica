# Veronica Batch E — Computer use (act on what she sees)

Date: 2026-09-18. Branch `batch-e`, merged to `master` as one unit. Approved by the user with the trust-window model.

## Global constraints (apply to every task)

- Brain stays on the user's Claude Code subscription login via `claude-agent-sdk`. **No API key**, ever.
- Confirm-gate stays strict: `veronica/brain/policy.py` `classify()` is the only thing that may auto-allow a tool, plus the **trust window** defined here (implemented in `Brain._can_use_tool`, logged, bounded, cleared on barge). Never set `allowed_tools`.
- Never add `Co-Authored-By` trailers or "Generated with Claude Code" to commits.
- Tests: `uv run pytest -q`. No test posts real CGEvents, calls real Vision/AX/CoreGraphics, or reads the real screen — every framework call goes through an injectable seam (`_quartz()`, `_vision()`, `_ax()` accessors patched in tests, like `_import_quartz` in `veronica/audio/hotkey.py`).
- Python 3.12, `uv`; PyObjC frameworks Quartz, Vision, ApplicationServices, AppKit (already deps). Follow `veronica/tools/screen.py` / `browser.py` patterns (`_ok`/`_err`/`_guard`, `create_sdk_mcp_server`).
- Voice UX copy short and spoken.

## Goal
"Click the Save button", "type hello in that box", "scroll down", "press enter", "double-click the file", "drag it over there" — Veronica acts on the frontmost app using the screenshot she already takes, verifies with another screenshot, and asks once per app/session before acting.

## E1 — Screenshot geometry + OCR (`veronica/tools/screen.py`, new `veronica/tools/ocr.py`)

- `capture_screenshot` also writes `latest.json` next to `latest.png` (0600): `{"region", "image_w", "image_h", "origin_x", "origin_y", "width_pt", "height_pt", "scale", "captured_at", "window": {"id", "app", "title", "x", "y", "w", "h"} | null}` where `origin/width/height` are the **captured area in screen points** (whole main display via `CGDisplayBounds(CGMainDisplayID())`; for `window` region the window bounds from `CGWindowListCopyWindowInfo` for `front_window_id()`), `image_w/h` the final (downscaled) PNG size, `scale = image_w / width_pt`. `load_geometry() -> Geometry | None` (dataclass) reads it; `Geometry.to_screen(x_img, y_img) -> (x_pt, y_pt)` = `(origin_x + x_img / scale, origin_y + y_img / scale)`; `Geometry.age_s`. The screenshot tool's text result becomes `"Screenshot of the screen: 1568×1019 px (screen 1470×956 pt). Coordinates you pass to computer_* tools are in these image pixels."`
- `veronica/tools/ocr.py`: `recognize_text(png_path: Path, *, vision=None) -> list[Word]` using `VNRecognizeTextRequest` (accurate level, `usesLanguageCorrection` True) on `VNImageRequestHandler.alloc().initWithURL_options_`; each observation → `Word(text, x, y, w, h, confidence)` in **image pixels, top-left origin** (Vision boxes are normalized bottom-left: `x = bx * image_w`, `y = (1 - by - bh) * image_h`). `find_text(words, query) -> list[Word]`: case-insensitive; exact line match first, then lines containing the query, then a fuzzy `difflib` ratio ≥ 0.8; results sorted top-to-bottom, left-to-right. Hindi/Devanagari recognition languages: `["en-US", "hi-IN"]` when available (`supportedRecognitionLanguagesAndReturnError_` — fall back to default on error).

## E2 — Input primitives (`veronica/tools/computer_events.py`)

- All Quartz/AX access through `_quartz()` / `_ax()` module accessors (lazy imports, patched in tests).
- `accessibility_trusted(prompt: bool = False) -> bool` via `AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: prompt})`.
- `PERMISSION_HINT = "Veronica isn't allowed to control this Mac yet — enable it in System Settings > Privacy & Security > Accessibility, then try again."`
- `move(x_pt, y_pt)`, `click(x_pt, y_pt, button="left"|"right"|"middle", double=False)` (move, down, up; `kCGMouseEventClickState` 2 for double, second down/up pair), `drag(x1, y1, x2, y2)` (down at start, 8 intermediate `LeftMouseDragged` moves over 200 ms, up at end), `scroll(x_pt, y_pt, dx, dy)` (`CGEventCreateScrollWheelEvent` pixel units, move first), `type_text(text)` (chunks of 20 UTF-16 code units via `CGEventKeyboardSetUnicodeString` on key-down/up with keycode 0; 10 ms between chunks), `key(combo)` (parse `"cmd+shift+s"`, `"enter"`, `"esc"`, `"tab"`, `"space"`, `"backspace"|"delete"`, `"up/down/left/right"`, `"home/end/pageup/pagedown"`, `"f1".."f12"`, letters/digits/punctuation via a US keycode table; modifiers → `CGEventSetFlags` on the key events; unknown key → `ValueError`). Events posted to `kCGHIDEventTap`. `sleep` injectable.
- `frontmost() -> Front(app: str, bundle_id: str, window_title: str, pid: int)` via `NSWorkspace.sharedWorkspace().frontmostApplication()` + `CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, 0)` first window with that pid and layer 0.
- `focused_is_secure() -> bool`: `AXUIElementCreateSystemWide()` → `kAXFocusedUIElementAttribute` → `kAXRoleAttribute == "AXSecureTextField"` (or `kAXSubroleAttribute == "AXSecureTextField"`); False on any error.
- `is_system_dialog(front: Front) -> bool`: bundle id in `{"com.apple.SecurityAgent", "com.apple.UserNotificationCenter", "com.apple.coreservices.uiagent", "com.apple.systempreferences" when title contains "Privacy" or "Security"}` — used to refuse "Allow"-style clicks and to force a confirm.

## E3 — `computer` MCP server (`veronica/tools/computer.py`)

Every tool: (1) `accessibility_trusted(prompt=True)` else `_err(PERMISSION_HINT)`; (2) `geometry = load_geometry()`; if None or `age_s > 120` → `_err("Take a screenshot first — I need a fresh view of the screen to know where things are.")` (except `computer_type`/`computer_key`, which don't need coordinates); (3) coordinates default to **image pixels** (`space="image"`), `space="screen"` accepts points; (4) after acting, `sleep(0.15)` and return `"done — frontmost: <app> — <window title>"`.

| tool | args | policy |
|---|---|---|
| `computer_move` | `x, y, space?` | allow |
| `computer_scroll` | `x, y, dx, dy, space?` | allow |
| `computer_find` | `text` → OCR on `latest.png`; returns up to 10 matches `"1. 'Save' at (812, 431) size 60×22 (conf 0.98)"` (image px, center coords) or `"not found"` | allow |
| `computer_click` | `x, y, button?, double?, space?` | confirm |
| `computer_click_text` | `text, index?=0, double?` → find + click the match's center; refuses `"no match for '<text>'"` | confirm |
| `computer_drag` | `x1, y1, x2, y2, space?` | confirm |
| `computer_type` | `text, submit?` → refuses if `focused_is_secure()` (`"That's a password field — I won't type into it."`); `submit` presses Enter after | confirm |
| `computer_key` | `combo` | confirm |

Extra refusals (before the confirm gate, in the tool): `computer_click_text` with a target in `{"allow", "always allow", "ok", "open system settings", "continue", "install", "trust"}` (case-insensitive) while `is_system_dialog(frontmost())` → `_err("I won't click through a system permission dialog — please do that one yourself.")`; any action tool while `is_system_dialog(front)` with title containing "Privacy & Security" is confirm-class regardless of trust window (see E4).

Summaries (`agent.summarize_detail`): `"Click (812, 431)"`, `"Double-click 'Save'"`, `"Click 'Save'"`, `"Drag (10, 10) → (300, 300)"`, `"Type 'hello' + Enter"` (text[:40]), `"Press cmd+s"`, `"Scroll down at (500, 400)"`, `"Move to (…)"`, `"Find 'Save' on screen"`. HUD shows these as the tool card.

Policy: `MCP_TOOL_RISK["computer"]` per the table.

Prompt (`prompts.py`, append): `"You can also act on the screen with the computer tools: take a screenshot, use computer_find to locate text, then computer_click_text/computer_click/computer_type/computer_key; coordinates are pixels of the last screenshot. After any action take a fresh screenshot before claiming it worked. Never type passwords or secrets, never click Allow/OK in system permission dialogs, and don't change settings under System Settings > Privacy & Security unless the user asked for exactly that."`

## E4 — Trust window (`veronica/brain/agent.py`)

- `Brain` gets `trust_window_s: float = 90` (from `Settings.computer_trust_s`, editable 0–600, 0 = off, live; listed under Brain in the settings page) and state `_trust_until: float`, `_trust_app: str | None`.
- In `_can_use_tool`, for tools with prefix `mcp__computer__` whose `classify` is `confirm`: `front = frontmost()` (injected callable, default `computer_events.frontmost`); if `now < _trust_until and front.bundle_id == _trust_app and not is_system_dialog(front)` → allow with `on_tool(summary, "auto")` and log `"trusted: %s"`; otherwise ask via `_confirm(summary, detail)`; on **yes** → `_trust_until = now + trust_window_s; _trust_app = front.bundle_id` (only when window > 0). `clear_trust()` resets both.
- Orchestrator: `_barge_teardown` and the `end`/`stop` intents call `self.brain.clear_trust()` (guarded `getattr`). The confirm prompt for the first computer action says the summary as usual; the HUD card for trusted actions shows the normal "auto" pill.
- Voice: "stop" during a sequence is the existing barge (interrupts the brain and clears trust).

## E5 — Docs
README "Computer use": what she can do, the permission (Accessibility for Veronica — already granted for PTT? No: PTT needed Input Monitoring; Accessibility is new — first action prompts), the trust window (90 s per app, settable), the safety rules, example phrases.

## Tests
- `tests/test_screen.py`: `latest.json` geometry written (fake `run` + fake Quartz bounds), `Geometry.to_screen`, age; window region geometry.
- `tests/test_ocr.py`: fake Vision request/handler returning observations → pixel boxes (y flip), `find_text` exact/contains/fuzzy ordering, language fallback.
- `tests/test_computer_events.py`: fake Quartz records posted events (types, coords, flags, click state), `key("cmd+shift+s")` keycode+flags, unknown key error, `type_text` chunking, `drag` intermediate moves, `accessibility_trusted` hint, `focused_is_secure` role check, `frontmost`, `is_system_dialog`.
- `tests/test_computer_tools.py`: each tool with fakes — permission hint, stale/missing geometry error, image→screen conversion, `space="screen"`, click_text via OCR, secure-field refusal, system-dialog refusal, result text with frontmost.
- `tests/test_policy.py` / `tests/test_agent.py`: risk table; summaries; trust window (allow inside window + same app; ask when app differs, expired, disabled (0), or system dialog; yes sets the window; `clear_trust`).
- `tests/test_orchestrator.py`: barge and end intent clear trust (fake brain records).
- `tests/test_config.py`/`test_settings_bridge.py`: `computer_trust_s`.

## Order & branching
`batch-e` off master: E1 → E2 → E3 → E4 (+E5). Reviewer per task, whole-branch review, merge `--no-ff`, `make app`, relaunch; then the user grants Accessibility to Veronica on first use.
