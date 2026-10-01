import os
import shlex
from collections.abc import Awaitable, Callable
from typing import Any

from veronica.brain.policy import TRUST_EXCLUDED_BUNDLES  # noqa: F401  (re-exported)


def _image_media_type(data: bytes) -> str:
    """image/jpeg for JPEG magic bytes (an oversized capture re-encoded by
    tools/screen.py), else image/png."""
    return "image/jpeg" if data[:2] == b"\xff\xd8" else "image/png"

# confirm(summary, detail, question=…) answers with something truthy only
# when the user approved: Orchestrator.ConfirmResult (`.outcome` of
# "approved" / "denied" / "other", plus `.heard` and `.always`), or a bare
# bool from older callers. `question` overrides the spoken wording; the gate
# passes it for a tool that can be auto-allowed.
Confirm = Callable[..., Awaitable[Any]]


MAC_PREFIX = "mcp__mac__"
PIM_PREFIX = "mcp__pim__"
MEMORY_PREFIX = "mcp__memory__"
SCREEN_PREFIX = "mcp__screen__"
MUSIC_PREFIX = "mcp__music__"
BROWSER_PREFIX = "mcp__browser__"
COMPUTER_PREFIX = "mcp__computer__"

# TRUST_EXCLUDED_BUNDLES (terminals) and the Enter/terminal/system-dialog
# rules live in policy.always_confirm now; the gate and the trust window
# both consult it.


# Summaries that already read as an action (the computer tools' "Click
# 'Save'", "Press cmd+s") are asked as themselves; anything else gets the
# generic "Run X?".
ACTION_SUMMARY_PREFIXES = ("Click ", "Double-click ", "Right-click ", "Type ", "Press ", "Drag ", "Scroll ")


def confirm_prompt(summary: str) -> str:
    """The spoken question for a tool `summary`: "Click 'Save'?" for a
    screen action, "Run Bash: ls?" for everything else."""
    if summary.startswith(ACTION_SUMMARY_PREFIXES):
        return f"{summary}?"
    return f"Run {summary}?"


def summarize_tool(tool_name: str, input: dict) -> str:
    description = input.get("description")
    if not (isinstance(description, str) and description.strip()):
        return summarize_detail(tool_name, input)
    desc = description.strip()
    if tool_name == "Bash":
        try:
            argv = shlex.split(str(input.get("command", "")))
        except ValueError:
            argv = []
        if argv:
            desc = f"{desc} via {argv[0]}"
    elif tool_name in ("Write", "Edit") and input.get("file_path"):
        desc = f"{desc} in {os.path.basename(str(input['file_path']))}"
    return desc[:80].rstrip(".")


def _pt(input: dict, xk: str = "x", yk: str = "y") -> str:
    def n(v):
        try:
            return str(round(float(v)))
        except (TypeError, ValueError):
            return str(v)
    return f"({n(input.get(xk, ''))}, {n(input.get(yk, ''))})"


def _summarize_computer(short: str, input: dict) -> str:
    if short == "computer_click":
        verb = "Double-click" if input.get("double") else ("Right-click" if input.get("button") == "right" else "Click")
        return f"{verb} {_pt(input)}"
    if short == "computer_click_text":
        verb = "Double-click" if input.get("double") else "Click"
        return f"{verb} '{input.get('text', '')}'"
    if short == "computer_drag":
        return f"Drag {_pt(input, 'x1', 'y1')} \u2192 {_pt(input, 'x2', 'y2')}"
    if short == "computer_type":
        desc = f"Type '{str(input.get('text', ''))[:40]}'"
        return desc + " + Enter" if input.get("submit") else desc
    if short == "computer_key":
        return f"Press {input.get('combo', '')}"
    if short == "computer_scroll":
        try:
            dx, dy = float(input.get("dx") or 0), float(input.get("dy") or 0)
        except (TypeError, ValueError):
            dx, dy = 0.0, 1.0
        if dy:
            direction = "down" if dy > 0 else "up"
        else:
            direction = "right" if dx > 0 else "left"
        return f"Scroll {direction} at {_pt(input)}"
    if short == "computer_move":
        return f"Move to {_pt(input)}"
    if short == "computer_find":
        return f"Find '{input.get('text', '')}' on screen"
    return short


def summarize_detail(tool_name: str, input: dict) -> str:
    if tool_name.startswith(MAC_PREFIX):
        short = tool_name[len(MAC_PREFIX):]
        if short == "open_app":
            return f"Open {input.get('name', '')}"
        if short == "open_url":
            return f"Open {input.get('url', '')}"
        if short == "clipboard_write":
            return "Copy to clipboard: " + str(input.get("text", ""))[:60]
        if short == "applescript":
            return "AppleScript: " + str(input.get("script", ""))[:60]
        if short == "run_shortcut":
            return f"Run the shortcut '{input.get('name', '')}'"
        return short
    if tool_name.startswith(PIM_PREFIX):
        short = tool_name[len(PIM_PREFIX):]
        if short == "calendar_events":
            return "Check calendar"
        if short == "calendar_create":
            return f"Create event {input.get('title', '')}"
        if short == "mail_unread":
            return "Read unread mail"
        if short == "mail_search":
            return f"Search mail: {input.get('query', '')}"
        if short == "mail_send":
            return f"Send mail to {input.get('to', '')}"
        if short == "message_send":
            return f"Message {input.get('to', '')}: " + str(input.get("body", ""))[:40]
        if short == "reminder_create":
            return f"Create reminder {input.get('title', '')}"
        if short == "reminders_due":
            return "Check reminders"
        if short == "notes_create":
            return f"Create note {input.get('title', '')}"
        if short == "timer_set":
            return f"Set timer {input.get('minutes', '')} min"
        if short == "timer_list":
            return "List timers"
        if short == "timer_cancel":
            return f"Cancel timer {input.get('label', '')}"
        return short
    if tool_name.startswith(MEMORY_PREFIX):
        short = tool_name[len(MEMORY_PREFIX):]
        if short == "recall":
            return f"Recall {input.get('query', '')}"
        if short == "facts_list":
            return "List remembered facts"
        if short == "fact_add":
            return f"Remember {input.get('text', '')}"
        if short == "fact_delete":
            return f"Forget {input.get('text', '')}"
        return short
    if tool_name.startswith(SCREEN_PREFIX):
        return "Look at screen"
    if tool_name.startswith(MUSIC_PREFIX):
        short = tool_name[len(MUSIC_PREFIX):]
        if short == "music_play":
            q = input.get("query", "")
            return f"Play {q}" if q else "Play music"
        if short == "music_pause":
            return "Pause music"
        if short == "music_next":
            return "Next track"
        if short == "music_prev":
            return "Previous track"
        if short == "music_now_playing":
            return "What's playing"
        if short == "music_volume":
            return f"Set music volume {input.get('level', '')}"
        return short
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
            text = str(input.get("text", ""))[:40]
            desc = f"Type '{text}' into '{input.get('target', '')}'"
            return desc + " and press Enter" if input.get("submit") else desc
        if short == "browser_scroll":
            return f"Scroll {input.get('direction', 'down')}"
        if short == "browser_back":
            return "Go back"
        return short
    if tool_name.startswith(COMPUTER_PREFIX):
        return _summarize_computer(tool_name[len(COMPUTER_PREFIX):], input)
    if tool_name in ("Write", "Edit") and "file_path" in input:
        return f"{tool_name} file {input['file_path']}"
    for key in ("command", "query", "url", "pattern", "file_path"):
        if key in input:
            return f"{tool_name}: {input[key]}"
    return tool_name


def __getattr__(name: str):
    # `Brain` is the Claude backend, kept here for existing imports. It is
    # resolved on first use rather than imported at the top: backends.claude
    # and gate.py import the helpers above, so an eager import here would be
    # a cycle whichever module happened to load first.
    if name == "Brain":
        from veronica.brain.backends.claude import ClaudeBrain
        return ClaudeBrain
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
