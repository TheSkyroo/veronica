import asyncio
import base64
import datetime as dt
import logging
import os
import shlex
import time
from collections.abc import AsyncIterator, Awaitable, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
)
from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

from veronica.brain.policy import classify
from veronica.brain.prompts import system_prompt
from veronica.brain.sentences import SentenceSplitter
from veronica.config import Settings
from veronica.tools.browser import browser_server
from veronica.tools.computer import computer_server
from veronica.tools.computer_events import Front, frontmost, is_system_dialog
from veronica.tools.mac import mac_server
from veronica.tools.memory_tools import memory_server
from veronica.tools.music import music_server
from veronica.tools.pim import pim_server
from veronica.tools.screen import latest_screenshot_path, screen_server

log = logging.getLogger("veronica.brain")


def _image_media_type(data: bytes) -> str:
    """image/jpeg for JPEG magic bytes (an oversized capture re-encoded by
    tools/screen.py), else image/png."""
    return "image/jpeg" if data[:2] == b"\xff\xd8" else "image/png"

Confirm = Callable[[str, str], Awaitable[bool]]

MAC_PREFIX = "mcp__mac__"
PIM_PREFIX = "mcp__pim__"
MEMORY_PREFIX = "mcp__memory__"
SCREEN_PREFIX = "mcp__screen__"
MUSIC_PREFIX = "mcp__music__"
BROWSER_PREFIX = "mcp__browser__"
COMPUTER_PREFIX = "mcp__computer__"


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


class Brain:
    """One resumable Claude Code session; every tool call goes through `confirm`."""

    _client_cls = ClaudeSDKClient  # swapped in tests

    def __init__(
        self,
        settings: Settings,
        confirm: Confirm,
        on_tool: Callable[[str, str], None] | None = None,
        memory=None,
        frontmost: Callable[[], Front] = frontmost,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.s = settings
        self._confirm = confirm
        self._on_tool = on_tool
        self._memory = memory
        self._frontmost = frontmost
        self._clock = clock
        self._client = None
        self._in_flight = False
        # Trust window (spec E4): after the user approves one confirm-class
        # screen action, further ones in the same app are auto-allowed until
        # `_trust_until` (monotonic seconds). Cleared on barge, "that's all",
        # or a "no".
        self._trust_until = 0.0
        self._trust_app: str | None = None

    # -- session persistence --------------------------------------------------
    def _load_session(self) -> str | None:
        f = self.s.session_file
        return f.read_text().strip() or None if f.exists() else None

    def _save_session(self, sid: str) -> None:
        self.s.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.s.session_file.write_text(sid)

    def _clear_session(self) -> None:
        if self.s.session_file.exists():
            self.s.session_file.unlink()

    # -- permission gate ------------------------------------------------------
    # Shell commands that only "work" under a different TCC identity than the
    # app (screencapture run by the Claude CLI child process needs its own
    # Screen Recording grant) — redirect the brain to the in-process tool.
    _REDIRECT_BASH = {
        "screencapture": "Use the screenshot tool instead of screencapture — it runs inside Veronica, which has the Screen Recording permission.",
    }

    def _bash_redirect(self, tool_name: str, input: dict) -> str | None:
        if tool_name != "Bash":
            return None
        try:
            argv = shlex.split(str(input.get("command", "")))
        except ValueError:
            return None
        for tok in argv:
            base = os.path.basename(tok)
            if base in self._REDIRECT_BASH:
                return self._REDIRECT_BASH[base]
        return None

    async def _can_use_tool(self, tool_name: str, input: dict, context):
        summary = summarize_tool(tool_name, input)
        redirect = self._bash_redirect(tool_name, input)
        if redirect is not None:
            log.info("tool redirected: %s -> %s", summary, redirect)
            return PermissionResultDeny(message=redirect)
        if classify(tool_name, input) == "allow":
            log.info("auto-allow: %s", summary)
            if self._on_tool:
                self._on_tool(summary, "auto")
            return PermissionResultAllow(updated_input=input)
        if tool_name.startswith(COMPUTER_PREFIX):
            return await self._gate_computer(tool_name, input, summary)
        log.info("tool request: %s", summary)
        if await self._confirm(summary, summarize_detail(tool_name, input)):
            return PermissionResultAllow(updated_input=input)
        return PermissionResultDeny(message="user declined")

    # -- trust window (E4) ----------------------------------------------------
    def clear_trust(self) -> None:
        self._trust_until = 0.0
        self._trust_app = None

    def _trusted(self, front: Front, now: float) -> bool:
        return (
            self._trust_app is not None
            and now < self._trust_until
            and front.bundle_id == self._trust_app
            and not is_system_dialog(front)
        )

    async def _gate_computer(self, tool_name: str, input: dict, summary: str):
        """Confirm gate for confirm-class `mcp__computer__*` tools. The
        frontmost app is looked up at gate time; a system permission dialog
        (`is_system_dialog`, keyed on bundle id — those windows have empty
        titles) never gets the trust exemption."""
        front = self._frontmost()
        now = self._clock()
        if self._trusted(front, now):
            log.info("trusted: %s", summary)
            if self._on_tool:
                self._on_tool(summary, "auto")
            return PermissionResultAllow(updated_input=input)
        log.info("tool request: %s", summary)
        if await self._confirm(summary, summarize_detail(tool_name, input)):
            window = self.s.computer_trust_s
            if window > 0 and front.bundle_id and not is_system_dialog(front):
                self._trust_until = now + window
                self._trust_app = front.bundle_id
                log.info("trust window opened for %s (%ss)", front.bundle_id, window)
            return PermissionResultAllow(updated_input=input)
        self.clear_trust()
        return PermissionResultDeny(message="user declined")

    def _options(self, resume: str | None) -> ClaudeAgentOptions:
        # Re-read facts/recent turns here (not cached) so a NEW client/session
        # picks up anything remembered since the last one was created; the
        # SDK session itself already carries context turn-to-turn within one
        # client, so this only matters right after a fresh session starts.
        facts: list[str] = []
        recent: list[tuple[str, str]] = []
        if self._memory is not None and self.s.memory_enabled:
            facts = [text for _id, _ts, text in self._memory.facts()]
            recent = [
                (heard, reply)
                for _ts, heard, reply in self._memory.recent(self.s.memory_recent_turns)
            ]
        kwargs = {}
        if self.s.max_turns is not None:
            kwargs["max_turns"] = self.s.max_turns
        return ClaudeAgentOptions(
            system_prompt=system_prompt(dt.date.today(), facts, recent),
            effort=self.s.effort,
            permission_mode="default",
            can_use_tool=self._can_use_tool,
            resume=resume,
            mcp_servers={
                "mac": mac_server, "pim": pim_server, "memory": memory_server,
                "screen": screen_server, "music": music_server,
                "browser": browser_server, "computer": computer_server,
            },
            cwd=str(self.s.brain_cwd),
            # do not set allowed_tools — it auto-approves and bypasses can_use_tool
            # Only our confirmation gate may allow tools; ignore any
            # ~/.claude/settings.json (or project/local) permissions.allow
            # rules that would otherwise bypass can_use_tool entirely.
            setting_sources=[],
            **kwargs,
        )

    async def _ensure_client(self):
        if self._client is not None:
            return self._client

        resume = self._load_session()
        client = self._client_cls(options=self._options(resume))
        try:
            await client.connect()
        except Exception:
            if resume is not None:
                log.warning("stale session cleared")
                self._clear_session()
                client = self._client_cls(options=self._options(None))
                try:
                    await client.connect()
                except Exception:
                    self._client = None
                    raise
            else:
                self._client = None
                raise

        self._client = client
        return self._client

    def _build_prompt(self, text: str, images: tuple[bytes, ...]):
        """Return `text` as-is for a plain query, or — when `images` is
        non-empty — an async iterable yielding one user message whose
        content is a list of blocks (text + one image block per image), per
        the SDK's streaming-input message shape (see ClaudeSDKClient.query
        docstring / client.py: `prompt: str | AsyncIterable[dict]`)."""
        if not images:
            return text

        async def _stream():
            content: list[dict] = [{"type": "text", "text": text}]
            for image_bytes in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _image_media_type(image_bytes),
                        "data": base64.b64encode(image_bytes).decode("ascii"),
                    },
                })
            # parent_tool_use_id: None for parity with the SDK's own user
            # message shape (see claude_agent_sdk client.py's streaming
            # example); some SDK versions read it unconditionally.
            yield {
                "type": "user",
                "message": {"role": "user", "content": content},
                "parent_tool_use_id": None,
            }

        return _stream()

    @staticmethod
    def _image_fallback_text(text: str) -> str:
        return (
            f"{text}\n\n(A screenshot of the screen was taken but could not be "
            f"attached to this message; it is saved at {latest_screenshot_path()} "
            "— use the Read tool to look at it.)"
        )

    # -- public ---------------------------------------------------------------
    async def ask(self, text: str, images: list[bytes] = ()) -> AsyncIterator[str]:
        client = await self._ensure_client()
        splitter = SentenceSplitter()
        try:
            # _in_flight means "the SDK turn started and hasn't yet been
            # observed to end" — independent of what happens to whoever is
            # consuming this generator. Set it BEFORE query() so a barge
            # landing while the write is still in flight still triggers an
            # interrupt(). It must stay True if the consuming task is
            # cancelled (e.g. barged), so a later interrupt() still sends the
            # control request and drains the stream; it's cleared only when
            # the turn actually ends (a ResultMessage is seen, in close(), or
            # after interrupt()'s drain completes).
            self._in_flight = True
            try:
                await client.query(self._build_prompt(text, tuple(images)))
            except Exception:
                if not images:
                    raise
                # The image content-block message shape is the least
                # battle-tested path through the SDK; rather than failing
                # the whole turn, fall back to a plain text query that
                # points Claude's Read tool at the saved capture.
                log.exception("image query failed; falling back to text-only")
                await client.query(self._image_fallback_text(text))
            it = client.receive_response().__aiter__()
            while True:
                try:
                    async with asyncio.timeout(self.s.brain_timeout_s):
                        msg = await anext(it, None)
                except TimeoutError:
                    log.warning("brain timeout after %ss", self.s.brain_timeout_s)
                    await self.close()
                    yield "Taking too long, cancelled."
                    return

                if msg is None:
                    break

                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            for sent in splitter.feed(block.text):
                                yield sent
                elif isinstance(msg, ResultMessage):
                    self._in_flight = False   # turn ended, error or not
                    if getattr(msg, "is_error", False):
                        errors = getattr(msg, "errors", None)
                        error_text = " ".join(
                            str(part) for part in (msg.result, errors) if part
                        ).lower()
                        # Substring heuristic, not a structured error code from the
                        # SDK — a false positive here just resets the session
                        # (loses conversation history) rather than mis-handling
                        # a genuinely different error, so it's a safe bias.
                        if any(
                            marker in error_text
                            for marker in (
                                "context",
                                "compact",
                                "too long",
                                "prompt is too long",
                                "max_tokens",
                            )
                        ):
                            old_sid = self._load_session()
                            log.warning(
                                "brain context overflow (session %s): %s %s",
                                old_sid, msg.result, errors,
                            )
                            self._clear_session()
                            await self.close()
                            yield "My memory got full, starting a fresh conversation."
                            return
                        log.error(
                            "brain error result: %s %s",
                            msg.result,
                            errors,
                        )
                        await self.close()
                        yield "Claude returned an error, check the log."
                        return
                    self._save_session(msg.session_id)
        except Exception:
            await self.close()
            raise
        for sent in splitter.flush():
            yield sent

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            finally:
                self._client = None
        self._in_flight = False

    async def interrupt(self) -> None:
        """Stop the in-flight turn, if any. Safe to call when idle.

        After interrupting, drains any leftover messages still in flight on
        the stream (the SDK may have buffered assistant text and a final
        ResultMessage before it noticed the interrupt) so the next ask()
        doesn't read a stale tail or a stale ResultMessage. If the drain
        hangs or fails, the client is closed so the next ask() reconnects
        (resuming the saved session).

        A no-op if no turn is currently in flight (e.g. barging in while
        Veronica is only replaying already-generated speech): sending an SDK
        control request and draining a stream that has nothing left to
        interrupt would just stall for interrupt_drain_s for no reason.
        """
        if self._client is None or not self._in_flight:
            return
        try:
            async with asyncio.timeout(self.s.interrupt_drain_s):
                await self._client.interrupt()
        except TimeoutError:
            log.warning("interrupt() timed out after %ss; closing client", self.s.interrupt_drain_s)
            await self.close()
            return
        except Exception:
            log.exception("interrupt failed; closing client")
            await self.close()
            return
        try:
            drained = 0
            async with asyncio.timeout(self.s.interrupt_drain_s):
                async for msg in self._client.receive_response():
                    drained += 1
                    if isinstance(msg, ResultMessage):
                        break
            log.info("drained %d message(s) after interrupt", drained)
            self._in_flight = False   # the drain saw the turn end (a ResultMessage or EOF)
        except Exception:
            log.exception("drain after interrupt failed; closing client")
            await self.close()
