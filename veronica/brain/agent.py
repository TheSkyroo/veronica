import asyncio
import datetime as dt
import logging
import os
import shlex
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
from veronica.tools.mac import mac_server

log = logging.getLogger("veronica.brain")

Confirm = Callable[[str, str], Awaitable[bool]]

MAC_PREFIX = "mcp__mac__"


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
    ) -> None:
        self.s = settings
        self._confirm = confirm
        self._on_tool = on_tool
        self._client = None
        self._in_flight = False

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
    async def _can_use_tool(self, tool_name: str, input: dict, context):
        summary = summarize_tool(tool_name, input)
        if classify(tool_name, input) == "allow":
            log.info("auto-allow: %s", summary)
            if self._on_tool:
                self._on_tool(summary, "auto")
            return PermissionResultAllow(updated_input=input)
        log.info("tool request: %s", summary)
        if await self._confirm(summary, summarize_detail(tool_name, input)):
            return PermissionResultAllow(updated_input=input)
        return PermissionResultDeny(message="user declined")

    def _options(self, resume: str | None) -> ClaudeAgentOptions:
        kwargs = {}
        if self.s.max_turns is not None:
            kwargs["max_turns"] = self.s.max_turns
        return ClaudeAgentOptions(
            system_prompt=system_prompt(dt.date.today()),
            effort=self.s.effort,
            permission_mode="default",
            can_use_tool=self._can_use_tool,
            resume=resume,
            mcp_servers={"mac": mac_server},
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

    # -- public ---------------------------------------------------------------
    async def ask(self, text: str) -> AsyncIterator[str]:
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
            await client.query(text)
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
