import asyncio
import datetime as dt
import logging
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

Confirm = Callable[[str], Awaitable[bool]]

MAC_PREFIX = "mcp__mac__"


def summarize_tool(tool_name: str, input: dict) -> str:
    if tool_name.startswith(MAC_PREFIX):
        short = tool_name[len(MAC_PREFIX):]
        if short == "open_app":
            return f"Open {input.get('name', '')}"
        if short == "open_url":
            return f"Open {input.get('url', '')}"
        if short == "clipboard_write":
            return "Copy to clipboard: " + str(input.get("text", ""))[:60]
        if short == "applescript":
            return "Run AppleScript: " + str(input.get("script", ""))[:60]
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

    def __init__(self, settings: Settings, confirm: Confirm) -> None:
        self.s = settings
        self._confirm = confirm
        self._client = None

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
            return PermissionResultAllow(updated_input=input)
        log.info("tool request: %s", summary)
        if await self._confirm(summary):
            return PermissionResultAllow(updated_input=input)
        return PermissionResultDeny(message="user declined")

    def _options(self, resume: str | None) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            system_prompt=system_prompt(dt.date.today()),
            effort=self.s.effort,
            max_turns=self.s.max_turns,
            permission_mode="default",
            can_use_tool=self._can_use_tool,
            resume=resume,
            mcp_servers={"mac": mac_server},
            # do not set allowed_tools — it auto-approves and bypasses can_use_tool
            # Only our confirmation gate may allow tools; ignore any
            # ~/.claude/settings.json (or project/local) permissions.allow
            # rules that would otherwise bypass can_use_tool entirely.
            setting_sources=[],
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
                    if getattr(msg, "is_error", False):
                        log.error(
                            "brain error result: %s %s",
                            msg.result,
                            getattr(msg, "errors", None),
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

    async def interrupt(self) -> None:
        """Stop the in-flight turn, if any. Safe to call when idle.

        After interrupting, drains any leftover messages still in flight on
        the stream (the SDK may have buffered assistant text and a final
        ResultMessage before it noticed the interrupt) so the next ask()
        doesn't read a stale tail or a stale ResultMessage. If the drain
        hangs or fails, the client is closed so the next ask() reconnects
        (resuming the saved session).
        """
        if self._client is None:
            return
        try:
            await self._client.interrupt()
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
        except Exception:
            log.exception("drain after interrupt failed; closing client")
            await self.close()
