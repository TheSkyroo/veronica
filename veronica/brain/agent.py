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

from veronica.brain.prompts import system_prompt
from veronica.brain.sentences import SentenceSplitter
from veronica.config import Settings

log = logging.getLogger("veronica.brain")

Confirm = Callable[[str], Awaitable[bool]]


def summarize_tool(tool_name: str, input: dict) -> str:
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

    # -- permission gate ------------------------------------------------------
    async def _can_use_tool(self, tool_name: str, input: dict, context):
        summary = summarize_tool(tool_name, input)
        log.info("tool request: %s", summary)
        if await self._confirm(summary):
            return PermissionResultAllow(updated_input=input)
        return PermissionResultDeny(message="user declined")

    def _options(self) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            system_prompt=system_prompt(dt.date.today()),
            effort=self.s.effort,
            max_turns=self.s.max_turns,
            permission_mode="default",
            can_use_tool=self._can_use_tool,
            resume=self._load_session(),
        )

    async def _ensure_client(self):
        if self._client is None:
            self._client = self._client_cls(options=self._options())
            await self._client.connect()
        return self._client

    # -- public ---------------------------------------------------------------
    async def ask(self, text: str) -> AsyncIterator[str]:
        client = await self._ensure_client()
        splitter = SentenceSplitter()
        await client.query(text)
        try:
            async with asyncio.timeout(self.s.brain_timeout_s):
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock):
                                for sent in splitter.feed(block.text):
                                    yield sent
                    elif isinstance(msg, ResultMessage):
                        self._save_session(msg.session_id)
        except TimeoutError:
            log.warning("brain timeout after %ss", self.s.brain_timeout_s)
            await self.close()
            yield "Taking too long, cancelled."
            return
        for sent in splitter.flush():
            yield sent

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.disconnect()
            finally:
                self._client = None
