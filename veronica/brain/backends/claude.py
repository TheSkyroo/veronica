import asyncio
import base64
import datetime as dt
import logging
import time
from collections.abc import AsyncIterator, Callable

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from claude_agent_sdk.types import PermissionResultAllow, PermissionResultDeny

from veronica.brain.agent import Confirm, _image_media_type
from veronica.brain.backends.cli import LIMIT_MARKERS, LimitError
from veronica.brain.gate import ToolGate, ToolStall
from veronica.brain.prompts import system_prompt
from veronica.brain.sentences import SentenceSplitter
from veronica.config import Settings
from veronica.tools.browser import browser_server
from veronica.tools.computer import computer_server
from veronica.tools.computer_events import Front, frontmost
from veronica.tools.mac import mac_server
from veronica.tools.memory_tools import memory_server
from veronica.tools.music import music_server
from veronica.tools.pim import pim_server
from veronica.tools.screen import latest_screenshot_path, screen_server

log = logging.getLogger("veronica.brain")


class ClaudeBrain:
    """One resumable Claude Code session; every tool call goes through the gate."""

    name = "claude"
    _client_cls = ClaudeSDKClient  # swapped in tests

    def __init__(
        self,
        settings: Settings,
        confirm: Confirm | None = None,
        on_tool: Callable[[str, str], None] | None = None,
        memory=None,
        frontmost: Callable[[], Front] = frontmost,
        clock: Callable[[], float] = time.monotonic,
        *,
        gate: ToolGate | None = None,
    ) -> None:
        self.s = settings
        # The orchestrator hands every backend one shared gate; building
        # one here keeps the old constructor (confirm/on_tool/...) working.
        self.gate = gate or ToolGate(settings, confirm, on_tool=on_tool, frontmost=frontmost, clock=clock)
        self._memory = memory
        self._client = None
        self._in_flight = False

    # -- gate proxies (the orchestrator keeps calling brain.<x>) ---------------
    def clear_trust(self) -> None:
        self.gate.clear_trust()

    def begin_turn(self, turn_id: int) -> None:
        self.gate.begin_turn(turn_id)

    def preapprove(self, turn_id: int, until: float) -> None:
        self.gate.preapprove(turn_id, until)

    @property
    def pending_redirect(self) -> str | None:
        return self.gate.pending_redirect

    @pending_redirect.setter
    def pending_redirect(self, value: str | None) -> None:
        self.gate.pending_redirect = value

    @property
    def _confirm(self) -> Confirm:
        # __main__'s text mode swaps in a stdin y/N confirm after construction.
        return self.gate._confirm

    @_confirm.setter
    def _confirm(self, value: Confirm) -> None:
        self.gate._confirm = value

    async def _can_use_tool(self, tool_name: str, input: dict, context):
        with self.gate.working():          # the stream is quiet until we answer
            d = await self.gate.decide(tool_name, input)
        if d.allow:
            return PermissionResultAllow(updated_input=input)
        return PermissionResultDeny(message=d.message)

    # -- session persistence --------------------------------------------------
    def _load_session(self) -> str | None:
        """The session to resume, or None to start a fresh one. A session
        that has been resumed for days grows without bound (the CLI replays
        the whole transcript on resume), and a cold resume of a big one can
        outlast brain_timeout_s — so one is retired once it reaches
        brain_session_max_age_h. The file is `<id>` (old format) or
        `<id>\n<unix seconds when it was started>`."""
        f = self.s.session_file
        if not f.exists():
            return None
        parts = f.read_text().split("\n")
        sid = parts[0].strip()
        if not sid:
            return None
        try:
            started = float(parts[1]) if len(parts) > 1 and parts[1].strip() else None
        except ValueError:
            started = None
        if started is None:
            # written before sessions were dated: date it from now, so it
            # still ages out — one extra long-lived session at most.
            self._save_session(sid)
            return sid
        max_age_s = self.s.brain_session_max_age_h * 3600
        if max_age_s > 0 and time.time() - started > max_age_s:
            log.info("retiring Claude session %s (older than %sh)", sid, self.s.brain_session_max_age_h)
            self._clear_session()
            return None
        return sid

    def _save_session(self, sid: str) -> None:
        self.s.session_file.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if self.s.session_file.exists():
            old = self.s.session_file.read_text().split("\n")
            if old and old[0].strip() == sid and len(old) > 1 and old[1].strip():
                try:
                    started = float(old[1])   # same session: keep its start
                except ValueError:
                    pass
        self.s.session_file.write_text(f"{sid}\n{started:.0f}")

    def _clear_session(self) -> None:
        if self.s.session_file.exists():
            self.s.session_file.unlink()

    def _options(self, resume: str | None) -> ClaudeAgentOptions:
        # Re-read facts/recent turns here (not cached) so a NEW client/session
        # picks up anything remembered since the last one was created; the
        # SDK session itself already carries context turn-to-turn within one
        # client, so this only matters right after a fresh session starts.
        facts: list[str] = []
        recent: list[tuple[str, str]] = []
        if self._memory is not None and self.s.memory_enabled:
            facts = self._memory.facts_for_prompt(self.s.memory_facts_max)
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
        self.pending_redirect = None   # a redirect belongs to the turn it was said in
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
            # tool_use ids without a result yet: the stream is quiet while
            # they run, and that isn't the model stalling.
            running: set[str] = set()
            while True:
                try:
                    msg = await self.gate.wait_quiet(anext(it, None), self.s.brain_timeout_s,
                                                     tool_running=bool(running))
                except ToolStall:
                    log.warning("turn ended early: reason=tool_stall detail=claude: tool call(s) %s "
                                "still running after %ss", sorted(running), self.gate.busy_ceiling_s)
                    await self.interrupt()
                    yield "That step never finished, so I stopped."
                    return
                except TimeoutError:
                    log.warning("turn ended early: reason=brain_timeout detail=claude: no output for %ss",
                                self.s.brain_timeout_s)
                    # interrupt(), not close(): dropping the client leaves the
                    # turn unanswered in the session, and the next ask()
                    # resumes that session — the CLI replays the abandoned
                    # turn ("Continue from where you left off.") and has to
                    # answer it before our new request, so one slow turn
                    # latches into a timeout on every turn after it.
                    # interrupt() closes the client itself if it can't drain.
                    await self.interrupt()
                    yield "Taking too long, cancelled."
                    return

                if msg is None:
                    break

                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            for sent in splitter.feed(block.text):
                                yield sent
                        elif isinstance(block, ToolUseBlock):
                            running.add(block.id)
                elif isinstance(msg, UserMessage):
                    for block in msg.content if isinstance(msg.content, list) else ():
                        if isinstance(block, ToolResultBlock):
                            running.discard(block.tool_use_id)
                elif isinstance(msg, ResultMessage):
                    self._in_flight = False   # turn ended, error or not
                    if getattr(msg, "is_error", False):
                        errors = getattr(msg, "errors", None)
                        error_text = " ".join(
                            str(part) for part in (msg.result, errors) if part
                        )
                        lowered = error_text.lower()
                        # Same markers the CLI backends use: a limit has to
                        # leave ask() as LimitError or the switcher can never
                        # fail over off Claude (spec 4a).
                        if any(marker in lowered for marker in LIMIT_MARKERS):
                            log.warning("turn ended early: reason=limit detail=claude: %s", error_text)
                            await self.close()
                            raise LimitError(error_text)
                        # Substring heuristic, not a structured error code from the
                        # SDK — a false positive here just resets the session
                        # (loses conversation history) rather than mis-handling
                        # a genuinely different error, so it's a safe bias.
                        if any(
                            marker in lowered
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
                                "turn ended early: reason=overflow detail=claude: context overflow (session %s): %s %s",
                                old_sid, msg.result, errors,
                            )
                            self._clear_session()
                            await self.close()
                            yield "My memory got full, starting a fresh conversation."
                            return
                        log.error(
                            "turn ended early: reason=error detail=claude: error result: %s %s",
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
