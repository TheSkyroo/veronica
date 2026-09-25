"""`LocalBrain`: llama.cpp on this Mac, for when there is no internet (or
the user just asks for it). Nothing leaves the machine.

The server (`llama-server`) is started lazily on the first turn — a model
load costs seconds — and left running until `close()` or ten idle minutes.
Turns are streamed `/v1/chat/completions` calls whose deltas feed the same
`SentenceSplitter` the other brains use, so TTS overlaps identically.

There is no vendor session to resume, so the conversation is kept here, in
memory, as a rolling window bounded by `local_ctx`.

Tools are Veronica's own MCP servers rendered as OpenAI-style function
schemas and run in-process; every call goes through `ToolGate.decide`
first, exactly like Claude's. A model that ignores the tool schema and
answers in words is fine — that is the normal outcome for the smaller
weights and must never be an error."""
import asyncio
import datetime as dt
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field

import httpx
from mcp.types import CallToolRequestParams

from veronica.brain.gate import ToolGate
from veronica.brain.prompts import system_prompt
from veronica.brain.sentences import SentenceSplitter
from veronica.config import Settings
from veronica.tools.browser import browser_server
from veronica.tools.computer import computer_server
from veronica.tools.mac import mac_server
from veronica.tools.memory_tools import memory_server
from veronica.tools.music import music_server
from veronica.tools.pim import pim_server
from veronica.tools.screen import screen_server

log = logging.getLogger("veronica.brain")

# The same servers ClaudeBrain hands the SDK, exposed here as function
# schemas under the same `mcp__<server>__<tool>` names — so policy.py, the
# HUD cards and the trust window all key off identical strings.
SERVERS = {
    "mac": mac_server, "pim": pim_server, "memory": memory_server,
    "screen": screen_server, "music": music_server,
    "browser": browser_server, "computer": computer_server,
}

START_FAILED = "The local model wouldn't start — check the Local settings."
# How long a cold `llama-server` gets to load the weights and answer /health.
START_TIMEOUT_S = 120.0
HEALTH_POLL_S = 0.25
# Stop the server after this long without a turn; the next one restarts it.
IDLE_SHUTDOWN_S = 600.0
# Tool rounds per turn. Small models loop ("call it once more") far more
# readily than the hosted ones, so this is a hard stop, not a budget.
MAX_TOOL_ROUNDS = 4

# A small local model reaches for the most general tool it can see when
# nothing fits, which means an out-of-nowhere "Run AppleScript…?" confirm
# for a question like "what's the battery at". These stay out of its
# catalogue: every one is either a catch-all or needs judgement the bigger
# brains have. Hiding them changes nothing about the gate — they are simply
# not offered.
HIDDEN_FROM_LOCAL = frozenset({"mcp__mac__applescript"})
HIDDEN_SERVERS_FOR_LOCAL = frozenset({"computer"})
# History budget. Rough on purpose: a token is ~3.5 characters of English,
# and the system prompt, the 40-odd tool schemas and the reply need most of
# the window, so history gets a share of what is left.
CHARS_PER_TOKEN = 3.5
HISTORY_SHARE = 0.35

_tools: tuple[list[dict], dict[str, tuple[object, str]]] | None = None


async def tool_catalog() -> tuple[list[dict], dict[str, tuple[object, str]]]:
    """(function schemas, `mcp__srv__tool` -> (server instance, bare name)),
    built once per process from the MCP servers' own tools/list."""
    global _tools
    if _tools is not None:
        return _tools
    schemas: list[dict] = []
    index: dict[str, tuple[object, str]] = {}
    for srv, server in SERVERS.items():
        inst = server["instance"]
        listed = await inst.get_request_handler("tools/list").handler(None, None)
        for t in listed.tools:
            full = f"mcp__{srv}__{t.name}"
            if full in HIDDEN_FROM_LOCAL or srv in HIDDEN_SERVERS_FOR_LOCAL:
                continue
            schemas.append({"type": "function", "function": {
                "name": full,
                "description": t.description or "",
                "parameters": t.input_schema or {"type": "object", "properties": {}},
            }})
            index[full] = (inst, t.name)
    _tools = (schemas, index)
    return _tools


@dataclass
class _Reply:
    """What one streamed completion produced, beside the spoken sentences."""
    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    error: str = ""


def _accumulate(calls: list[dict], deltas: list[dict]) -> None:
    """Fold streamed `tool_calls` deltas into `calls`. The name and id
    arrive on the first delta for an index, the arguments in pieces."""
    for d in deltas:
        i = int(d.get("index", 0))
        while len(calls) <= i:
            calls.append({"id": "", "name": "", "arguments": ""})
        fn = d.get("function") or {}
        if d.get("id"):
            calls[i]["id"] = d["id"]
        if fn.get("name"):
            calls[i]["name"] = fn["name"]
        calls[i]["arguments"] += fn.get("arguments") or ""


class LocalBrain:
    name = "local"
    label = "Local"
    idle_shutdown_s: float = IDLE_SHUTDOWN_S
    start_timeout_s: float = START_TIMEOUT_S

    def __init__(
        self,
        settings: Settings,
        gate: ToolGate,
        on_tool: Callable[[str, str], None] | None = None,
        memory=None,
        *,
        spawn=None,
        client=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.s = settings
        self.gate = gate
        self._on_tool = on_tool
        self._memory = memory
        self._spawn = spawn or self._subprocess_spawn
        self._clock = clock
        self._client = client or httpx.AsyncClient()
        self._proc = None
        self._adopted = False      # a server that was already listening: not ours to kill
        self._history: list[dict] = []
        self._response = None      # the in-flight streamed response, for interrupt()
        self._interrupted = False
        self._last_used = clock()
        self._idle_task: asyncio.Task | None = None

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

    # -- server ---------------------------------------------------------------
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.s.local_port}"

    def argv(self) -> list[str]:
        """The llama.cpp server command line. `--jinja` is what makes the
        model's own chat template (and so its tool calling) work; the web
        UI is off because nothing here browses to it."""
        return [
            str(self.s.local_server_bin),
            "--model", str(self.s.local_model),
            "--ctx-size", str(self.s.local_ctx),
            "--host", "127.0.0.1",
            "--port", str(self.s.local_port),
            "--jinja",
            "--no-webui",
        ]

    async def _subprocess_spawn(self, argv: list[str]):
        return await asyncio.create_subprocess_exec(
            *argv, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            stdin=asyncio.subprocess.DEVNULL,
        )

    async def _healthy(self, timeout: float) -> bool:
        try:
            r = await self._client.get(f"{self.base_url}/health", timeout=timeout)
            return r.status_code == 200
        except Exception:
            return False

    async def _ensure_server(self) -> None:
        """Start `llama-server` and wait for /health. A server already
        listening on our port (the user's own, or one we left running
        across a reload) is adopted rather than fought with."""
        if self._proc is not None and self._proc.returncode is None:
            return
        if await self._healthy(1.0):
            self._adopted = True
            return
        argv = self.argv()
        log.info("local: spawning %s", argv)
        self._proc = await self._spawn(argv)
        self._adopted = False
        deadline = self._clock() + self.start_timeout_s
        while True:
            if await self._healthy(2.0):
                log.info("local: server ready on %s", self.base_url)
                return
            if self._proc.returncode is not None:
                raise RuntimeError(f"llama-server exited {self._proc.returncode}")
            if self._clock() >= deadline:
                await self._stop_server()
                raise TimeoutError(f"llama-server wasn't healthy within {self.start_timeout_s}s")
            await asyncio.sleep(HEALTH_POLL_S)

    async def _stop_server(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or self._adopted:
            return
        try:
            if proc.returncode is None:
                proc.terminate()
            async with asyncio.timeout(self.s.interrupt_drain_s):
                await proc.wait()
        except TimeoutError:
            proc.kill()
        except Exception:
            log.debug("local: stopping the server failed", exc_info=True)

    def _touch(self) -> None:
        """Mark the server used and (re)arm the idle shutdown."""
        self._last_used = self._clock()
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_watch())

    async def _idle_watch(self) -> None:
        """Stop the server once it has gone `idle_shutdown_s` without a turn
        — a loaded model holds gigabytes, and the next turn just reloads."""
        while True:
            left = self.idle_shutdown_s - (self._clock() - self._last_used)
            if left <= 0:
                break
            await asyncio.sleep(left)
        log.info("local: idle for %.0fs, stopping the server", self.idle_shutdown_s)
        await self._stop_server()

    # -- history ----------------------------------------------------------------
    def _system_prompt(self) -> str:
        facts: list[str] = []
        recent: list[tuple[str, str]] = []
        if self._memory is not None and self.s.memory_enabled:
            facts = [text for _id, _ts, text in self._memory.facts()]
            recent = [(heard, reply) for _ts, heard, reply in self._memory.recent(self.s.memory_recent_turns)]
        return system_prompt(dt.date.today(), facts, recent) + (
            " You are running offline on this Mac, so you cannot search the web or open "
            "a page — say so instead of guessing. Use a tool only when the request needs "
            "one; otherwise just answer."
        )

    def _trim(self) -> None:
        """Drop the oldest messages until the history fits its share of the
        context window. Always cuts back to a `user` message: an assistant
        turn's tool calls and their results have to stay together or the
        chat template breaks."""
        budget = int(self.s.local_ctx * HISTORY_SHARE * CHARS_PER_TOKEN)
        while self._history and sum(len(json.dumps(m)) for m in self._history) > budget:
            del self._history[0]
            while self._history and self._history[0].get("role") != "user":
                del self._history[0]

    # -- one streamed completion ------------------------------------------------
    async def _round(self, messages: list[dict], tools: list[dict],
                     splitter: SentenceSplitter, out: _Reply) -> AsyncIterator[str]:
        """POST one completion and yield sentences as the deltas arrive.
        `reasoning_content` deltas (Granite and friends think out loud) are
        dropped on the floor — they are not for speaking."""
        payload = {"messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools
        timeout = httpx.Timeout(self.s.brain_timeout_s, connect=5.0)
        try:
            async with self._client.stream(
                "POST", f"{self.base_url}/v1/chat/completions", json=payload, timeout=timeout
            ) as response:
                self._response = response
                if response.status_code != 200:
                    await response.aread()
                    out.error = f"HTTP {response.status_code}"
                    return
                async for line in response.aiter_lines():
                    if self._interrupted:
                        return
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        choice = json.loads(data)["choices"][0]
                    except (ValueError, LookupError):
                        log.debug("local: unparsable chunk %.200s", data)
                        continue
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        out.content += delta["content"]
                        for sent in splitter.feed(delta["content"]):
                            yield sent
                    if delta.get("tool_calls"):
                        _accumulate(out.tool_calls, delta["tool_calls"])
        except (httpx.HTTPError, asyncio.IncompleteReadError) as e:
            if not self._interrupted:
                out.error = str(e) or type(e).__name__
        finally:
            self._response = None

    # -- tools ------------------------------------------------------------------
    async def _call_tool(self, index: dict, call: dict) -> str:
        """Gate, then run, one tool call. Everything the model asks for goes
        through `ToolGate.decide`; a refusal comes back as the tool result so
        it can re-plan, same as every other backend."""
        name = call.get("name") or ""
        try:
            args = json.loads(call.get("arguments") or "{}")
        except ValueError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        entry = index.get(name)
        if entry is None:
            log.warning("local: model asked for unknown tool %r", name)
            return f"error: there is no tool called {name}"
        decision = await self.gate.decide(name, args)
        if not decision.allow:
            return f"Not allowed: {decision.message}"
        inst, bare = entry
        try:
            result = await inst.get_request_handler("tools/call").handler(
                None, CallToolRequestParams(name=bare, arguments=args))
        except Exception as e:
            log.exception("local: tool %s failed", name)
            return f"error: {e}"
        return "\n".join(
            c.text for c in (result.content or []) if getattr(c, "type", None) == "text"
        ) or "ok"

    # -- public -----------------------------------------------------------------
    async def ask(self, text: str, images: list[bytes] = ()) -> AsyncIterator[str]:
        self.pending_redirect = None   # a redirect belongs to the turn it was said in
        self._interrupted = False
        if images:
            # No vision model here; say so rather than answering about a
            # screenshot the weights never saw.
            log.info("local: %d image(s) dropped — the local model is text-only", len(images))
            text = f"{text}\n\n(A screenshot was taken but you cannot see images; say so.)"
        try:
            await self._ensure_server()
        except Exception:
            log.exception("local: server would not start")
            yield START_FAILED
            return
        self._touch()

        schemas, index = await tool_catalog()
        self._history.append({"role": "user", "content": text})
        self._trim()
        splitter = SentenceSplitter()
        for _attempt in range(MAX_TOOL_ROUNDS + 1):
            out = _Reply()
            messages = [{"role": "system", "content": self._system_prompt()}, *self._history]
            async for sent in self._round(messages, schemas, splitter, out):
                yield sent
            if self._interrupted:
                if out.content:
                    self._history.append({"role": "assistant", "content": out.content})
                return
            if out.error:
                log.error("local: %s", out.error)
                self._history.append({"role": "assistant", "content": out.content})
                for sent in splitter.flush():
                    yield sent
                if not out.content:
                    yield "The local model returned an error, check the log."
                self._touch()
                return
            if not out.tool_calls:
                self._history.append({"role": "assistant", "content": out.content})
                for sent in splitter.flush():
                    yield sent
                self._touch()
                return
            self._history.append({
                "role": "assistant", "content": out.content,
                "tool_calls": [{"id": c["id"] or f"call_{i}", "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                               for i, c in enumerate(out.tool_calls)],
            })
            for i, call in enumerate(out.tool_calls):
                self._history.append({
                    "role": "tool",
                    "tool_call_id": call["id"] or f"call_{i}",
                    "content": await self._call_tool(index, call),
                })
            self._trim()
        # Out of rounds: the model kept asking for tools and never spoke.
        log.warning("local: stopped after %d tool rounds", MAX_TOOL_ROUNDS)
        for sent in splitter.flush():
            yield sent
        self._touch()

    async def interrupt(self) -> None:
        """Stop the in-flight completion. Closing the response drops the
        connection, which frees the server's slot; the history keeps what
        was already said. Safe when idle."""
        self._interrupted = True
        response, self._response = self._response, None
        if response is None:
            return
        try:
            await response.aclose()
        except Exception:
            log.debug("local: closing the stream failed", exc_info=True)

    async def close(self) -> None:
        await self.interrupt()
        if self._idle_task is not None:
            self._idle_task.cancel()
            self._idle_task = None
        await self._stop_server()
        try:
            await self._client.aclose()
        except Exception:
            log.debug("local: closing the http client failed", exc_info=True)
