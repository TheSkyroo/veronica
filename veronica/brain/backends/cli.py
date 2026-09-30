"""`CliBrain`: the base for brains that are a vendor CLI run as a
subprocess. The subclass says how to spawn it (`argv`, `env`), what to
write into its workspace before a turn (`prepare_workspace`), and how
its JSON stream maps onto our small `Event` set (`parse`); the base
does everything else — streaming sentences, sessions, images, the
timeout, interrupt, limit/overflow detection, and the canary that
proves the CLI's pre-tool hook is really firing.

Two modes. `per_turn` (default) spawns one process per `ask` and the
prompt travels in `argv`. `persistent` spawns once with a stdin pipe;
the first events must include a `Session`, then each `ask` writes one
line (`turn_message`) and reads until the turn's `Done`/`Error`. An
interrupt kills the child either way; the next `ask` respawns, resuming
the saved session id through `argv`.

The canary: with the vendor CLI in auto-approve mode, our hook is the
only gate on its own shell/file tools. Every native, non-read-only tool
call therefore has to show up in `hook.log` (the hook writes a line
before it asks the gate), checked when the call starts and again when it
ends. A call without one means the hook didn't fire — the child is
killed, the backend's native tools are switched off (persisted), and the
turn is retried tools-off. That second pass runs with native tools off,
where the CLI's own read-only/deny mode is the gate, so the canary is
not armed there: tripping it twice would kill the turn silently."""
import asyncio
import datetime as dt
import json
import logging
import os
import signal
import subprocess
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from veronica import prefs
from veronica.brain import hook
from veronica.brain.gate import ToolGate, ToolStall
from veronica.brain.prompts import system_prompt
from veronica.brain.sentences import SentenceSplitter
from veronica.config import Settings

log = logging.getLogger("veronica.brain")


# -- events a subclass's parse() turns CLI output into -------------------------
@dataclass
class Text:
    delta: str


@dataclass
class ToolStart:
    call_id: str
    tool: str
    input: dict
    native: bool      # the CLI's own tool (hook-gated) vs one of our MCP tools (serve-gated)


@dataclass
class ToolEnd:
    call_id: str


@dataclass
class Session:
    id: str


@dataclass
class Done:
    final_text: str = ""


@dataclass
class Error:
    message: str


Event = Text | ToolStart | ToolEnd | Session | Done | Error

# "You've hit your weekly limit - resets 6:30am (Asia/Calcutta)" is what
# Claude Code says on a subscription limit, hence the per-period markers.
# One stream-json line. A CLI echoes the tool result back in its own
# stream, and a screenshot's base64 rides in it, so asyncio's 64 KiB
# default is far too small — a "look at my screen" turn died on
# "Separator is not found, and chunk exceed the limit". This is the same
# 1 MiB ceiling the Agent SDK's reader uses, and the reason screen.py
# keeps a capture under MAX_PNG_BYTES.
STDOUT_LINE_LIMIT = 1024 * 1024

LIMIT_MARKERS = ("usage limit", "rate limit", "rate_limit", "429", "quota", "resource exhausted",
                 "too many requests", "limit reached", "out of credits", "insufficient_quota", "overloaded",
                 "hourly limit", "daily limit", "weekly limit", "monthly limit")
OVERFLOW_MARKERS = ("context", "compact", "too long", "prompt is too long", "max_tokens")

Mode = Literal["per_turn", "persistent"]


class LimitError(Exception):
    """The CLI reported a usage/rate limit; raised out of ask() so the
    switcher can fail over to another brain."""


@dataclass
class _Outcome:
    kind: str = "ok"          # ok | canary | limit | error
    message: str = ""


class CliBrain:
    """A brain backed by a vendor CLI subprocess. Subclasses set `name`,
    `label`, `binary`, optionally `mode`, and implement the hooks."""

    name: str = ""
    label: str = ""
    binary: str = ""
    mode: Mode = "per_turn"
    # How long ToolEnd waits for the hook's log line before calling the
    # canary dead. The hook writes before the tool runs, so by the time
    # the CLI reports the call finished the line is normally there.
    canary_grace_s: float = 0.5
    # Same check at ToolStart, to cut the window in which an ungated call
    # runs: a long command would otherwise finish before the ToolEnd check
    # sees anything. Its grace is the longer of the two because at that
    # point the hook may still be starting up (a Python process spawn); it
    # only ever waits when the line is missing, i.e. on the way to a trip.
    canary_start_grace_s: float = 3.0

    def __init__(
        self,
        settings: Settings,
        gate: ToolGate,
        on_tool: Callable[[str, str], None] | None = None,
        memory=None,
        *,
        spawn=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.s = settings
        self.gate = gate
        self._on_tool = on_tool
        self._memory = memory
        self._spawn = spawn or self._subprocess_spawn
        self._clock = clock
        # `subprocess.run` for one-off vendor commands (e.g. `agy mcp add`);
        # tests swap in a recorder so no real CLI is ever called.
        self._run = subprocess.run
        self.workspace: Path = settings.backend_dir(self.name)
        self.hook_log: Path = self.workspace / "hook.log"
        self._proc = None
        self._proc_native: bool | None = None
        self._stderr_task: asyncio.Task | None = None
        self._stderr_tail = ""
        self._interrupted = False

    # -- subclass hooks -------------------------------------------------------
    def argv(self, text: str, session_id: str | None, image_paths: list[Path], native: bool) -> list[str]:
        """Command line for one turn (per_turn) or for the long-lived child
        (persistent; `text`/`image_paths` are then empty)."""
        raise NotImplementedError

    def env(self) -> dict[str, str]:
        """Extra environment for the child (the gate/hook variables are added by the base)."""
        return {}

    def prepare_workspace(self, prompt_text: str, native: bool) -> None:
        """Write the system prompt / hook / MCP config the CLI reads. Called before every turn."""
        raise NotImplementedError

    def parse(self, line: str) -> list[Event]:
        """One line of the CLI's stream -> zero or more events."""
        raise NotImplementedError

    def native_key(self, tool: str, input: dict) -> str:
        """The string the hook logs for this native call (must match what hook.canary_key logs)."""
        return hook.canary_key(tool, input)

    def canary_matches(self, logged_key: str, stream_key: str) -> bool:
        """Does a hook.log line's key account for the `native_key` seen in the
        stream? Equality by default; a CLI whose stream wraps the command
        (Codex: `/bin/zsh -lc '<cmd>'`) loosens this."""
        return logged_key == stream_key

    def turn_message(self, text: str, image_paths: list[Path]) -> str:
        """persistent mode: the one stdin line that starts a turn."""
        raise NotImplementedError

    def session_started(self, session_id: str) -> None:
        """persistent mode: called after the handshake, before the first prompt."""

    def readonly_summary(self, tool: str, input: dict) -> str:
        """HUD card text for a read-only native tool."""
        what = (input.get("file_path") or input.get("AbsolutePath") or input.get("TargetFile")
                or input.get("pattern") or input.get("Query") or input.get("query") or "")
        return f"{tool} {what}".strip()

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

    # -- settings / session ---------------------------------------------------
    def native_tools_enabled(self) -> bool:
        return bool(getattr(self.s, f"{self.name}_native_tools", True))

    def _session_file(self) -> Path:
        return self.s.session_file_for(self.name)

    def _load_session(self) -> str | None:
        f = self._session_file()
        return f.read_text().strip() or None if f.exists() else None

    def _save_session(self, sid: str) -> None:
        f = self._session_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(sid)

    def _clear_session(self) -> None:
        f = self._session_file()
        if f.exists():
            f.unlink()

    def _system_prompt(self) -> str:
        # Re-read facts/recent turns every turn, like ClaudeBrain._options.
        facts: list[str] = []
        recent: list[tuple[str, str]] = []
        if self._memory is not None and self.s.memory_enabled:
            facts = self._memory.facts_for_prompt(self.s.memory_facts_max)
            recent = [(heard, reply) for _ts, heard, reply in self._memory.recent(self.s.memory_recent_turns)]
        return system_prompt(dt.date.today(), facts, recent)

    def _write_images(self, images: tuple[bytes, ...]) -> list[Path]:
        paths = []
        for i, data in enumerate(images, 1):
            ext = "jpg" if data[:2] == b"\xff\xd8" else "png"
            p = self.workspace / f"img-{i}.{ext}"
            p.write_bytes(data)
            paths.append(p)
        return paths

    # -- process ----------------------------------------------------------------
    async def _subprocess_spawn(self, argv: list[str], cwd: str, env: dict[str, str]):
        stdin = asyncio.subprocess.PIPE if self.mode == "persistent" else asyncio.subprocess.DEVNULL
        return await asyncio.create_subprocess_exec(
            *argv, cwd=cwd, env=env, limit=STDOUT_LINE_LIMIT,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdin=stdin,
        )

    @staticmethod
    async def _readline(proc) -> bytes:
        """One stream-json line. A line past STDOUT_LINE_LIMIT is dropped
        rather than raised: the stream is still usable, and losing one
        event beats losing the turn."""
        while True:
            try:
                return await proc.stdout.readline()
            except ValueError:
                log.warning("dropped a stream line over %d bytes", STDOUT_LINE_LIMIT)

    async def _start(self, argv: list[str]):
        env = {
            **os.environ, **self.env(),
            "VERONICA_GATE_SOCK": str(self.s.gate_socket),
            "VERONICA_BRAIN": self.name,
            "VERONICA_HOOK_LOG": str(self.hook_log),
        }
        log.info("%s: spawning %s", self.name, argv)
        proc = await self._spawn(argv, str(self.workspace), env)
        self._proc = proc
        self._stderr_tail = ""
        self._stderr_task = asyncio.create_task(self._drain_stderr(proc))
        return proc

    async def _drain_stderr(self, proc) -> None:
        """Copy the child's stderr to workspace/last-stderr.log and keep a tail for error messages."""
        try:
            with open(self.workspace / "last-stderr.log", "wb") as f:
                while True:
                    chunk = await proc.stderr.readline()
                    if not chunk:
                        break
                    f.write(chunk)
                    self._stderr_tail = (self._stderr_tail + chunk.decode(errors="replace"))[-2000:]
        except Exception:
            log.debug("stderr drain ended", exc_info=True)

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        self._proc_native = None
        if proc is None:
            return
        try:
            if proc.returncode is None:
                proc.kill()
        except ProcessLookupError:
            pass
        try:
            async with asyncio.timeout(self.s.interrupt_drain_s):
                await proc.wait()
        except (TimeoutError, Exception):
            pass

    async def _reap(self) -> None:
        """per_turn: the child is done; wait for it and forget it."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            async with asyncio.timeout(self.s.interrupt_drain_s):
                await proc.wait()
        except TimeoutError:
            proc.kill()

    async def _ensure_persistent(self, native: bool):
        """persistent mode: the live child, spawned (or respawned after a
        kill / native-tools change) and handshaken — its first events
        must carry the `Session`, which is saved and announced before the
        first prompt goes in."""
        if self._proc is not None and self._proc_native == native:
            return self._proc
        await self._kill()
        sid = self._load_session()
        try:
            proc = await self._handshake(self.argv("", sid, [], native))
        except Exception:
            if sid is None:
                raise
            log.warning("%s: resume of %s failed; starting fresh", self.name, sid)
            self._clear_session()
            proc = await self._handshake(self.argv("", None, [], native))
        self._proc_native = native
        return proc

    async def _handshake(self, argv: list[str]):
        proc = await self._start(argv)
        try:
            async with asyncio.timeout(self.s.brain_timeout_s):
                while True:
                    raw = await self._readline(proc)
                    if not raw:
                        raise RuntimeError(f"{self.label} exited before announcing a session: {self._stderr_tail[-300:]}")
                    for ev in self._parse_line(raw):
                        if isinstance(ev, Session):
                            self._save_session(ev.id)
                            self.session_started(ev.id)
                            return proc
                        if isinstance(ev, Error):
                            raise RuntimeError(ev.message)
        except BaseException:
            await self._kill()
            raise

    def _parse_line(self, raw: bytes) -> list[Event]:
        line = raw.decode(errors="replace").strip()
        if not line:
            return []
        try:
            return self.parse(line)
        except Exception:
            log.warning("%s: unparsable line: %.200s", self.name, line, exc_info=True)
            return []

    # -- canary -----------------------------------------------------------------
    async def _trip_canary(self, key: str, out: _Outcome) -> None:
        log.error("turn ended early: reason=canary detail=%s: hook never logged native call %r", self.name, key)
        await self._kill()
        out.kind = "canary"

    async def _hook_logged(self, key: str, since: float, grace: float | None = None) -> bool:
        deadline = self._clock() + (self.canary_grace_s if grace is None else grace)
        while True:
            try:
                for line in self.hook_log.read_text().splitlines():
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    if float(entry.get("ts", 0)) >= since and self.canary_matches(str(entry.get("key", "")), key):
                        return True
            except FileNotFoundError:
                pass
            if self._clock() >= deadline:
                return False
            await asyncio.sleep(0.05)

    # -- one turn ---------------------------------------------------------------
    async def _turn(self, text: str, images: tuple[bytes, ...], native: bool, out: _Outcome) -> AsyncIterator[str]:
        """Run one turn, yielding sentences as they arrive; `out.kind` says
        how it ended (ok / canary / limit / error)."""
        splitter = SentenceSplitter()
        image_paths = self._write_images(images)
        self.prepare_workspace(self._system_prompt(), native)
        self.hook_log.write_text("")
        turn_start = time.time()   # wall clock: the hook stamps its lines with time.time()
        self._interrupted = False
        try:
            if self.mode == "persistent":
                proc = await self._ensure_persistent(native)
                proc.stdin.write((self.turn_message(text, image_paths) + "\n").encode())
                await proc.stdin.drain()
            else:
                proc = await self._start(self.argv(text, self._load_session(), image_paths, native))
        except Exception:
            log.exception("turn ended early: reason=error detail=%s: could not start", self.name)
            await self._kill()
            out.kind = "error"
            yield f"{self.label} returned an error, check the log."
            return

        pending: dict[str, str] = {}   # call_id -> canary key, native non-read-only calls in flight
        running: set[str] = set()      # call_id of every tool call in flight, ours or native
        saw_text = False
        while True:
            # brain_timeout_s is for the model going quiet. A tool call in
            # flight, or the gate asking/running something, is quiet too —
            # and killing the child there dropped the task part-way.
            try:
                raw = await self.gate.wait_quiet(self._readline(proc), self.s.brain_timeout_s,
                                                 tool_running=bool(running))
            except ToolStall:
                log.warning("turn ended early: reason=tool_stall detail=%s: tool call(s) %s still running after %ss",
                            self.name, sorted(running), self.gate.busy_ceiling_s)
                await self._kill()
                out.kind = "error"
                yield "That step never finished, so I stopped."
                return
            except TimeoutError:
                log.warning("turn ended early: reason=brain_timeout detail=%s: no output for %ss",
                            self.name, self.s.brain_timeout_s)
                await self._kill()
                out.kind = "error"
                yield "Taking too long, cancelled."
                return

            if not raw:
                if self._interrupted:
                    return
                await self._reap()
                rc = getattr(proc, "returncode", 0) or 0
                if rc != 0:
                    log.error("%s: exited %s without a result: %s", self.name, rc, self._stderr_tail[-500:])
                    async for sent in self._on_error(Error(self._stderr_tail or f"exit {rc}"), out):
                        yield sent
                    return
                for sent in splitter.flush():
                    yield sent
                return

            for ev in self._parse_line(raw):
                if isinstance(ev, Text):
                    saw_text = True
                    for sent in splitter.feed(ev.delta):
                        yield sent
                elif isinstance(ev, ToolStart):
                    running.add(ev.call_id)
                    if not ev.native:
                        continue       # our MCP tool: tools.serve gated it and carded it
                    if ev.tool in hook.READONLY_TOOLS:
                        if self._on_tool:
                            self._on_tool(self.readonly_summary(ev.tool, ev.input), "auto")
                    elif native:
                        # With native tools off there is nothing for the hook to
                        # gate: the CLI's own read-only/deny mode is the
                        # enforcement, and tripping here would kill the fallback
                        # turn and leave the user with no answer at all.
                        key = pending[ev.call_id] = self.native_key(ev.tool, ev.input)
                        if not await self._hook_logged(key, turn_start, self.canary_start_grace_s):
                            await self._trip_canary(key, out)
                            return
                elif isinstance(ev, ToolEnd):
                    running.discard(ev.call_id)
                    key = pending.pop(ev.call_id, None)
                    if key is not None and not await self._hook_logged(key, turn_start):
                        await self._trip_canary(key, out)
                        return
                elif isinstance(ev, Session):
                    self._save_session(ev.id)
                elif isinstance(ev, Error):
                    await self._kill()
                    async for sent in self._on_error(ev, out):
                        yield sent
                    return
                elif isinstance(ev, Done):
                    if not saw_text and ev.final_text:
                        for sent in splitter.feed(ev.final_text):
                            yield sent
                    for sent in splitter.flush():
                        yield sent
                    if self.mode == "per_turn":
                        await self._reap()
                    return

    async def _on_error(self, ev: Error, out: _Outcome) -> AsyncIterator[str]:
        msg = ev.message.lower()
        if any(m in msg for m in LIMIT_MARKERS):
            log.warning("turn ended early: reason=limit detail=%s: %s", self.name, ev.message)
            out.kind, out.message = "limit", ev.message
            return
        if any(m in msg for m in OVERFLOW_MARKERS):
            # Same substring heuristic as ClaudeBrain: a false positive only
            # costs the conversation history.
            log.warning("turn ended early: reason=overflow detail=%s: context overflow (session %s): %s",
                        self.name, self._load_session(), ev.message)
            self._clear_session()
            out.kind = "error"
            yield "My memory got full, starting a fresh conversation."
            return
        log.error("turn ended early: reason=error detail=%s: error result: %s", self.name, ev.message)
        out.kind = "error"
        yield f"{self.label} returned an error, check the log."

    # -- public ---------------------------------------------------------------
    async def ask(self, text: str, images: list[bytes] = ()) -> AsyncIterator[str]:
        self.pending_redirect = None   # a redirect belongs to the turn it was said in
        native = self.native_tools_enabled()
        for attempt in range(2):        # second pass = canary fallback, tools off
            out = _Outcome()
            async for sent in self._turn(text, tuple(images), native, out):
                yield sent
            if out.kind == "ok":
                return
            if out.kind == "canary" and attempt == 0:
                yield f"Hooks aren't running on {self.label}, so I've turned off its shell. Tools still work."
                prefs.save_settings_override(f"{self.name}_native_tools", False)
                try:
                    setattr(self.s, f"{self.name}_native_tools", False)
                except (ValueError, AttributeError):      # not a Settings field (tests)
                    object.__setattr__(self.s, f"{self.name}_native_tools", False)
                native = False
                continue
            if out.kind == "limit":
                raise LimitError(out.message)
            return

    async def interrupt(self) -> None:
        """Stop the child: SIGINT, a short wait, then kill. The next ask()
        respawns (resuming the saved session). Safe when idle."""
        proc = self._proc
        if proc is None:
            return
        self._interrupted = True
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            async with asyncio.timeout(self.s.interrupt_drain_s):
                await proc.wait()
        except TimeoutError:
            log.warning("%s: no exit %ss after SIGINT; killing", self.name, self.s.interrupt_drain_s)
        except Exception:
            log.exception("%s: wait after SIGINT failed", self.name)
        await self._kill()

    async def close(self) -> None:
        await self.interrupt()
