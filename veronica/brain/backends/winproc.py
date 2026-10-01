"""Windows process plumbing for the vendor-CLI brains: finding the real
program behind an npm shim, spawning it so it can be interrupted, killing
its whole tree, and writing hook command lines that survive a shell.

Shims. `npm i -g` puts a `codex.cmd` / `copilot.cmd` (plus `.ps1` and a
sh script) on PATH. Running a `.cmd` means running cmd.exe, and cmd.exe
re-parses the command line with its own rules: a prompt containing `&`,
`|`, `%VAR%` or `"` can run something else entirely (the "BatBadBut"
class of bug; CPython does not escape for it). The prompt is the user's
speech, so it never goes through cmd.exe: `resolve_cli` reads the shim,
finds the JavaScript entry point it would have started, and runs
`node <script.js> <args>` directly (the shim's own `node.exe` beside it
when there is one, else `node` on PATH). The packages' entry scripts
start their native binaries themselves with an argument array, never a
shell. A real `.exe` on PATH (a native install, `agy`) is run as is. A
`.cmd`/`.bat` we cannot see through is refused rather than run.

Interrupt. Windows has no SIGINT for another process. The child is
started in its own process group (CREATE_NEW_PROCESS_GROUP), so
CTRL_BREAK_EVENT reaches it and nothing else; that only works when we
share a console with it, so when Veronica runs windowless the event
fails and the caller goes straight to `kill_tree`. `kill_tree` uses
taskkill /T: the CLIs start helpers (the native binary under node, MCP
servers, shells) that a plain TerminateProcess would orphan."""
import ctypes
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger("veronica.brain")

# getattr: these only exist on Windows; the values are the Win32 ones.
CREATE_NEW_PROCESS_GROUP: int = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
CREATE_NO_WINDOW: int = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
CTRL_BREAK_EVENT: int = getattr(signal, "CTRL_BREAK_EVENT", 1)

# `"%dp0%\node_modules\@openai\codex\bin\codex.js"` in a cmd-shim .cmd,
# `"$basedir/node_modules/@github/copilot/index.js"` in its .ps1 twin.
_CMD_SCRIPT = re.compile(r'"%~?dp0%\\?([^"%*]+?\.[cm]?js)"', re.IGNORECASE)
_PS1_SCRIPT = re.compile(r'"\$basedir[\\/]([^"$]+?\.[cm]?js)"', re.IGNORECASE)


class UnsafeShim(RuntimeError):
    """A batch file we would have to run through cmd.exe to start."""


def _has_console() -> bool:
    try:
        return bool(ctypes.windll.kernel32.GetConsoleWindow())   # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return True


def console_python(executable: str | None = None) -> str:
    """The interpreter for a child that speaks over stdio (the hook, an MCP
    server): `python.exe`, never `pythonw.exe`. Veronica itself usually
    runs windowless under pythonw, whose stdin/stdout are None — a hook or
    MCP server started with it would have nothing to talk on."""
    exe = Path(executable or sys.executable)
    if exe.name.lower() == "pythonw.exe":
        console = exe.with_name("python.exe")
        if console.is_file():
            return str(console)
    return str(exe)


def spawn_flags(has_console: Callable[[], bool] = _has_console) -> int:
    """creationflags for a CLI child: its own process group (so a
    CTRL_BREAK can be aimed at it), and no console window popping up when
    Veronica itself runs without one."""
    if os.name != "nt":
        return 0
    return CREATE_NEW_PROCESS_GROUP | (0 if has_console() else CREATE_NO_WINDOW)


def no_window_flags() -> int:
    """creationflags for a short helper (taskkill, `agy mcp add`)."""
    return CREATE_NO_WINDOW if os.name == "nt" else 0


def _shim_script(shim: Path) -> Path | None:
    try:
        text = shim.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    pattern = _PS1_SCRIPT if shim.suffix.lower() == ".ps1" else _CMD_SCRIPT
    m = pattern.search(text)
    if not m:
        return None
    script = shim.parent / m.group(1).replace("\\", os.sep).replace("/", os.sep)
    return script if script.is_file() else None


def resolve_cli(argv: list[str], *, which: Callable[[str], str | None] = shutil.which) -> list[str]:
    """`argv` with argv[0] replaced by what to actually execute (see the
    module docstring). A name that isn't on PATH is left alone, so the
    spawn fails with the usual FileNotFoundError."""
    if not argv:
        return list(argv)
    found = which(argv[0])
    if found is None:
        return list(argv)
    exe = Path(found)
    if not exe.suffix and exe.with_suffix(".cmd").is_file():
        exe = exe.with_suffix(".cmd")         # npm's sh shim found instead of its .cmd twin
    if exe.suffix.lower() not in (".cmd", ".bat", ".ps1"):
        return [str(exe), *argv[1:]]
    script = _shim_script(exe)
    if script is None:
        raise UnsafeShim(f"{exe} is a batch shim with no script Veronica can run directly")
    local_node = exe.parent / "node.exe"
    node = str(local_node) if local_node.is_file() else which("node")
    if node is None:
        raise UnsafeShim(f"{exe} needs node, and node isn't on PATH")
    return [node, str(script), *argv[1:]]


def interrupt(proc) -> bool:
    """Send the child CTRL_BREAK. False when it can't be delivered (no
    shared console, already gone): kill it instead."""
    try:
        proc.send_signal(CTRL_BREAK_EVENT)
        return True
    except (OSError, ValueError) as e:      # ProcessLookupError is an OSError
        log.debug("CTRL_BREAK not delivered: %s", e)
        return False


def kill_tree(proc, run: Callable = subprocess.run) -> None:
    """Kill the child and everything it started. taskkill first (it needs
    the parent alive to find the children), then the child itself in case
    taskkill wasn't there or didn't finish it."""
    pid = getattr(proc, "pid", None)
    if pid and getattr(proc, "returncode", None) is None:
        try:
            run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=5,
                creationflags=no_window_flags())
        except (OSError, subprocess.SubprocessError) as e:
            log.debug("taskkill failed: %s", e)
    try:
        if proc.returncode is None:
            proc.kill()
    except ProcessLookupError:
        pass


# -- hook command lines ----------------------------------------------------------
# The CLIs keep their pre-tool hook as ONE command string and hand it to a
# shell. Which shell is up to the CLI (cmd.exe, PowerShell), and the two
# disagree on quoting: `"C:\a b\python.exe" -m x` is fine in cmd and a
# parse error in PowerShell, `& 'C:\a b\python.exe'` the other way round.
# A command line without spaces needs no quotes in either, so every path is
# turned into its 8.3 short form when it has a space (C:\Users\MANIKU~1\...).
# Only when a volume has short names switched off does a path keep its
# space; it is then double-quoted, which suits cmd.exe.
_PLAIN = re.compile(r"^[A-Za-z0-9_\-+=.:/\\~]+$")


def _short_path_win(path: str) -> str:
    try:
        fn = ctypes.windll.kernel32.GetShortPathNameW   # type: ignore[attr-defined]
    except AttributeError:
        return path
    buf = ctypes.create_unicode_buffer(32768)
    n = fn(path, buf, len(buf))
    return buf.value if 0 < n < len(buf) else path


def short_path(path: str | os.PathLike, shorten: Callable[[str], str] | None = None) -> str:
    """`path` with every existing part in 8.3 form, the not-yet-existing
    tail (a log file not written yet) appended as is."""
    shorten = shorten or _short_path_win
    p = Path(path)
    tail: list[str] = []
    while not p.exists() and p.parent != p:
        tail.append(p.name)
        p = p.parent
    head = shorten(str(p)) if p.exists() else str(p)
    return str(Path(head, *reversed(tail)))


def neutral_arg(arg: str, shorten: Callable[[str], str] | None = None) -> str:
    """One hook-command token that reads the same to cmd.exe and
    PowerShell when at all possible (see above)."""
    if _PLAIN.match(arg):
        return arg
    if os.path.isabs(arg):
        short = short_path(arg, shorten)
        if _PLAIN.match(short):
            return short
        arg = short
    return subprocess.list2cmdline([arg])


def neutral_command(argv: list[str], shorten: Callable[[str], str] | None = None) -> str:
    return " ".join(neutral_arg(str(a), shorten) for a in argv)


def ps_quote(arg: str) -> str:
    """A PowerShell single-quoted (fully literal) string. PowerShell also
    takes the typographic quotes as quote characters, so those are doubled
    too."""
    return "'" + re.sub(r"(['\u2018\u2019\u201a\u201b])", r"\1\1", arg) + "'"


def ps_command(argv: list[str]) -> str:
    """`argv` as one PowerShell command: the call operator and every
    token literal, so nothing in a path is expanded or split."""
    return "& " + " ".join(ps_quote(str(a)) for a in argv)
