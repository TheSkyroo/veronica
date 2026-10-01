"""Risk classifier for tool calls: decide whether a call runs without asking,
and which calls are asked about every single time (`always_confirm`).

The brains' shell on Windows is PowerShell (Codex, Copilot, Antigravity)
or Git Bash (Claude Code), and a command may also reach cmd.exe. The
tables below hold all three. Windows command names are case-insensitive
and `.exe` is optional, so a command head is compared as its lowercased
basename without `.exe` (`C:\\Windows\\System32\\SHUTDOWN.EXE` is `shutdown`)."""
import ipaddress
import ntpath
import shlex
from collections.abc import Callable
from typing import Literal
from urllib.parse import urlsplit

from veronica.tools.computer_events import Front, is_system_dialog, normalize_combo

Decision = Literal["allow", "confirm"]

ALLOW_TOOLS = frozenset({"Read", "Glob", "Grep", "WebSearch", "WebFetch"})

# Read-only commands, by normalized head (see `command_head`): Git Bash
# coreutils, PowerShell cmdlets and their aliases, and cmd/Windows tools
# whose every argument only reads. Nothing here may write, delete or
# change state whatever it is given (so not ipconfig: /release, /flushdns).
SAFE_BASH = frozenset({
    # Git Bash
    "ls", "cat", "head", "tail", "pwd", "date", "cal", "whoami", "df",
    "du", "ps", "which", "echo", "uptime", "wc", "file", "stat",
    # PowerShell (cmdlets and their read-only aliases)
    "get-childitem", "gci", "dir", "get-content", "gc", "type",
    "get-date", "get-location", "gl", "get-clipboard", "get-process", "gps",
    "get-item", "gi", "test-path", "resolve-path", "get-command", "gcm",
    "get-uptime", "get-computerinfo", "write-output", "write-host",
    # Windows / cmd
    "hostname", "tasklist", "ver", "systeminfo", "where",
})

# any of these anywhere in a shell command → confirm. Covers pipes, chains,
# redirects and substitution in all three shells: $(…) and `…` (bash),
# (…), {…}, $var and the ` escape (PowerShell), %VAR% and the ^ escape (cmd).
FORBIDDEN = "|&;><`$\n\r(){}%^"

CURL_SAFE_LONG_NO_ARG = frozenset({"--silent", "--location", "--compressed", "--fail"})
CURL_SAFE_SHORT_CHARS = frozenset("sSLf")
CURL_SAFE_WITH_ARG = frozenset({
    "--max-time", "-m", "-H", "--header", "-A", "--user-agent",
})
CURL_HEADER_NAME_ALLOW = frozenset({
    "accept", "accept-language", "accept-encoding", "user-agent", "cache-control",
})

# Per-in-process-MCP-server risk table, keyed by server name; tool calls
# from server `X` arrive as `mcp__X__<tool>` (MCP_PREFIX_FMT).
MCP_TOOL_RISK: dict[str, dict[str, Decision]] = {
    "system": {
        "open_app": "allow",
        "open_url": "allow",
        "clipboard_read": "allow",
        "clipboard_write": "confirm",
        "notify": "allow",
        "volume_get": "allow",
        "volume_set": "allow",
        # Arbitrary script with the user's rights: confirm, and
        # `always_confirm` keeps it asked every single time.
        "powershell": "confirm",
    },
    "pim": {
        "calendar_events": "allow",
        "calendar_create": "confirm",
        "mail_unread": "allow",
        "mail_search": "allow",
        # Listed for completeness; `always_confirm` below keeps it asked
        # every time, whatever this says.
        "mail_send": "confirm",
        "reminder_create": "confirm",
        "reminders_due": "allow",
        # Append-only and harmless (a mistaken note is trivially deleted in
        # Outlook), so — unlike calendar_create/mail_send/reminder_create
        # — this doesn't need a confirm gate.
        "notes_create": "allow",
        "timer_set": "allow",
        "timer_list": "allow",
        "timer_cancel": "allow",
    },
    "memory": {
        "recall": "allow",
        "facts_list": "allow",
        # A fact persists across every future session (it's injected into
        # the system prompt of every new client), unlike a normal reply, so
        # it gets the same confirm gate as anything else that changes
        # standing state rather than just answering the current turn.
        "fact_add": "confirm",
        "fact_delete": "confirm",
    },
    "screen": {
        # Read-only and local: no network, no file changes outside our own
        # scratch dir.
        "screenshot": "allow",
    },
    "music": {
        "music_play": "allow",
        "music_pause": "allow",
        "music_next": "allow",
        "music_prev": "allow",
        "music_now_playing": "allow",
        "music_volume": "allow",
    },
    "browser": {
        # Read-only / navigation: same risk as system.open_url.
        "browser_tabs": "allow",
        "browser_open": "allow",
        "browser_read": "allow",
        "browser_find": "allow",
        "browser_scroll": "allow",
        "browser_back": "allow",
        # Acts inside the user's logged-in session: confirm.
        "browser_click": "confirm",
        "browser_type": "confirm",
    },
    "computer": {
        # Pointer moves, scrolling and OCR change nothing on their own.
        "computer_move": "allow",
        "computer_scroll": "allow",
        "computer_find": "allow",
        # Anything that presses a button or key acts in the frontmost app:
        # confirm (the trust window in Brain._can_use_tool may auto-allow a
        # follow-up in the same app for a short while).
        "computer_click": "confirm",
        "computer_click_text": "confirm",
        "computer_drag": "confirm",
        "computer_type": "confirm",
        "computer_key": "confirm",
    },
}

MCP_PREFIX_FMT = "mcp__{server}__"

# Back-compat aliases (kept for anything still importing the old names).
SYSTEM_TOOL_RISK: dict[str, Decision] = MCP_TOOL_RISK["system"]
SYSTEM_PREFIX = MCP_PREFIX_FMT.format(server="system")


def _is_safe_short_combo(tok: str) -> bool:
    # e.g. -s, -sS, -sL, -fsSL — a single dash followed only by chars from
    # {s,S,L,f}. Anything else attached to a dash (e.g. -m5, -H"x") is NOT
    # matched here and falls through to "confirm".
    return (
        tok.startswith("-") and not tok.startswith("--")
        and len(tok) > 1
        and all(c in CURL_SAFE_SHORT_CHARS for c in tok[1:])
    )


def _curl_header_name_allowed(value: str) -> bool:
    name = value.split(":", 1)[0].strip().lower()
    return name in CURL_HEADER_NAME_ALLOW


_PRIVATE_LOCAL_HOSTNAMES = frozenset({"localhost"})


def _is_public_https_url(url: str) -> bool:
    """True iff `url` is a well-formed http(s) URL with no embedded
    credentials and a host that isn't loopback/private/link-local/metadata."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    if "@" in parts.netloc:
        return False  # userinfo (credentials) in the URL
    if not host:
        return False
    host = host.lower()
    if ":" in host:
        return False  # bracketed IPv6 literal — always confirm
    if host in _PRIVATE_LOCAL_HOSTNAMES or host.endswith(".localhost"):
        return False
    try:
        ip = ipaddress.IPv4Address(host)
    except ValueError:
        ip = None
    if ip is not None and (
        ip.is_loopback or ip.is_private or ip.is_link_local
        or ip.is_reserved or ip.is_unspecified or ip.is_multicast
    ):
        return False
    return True


def _curl_is_safe(argv: list[str]) -> bool:
    urls = 0
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok.startswith(("http://", "https://")):
            if not _is_public_https_url(tok):
                return False
            urls += 1
            i += 1
            continue
        if tok in CURL_SAFE_LONG_NO_ARG or _is_safe_short_combo(tok):
            i += 1
            continue
        if tok in CURL_SAFE_WITH_ARG:
            if i + 1 >= len(argv):
                return False
            val = argv[i + 1]
            if val.startswith("@"):
                return False  # -H @file / -A @file reads a local file
            if tok in ("-H", "--header") and not _curl_header_name_allowed(val):
                return False
            i += 2
            continue
        return False
    return urls == 1


def split_command(command: str) -> list[str]:
    """Tokens of one shell command. Quotes group, but a backslash is a
    plain character — it is the path separator here, not an escape (the
    shells' own escapes, ` and ^, never reach a parse that matters:
    FORBIDDEN refuses them for the allow path, and an unparsable command
    always confirms). Raises ValueError on an unbalanced quote."""
    lex = shlex.shlex(command, posix=True)
    lex.whitespace_split = True
    lex.escape = ""
    lex.commenters = ""
    return list(lex)


def command_head(token: str) -> str:
    """`C:\\Windows\\System32\\Shutdown.EXE` -> `shutdown`: the basename
    (either separator), lowercased, `.exe` dropped."""
    return ntpath.basename(token).lower().removesuffix(".exe")


def _bash_is_safe(command: str) -> bool:
    if not command or any(ch in command for ch in FORBIDDEN):
        return False
    try:
        argv = split_command(command)
    except ValueError:
        return False
    if not argv:
        return False
    head = command_head(argv[0])
    if head == "curl":
        return _curl_is_safe(argv)
    if head in ("start-process", "saps", "start"):
        # Opening a web page in the default browser: same as open_url.
        return len(argv) == 2 and argv[1].startswith(("http://", "https://"))
    return head in SAFE_BASH


# Confirm-class tools the user may switch off one by one, in Settings ›
# Brain or by answering a confirm with "always". Nothing outside this set
# can ever be auto-allowed, however it gets into the setting: sending
# mail, PowerShell, screen control and the shell always ask.
AUTO_ALLOWABLE = frozenset({
    "mcp__system__clipboard_write",
    "mcp__pim__calendar_create",
    "mcp__pim__reminder_create",
    "mcp__memory__fact_add",
    "mcp__memory__fact_delete",
    "mcp__browser__browser_click",
    "mcp__browser__browser_type",
})


def _auto_allowed(tool_name: str, auto_allow) -> bool:
    """True iff the user listed `tool_name` AND it is one of the tools that
    may be on the list at all."""
    if tool_name not in AUTO_ALLOWABLE:
        return False
    return any(str(entry or "").strip() == tool_name for entry in auto_allow or ())


def classify(tool_name: str, tool_input: dict, auto_allow=()) -> Decision:
    """`auto_allow` is Settings.auto_allow_tools — confirm-class tools the
    user has approved for good. Only AUTO_ALLOWABLE names count, and
    `always_confirm` is consulted first, so anything on the never-list that
    was hand-typed into the setting still gets its own yes/no."""
    decision = _classify(tool_name, tool_input)
    if (
        decision == "confirm"
        and _auto_allowed(tool_name, auto_allow)
        and not always_confirm(tool_name, tool_input)
    ):
        return "allow"
    return decision


def _classify(tool_name: str, tool_input: dict) -> Decision:
    if tool_name in ALLOW_TOOLS:
        return "allow"
    if tool_name == "Bash":
        return "allow" if _bash_is_safe(str(tool_input.get("command", ""))) else "confirm"
    if tool_name.startswith("mcp__"):
        rest = tool_name[len("mcp__"):]
        server, _, short = rest.partition("__")
        risk_table = MCP_TOOL_RISK.get(server)
        if risk_table is not None:
            return risk_table.get(short, "confirm")
    return "confirm"


# --- always-confirm: asked every time, whatever else is going on ----------------
# The confirm gate has two ways of skipping the question — the screen-control
# trust window and pre-approval by request wording ("just do it"). Nothing in
# this table is ever covered by either: it sends something, destroys
# something, or presses Enter/submits in a place where that runs a command.

# Apps where the trust window never opens and never applies: a click or
# keystroke in a terminal runs whatever is on the prompt line, so every
# screen action there is confirmed on its own. Lowercase executable names,
# what `Front.bundle_id` holds on Windows (see `is_terminal`).
TRUST_EXCLUDED_BUNDLES = frozenset({
    "windowsterminal.exe",
    "wt.exe",
    "openconsole.exe",
    "conhost.exe",
    "cmd.exe",
    "powershell.exe",
    "powershell_ise.exe",
    "pwsh.exe",
    "mintty.exe",           # Git Bash, MSYS2, Cygwin
    "bash.exe",
    "wsl.exe",
    "wslhost.exe",
    "alacritty.exe",
    "wezterm-gui.exe",
    "conemu64.exe",
    "conemu.exe",
    "tabby.exe",
    "hyper.exe",
    "warp.exe",
    "putty.exe",
})
_ENTER_KEYS = frozenset({"enter", "return"})


def is_terminal(front: Front) -> bool:
    """Is `front` a terminal? Compared on the executable's lowercased
    basename, with or without `.exe`."""
    name = ntpath.basename(front.bundle_id or "").lower()
    if not name:
        return False
    return (name if name.endswith(".exe") else name + ".exe") in TRUST_EXCLUDED_BUNDLES


def _ps_switch(tok: str, name: str, min_len: int = 1) -> bool:
    """Is `tok` PowerShell's `-<name>` switch? PowerShell takes any unique
    prefix and `-Name:$true`, so `-r`, `-Rec` and `-recurse:$true` are all
    -Recurse. `min_len` is the shortest prefix taken as this switch."""
    if not tok.startswith("-") or tok.startswith("--"):
        return False
    word = tok[1:].split(":", 1)[0].lower()
    return len(word) >= min_len and name.startswith(word)


def _any_switch(argv: list[str], *names: tuple[str, int]) -> bool:
    return any(_ps_switch(tok, n, m) for tok in argv for n, m in names)


def _slash(argv: list[str], *flags: str) -> bool:
    """A cmd.exe-style `/x` flag (case-insensitive)."""
    wanted = {f.lower() for f in flags}
    return any(tok.lower() in wanted for tok in argv)


_RECURSE_FORCE = (("recurse", 1), ("force", 2))
_REGISTRY = ("hklm:", "hkcu:", "hkcr:", "hku:", "hkcc:", "registry::")


def _touches_registry(argv: list[str]) -> bool:
    return any(tok.lower().startswith(_REGISTRY) for tok in argv)


def _rm(argv: list[str]) -> bool:
    """`rm` is coreutils in Git Bash and Remove-Item in PowerShell."""
    return (any(_has_flag_char(tok, "rR") for tok in argv) or _any_switch(argv, *_RECURSE_FORCE)
            or _touches_registry(argv))


# Shell: command head (`command_head`) -> predicate on the rest of that
# command's argv. Checked per command — a chain like `cd x; rm -r y` is
# split on the chain/pipe/grouping characters first, so `rm` is seen as a
# head.
_ALWAYS_BASH: dict[str, Callable[[list[str]], bool]] = {
    # Deleting trees.
    "rm": _rm,
    "remove-item": lambda argv: _any_switch(argv, *_RECURSE_FORCE) or _touches_registry(argv),
    "ri": lambda argv: _any_switch(argv, *_RECURSE_FORCE) or _touches_registry(argv),
    "del": lambda argv: _slash(argv, "/s", "/q") or _any_switch(argv, *_RECURSE_FORCE),
    "erase": lambda argv: _slash(argv, "/s", "/q") or _any_switch(argv, *_RECURSE_FORCE),
    "rd": lambda argv: _slash(argv, "/s", "/q") or _any_switch(argv, *_RECURSE_FORCE),
    "rmdir": lambda argv: True,
    "clear-recyclebin": lambda argv: True,
    "cipher": lambda argv: any(tok.lower().startswith("/w") for tok in argv),
    "git": lambda argv: bool(argv) and argv[0] == "push"
        and any(tok in ("-f", "--force", "--force-with-lease") or tok.startswith("--force-with-lease=")
                for tok in argv[1:]),
    # Power and session.
    "shutdown": lambda argv: True,
    "stop-computer": lambda argv: True,
    "restart-computer": lambda argv: True,
    "logoff": lambda argv: True,
    "rundll32": lambda argv: True,       # powrprof SetSuspendState, user32 LockWorkStation, anything
    # Privilege.
    "sudo": lambda argv: True,
    "gsudo": lambda argv: True,
    "runas": lambda argv: True,
    "start-process": lambda argv: any(tok.lower().strip("'\"") == "runas" for tok in argv),
    "saps": lambda argv: any(tok.lower().strip("'\"") == "runas" for tok in argv),
    "start": lambda argv: any(tok.lower().strip("'\"") == "runas" for tok in argv),
    "set-executionpolicy": lambda argv: True,
    # Registry and system configuration.
    "reg": lambda argv: bool(argv) and argv[0].lower() in ("add", "delete", "import", "restore", "load",
                                                           "unload", "copy"),
    "set-itemproperty": _touches_registry,
    "sp": _touches_registry,
    "new-itemproperty": _touches_registry,
    "remove-itemproperty": _touches_registry,
    "rp": _touches_registry,
    "new-item": _touches_registry,
    "ni": _touches_registry,
    "set-item": _touches_registry,
    "si": _touches_registry,
    "set-mppreference": lambda argv: True,
    "add-mppreference": lambda argv: True,
    "netsh": lambda argv: any(tok.lower() in ("set", "add", "delete", "reset") for tok in argv),
    "net": lambda argv: bool(argv) and argv[0].lower() in ("user", "localgroup", "accounts", "stop", "share"),
    "schtasks": lambda argv: _slash(argv, "/create", "/delete", "/change"),
    "register-scheduledtask": lambda argv: True,
    "unregister-scheduledtask": lambda argv: True,
    "sc": lambda argv: bool(argv) and argv[0].lower() in ("config", "delete", "create"),
    "takeown": lambda argv: True,
    "icacls": lambda argv: True,
    # Disks and boot.
    "format": lambda argv: True,
    "format-volume": lambda argv: True,
    "clear-disk": lambda argv: True,
    "initialize-disk": lambda argv: True,
    "remove-partition": lambda argv: True,
    "diskpart": lambda argv: True,
    "bcdedit": lambda argv: True,
    "vssadmin": lambda argv: True,
    "wmic": lambda argv: True,
    # Killing processes by force.
    "stop-process": lambda argv: _any_switch(argv, ("force", 1)),
    "spps": lambda argv: _any_switch(argv, ("force", 1)),
    "kill": lambda argv: _any_switch(argv, ("force", 1)) or any(tok in ("-9", "-KILL", "-SIGKILL") for tok in argv),
    "taskkill": lambda argv: _slash(argv, "/f"),
    # Wrappers that hide the real command from this table: whatever they run
    # is opaque here, so they always get their own yes/no.
    "invoke-expression": lambda argv: True,
    "iex": lambda argv: True,
    "invoke-command": lambda argv: True,
    "icm": lambda argv: True,
    "powershell": lambda argv: True,
    "pwsh": lambda argv: True,
    "cmd": lambda argv: True,
    "wsl": lambda argv: True,
    "sh": lambda argv: "-c" in argv,
    "bash": lambda argv: "-c" in argv,
    "python": lambda argv: "-c" in argv,
    "python3": lambda argv: "-c" in argv,
    "py": lambda argv: "-c" in argv,
    "perl": lambda argv: "-e" in argv,
    "ruby": lambda argv: "-e" in argv,
    "node": lambda argv: "-e" in argv,
    "xargs": lambda argv: True,
    "eval": lambda argv: True,
    "exec": lambda argv: True,
    "find": lambda argv: any(tok in ("-delete", "-exec", "-execdir", "-ok") for tok in argv),
}
# Chain, pipe and grouping characters of bash, PowerShell and cmd: each
# piece between them is judged as a command of its own, so a scriptblock
# or a subexpression can't hide one.
_CHAIN_CHARS = "|&;\n\r(){}"


def _has_flag_char(tok: str, chars: str) -> bool:
    """`-r`, `-rf`, `-fR`… — a short-flag cluster containing one of `chars`."""
    return tok.startswith("-") and not tok.startswith("--") and any(c in tok[1:] for c in chars)


def _bash_always_confirms(command: str) -> bool:
    for part in _split_chain(command):
        try:
            argv = split_command(part)
        except ValueError:
            return True   # unparsable: assume the worst
        # PowerShell's call operator and a leading `$` (`$(...)` was split
        # off already) don't change which command runs.
        while argv and argv[0] in ("&", ".", "$", "@"):
            argv = argv[1:]
        if not argv:
            continue
        check = _ALWAYS_BASH.get(command_head(argv[0]))
        if check is not None and check(argv[1:]):
            return True
    return False


# Tools that put something in front of another person. Named here so the
# rule is explicit, with the pattern below still catching future ones.
SEND_TOOLS = frozenset({"mail_send"})


def _is_send_tool(server: str, short: str) -> bool:
    """mail_send today; any future messages/mail send on any server."""
    if short in SEND_TOOLS:
        return True
    return (short.endswith("_send") and "message" in short) or (server == "messages" and short == "send")


def _split_chain(command: str) -> list[str]:
    parts, cur = [], []
    for ch in command:
        if ch in _CHAIN_CHARS:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return parts


def always_confirm(tool_name: str, tool_input: dict, front: Front | None = None) -> bool:
    """True for calls that get their own yes/no no matter what: sending
    mail, destructive or power/privilege/registry shell commands, every
    PowerShell script, and screen actions that press Enter, type into a
    terminal, or touch a system dialog. `front` is the
    frontmost app for computer tools (None: unknown, only the input is
    judged)."""
    if tool_name == "Bash":
        return _bash_always_confirms(str(tool_input.get("command", "")))
    if not tool_name.startswith("mcp__"):
        return False
    rest = tool_name[len("mcp__"):]
    server, _, short = rest.partition("__")
    if _is_send_tool(server, short):
        return True
    if server == "system" and short == "powershell":
        return True
    if server == "computer":
        if front is not None and is_system_dialog(front):
            return True
        if short == "computer_type":
            if tool_input.get("submit"):
                return True
            return front is not None and is_terminal(front)
        if short == "computer_key":
            try:
                return normalize_combo(str(tool_input.get("combo") or "")) in _ENTER_KEYS
            except ValueError:
                return False
    return False
