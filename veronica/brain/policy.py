"""Risk classifier for tool calls: decide whether a call runs without asking,
and which calls are asked about every single time (`always_confirm`)."""
import ipaddress
import os
import shlex
from collections.abc import Callable
from typing import Literal
from urllib.parse import urlsplit

from veronica.tools.computer_events import Front, is_system_dialog, normalize_combo

Decision = Literal["allow", "confirm"]

ALLOW_TOOLS = frozenset({"Read", "Glob", "Grep", "WebSearch", "WebFetch"})

SAFE_BASH = frozenset({
    "ls", "cat", "head", "tail", "pwd", "date", "cal", "whoami", "pbpaste", "df",
    "du", "ps", "which", "echo", "uptime", "wc", "file", "stat",
})

# any of these anywhere in a Bash command → confirm (covers $(…), pipes, chains, redirects)
FORBIDDEN = "|&;><`$\n"

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
    "mac": {
        "open_app": "allow",
        "open_url": "allow",
        "clipboard_read": "allow",
        "clipboard_write": "confirm",
        "notify": "allow",
        "volume_get": "allow",
        "volume_set": "allow",
        "applescript": "confirm",
    },
    "pim": {
        "calendar_events": "allow",
        "calendar_create": "confirm",
        "mail_unread": "allow",
        "mail_search": "allow",
        "mail_send": "confirm",
        "reminder_create": "confirm",
        "reminders_due": "allow",
        # Append-only and harmless (a mistaken note is trivially deleted in
        # Notes.app), so — unlike calendar_create/mail_send/reminder_create
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
        # Read-only / navigation: same risk as mac.open_url.
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
MAC_TOOL_RISK: dict[str, Decision] = MCP_TOOL_RISK["mac"]
MAC_PREFIX = MCP_PREFIX_FMT.format(server="mac")


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


def _bash_is_safe(command: str) -> bool:
    if not command or any(ch in command for ch in FORBIDDEN):
        return False
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    head = argv[0]
    if head == "curl":
        return _curl_is_safe(argv)
    if head == "open":
        if len(argv) == 3 and argv[1] == "-a":
            app_name = argv[2]
            # only allow bare app names: no "/" and doesn't start with "." or "-"
            if "/" not in app_name and not app_name.startswith((".", "-")):
                return True
            return False
        if len(argv) == 2 and argv[1].startswith(("http://", "https://")):
            return True
        return False
    return head in SAFE_BASH


def classify(tool_name: str, tool_input: dict) -> Decision:
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
# screen action there is confirmed on its own.
TRUST_EXCLUDED_BUNDLES = frozenset({
    "com.apple.Terminal",
    "com.googlecode.iterm2",
    "dev.warp.Warp-Stable",
    "net.kovidgoyal.kitty",
    "com.github.wez.wezterm",
    "io.alacritty",
    "com.mitchellh.ghostty",
})
_ENTER_KEYS = frozenset({"enter", "return"})

# Bash: command basename -> predicate on the rest of that command's argv.
# Checked per shell command — a chain like `cd x && rm -rf y` is split on
# the chain/pipe characters first, so `rm` is seen as a head.
_ALWAYS_BASH: dict[str, Callable[[list[str]], bool]] = {
    "rm": lambda argv: any(_has_flag_char(tok, "rR") for tok in argv),
    "git": lambda argv: bool(argv) and argv[0] == "push"
        and any(tok in ("-f", "--force", "--force-with-lease") or tok.startswith("--force-with-lease=")
                for tok in argv[1:]),
    "shutdown": lambda argv: True,
    "reboot": lambda argv: True,
    "halt": lambda argv: True,
    "pmset": lambda argv: any(tok in ("sleepnow", "restart", "shutdown") for tok in argv),
    "osascript": lambda argv: _mentions(" ".join(argv), _POWER_PHRASES),
    "sudo": lambda argv: True,
    "killall": lambda argv: True,
    "diskutil": lambda argv: True,
    "launchctl": lambda argv: bool(argv) and argv[0] in ("unload", "bootout"),
    "defaults": lambda argv: len(argv) >= 2 and argv[0] == "write" and argv[1].startswith("com.apple."),
    "tccutil": lambda argv: True,
}
_POWER_PHRASES = ("shut down", "restart", "log out", "sleep")
_APPLESCRIPT_PHRASES = _POWER_PHRASES[:3] + ("delete", "empty trash", "keystroke return", "key code 36")
_CHAIN_CHARS = "|&;\n"


def _has_flag_char(tok: str, chars: str) -> bool:
    """`-r`, `-rf`, `-fR`… — a short-flag cluster containing one of `chars`."""
    return tok.startswith("-") and not tok.startswith("--") and any(c in tok[1:] for c in chars)


def _mentions(text: str, phrases: tuple[str, ...]) -> bool:
    low = text.lower()
    return any(p in low for p in phrases)


def _bash_always_confirms(command: str) -> bool:
    for part in _split_chain(command):
        try:
            argv = shlex.split(part)
        except ValueError:
            return True   # unparsable: assume the worst
        if not argv:
            continue
        head = os.path.basename(argv[0])
        check = _ALWAYS_BASH.get(head)
        if check is not None and check(argv[1:]):
            return True
    return False


def _is_send_tool(server: str, short: str) -> bool:
    """mail_send today; any future messages/mail send on any server."""
    if short == "mail_send":
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
    mail/messages, destructive or power/privilege shell commands, an
    AppleScript that does the same, and screen actions that press Enter,
    type into a terminal, or touch a system dialog. `front` is the
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
    if server == "mac" and short == "applescript":
        return _mentions(str(tool_input.get("script", "")), _APPLESCRIPT_PHRASES)
    if server == "computer":
        if front is not None and is_system_dialog(front):
            return True
        if short == "computer_type":
            if tool_input.get("submit"):
                return True
            return front is not None and front.bundle_id in TRUST_EXCLUDED_BUNDLES
        if short == "computer_key":
            try:
                return normalize_combo(str(tool_input.get("combo") or "")) in _ENTER_KEYS
            except ValueError:
                return False
    return False
