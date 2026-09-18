"""Risk classifier for tool calls: decide whether a call runs without asking."""
import ipaddress
import shlex
from typing import Literal
from urllib.parse import urlsplit

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
