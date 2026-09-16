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

MAC_TOOL_RISK: dict[str, Decision] = {
    "open_app": "allow",
    "open_url": "allow",
    "clipboard_read": "allow",
    "clipboard_write": "confirm",
    "notify": "allow",
    "volume_get": "allow",
    "volume_set": "allow",
    "applescript": "confirm",
}

MAC_PREFIX = "mcp__mac__"


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
    if tool_name.startswith(MAC_PREFIX):
        return MAC_TOOL_RISK.get(tool_name[len(MAC_PREFIX):], "confirm")
    return "confirm"
