"""Risk classifier for tool calls: decide whether a call runs without asking."""
import shlex
from typing import Literal

Decision = Literal["allow", "confirm"]

ALLOW_TOOLS = frozenset({"Read", "Glob", "Grep", "WebSearch", "WebFetch"})

SAFE_BASH = frozenset({
    "ls", "cat", "head", "tail", "pwd", "date", "cal", "whoami", "pbpaste", "df",
    "du", "ps", "which", "echo", "uptime", "wc", "file", "stat",
})

# any of these anywhere in a Bash command → confirm (covers $(…), pipes, chains, redirects)
FORBIDDEN = "|&;><`$\n"

CURL_SAFE_NO_ARG = frozenset({"-s", "-S", "-L", "--silent", "--location", "--compressed"})
CURL_SAFE_WITH_ARG = frozenset({
    "--max-time", "-m", "-H", "--header", "-A", "--user-agent",
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


def _curl_is_safe(argv: list[str]) -> bool:
    urls = 0
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok.startswith(("http://", "https://")):
            urls += 1
            i += 1
            continue
        if tok in CURL_SAFE_NO_ARG:
            i += 1
            continue
        if tok in CURL_SAFE_WITH_ARG:
            if i + 1 >= len(argv):
                return False
            i += 2  # consumes this flag's argument, whatever it is
            continue
        if "=" in tok and tok.split("=", 1)[0] in CURL_SAFE_WITH_ARG:
            i += 1
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
