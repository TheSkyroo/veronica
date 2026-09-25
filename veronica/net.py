"""Is the network up? One TCP connect to the host the active brain needs,
cached for a few seconds so every turn can ask without paying for it.

Deliberately not an HTTP request: a connect (DNS + SYN) is enough to tell
"the Wi-Fi is off / the cafe portal ate it" from "we're online", and it
fails fast. A captive portal that answers the SYN looks online here — the
brain's own error then handles it, as before."""
import logging
import socket
import time
from collections.abc import Callable

log = logging.getLogger("veronica.net")

# Where each brain actually talks. Used as the probe target so the check
# answers the question that matters ("can THIS brain work"), not a generic
# "is there internet".
VENDOR_HOSTS: dict[str, str] = {
    "codex": "api.openai.com",
    "antigravity": "cloudcode-pa.googleapis.com",
    "claude": "api.anthropic.com",
    "copilot": "api.githubcopilot.com",
}
DEFAULT_HOST = "api.openai.com"
CACHE_S = 20.0

# host -> (monotonic time of the probe, result)
_cache: dict[str, tuple[float, bool]] = {}


def _connect(host: str, timeout: float) -> bool:
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError as e:      # DNS failure, no route, refused, timeout
        log.debug("offline probe to %s failed: %s", host, e)
        return False


def online(
    host: str | None = None,
    *,
    timeout: float = 1.5,
    connect: Callable[[str, float], bool] = _connect,
    clock: Callable[[], float] = time.monotonic,
) -> bool:
    """True if `host` (default: OpenAI's, see VENDOR_HOSTS) accepts a TCP
    connection within `timeout`. Cached per host for CACHE_S seconds."""
    host = host or DEFAULT_HOST
    now = clock()
    hit = _cache.get(host)
    if hit is not None and now - hit[0] < CACHE_S:
        return hit[1]
    result = connect(host, timeout)
    _cache[host] = (now, result)
    return result


def forget() -> None:
    """Drop the cache (tests, and anything that knows the answer changed)."""
    _cache.clear()
