"""Veronica's own memory, exposed to Claude as in-process MCP tools: recall
past conversation turns and manage explicit facts the user asked to be
remembered. Backed by `veronica.memory.store.MemoryStore`, bound at startup
via `bind()` (same pattern as `tools/pim.py`'s timer service)."""
import asyncio

from claude_agent_sdk import create_sdk_mcp_server, tool

RECALL_LIMIT_MAX = 20


def _ok(text: str = "ok") -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"content": [{"type": "text", "text": f"error: {text}"}], "is_error": True}


def _guard(fn):
    """Wrap a handler so malformed args (missing keys, bad types) return
    `_err(...)` instead of raising."""
    async def wrapper(args: dict) -> dict:
        try:
            return await fn(args)
        except Exception as exc:
            return _err(str(exc))
    return wrapper


def _clamp(value, lo, hi, default):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


store = None  # bound by build_orchestrator via bind()


def bind(new_store) -> None:
    """Replace the module-level memory store (called once at startup)."""
    global store
    store = new_store


@tool("recall", "Search past conversation turns (what was heard / what Veronica replied) for a query", {"query": str, "limit": int})
@_guard
async def recall(args: dict) -> dict:
    if store is None:
        return _err("memory store not available")
    query = str(args.get("query", "")).strip()
    if not query:
        return _err("query is required")
    limit = _clamp(args.get("limit", 5), 1, RECALL_LIMIT_MAX, 5)
    rows = await asyncio.to_thread(store.search, query, limit)
    if not rows:
        return _ok("No matching past conversation.")
    lines = [f"{ts}  heard: {heard}  reply: {reply}" for ts, heard, reply in rows]
    return _ok("\n".join(lines))


@tool("facts_list", "List all facts remembered about the user", {})
@_guard
async def facts_list(args: dict) -> dict:
    if store is None:
        return _err("memory store not available")
    rows = await asyncio.to_thread(store.facts)
    if not rows:
        return _ok("No facts remembered.")
    return _ok("\n".join(text for _id, _ts, text in rows))


@tool("fact_add", "Remember a new fact about the user", {"text": str})
@_guard
async def fact_add(args: dict) -> dict:
    if store is None:
        return _err("memory store not available")
    text = str(args.get("text", "")).strip()
    if not text:
        return _err("text is required")
    await asyncio.to_thread(store.add_fact, text)
    return _ok(f"Remembered: {text}")


@tool("fact_delete", "Forget a previously remembered fact matching the given text", {"text": str})
@_guard
async def fact_delete(args: dict) -> dict:
    if store is None:
        return _err("memory store not available")
    text = str(args.get("text", "")).strip()
    if not text:
        return _err("text is required")
    n = await asyncio.to_thread(store.delete_fact_matching, text)
    return _ok(f"Forgot {n} fact(s) matching {text!r}") if n else _err(f"no fact matching {text!r}")


TOOLS = [recall, facts_list, fact_add, fact_delete]
MEMORY_TOOL_NAMES = [t.name for t in TOOLS]
memory_server = create_sdk_mcp_server(name="memory", version="1.0.0", tools=TOOLS)
