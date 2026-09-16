import pytest

from veronica.tools import memory_tools


def text(res):
    return res["content"][0]["text"]


class FakeStore:
    def __init__(self):
        self._facts = []  # list[(id, ts, text)]
        self._turns = []  # list[(ts, heard, reply)]
        self._next_id = 1

    def add_fact(self, t):
        self._facts.append((self._next_id, "2026-09-16T00:00:00", t))
        self._next_id += 1
        return self._next_id - 1

    def facts(self):
        return list(self._facts)

    def delete_fact_matching(self, text_):
        before = len(self._facts)
        self._facts = [f for f in self._facts if text_.lower() not in f[2].lower()]
        return before - len(self._facts)

    def search(self, query, limit=5):
        rows = [t for t in self._turns if query.lower() in t[1].lower() or query.lower() in t[2].lower()]
        return rows[:limit]


@pytest.fixture(autouse=True)
def _unbind():
    yield
    memory_tools.bind(None)


async def test_tools_without_bound_store():
    memory_tools.bind(None)
    for handler, args in (
        (memory_tools.recall, {"query": "x"}),
        (memory_tools.facts_list, {}),
        (memory_tools.fact_add, {"text": "x"}),
        (memory_tools.fact_delete, {"text": "x"}),
    ):
        res = await handler.handler(args)
        assert res["is_error"]


async def test_fact_add_and_list():
    store = FakeStore()
    memory_tools.bind(store)
    res = await memory_tools.fact_add.handler({"text": "likes tea"})
    assert not res.get("is_error")
    assert "likes tea" in text(res)
    res = await memory_tools.facts_list.handler({})
    assert text(res) == "likes tea"


async def test_fact_add_requires_text():
    memory_tools.bind(FakeStore())
    res = await memory_tools.fact_add.handler({"text": "  "})
    assert res["is_error"]


async def test_facts_list_empty():
    memory_tools.bind(FakeStore())
    res = await memory_tools.facts_list.handler({})
    assert text(res) == "No facts remembered."


async def test_fact_delete_found_and_not_found():
    store = FakeStore()
    memory_tools.bind(store)
    await memory_tools.fact_add.handler({"text": "likes tea"})
    res = await memory_tools.fact_delete.handler({"text": "tea"})
    assert not res.get("is_error")
    res = await memory_tools.fact_delete.handler({"text": "coffee"})
    assert res["is_error"]


async def test_fact_delete_requires_text():
    memory_tools.bind(FakeStore())
    res = await memory_tools.fact_delete.handler({"text": ""})
    assert res["is_error"]


async def test_recall_requires_query():
    memory_tools.bind(FakeStore())
    res = await memory_tools.recall.handler({"query": ""})
    assert res["is_error"]


async def test_recall_no_match():
    memory_tools.bind(FakeStore())
    res = await memory_tools.recall.handler({"query": "nothing"})
    assert text(res) == "No matching past conversation."


async def test_recall_formats_matches():
    store = FakeStore()
    store._turns = [("2026-09-16T09:00:00", "what's the weather", "sunny")]
    memory_tools.bind(store)
    res = await memory_tools.recall.handler({"query": "weather"})
    out = text(res)
    assert "weather" in out and "sunny" in out


async def test_recall_limit_clamped():
    store = FakeStore()
    store._turns = [("t", f"weather {i}", "ok") for i in range(30)]
    memory_tools.bind(store)
    res = await memory_tools.recall.handler({"query": "weather", "limit": 999})
    assert len(text(res).split("\n")) == memory_tools.RECALL_LIMIT_MAX


def test_server_and_names():
    assert memory_tools.memory_server["name"] == "memory"
    assert set(memory_tools.MEMORY_TOOL_NAMES) == {
        "recall", "facts_list", "fact_add", "fact_delete",
    }
