import pytest

from veronica.memory.store import MemoryStore, _fts_match_expr


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "memory.db")
    yield s
    s.close()


def test_creates_db_file(tmp_path):
    s = MemoryStore(tmp_path / "sub" / "memory.db")
    try:
        assert (tmp_path / "sub" / "memory.db").exists()
    finally:
        s.close()


def test_add_and_recent_turns(store):
    store.add_turn("hi", "hello")
    store.add_turn("what time is it", "it's noon")
    rows = store.recent(10)
    assert [(h, r) for _, h, r in rows] == [
        ("hi", "hello"),
        ("what time is it", "it's noon"),
    ]


def test_recent_respects_limit_and_order(store):
    for i in range(5):
        store.add_turn(f"h{i}", f"r{i}")
    rows = store.recent(2)
    assert [(h, r) for _, h, r in rows] == [("h3", "r3"), ("h4", "r4")]


def test_recent_empty(store):
    assert store.recent(5) == []


def test_search_turns(store):
    store.add_turn("what's the weather in paris", "sunny")
    store.add_turn("remind me to buy milk", "ok")
    store.add_turn("weather tomorrow", "rainy")
    rows = store.search("weather")
    heards = [h for _, h, _ in rows]
    assert "what's the weather in paris" in heards
    assert "weather tomorrow" in heards
    assert "remind me to buy milk" not in heards


def test_search_limit(store):
    for i in range(10):
        store.add_turn(f"paris trip {i}", "ok")
    rows = store.search("paris", limit=3)
    assert len(rows) == 3


def test_search_no_match(store):
    store.add_turn("hello", "hi")
    assert store.search("nonexistentword") == []


def test_search_empty_query(store):
    store.add_turn("hello", "hi")
    assert store.search("") == []


def test_add_fact_and_list(store):
    id1 = store.add_fact("likes tea")
    id2 = store.add_fact("works at Acme")
    facts = store.facts()
    assert [f[0] for f in facts] == [id1, id2]
    assert [f[2] for f in facts] == ["likes tea", "works at Acme"]


def test_delete_fact_matching(store):
    store.add_fact("likes tea")
    store.add_fact("works at Acme")
    n = store.delete_fact_matching("tea")
    assert n == 1
    remaining = [f[2] for f in store.facts()]
    assert remaining == ["works at Acme"]


def test_delete_fact_matching_no_match(store):
    store.add_fact("likes tea")
    n = store.delete_fact_matching("coffee")
    assert n == 0
    assert len(store.facts()) == 1


def test_delete_fact_matching_multiple(store):
    store.add_fact("likes tea and coffee")
    store.add_fact("hates coffee")
    n = store.delete_fact_matching("coffee")
    assert n == 2
    assert store.facts() == []


def test_delete_fact_matching_empty_text(store):
    store.add_fact("likes tea")
    assert store.delete_fact_matching("") == 0


def test_fts_match_expr_sanitizes_tokens():
    assert _fts_match_expr('weather "paris"! -OR*') == '"weather" OR "paris" OR "OR"'
    assert _fts_match_expr("") == ""
    assert _fts_match_expr(None) == ""


def test_fts5_available_on_this_python(store):
    # If this ever flips false, search()/delete_fact_matching() silently
    # fall back to LIKE; the other tests here still pass either way.
    assert store.fts_enabled is True


def test_delete_fact_matching_is_precise_across_similar_facts(store):
    store.add_fact("gym session at 7")
    store.add_fact("dinner reservation at 7")
    store.add_fact("call mom")
    store.add_fact("buy milk")
    n = store.delete_fact_matching("my gym is at 7")
    assert n == 1
    remaining = sorted(f[2] for f in store.facts())
    assert remaining == ["buy milk", "call mom", "dinner reservation at 7"]


def test_delete_fact_matching_exact_match_preferred(store):
    store.add_fact("likes tea")
    store.add_fact("likes tea and coffee")
    n = store.delete_fact_matching("likes tea")
    assert n == 1
    remaining = [f[2] for f in store.facts()]
    assert remaining == ["likes tea and coffee"]


def test_forget_it_deletes_nothing(store):
    store.add_fact("likes tea")
    n = store.delete_fact_matching("it")
    assert n == 0
    assert len(store.facts()) == 1


def test_forget_everything_deletes_nothing(store):
    store.add_fact("likes tea")
    store.add_fact("works at Acme")
    n = store.delete_fact_matching("everything")
    assert n == 0
    assert len(store.facts()) == 2


def test_forget_that_alone_deletes_nothing(store):
    store.add_fact("likes tea")
    n = store.delete_fact_matching("that")
    assert n == 0


def test_add_fact_normalizes_whitespace(store):
    store.add_fact("  likes   tea  \n\n and coffee ")
    assert [f[2] for f in store.facts()] == ["likes tea and coffee"]


def test_add_turn_normalizes_whitespace(store):
    store.add_turn(" hello   there ", "hi \n there ")
    rows = store.recent(1)
    assert [(h, r) for _, h, r in rows] == [("hello there", "hi there")]


def test_close_then_reopen(tmp_path):
    s1 = MemoryStore(tmp_path / "memory.db")
    s1.add_fact("persisted")
    s1.close()
    s2 = MemoryStore(tmp_path / "memory.db")
    try:
        assert [f[2] for f in s2.facts()] == ["persisted"]
    finally:
        s2.close()
