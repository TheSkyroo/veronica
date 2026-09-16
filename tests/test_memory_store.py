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


def test_turns_newest_first_with_limit_and_offset(store):
    for i in range(5):
        store.add_turn(f"h{i}", f"r{i}")
    rows = store.turns(limit=2, offset=1)
    assert [(r["heard"], r["reply"]) for r in rows] == [("h3", "r3"), ("h2", "r2")]


def test_turns_default_newest_first(store):
    store.add_turn("hi", "hello")
    store.add_turn("bye", "later")
    rows = store.turns()
    assert [r["heard"] for r in rows] == ["bye", "hi"]


def test_turns_row_shape(store):
    turn_id = store.add_turn("hi", "hello")
    rows = store.turns()
    assert rows[0]["id"] == turn_id
    assert set(rows[0].keys()) == {"id", "ts", "heard", "reply"}
    assert isinstance(rows[0]["ts"], str)


def test_turns_empty(store):
    assert store.turns() == []


def test_turns_query_hits_heard_and_reply(store):
    store.add_turn("what's the weather in paris", "sunny")
    store.add_turn("remind me to buy milk", "the weather looks fine too")
    store.add_turn("call mom", "ok")
    rows = store.turns(query="weather")
    heards = [r["heard"] for r in rows]
    assert "what's the weather in paris" in heards
    assert "remind me to buy milk" in heards
    assert "call mom" not in heards


def test_turns_query_no_match(store):
    store.add_turn("hello", "hi")
    assert store.turns(query="nonexistentword") == []


def test_turns_query_empty_returns_all(store):
    store.add_turn("hi", "hello")
    store.add_turn("bye", "later")
    rows = store.turns(query="")
    assert len(rows) == 2


def test_turns_query_without_fts(store):
    store.fts_enabled = False
    store.add_turn("what's the weather in paris", "sunny")
    store.add_turn("remind me to buy milk", "ok")
    rows = store.turns(query="weather")
    assert [r["heard"] for r in rows] == ["what's the weather in paris"]


def test_turns_without_fts_still_lists(store):
    store.fts_enabled = False
    store.add_turn("hi", "hello")
    store.add_turn("bye", "later")
    rows = store.turns()
    assert [r["heard"] for r in rows] == ["bye", "hi"]


def test_delete_turn_returns_true_and_removes(store):
    id1 = store.add_turn("hi", "hello")
    id2 = store.add_turn("bye", "later")
    assert store.delete_turn(id1) is True
    remaining = [r["id"] for r in store.turns()]
    assert remaining == [id2]


def test_delete_turn_returns_false_when_missing(store):
    store.add_turn("hi", "hello")
    assert store.delete_turn(9999) is False


def test_delete_turn_also_removes_from_fts(store):
    turn_id = store.add_turn("weather in paris", "sunny")
    store.delete_turn(turn_id)
    assert store.search("weather") == []


def test_clear_turns_returns_count_and_empties(store):
    store.add_turn("hi", "hello")
    store.add_turn("bye", "later")
    n = store.clear_turns()
    assert n == 2
    assert store.recent(10) == []
    assert store.turns() == []


def test_clear_turns_empty_store(store):
    assert store.clear_turns() == 0


def test_clear_turns_clears_fts_too(store):
    store.add_turn("weather in paris", "sunny")
    store.clear_turns()
    assert store.search("weather") == []


def test_close_then_reopen(tmp_path):
    s1 = MemoryStore(tmp_path / "memory.db")
    s1.add_fact("persisted")
    s1.close()
    s2 = MemoryStore(tmp_path / "memory.db")
    try:
        assert [f[2] for f in s2.facts()] == ["persisted"]
    finally:
        s2.close()
