import sqlite3

import pytest

from veronica.memory.store import MemoryStore, _fts_match_expr, infer_kind


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


# -- typed facts, dedupe, topic forget, use tracking (F5) ---------------------

@pytest.mark.parametrize("text,kind", [
    ("I like my coffee black", "preference"),
    ("allergic to peanuts", "preference"),
    ("hates coriander", "preference"),
    ("standup is every day at 9:30", "routine"),
    ("goes to the gym every morning", "routine"),
    ("my sister's name is Priya", "person"),
    ("Rahul is my manager", "person"),
    ("the office is in Bandra", "place"),
    ("lives in Pune", "place"),
    ("the wifi password is hunter2", "other"),
    ("", "other"),
])
def test_infer_kind_table(text, kind):
    assert infer_kind(text) == kind


def test_add_fact_stores_inferred_kind(store):
    store.add_fact("likes tea")
    store.add_fact("the office is in Bandra")
    assert store.facts_by_kind()["preference"] == ["likes tea"]
    assert store.facts_by_kind()["place"] == ["the office is in Bandra"]


def test_facts_by_kind_skips_empty_kinds_and_keeps_kind_order(store):
    store.add_fact("the office is in Bandra")
    store.add_fact("likes tea")
    assert list(store.facts_by_kind()) == ["preference", "place"]


def test_near_duplicate_fact_replaces_instead_of_adding(store):
    store.add_fact("likes tea in the morning")
    fact_id, replaced = store.remember("Likes tea in the mornings")
    assert replaced is True
    assert [f[2] for f in store.facts()] == ["Likes tea in the mornings"]
    assert [f[0] for f in store.facts()] == [fact_id]


def test_distinct_facts_are_both_kept(store):
    store.add_fact("likes tea in the morning")
    _id, replaced = store.remember("allergic to peanuts")
    assert replaced is False
    assert len(store.facts()) == 2


def test_replacing_a_fact_reinfers_its_kind_and_updates_search(store):
    store.add_fact("the office is at Church Street")
    store.remember("the office is on Church Street")
    assert store.facts_by_kind()["place"] == ["the office is on Church Street"]
    # the full-text index followed the rewrite rather than keeping both
    assert store._conn.execute("SELECT text FROM facts_fts").fetchall() == [
        ("the office is on Church Street",)
    ]


def test_delete_facts_about_topic_counts_and_spares_others(store):
    store.add_fact("the office wifi password is hunter2")
    store.add_fact("my desk at the office is by the window")
    store.add_fact("office lunch is at one")
    store.add_fact("likes tea")
    assert store.delete_facts_about("the office") == 3
    assert [f[2] for f in store.facts()] == ["likes tea"]


def test_delete_facts_about_ignores_stopword_only_topics(store):
    store.add_fact("likes tea")
    assert store.delete_facts_about("everything") == 0
    assert store.delete_facts_about("") == 0
    assert len(store.facts()) == 1


def test_facts_for_prompt_caps_and_orders_by_last_use(store):
    for text in ("likes tea", "allergic to peanuts", "the office is in Bandra",
                 "the wifi password is hunter2", "standup is every day at nine"):
        store.add_fact(text)
    store.touch_facts_used("Peanuts are out, you're allergic.")
    assert store.facts_for_prompt(3)[0] == "allergic to peanuts"
    assert len(store.facts_for_prompt(3)) == 3
    assert store.facts_for_prompt(0) == []


def test_touch_facts_used_bumps_only_mentioned_facts(store):
    store.add_fact("likes tea")
    store.add_fact("allergic to peanuts")
    assert store.touch_facts_used("I know you like tea, so here it is.") == 1
    assert store.facts_for_prompt(10)[0] == "likes tea"
    assert store.touch_facts_used("It's sunny in Paris.") == 0


def test_migrates_a_pre_f5_facts_table(tmp_path):
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    with conn:
        conn.execute(
            "CREATE TABLE facts (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT NOT NULL, text TEXT NOT NULL)"
        )
        conn.execute("INSERT INTO facts (ts, text) VALUES (?, ?)", ("2026-01-01T00:00:00", "likes tea"))
        conn.execute("INSERT INTO facts (ts, text) VALUES (?, ?)",
                     ("2026-01-02T00:00:00", "the office is in Bandra"))
    conn.close()
    s = MemoryStore(path)
    try:
        assert [f[2] for f in s.facts()] == ["likes tea", "the office is in Bandra"]
        by_kind = s.facts_by_kind()
        assert by_kind["preference"] == ["likes tea"]
        assert by_kind["place"] == ["the office is in Bandra"]
        # last_used was backfilled from ts, so prompt order is newest-first.
        assert s.facts_for_prompt(10) == ["the office is in Bandra", "likes tea"]
    finally:
        s.close()
