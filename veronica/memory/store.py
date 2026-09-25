"""SQLite-backed memory: conversation turns and explicit facts.

Uses FTS5 for `search`/`delete_fact_matching` when the sqlite3 build has it
(verified at open time), falling back to a `LIKE`-based scan otherwise.
Called from the asyncio thread only, but every public method is guarded by
an internal lock so the store is safe to share across threads too.
"""
import datetime as dt
import difflib
import re
import sqlite3
import threading
from pathlib import Path


def _has_fts5(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts5_probe")
        return True
    except sqlite3.OperationalError:
        return False


def _fts_match_expr(query: str) -> str:
    """Sanitize free text into an FTS5 MATCH expression: each word-token is
    quoted (so punctuation/operators like `-`, `"`, `*` in user text can't be
    read as FTS5 query syntax) and tokens are OR'd together."""
    tokens = re.findall(r"\w+", query or "")
    return " OR ".join(f'"{t}"' for t in tokens)


# Words too generic to identify a specific fact on their own — "forget it" /
# "forget everything" must not turn into a delete-everything-that-matches-
# "it" scan; if only these are left after removing them, delete_fact_matching
# treats the query as having no real content and deletes nothing.
FORGET_STOPWORDS = frozenset({
    "a", "an", "the", "that", "this", "it", "is", "am", "are", "was", "were",
    "to", "of", "in", "on", "at", "for", "and", "or", "my", "i", "me", "you",
    "your", "be", "been", "being", "with", "about", "everything", "all",
    "stuff", "thing", "things", "please",
})


def _normalize_ws(s: str) -> str:
    return " ".join((s or "").split())


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _now_used() -> str:
    """Microseconds, unlike _now(): several facts can be written or bumped
    inside the same second and the facts block still has to order them."""
    return dt.datetime.now().isoformat()


# -- fact kinds ------------------------------------------------------------
# A fact's kind is inferred from its wording at write time — no model call,
# just cue words. The order below is the precedence: the first kind whose
# cues appear wins, so "I go to the gym every morning" is a routine (cue
# "every") rather than a place (cue "gym"), and an explicit liking beats
# everything because that is what the sentence is actually about.
FACT_KINDS: tuple[str, ...] = ("preference", "person", "place", "routine", "other")
# Spoken/printed headings for the groups facts_by_kind returns.
KIND_LABELS: dict[str, str] = {
    "preference": "Preferences", "person": "People", "place": "Places",
    "routine": "Routines", "other": "Other",
}

_KIND_CUES: tuple[tuple[str, frozenset[str], tuple[str, ...]], ...] = (
    ("preference", frozenset({
        "like", "likes", "liked", "love", "loves", "hate", "hates", "dislike",
        "dislikes", "prefer", "prefers", "preferred", "favourite",
        "favorite", "allergic", "vegetarian", "vegan", "enjoys", "enjoy",
        "avoids", "avoid", "never",
    }), ("can't stand", "cant stand", "doesn't like", "does not like",
         "would rather", "i take my", "takes my", "i drink", "drinks")),
    ("routine", frozenset({
        "every", "daily", "weekly", "monthly", "always", "usually", "routine",
        "schedule", "standup", "habit", "mornings", "evenings", "nights",
        "weekdays", "weekends", "mondays", "tuesdays", "wednesdays",
        "thursdays", "fridays", "saturdays", "sundays",
    }), ("each morning", "each day", "each week", "before bed", "after work")),
    ("person", frozenset({
        "wife", "husband", "mom", "mum", "mother", "dad", "father", "sister",
        "brother", "son", "daughter", "boss", "manager", "colleague",
        "coworker", "teammate", "friend", "doctor", "dentist", "landlord",
        "partner", "girlfriend", "boyfriend", "neighbour", "neighbor",
        "cousin", "uncle", "aunt", "grandma", "grandpa", "kid", "kids",
        "children", "nephew", "niece", "therapist", "barber",
    }), ("name is", "'s name")),
    ("place", frozenset({
        "office", "home", "address", "apartment", "flat", "street", "road",
        "city", "town", "neighbourhood", "neighborhood", "gym", "cafe",
        "restaurant", "airport", "hotel", "desk", "building", "campus",
        "hometown",
    }), ("lives in", "lives at", "located at", "based in", "i live in")),
)


def infer_kind(text: str) -> str:
    """The kind of fact `text` states — the first cue set it hits, else
    "other". Deliberately dumb and readable: it only has to be right often
    enough to make `facts_list` easier to listen to."""
    low = _normalize_ws(text).lower()
    words = set(re.findall(r"[\w']+", low))
    for kind, cue_words, cue_phrases in _KIND_CUES:
        if words & cue_words or any(phrase in low for phrase in cue_phrases):
            return kind
    return "other"


# -- fuzzy matching between a fact and free text ---------------------------
# A rewording of a fact already stored replaces it rather than piling up a
# near-copy; 0.9 is tight enough that "likes tea" and "hates tea" stay two
# facts.
DUPLICATE_RATIO = 0.9
# "Used in a reply" is a stem-prefix overlap: the brain reports nothing, so
# this is all we have, and it only has to be good enough to order the facts
# block. Four characters so "like"/"likes" and "office"/"offices" match.
_STEM = 4
FACT_USE_RATIO = 0.6


def _fold(text: str) -> str:
    """Lowercased, punctuation-free, single-spaced — the form both the
    duplicate ratio and the topic substring test compare."""
    return " ".join(re.findall(r"\w+", (text or "").lower()))


def _stems(text: str) -> set[str]:
    return {w[:_STEM] for w in re.findall(r"\w+", (text or "").lower())}


def _key_words(text: str) -> list[str]:
    """The words of a fact that could identify it in a spoken reply: the
    generic ones carry no signal."""
    return [w for w in re.findall(r"\w+", (text or "").lower())
            if len(w) >= 4 and w not in FORGET_STOPWORDS]


def _mentioned_in(fact: str, spoken: str, spoken_stems: set[str]) -> bool:
    words = _key_words(fact)
    if not words:
        return _fold(fact) in _fold(spoken)
    hits = sum(1 for w in words if w[:_STEM] in spoken_stems)
    return hits / len(words) >= FACT_USE_RATIO


class MemoryStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = None
        with self._lock:
            self.fts_enabled = _has_fts5(self._conn)
            self._create_schema()

    # -- schema -----------------------------------------------------------
    def _create_schema(self) -> None:
        with self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS turns ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts TEXT NOT NULL, heard TEXT NOT NULL, reply TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS facts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "ts TEXT NOT NULL, text TEXT NOT NULL, "
                "kind TEXT NOT NULL DEFAULT 'other', "
                "last_used TEXT NOT NULL DEFAULT '')"
            )
            self._migrate_facts()
            if self.fts_enabled:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(heard, reply)"
                )
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text)"
                )

    def _migrate_facts(self) -> None:
        """Bring a pre-F5 `facts` table (ts, text) up to date in place: the
        kind is backfilled by running the same inference over the rows that
        are already there, and "last used" starts at the row's own
        timestamp so an untouched memory still orders newest-first. Runs
        inside _create_schema's transaction; no-op once both columns exist."""
        have = {row[1] for row in self._conn.execute("PRAGMA table_info(facts)")}
        if "kind" not in have:
            self._conn.execute("ALTER TABLE facts ADD COLUMN kind TEXT NOT NULL DEFAULT 'other'")
            rows = self._conn.execute("SELECT id, text FROM facts").fetchall()
            self._conn.executemany(
                "UPDATE facts SET kind = ? WHERE id = ?",
                [(infer_kind(text), fact_id) for fact_id, text in rows],
            )
        if "last_used" not in have:
            self._conn.execute("ALTER TABLE facts ADD COLUMN last_used TEXT NOT NULL DEFAULT ''")
            self._conn.execute("UPDATE facts SET last_used = ts WHERE last_used = ''")

    # -- turns --------------------------------------------------------------
    def add_turn(self, heard: str, reply: str) -> int:
        heard = _normalize_ws(heard)
        reply = _normalize_ws(reply)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO turns (ts, heard, reply) VALUES (?, ?, ?)",
                (_now(), heard, reply),
            )
            turn_id = cur.lastrowid
            if self.fts_enabled:
                self._conn.execute(
                    "INSERT INTO turns_fts (rowid, heard, reply) VALUES (?, ?, ?)",
                    (turn_id, heard, reply),
                )
            return turn_id

    def recent(self, n: int) -> list[tuple[str, str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, heard, reply FROM turns ORDER BY id DESC LIMIT ?", (max(0, n),)
            ).fetchall()
        return list(reversed(rows))  # chronological order

    def search(self, query: str, limit: int = 5) -> list[tuple[str, str, str]]:
        limit = max(1, limit)
        with self._lock:
            if self.fts_enabled:
                expr = _fts_match_expr(query)
                if not expr:
                    return []
                rows = self._conn.execute(
                    "SELECT t.ts, t.heard, t.reply FROM turns_fts f "
                    "JOIN turns t ON t.id = f.rowid "
                    "WHERE turns_fts MATCH ? ORDER BY rank LIMIT ?",
                    (expr, limit),
                ).fetchall()
                return rows
            tokens = re.findall(r"\w+", query or "")
            if not tokens:
                return []
            clauses = " OR ".join(["heard LIKE ? OR reply LIKE ?"] * len(tokens))
            params: list = []
            for t in tokens:
                like = f"%{t}%"
                params.extend([like, like])
            rows = self._conn.execute(
                f"SELECT ts, heard, reply FROM turns WHERE {clauses} ORDER BY id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
            return rows

    def turns(self, limit: int = 200, offset: int = 0, query: str = "") -> list[dict]:
        limit = max(0, limit)
        offset = max(0, offset)
        with self._lock:
            query = query or ""
            if not query:
                rows = self._conn.execute(
                    "SELECT id, ts, heard, reply FROM turns ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            elif self.fts_enabled:
                expr = _fts_match_expr(query)
                if not expr:
                    return []
                rows = self._conn.execute(
                    "SELECT t.id, t.ts, t.heard, t.reply FROM turns_fts f "
                    "JOIN turns t ON t.id = f.rowid "
                    "WHERE turns_fts MATCH ? ORDER BY rank LIMIT ? OFFSET ?",
                    (expr, limit, offset),
                ).fetchall()
            else:
                tokens = re.findall(r"\w+", query)
                if not tokens:
                    return []
                clauses = " OR ".join(["heard LIKE ? OR reply LIKE ?"] * len(tokens))
                params: list = []
                for t in tokens:
                    like = f"%{t}%"
                    params.extend([like, like])
                rows = self._conn.execute(
                    f"SELECT id, ts, heard, reply FROM turns WHERE {clauses} "
                    "ORDER BY id DESC LIMIT ? OFFSET ?",
                    (*params, limit, offset),
                ).fetchall()
        return [
            {"id": rid, "ts": ts, "heard": heard, "reply": reply}
            for rid, ts, heard, reply in rows
        ]

    def delete_turn(self, turn_id: int) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM turns WHERE id = ?", (turn_id,))
            if self.fts_enabled:
                self._conn.execute("DELETE FROM turns_fts WHERE rowid = ?", (turn_id,))
            return cur.rowcount > 0

    def clear_turns(self) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM turns")
            if self.fts_enabled:
                self._conn.execute("DELETE FROM turns_fts")
            return cur.rowcount

    # -- facts --------------------------------------------------------------
    def remember(self, text: str) -> tuple[int, bool]:
        """Store `text` as a fact and return (id, replaced). A near-duplicate
        of a fact already there (ratio >= DUPLICATE_RATIO on the folded text)
        is rewritten in place — same row, fresh wording, re-inferred kind —
        rather than left as a second near-copy the facts block then pays for
        twice."""
        text = _normalize_ws(text)
        kind = infer_kind(text)
        now, used = _now(), _now_used()
        with self._lock, self._conn:
            fact_id = self._near_duplicate_id(text)
            replaced = fact_id is not None
            if replaced:
                self._conn.execute(
                    "UPDATE facts SET ts = ?, text = ?, kind = ?, last_used = ? WHERE id = ?",
                    (now, text, kind, used, fact_id),
                )
            else:
                cur = self._conn.execute(
                    "INSERT INTO facts (ts, text, kind, last_used) VALUES (?, ?, ?, ?)",
                    (now, text, kind, used),
                )
                fact_id = cur.lastrowid
            if self.fts_enabled:
                if replaced:
                    self._conn.execute("DELETE FROM facts_fts WHERE rowid = ?", (fact_id,))
                self._conn.execute(
                    "INSERT INTO facts_fts (rowid, text) VALUES (?, ?)", (fact_id, text)
                )
            return fact_id, replaced

    def add_fact(self, text: str) -> int:
        """remember(), for the callers that only need the row id."""
        return self.remember(text)[0]

    def _near_duplicate_id(self, text: str) -> int | None:
        """The id of the stored fact `text` is a rewording of, if any. Caller
        holds the lock."""
        folded = _fold(text)
        if not folded:
            return None
        best, best_ratio = None, DUPLICATE_RATIO
        for fact_id, other in self._conn.execute("SELECT id, text FROM facts"):
            ratio = difflib.SequenceMatcher(None, folded, _fold(other)).ratio()
            if ratio >= best_ratio:
                best, best_ratio = fact_id, ratio
        return best

    def facts(self) -> list[tuple[int, str, str]]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, ts, text FROM facts ORDER BY id ASC"
            ).fetchall()

    def facts_by_kind(self) -> dict[str, list[str]]:
        """Facts grouped for reciting: FACT_KINDS order, kinds with nothing
        in them left out, oldest-first within a kind."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind, text FROM facts ORDER BY id ASC"
            ).fetchall()
        grouped: dict[str, list[str]] = {}
        for kind in FACT_KINDS:
            texts = [text for k, text in rows if k == kind]
            if texts:
                grouped[kind] = texts
        return grouped

    def facts_for_prompt(self, limit: int) -> list[str]:
        """Up to `limit` facts for the system prompt, most-recently-used
        first (see touch_facts_used), newest first among never-used ones."""
        limit = max(0, limit)
        if not limit:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT text FROM facts ORDER BY last_used DESC, id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [text for (text,) in rows]

    def touch_facts_used(self, spoken: str) -> int:
        """Mark every fact whose distinctive words turn up in `spoken` as
        just used, and return how many. Called once per turn after the reply
        is out — a table scan plus a set lookup per fact, never on the path
        between the brain and the speaker."""
        folded = _fold(spoken)
        if not folded:
            return 0
        spoken_stems = _stems(folded)
        with self._lock:
            rows = self._conn.execute("SELECT id, text FROM facts").fetchall()
            ids = [fid for fid, text in rows if _mentioned_in(text, folded, spoken_stems)]
            if not ids:
                return 0
            with self._conn:
                qmarks = ",".join("?" * len(ids))
                self._conn.execute(
                    f"UPDATE facts SET last_used = ? WHERE id IN ({qmarks})", (_now_used(), *ids)
                )
            return len(ids)

    def delete_facts_about(self, topic: str) -> int:
        """"Forget everything about X": delete every fact the topic turns up
        in and return the count. Looser than delete_fact_matching on purpose
        — one shared content word is enough — but it keeps the same
        stopword guard, so "forget everything about it" deletes nothing."""
        folded = _fold(topic)
        words = [w for w in folded.split() if w not in FORGET_STOPWORDS]
        if not words:
            return 0
        wanted = {w[:_STEM] for w in words}
        with self._lock:
            rows = self._conn.execute("SELECT id, text FROM facts").fetchall()
            ids = [
                fid for fid, text in rows
                if folded in _fold(text) or (wanted & _stems(text))
            ]
            if not ids:
                return 0
            with self._conn:
                qmarks = ",".join("?" * len(ids))
                self._conn.execute(f"DELETE FROM facts WHERE id IN ({qmarks})", ids)
                if self.fts_enabled:
                    self._conn.execute(f"DELETE FROM facts_fts WHERE rowid IN ({qmarks})", ids)
            return len(ids)

    def delete_fact_matching(self, text: str) -> int:
        """Delete facts matching `text`, as precisely as possible so a short
        or generic "forget X" can't sweep up unrelated facts:

        1. An exact (case-insensitive, whitespace-normalized) match on a
           fact's full text — the common case, since `text` here is usually
           exactly what was originally remembered.
        2. Else, a substring match (case-insensitive) — `text` names part of
           a fact.
        3. Else, an FTS AND-match on every non-stopword token in `text` — a
           looser paraphrase still has to hit every content word. If nothing
           but stopwords are left (e.g. "it", "that", "everything"), nothing
           is deleted rather than guessing.
        """
        norm_query = _normalize_ws(text).lower()
        if not norm_query:
            return 0
        with self._lock:
            rows = self._conn.execute("SELECT id, text FROM facts").fetchall()
            ids = [rid for rid, t in rows if _normalize_ws(t).lower() == norm_query]
            if not ids:
                ids = [rid for rid, t in rows if norm_query in _normalize_ws(t).lower()]
            if not ids:
                tokens = [
                    tok for tok in re.findall(r"\w+", text or "")
                    if tok.lower() not in FORGET_STOPWORDS
                ]
                if not tokens:
                    return 0
                if self.fts_enabled:
                    expr = " AND ".join(f'"{tok}"' for tok in tokens)
                    ids = [
                        row[0]
                        for row in self._conn.execute(
                            "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ?", (expr,)
                        ).fetchall()
                    ]
                else:
                    clauses = " AND ".join(["text LIKE ?"] * len(tokens))
                    params = [f"%{tok}%" for tok in tokens]
                    ids = [
                        row[0]
                        for row in self._conn.execute(
                            f"SELECT id FROM facts WHERE {clauses}", params
                        ).fetchall()
                    ]
            if not ids:
                return 0
            with self._conn:
                qmarks = ",".join("?" * len(ids))
                self._conn.execute(f"DELETE FROM facts WHERE id IN ({qmarks})", ids)
                if self.fts_enabled:
                    self._conn.execute(f"DELETE FROM facts_fts WHERE rowid IN ({qmarks})", ids)
            return len(ids)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
