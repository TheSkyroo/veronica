"""SQLite-backed memory: conversation turns and explicit facts.

Uses FTS5 for `search`/`delete_fact_matching` when the sqlite3 build has it
(verified at open time), falling back to a `LIKE`-based scan otherwise.
Called from the asyncio thread only, but every public method is guarded by
an internal lock so the store is safe to share across threads too.
"""
import datetime as dt
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
                "ts TEXT NOT NULL, text TEXT NOT NULL)"
            )
            if self.fts_enabled:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(heard, reply)"
                )
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text)"
                )

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

    # -- facts --------------------------------------------------------------
    def add_fact(self, text: str) -> int:
        text = _normalize_ws(text)
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO facts (ts, text) VALUES (?, ?)", (_now(), text)
            )
            fact_id = cur.lastrowid
            if self.fts_enabled:
                self._conn.execute(
                    "INSERT INTO facts_fts (rowid, text) VALUES (?, ?)", (fact_id, text)
                )
            return fact_id

    def facts(self) -> list[tuple[int, str, str]]:
        with self._lock:
            return self._conn.execute(
                "SELECT id, ts, text FROM facts ORDER BY id ASC"
            ).fetchall()

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
