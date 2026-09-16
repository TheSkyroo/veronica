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
        with self._lock:
            if self.fts_enabled:
                expr = _fts_match_expr(text)
                if not expr:
                    return 0
                ids = [
                    row[0]
                    for row in self._conn.execute(
                        "SELECT rowid FROM facts_fts WHERE facts_fts MATCH ?", (expr,)
                    ).fetchall()
                ]
            else:
                tokens = re.findall(r"\w+", text or "")
                if not tokens:
                    return 0
                clauses = " OR ".join(["text LIKE ?"] * len(tokens))
                params = [f"%{t}%" for t in tokens]
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
