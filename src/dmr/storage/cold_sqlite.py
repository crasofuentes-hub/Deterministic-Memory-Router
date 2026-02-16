from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class ColdRow:
    tenant_id: str
    user_id: str
    turn_id: str
    signature: str
    ts: float
    text: str
    rank: float = 0.0


class SQLiteColdStore:
    """
    Cold storage with standalone FTS5.
    cold_rows = source of truth, cold_fts = derived index.
    Auto-migrates schema (adds ts if missing) and can rebuild FTS safely.
    """

    def __init__(self, path: str):
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")
        con.execute("PRAGMA foreign_keys=ON;")
        return con

    def _has_column(self, con: sqlite3.Connection, table: str, col: str) -> bool:
        cur = con.execute(f"PRAGMA table_info({table});")
        cols = [r[1] for r in cur.fetchall()]
        return col in cols

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS cold_rows(
                    tenant_id TEXT NOT NULL,
                    user_id   TEXT NOT NULL,
                    turn_id   TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    text      TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, user_id, turn_id)
                );
                """
            )

            # Migration: add ts if missing
            if not self._has_column(con, "cold_rows", "ts"):
                con.execute("ALTER TABLE cold_rows ADD COLUMN ts REAL NOT NULL DEFAULT 0;")

            # Standalone FTS (NOT external content) => avoids missing-row issues
            con.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS cold_fts
                USING fts5(
                    tenant_id UNINDEXED,
                    user_id   UNINDEXED,
                    turn_id   UNINDEXED,
                    signature UNINDEXED,
                    ts        UNINDEXED,
                    text,
                    tokenize = 'unicode61'
                );
                """
            )

            con.execute("DROP INDEX IF EXISTS cold_rows_tut;")
            con.execute("CREATE INDEX IF NOT EXISTS cold_rows_tut ON cold_rows(tenant_id, user_id, ts);")
            con.commit()

    def put_many(self, rows: Iterable[ColdRow]) -> None:
        rows_list = list(rows)
        if not rows_list:
            return

        with self._connect() as con:
            cur = con.cursor()
            cur.execute("BEGIN;")

            cur.executemany(
                """
                INSERT OR REPLACE INTO cold_rows(tenant_id,user_id,turn_id,signature,ts,text)
                VALUES(?,?,?,?,?,?);
                """,
                [(r.tenant_id, r.user_id, r.turn_id, r.signature, float(r.ts), r.text) for r in rows_list],
            )

            for r in rows_list:
                cur.execute(
                    "DELETE FROM cold_fts WHERE tenant_id=? AND user_id=? AND turn_id=?;",
                    (r.tenant_id, r.user_id, r.turn_id),
                )
                cur.execute(
                    """
                    INSERT INTO cold_fts(tenant_id,user_id,turn_id,signature,ts,text)
                    VALUES(?,?,?,?,?,?);
                    """,
                    (r.tenant_id, r.user_id, r.turn_id, r.signature, float(r.ts), r.text),
                )

            cur.execute("COMMIT;")

    def repair_fts(self) -> None:
        with self._connect() as con:
            cur = con.cursor()
            cur.execute("BEGIN;")
            cur.execute("DELETE FROM cold_fts;")
            cur.execute(
                """
                INSERT INTO cold_fts(tenant_id,user_id,turn_id,signature,ts,text)
                SELECT tenant_id,user_id,turn_id,signature,ts,text
                FROM cold_rows;
                """
            )
            cur.execute("COMMIT;")

    def search_fts(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        limit: int = 20,
        budget_ms: float = 50.0,
        _retry: bool = True,
    ) -> List[ColdRow]:
        t0 = time.perf_counter()
        with self._connect() as con:
            try:
                cur = con.execute(
                    """
                    SELECT tenant_id,user_id,turn_id,signature,ts,text, bm25(cold_fts) AS rank
                    FROM cold_fts
                    WHERE cold_fts MATCH ?
                      AND tenant_id = ?
                      AND user_id   = ?
                    ORDER BY rank ASC, turn_id ASC
                    LIMIT ?;
                    """,
                    (query, tenant_id, user_id, int(limit)),
                )
                out: List[ColdRow] = []
                for row in cur.fetchall():
                    if (time.perf_counter() - t0) * 1000.0 > float(budget_ms):
                        break
                    out.append(ColdRow(row[0], row[1], row[2], row[3], float(row[4]), row[5], float(row[6])))
                return out
            except sqlite3.DatabaseError as e:
                msg = str(e).lower()
                if _retry and ("fts5:" in msg or "cold_fts" in msg or "missing row" in msg):
                    pass
                else:
                    raise

        self.repair_fts()
        return self.search_fts(tenant_id, user_id, query, limit=limit, budget_ms=budget_ms, _retry=False)