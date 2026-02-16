from __future__ import annotations
import time
from pathlib import Path
from dmr.storage.cold_sqlite import SQLiteColdStore, ColdRow

def test_cold_fts_invariance(tmp_path: Path):
    db = SQLiteColdStore(path=str(tmp_path / "cold.sqlite3"))
    tenant, user = "T", "U"
    now = time.time()
    rows = [
        ColdRow(tenant, user, "c1", "s1", now, "Human: alpha memory\nAI: ok"),
        ColdRow(tenant, user, "c2", "s2", now, "Human: alpha beta gamma\nAI: ok"),
        ColdRow(tenant, user, "c3", "s3", now, "Human: beta only\nAI: ok"),
    ]
    db.put_many(rows)
    r1 = db.search_fts(tenant, user, "alpha", limit=10, budget_ms=50.0)
    r2 = db.search_fts(tenant, user, "alpha", limit=10, budget_ms=50.0)
    assert r1 == r2