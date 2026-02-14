from pathlib import Path
from memory_router.batch.runner import run_offline

def test_offline_runner_generate_deterministic(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[2]
    corpus = repo_root / "tests" / "fixtures" / "corpus.jsonl"
    schemas = repo_root / "src" / "memory_router" / "config" / "schemas"
    out = tmp_path / "run"

    m1 = run_offline(corpus, out, schemas, "router_v1.0.0", "trace_offline_0001", mode="generate")
    m2 = run_offline(corpus, out, schemas, "router_v1.0.0", "trace_offline_0001", mode="generate")

    assert m1["items_count"] == m2["items_count"]
    assert m1["items_hash"] == m2["items_hash"]