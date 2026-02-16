from pathlib import Path
from memory_router.batch.runner import run_offline


def test_offline_runner_check_matches_golden():
    repo_root = Path(__file__).resolve().parents[2]
    corpus = repo_root / "tests" / "fixtures" / "corpus.jsonl"
    schemas = repo_root / "src" / "memory_router" / "config" / "schemas"
    golden_out = repo_root / "tests" / "fixtures" / "golden" / "router_v1.0.0"

    # Debe existir (golden versionado en repo)
    assert (golden_out / "run_manifest.json").exists()

    # CHECK no regenera; compara contra run_manifest.json
    run_offline(
        corpus_path=corpus,
        out_dir=golden_out,
        schemas_dir=schemas,
        config_version="router_v1.0.0",
        trace_id="trace_offline_0001",
        mode="check",
    )
