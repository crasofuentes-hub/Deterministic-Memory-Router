
from __future__ import annotations

from pathlib import Path

from memory_router.utils.schema_validator import SchemaRegistry, validate_payload
from memory_router.core.mini_summarizer import make_mini_summary
from memory_router.core.fuser import fuse


def test_mini_summary_schema_valid():
    repo_root = Path(__file__).resolve().parents[2]
    reg = SchemaRegistry(repo_root / "src" / "memory_router" / "config" / "schemas")

    ms = make_mini_summary(
        agent="conversation",
        stable_id=1,
        source_block_ids=["b1"],
        key_sentences=["hello world. second."],
        scores={"match_query": 0.2, "recency": 0.9, "priority": 0.7, "score_global": 0.5},
        config_version="router_v1.0.0",
        trace_id="t1",
        registry=reg,
        max_tokens=40,
    )

    validate_payload(ms, "mini_summary.v1.1", reg)


def test_fuser_output_schema_and_snippet_refs_ok():
    repo_root = Path(__file__).resolve().parents[2]
    reg = SchemaRegistry(repo_root / "src" / "memory_router" / "config" / "schemas")

    ms_code = make_mini_summary(
        agent="code",
        stable_id=7,
        source_block_ids=["b7"],
        key_sentences=["def foo(): return 1"],
        scores={"match_query": 0.9, "recency": 0.5, "priority": 0.9, "score_global": 0.8},
        config_version="router_v1.0.0",
        trace_id="t1",
        registry=reg,
        max_tokens=40,
    )

    ms_pref = make_mini_summary(
        agent="preferences",
        stable_id=2,
        source_block_ids=["b2"],
        key_sentences=["prefiero Linux"],
        scores={"match_query": 0.8, "recency": 0.5, "priority": 1.0, "score_global": 0.82},
        config_version="router_v1.0.0",
        trace_id="t1",
        registry=reg,
        max_tokens=40,
    )

    out = fuse([ms_pref, ms_code], config_version="router_v1.0.0", trace_id="t1", registry=reg)

    validate_payload(out, "fuser_output.v1", reg)
    assert out["relevant_code"][0]["snippet_ref"] == "code_block_7"
