from __future__ import annotations
from typing import Dict, Any, List
from ..utils.schema_validator import SchemaRegistry, validate_payload


def _cap_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return " ".join(words).strip()
    return " ".join(words[:max_words]).strip()


def make_mini_summary(
    *,
    agent: str,
    stable_id: int,
    source_block_ids: List[str],
    key_sentences: List[str],
    scores: Dict[str, float],
    config_version: str,
    trace_id: str,
    registry: SchemaRegistry,
    max_tokens: int = 40,
) -> Dict[str, Any]:
    # tokens aproximados -> palabras (conservador)
    max_words = max(10, min(80, int(max_tokens)))  # 1 palabra ~ 1 token aprox en límite
    base = " ".join([s.strip() for s in key_sentences if s.strip()]).strip()
    if not base:
        base = "(empty)"
    summary = _cap_words(base, max_words=max_words)
    payload = {
        "schema_id": "mini_summary.v1.1",
        "config_version": config_version,
        "trace_id": trace_id,
        "agent": agent,
        "stable_id": int(stable_id),
        "source_block_ids": list(source_block_ids),
        "summary": summary,
        "max_tokens": int(max_tokens),
        "scores": {
            "match_query": float(scores["match_query"]),
            "recency": float(scores["recency"]),
            "priority": float(scores["priority"]),
            "score_global": float(scores["score_global"]),
        },
    }
    validate_payload(payload, "mini_summary.v1.1", registry)
    return payload
