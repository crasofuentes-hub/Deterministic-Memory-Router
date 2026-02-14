from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from ..models.memory_item import MemoryItem
from ..models.retrieval import RetrievedItem
from .similarity import tokenize, idf, tfidf_vec, cos_sim

AGENT_PRIORITY = {
    "preferences": 1.0,
    "code": 0.9,
    "conversation": 0.7,
}

@dataclass(frozen=True)
class ScoreWeights:
    alpha: float = 0.55  # match
    beta: float  = 0.30  # recency
    gamma: float = 0.15  # priority

def _norm_recency(ts_unix: int, now_unix: int, horizon_sec: int = 30 * 24 * 3600) -> float:
    dt = max(0, now_unix - int(ts_unix))
    if dt >= horizon_sec:
        return 0.0
    return float(1.0 - (dt / float(horizon_sec)))

def retrieve_topk(
    query: str,
    items: List[MemoryItem],
    now_unix: int,
    topk: int = 5,
    weights: ScoreWeights = ScoreWeights(),
    agent_priority: Dict[str, float] = AGENT_PRIORITY,
) -> List[RetrievedItem]:
    docs_tokens = [tokenize(it.text) for it in items] + [tokenize(query)]
    idf_map = idf(docs_tokens)
    qv = tfidf_vec(tokenize(query), idf_map)

    scored: List[RetrievedItem] = []
    for it in items:
        iv = tfidf_vec(tokenize(it.text), idf_map)
        match = cos_sim(qv, iv)
        rec = _norm_recency(it.ts_unix, now_unix)
        pri = float(agent_priority.get(it.agent, 0.5))
        sg = float(weights.alpha * match + weights.beta * rec + weights.gamma * pri)

        scored.append(RetrievedItem(
            agent=it.agent,
            stable_id=it.stable_id,
            text=it.text,
            ts_unix=it.ts_unix,
            match_query=float(match),
            recency=float(rec),
            priority=float(pri),
            score_global=float(sg),
            meta=dict(it.meta),
        ))

    scored.sort(key=lambda r: (-r.score_global, -r.recency, -r.priority, r.stable_id))
    return scored[: max(0, int(topk))]