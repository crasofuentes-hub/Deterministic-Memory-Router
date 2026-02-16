from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Iterable, Optional
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
    beta: float = 0.30  # recency
    gamma: float = 0.15  # priority


def _norm_recency(
    ts_unix: int, now_unix: int, horizon_sec: int = 30 * 24 * 3600
) -> float:
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

        scored.append(
            RetrievedItem(
                agent=it.agent,
                stable_id=it.stable_id,
                text=it.text,
                ts_unix=it.ts_unix,
                match_query=float(match),
                recency=float(rec),
                priority=float(pri),
                score_global=float(sg),
                meta=dict(it.meta),
            )
        )

    scored.sort(key=lambda r: (-r.score_global, -r.recency, -r.priority, r.stable_id))
    return scored[: max(0, int(topk))]


# =========================
# Compatibility layer (M2/M3) â€” wrappers expected by batch.runner/router_scoring
# These are deterministic and only depend on retrieve_topk + stable tie-break rules.
# =========================
def _stable_sort(items: list, key_fn):
    # Deterministic: Python sort is stable; we add explicit tie-break by stable_id if present.
    def k(x):
        kk = key_fn(x)
        sid = getattr(x, "stable_id", None)
        if sid is None and isinstance(x, dict):
            sid = x.get("stable_id")
        return (kk, sid if sid is not None else 10**18)

    return sorted(items, key=k)


def retrieve_by_agent(
    memory_items: list,
    query: str,
    agent: str,
    now_unix: int,
    top_k: int = 5,
    weights: Optional["ScoreWeights"] = None,
) -> list:
    """
    Retrieve top-K for a single agent using retrieve_topk() and deterministic filtering.

    - memory_items: lista de MemoryItem / dicts con campo agent
    - query: string
    - agent: "conversation" | "preferences" | "code" | "other"
    - now_unix: timestamp determinista (int) para recency
    - top_k: K por agente
    - weights: ScoreWeights opcional
    """

    def get_agent(mi):
        return (
            getattr(mi, "agent", None) if not isinstance(mi, dict) else mi.get("agent")
        )

    # 1) Filter por agent preservando orden (determinista)
    filt = [mi for mi in memory_items if get_agent(mi) == agent]

    # 2) Llama retrieve_topk sin asumir nombres exactos de parÃ¡metros
    import inspect

    sig = inspect.signature(retrieve_topk)
    params = [p.lower() for p in sig.parameters.keys()]

    kwargs = {}
    if "topk" in params:
        kwargs["topk"] = int(top_k)

    if "weights" in params:
        kwargs["weights"] = weights

    if "now_unix" in params:
        kwargs["now_unix"] = int(now_unix)
        return retrieve_topk(filt, query, **kwargs)

    if "now" in params:
        kwargs["now"] = int(now_unix)
        return retrieve_topk(filt, query, **kwargs)

    # fallback: 3er posicional es now_unix (segÃºn tu traceback)
    if kwargs:
        return retrieve_topk(filt, query, int(now_unix), **kwargs)

    return retrieve_topk(filt, query, int(now_unix))


def merge_global(per_agent_lists: Iterable[list], top_k: int = 8) -> list:
    """
    Merge multiple per-agent retrieval lists into a single deterministic ranked list.
    Sort by score_global desc, tie-break by stable_id asc.
    """
    merged: list = []
    for lst in per_agent_lists:
        merged.extend(lst or [])

    # score_global (preferred) else score else 0
    def score_of(x):
        if isinstance(x, dict):
            return float(x.get("score_global") or x.get("score") or 0.0)
        return float(
            getattr(x, "score_global", None) or getattr(x, "score", 0.0) or 0.0
        )

    ranked = sorted(
        merged,
        key=lambda x: (
            -score_of(x),
            (
                x.get("stable_id")
                if isinstance(x, dict)
                else getattr(x, "stable_id", 10**18)
            ),
        ),
    )
    return ranked[:top_k]


def build_mini_inputs(
    retrieved: list, max_sentences: int = 4, max_chars: int = 240
) -> list:
    """
    Extract-first: build minimal inputs (key sentences) from retrieved items deterministically.
    Returns list of dicts: {stable_id, agent, source_block_ids, key_sentences, scores}
    """
    out = []
    for it in retrieved:
        if isinstance(it, dict):
            text = str(it.get("text") or it.get("content") or "")
            agent = it.get("agent")
            stable_id = it.get("stable_id")
            block_id = it.get("block_id") or it.get("id")
            scores = it.get("scores") or {}
            score_global = it.get("score_global") or it.get("score")
        else:
            text = str(getattr(it, "text", "") or getattr(it, "content", "") or "")
            agent = getattr(it, "agent", None)
            stable_id = getattr(it, "stable_id", None)
            block_id = getattr(it, "block_id", None) or getattr(it, "id", None)
            scores = getattr(it, "scores", None) or {}
            score_global = getattr(it, "score_global", None) or getattr(
                it, "score", None
            )

        # Simple deterministic sentence split (no regex heavy)
        parts = [p.strip() for p in text.replace("\r", "\n").split("\n") if p.strip()]
        if not parts:
            parts = [text.strip()] if text.strip() else []

        key_sent = []
        for p in parts:
            if len(key_sent) >= max_sentences:
                break
            if p and p not in key_sent:
                key_sent.append(p[:max_chars])

        if "score_global" not in scores and score_global is not None:
            try:
                scores = dict(scores)
                scores["score_global"] = float(score_global)
            except Exception:
                pass

        out.append(
            {
                "stable_id": stable_id,
                "agent": agent,
                "source_block_ids": [block_id] if block_id else [],
                "key_sentences": key_sent,
                "scores": scores,
            }
        )
    return out
