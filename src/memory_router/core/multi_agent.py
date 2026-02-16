from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .budgets import Budgets, Thresholds
from .faiss_store import FaissShard, MemoryChunk, Retrieved
from .tokens import _clip_to_tokens, _simple_token_count


class DeterministicFusion:
    """
    Deterministic fusion:
    - No voting.
    - Prefer non-contradictory overlap (dedupe lines).
    - If conflict, newest agent wins (by turn_end, then similarity).
    """

    def fuse(self, per_agent_summaries: List[Tuple[str, float, int, str]]) -> str:
        # each tuple: (agent, similarity, turn_end, summary_text)
        if not per_agent_summaries:
            return ""

        # "newest wins": higher turn_end, then similarity, then agent name
        per_agent_summaries = sorted(
            per_agent_summaries,
            key=lambda t: (t[2], t[1], t[0]),
            reverse=True,
        )

        # keep unique lines deterministically
        lines_out: List[str] = []
        seen = set()

        for agent, sim, turn_end, text in per_agent_summaries:
            for line in (text or "").split("\n"):
                ln = line.strip()
                if not ln:
                    continue
                key = ln.lower()
                if key in seen:
                    continue
                seen.add(key)
                lines_out.append(ln)

        return "\n".join(lines_out).strip()


class MultiAgentMemorySystem:
    """
    5-agent FAISS memory system with:
    - sharding by turn windows (10 turns per agent, rolling)
    - retrieval over all agents (order stable; caller may parallelize)
    - threshold gating
    - deterministic fusion layer
    - strict budgets
    """

    def __init__(
        self,
        store_dir: Path,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        agents: Optional[List[str]] = None,
        turns_per_agent: int = 10,
        budgets: Budgets = Budgets(),
        thresholds: Thresholds = Thresholds(),
    ):
        self.store_dir = store_dir
        self.model_name = model_name
        self.agents = agents or ["agent1", "agent2", "agent3", "agent4", "agent5"]
        self.turns_per_agent = int(turns_per_agent)
        self.budgets = budgets
        self.thresholds = thresholds

        self.model = SentenceTransformer(model_name)
        dim = int(self.model.get_sentence_embedding_dimension())

        self.shards: Dict[str, FaissShard] = {
            a: FaissShard(store_dir, a, dim) for a in self.agents
        }
        for s in self.shards.values():
            s.load()

        self.fuser = DeterministicFusion()

    def _agent_for_turn(self, turn_index_1based: int) -> str:
        block = (turn_index_1based - 1) // self.turns_per_agent
        agent_ix = block % len(self.agents)
        return self.agents[agent_ix]

    def index_turn(
        self,
        stable_id: int,
        turn_index_1based: int,
        text: str,
        ts_unix: int,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        agent = self._agent_for_turn(turn_index_1based)
        turn_start = (
            (turn_index_1based - 1) // self.turns_per_agent
        ) * self.turns_per_agent + 1
        turn_end = turn_start + self.turns_per_agent - 1

        emb = self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        m = dict(meta or {})
        # IMPORTANT: include turn_end for "newest wins" at query-time
        m.setdefault("turn_end", int(turn_end))
        m.setdefault("turn_start", int(turn_start))

        ch = MemoryChunk(
            stable_id=int(stable_id),
            agent=str(agent),
            turn_start=int(turn_start),
            turn_end=int(turn_end),
            text=str(text),
            ts_unix=int(ts_unix),
            meta=m,
        )
        self.shards[agent].add(emb, ch)

    def persist(self) -> None:
        for s in self.shards.values():
            s.save()

    def _summarize_agent(self, retrieved: List[Retrieved]) -> str:
        if not retrieved:
            return ""
        retrieved = sorted(
            retrieved, key=lambda r: (r.score, r.ts_unix, -r.stable_id), reverse=True
        )
        joined = "\n".join(
            [r.text.strip() for r in retrieved if (r.text or "").strip()]
        )
        return _clip_to_tokens(joined, self.budgets.max_agent_summary_tokens)

    def query(self, text: str, now_unix: int) -> Dict[str, Any]:
        q = self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)

        per_agent: List[Dict[str, Any]] = []
        summaries_for_fuse: List[Tuple[str, float, int, str]] = []

        for agent in self.agents:
            got = self.shards[agent].search(q, self.budgets.topk_per_agent)
            best = max([g.score for g in got], default=0.0)

            newest_turn_end = 0
            if got and isinstance(got[0].meta, dict):
                newest_turn_end = int(got[0].meta.get("turn_end", 0))

            summary = ""
            passed = bool(best >= float(self.thresholds.similarity_gate))
            if passed:
                summary = self._summarize_agent(got)
                summaries_for_fuse.append(
                    (agent, float(best), int(newest_turn_end), summary)
                )

            per_agent.append(
                {
                    "agent": agent,
                    "best_similarity": float(best),
                    "passed_gate": passed,
                    "summary": summary,
                }
            )

        fused = self.fuser.fuse(summaries_for_fuse)
        fused = _clip_to_tokens(fused, int(self.budgets.max_recall_tokens))

        return {
            "query": str(text),
            "now_unix": int(now_unix),
            "threshold": float(self.thresholds.similarity_gate),
            "budgets": asdict(self.budgets),
            "per_agent": per_agent,
            "fused_context": fused,
            "fused_tokens": _simple_token_count(fused),
        }
