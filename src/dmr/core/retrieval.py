from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from dmr.vectorize import DeterministicVectorizer
from dmr.index.faiss_hot import FaissHNSWHotIndex
from dmr.storage.cold_sqlite import SQLiteColdStore, ColdRow


@dataclass(frozen=True)
class RetrievalPolicy:
    threshold: float = 0.60
    k_final: int = 5
    max_chars: int = 800
    k_hot_candidates: int = 30
    k_cold_candidates: int = 30
    budget_ms_hot: float = 50.0
    budget_ms_cold: float = 50.0


@dataclass(frozen=True)
class EvidenceItem:
    turn_id: str
    signature: str
    score: float
    source: str  # "hot" | "cold"
    text: str


class DeterministicRetriever:
    def __init__(
        self,
        vectorizer: DeterministicVectorizer,
        hot_index: FaissHNSWHotIndex,
        hot_storage,
        cold_store: SQLiteColdStore,
        policy: Optional[RetrievalPolicy] = None,
    ):
        self.vectorizer = vectorizer
        self.hot_index = hot_index
        self.hot_storage = hot_storage
        self.cold_store = cold_store
        self.policy = policy or RetrievalPolicy()

    def retrieve(self, tenant_id: str, user_id: str, query: str) -> List[EvidenceItem]:
        user_key = f"{tenant_id}:{user_id}"
        qv = self.vectorizer.text_to_vector(query).astype(np.float32)

        hot = self._retrieve_hot(user_key, qv)
        cold = self._retrieve_cold(tenant_id, user_id, query)

        merged = hot + cold
        merged.sort(key=lambda e: (-e.score, e.turn_id))

        out: List[EvidenceItem] = []
        total_chars = 0
        for e in merged:
            if len(out) >= int(self.policy.k_final):
                break
            if e.score < float(self.policy.threshold):
                continue
            if total_chars + len(e.text) > int(self.policy.max_chars):
                continue
            out.append(e)
            total_chars += len(e.text)

        return out

    def _retrieve_hot(self, user_key: str, qv: np.ndarray) -> List[EvidenceItem]:
        # Hot path is optional (Redis may be unavailable). Retrieval must degrade safely.
        try:
            k = int(self.policy.k_hot_candidates)
            dists, idxs = self.hot_index.search(user_key, qv, k=k)
            if len(idxs) == 0:
                return []
            indices = [int(i) for i in idxs if int(i) >= 0]
            if not indices:
                return []

            # Map FAISS indices to turn_ids via hot storage
            turn_ids: Sequence[str] = self.hot_storage.idxmap_mget(user_key, indices)
            out: List[EvidenceItem] = []
            for i, tid in enumerate(turn_ids):
                if not tid:
                    continue
                if self.hot_storage.tombstoned(user_key, tid):
                    continue
                rec = self.hot_storage.get_turn(user_key, tid)
                if not rec:
                    continue
                dist = float(dists[i]) if i < len(dists) else 1e9
                score = 1.0 / (1.0 + max(dist, 0.0))
                out.append(
                    EvidenceItem(
                        turn_id=str(rec.get("turn_id", tid)),
                        signature=str(rec.get("signature", "")),
                        score=float(score),
                        source="hot",
                        text=str(rec.get("text", "")),
                    )
                )
            return out
        except Exception:
            return []

    def _retrieve_cold(
        self, tenant_id: str, user_id: str, query: str
    ) -> List[EvidenceItem]:
        rows: List[ColdRow] = self.cold_store.search_fts(
            tenant_id,
            user_id,
            query,
            limit=int(self.policy.k_cold_candidates),
            budget_ms=float(self.policy.budget_ms_cold),
        )

        out: List[EvidenceItem] = []
        for r in rows:
            # cold rows don't have vector distance; we use stable rank proxy:
            # score derived deterministically from text length and exact token presence.
            # (FTS already filtered; this keeps deterministic monotonic behavior.)
            score = 0.50
            if query.lower() in r.text.lower():
                score = 0.75

            out.append(
                EvidenceItem(
                    turn_id=str(r.turn_id),
                    signature=str(r.signature),
                    score=float(score),
                    source="cold",
                    text=str(r.text),
                )
            )
        return out
