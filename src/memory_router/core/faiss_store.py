from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import json
import numpy as np

import faiss  # type: ignore

@dataclass(frozen=True)
class MemoryChunk:
    stable_id: int
    agent: str
    turn_start: int
    turn_end: int
    text: str
    ts_unix: int
    meta: Dict[str, Any]

@dataclass(frozen=True)
class Retrieved:
    stable_id: int
    agent: str
    score: float
    text: str
    ts_unix: int
    meta: Dict[str, Any]

class FaissShard:
    """
    A single agent's FAISS store (IndexFlatIP).
    - vectors: L2-normalized; inner product ~= cosine similarity.
    - metadata: JSONL aligned by insertion order (deterministic).
    """

    def __init__(self, root: Path, agent: str, dim: int):
        self.root = root
        self.agent = agent
        self.dim = dim
        self.index_path = root / f"{agent}.faiss"
        self.meta_path = root / f"{agent}.jsonl"
        self.index = faiss.IndexFlatIP(dim)
        self._chunks: List[MemoryChunk] = []

    def load(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._chunks = []
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    self._chunks.append(MemoryChunk(**obj))

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with self.meta_path.open("w", encoding="utf-8") as f:
            for ch in self._chunks:
                f.write(json.dumps(asdict(ch), ensure_ascii=False) + "\n")

    @property
    def count(self) -> int:
        return len(self._chunks)

    def add(self, emb: np.ndarray, chunk: MemoryChunk) -> None:
        # emb shape: (1, dim), float32, normalized
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32)
        self.index.add(emb)
        self._chunks.append(chunk)

    def search(self, q: np.ndarray, topk: int) -> List[Retrieved]:
        if self.count == 0:
            return []
        if q.dtype != np.float32:
            q = q.astype(np.float32)
        scores, idxs = self.index.search(q, int(topk))
        out: List[Retrieved] = []
        for j, ix in enumerate(idxs[0].tolist()):
            if ix < 0 or ix >= self.count:
                continue
            ch = self._chunks[ix]
            out.append(
                Retrieved(
                    stable_id=ch.stable_id,
                    agent=ch.agent,
                    score=float(scores[0][j]),
                    text=ch.text,
                    ts_unix=ch.ts_unix,
                    meta=dict(ch.meta),
                )
            )
        return out