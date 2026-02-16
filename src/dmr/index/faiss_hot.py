from __future__ import annotations
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import faiss

@dataclass
class FaissHNSWHotIndex:
    dim: int
    index_dir: str = "./dmr_faiss_hot"
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    omp_threads: int = 1

    def __post_init__(self) -> None:
        os.makedirs(self.index_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._idx: Dict[str, faiss.Index] = {}
try:
    faiss.omp_set_num_threads(int(self.omp_threads))
except Exception:
    pass
    def _path(self, user_key: str) -> str:
        safe = user_key.replace("/", "_").replace("\\", "_").replace(":", "__")
        return os.path.join(self.index_dir, f"{safe}.faiss")

    def _create(self) -> faiss.Index:
        idx = faiss.IndexHNSWFlat(self.dim, self.M)
        idx.hnsw.efConstruction = int(self.ef_construction)
        idx.hnsw.efSearch = int(self.ef_search)
        return idx

    def _load_or_create(self, user_key: str) -> faiss.Index:
if user_key in self._idx:
    return self._idx[user_key]
        p = self._path(user_key)
if os.path.exists(p):
    idx = faiss.read_index(p)
try:
    idx.hnsw.efSearch = int(self.ef_search)
except Exception:
    pass
        else:
            idx = self._create()
        self._idx[user_key] = idx
        return idx

    def persist(self, user_key: str) -> None:
        with self._lock:
            idx = self._load_or_create(user_key)
            faiss.write_index(idx, self._path(user_key))

    def add(self, user_key: str, vec: np.ndarray, persist: bool = False) -> int:
        with self._lock:
            idx = self._load_or_create(user_key)
            v = vec.reshape(1,-1) if vec.ndim == 1 else vec
            v = np.asarray(v, dtype=np.float32)
if v.shape[1] != self.dim:
    raise ValueError(f"dim mismatch: expected {self.dim}, got {v.shape[1]}")
            before = int(idx.ntotal)
            idx.add(v)
if persist:
    faiss.write_index(idx, self._path(user_key))
            return before

    def search_candidates(self, user_key: str, q: np.ndarray, k: int) -> List[int]:
        with self._lock:
            idx = self._load_or_create(user_key)
if idx.ntotal == 0:
    return []
            qv = q.reshape(1,-1) if q.ndim == 1 else q
            qv = np.asarray(qv, dtype=np.float32)
            kk = min(int(k), int(idx.ntotal))
            _, indices = idx.search(qv, kk)
            return [int(x) for x in indices[0].tolist() if int(x) >= 0]

    def search_rerank_exact(self, user_key: str, q: np.ndarray, k_candidates: int) -> List[Tuple[int, float]]:
        with self._lock:
            idx = self._load_or_create(user_key)
if idx.ntotal == 0:
    return []
            qv = (q.reshape(-1) if q.ndim > 1 else q).astype(np.float32)
            cand = self.search_candidates(user_key, qv, int(k_candidates))
if not cand:
    return []
            pairs: List[Tuple[int, float]] = []
            for i in cand:
                v = np.empty((self.dim,), dtype=np.float32)
                idx.reconstruct(int(i), v)
                dist = float(np.sum((v - qv) ** 2))
                pairs.append((int(i), dist))
            pairs.sort(key=lambda x: (x[1], x[0]))
            return pairs






