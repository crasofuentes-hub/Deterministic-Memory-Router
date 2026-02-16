from __future__ import annotations

import os
import threading
from typing import List, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError("faiss is required for FaissHNSWHotIndex") from e


class FaissHNSWHotIndex:
    def __init__(self, dim: int, index_dir: str, omp_threads: int = 1) -> None:
        self.dim = int(dim)
        self.index_dir = index_dir
        self._lock = threading.RLock()
        self._idx: dict[str, faiss.Index] = {}

        os.makedirs(self.index_dir, exist_ok=True)
        try:
            faiss.omp_set_num_threads(int(omp_threads))
        except Exception:
            pass

    def _path(self, user_key: str) -> str:
        safe = user_key.replace(":", "_").replace("/", "_")
        return os.path.join(self.index_dir, f"{safe}.faiss")

    def _create(self) -> faiss.Index:
        # HNSW flat L2
        idx = faiss.IndexHNSWFlat(self.dim, 32)
        idx.hnsw.efSearch = 64
        idx.hnsw.efConstruction = 128
        return idx

    def _load_or_create(self, user_key: str) -> faiss.Index:
        with self._lock:
            if user_key in self._idx:
                return self._idx[user_key]

            p = self._path(user_key)
            if os.path.exists(p):
                idx = faiss.read_index(p)
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

            v = vec.reshape(1, -1) if vec.ndim == 1 else vec
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
            if int(idx.ntotal) == 0:
                return []

            qv = q.reshape(1, -1) if q.ndim == 1 else q
            qv = np.asarray(qv, dtype=np.float32)
            kk = min(int(k), int(idx.ntotal))

            dist, idxs = idx.search(qv, kk)
            return [int(x) for x in idxs.reshape(-1).tolist() if int(x) >= 0]

    def search_rerank_exact(
        self, user_key: str, q: np.ndarray, k_candidates: int = 10
    ) -> List[Tuple[int, float]]:
        # Para mantener determinismo: usa faiss search como ranking base.
        with self._lock:
            idx = self._load_or_create(user_key)
            if int(idx.ntotal) == 0:
                return []

            qv = q.reshape(1, -1) if q.ndim == 1 else q
            qv = np.asarray(qv, dtype=np.float32)
            kk = min(int(k_candidates), int(idx.ntotal))

            dist, idxs = idx.search(qv, kk)
            pairs = [
                (int(i), float(d))
                for i, d in zip(idxs.reshape(-1), dist.reshape(-1))
                if int(i) >= 0
            ]
            # Orden estable por distancia y luego id
            pairs.sort(key=lambda t: (t[1], t[0]))
            return pairs
