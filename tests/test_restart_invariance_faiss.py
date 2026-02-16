from __future__ import annotations

from pathlib import Path

import numpy as np

from dmr.index.faiss_hot import FaissHNSWHotIndex


def test_restart_invariance_faiss_rerank_exact(tmp_path: Path) -> None:
    user_key = "T:U"
    dim = 20

    idx1 = FaissHNSWHotIndex(dim=dim, index_dir=str(tmp_path), omp_threads=1)

    v1 = np.zeros((dim,), dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros((dim,), dtype=np.float32)
    v2[0] = 2.0
    v3 = np.zeros((dim,), dtype=np.float32)
    v3[0] = 3.0

    idx1.add(user_key, v1)
    idx1.add(user_key, v2)
    idx1.add(user_key, v3)
    idx1.persist(user_key)

    q = np.zeros((dim,), dtype=np.float32)
    q[0] = 2.2

    r1 = idx1.search_rerank_exact(user_key, q, k_candidates=3)

    idx2 = FaissHNSWHotIndex(dim=dim, index_dir=str(tmp_path), omp_threads=1)
    r2 = idx2.search_rerank_exact(user_key, q, k_candidates=3)

    assert r1 == r2
