from __future__ import annotations

import os
import time
import redis
import numpy as np

from dmr.vectorize import DeterministicVectorizer
from dmr.index.faiss_hot import FaissHNSWHotIndex
from dmr.storage.redis_hot import RedisHotStorage
from dmr.storage.cold_sqlite import SQLiteColdStore, ColdRow
from dmr.core.retrieval import DeterministicRetriever, RetrievalPolicy


def test_no_saturation_contract(tmp_path):
    url = os.environ.get("DMR_TEST_REDIS_URL", "").strip()
    if not url:
        return

    r = redis.Redis.from_url(url, decode_responses=True)
    try:
        r.ping()
    except Exception:
        return

    for k in r.scan_iter("dmr_test:*"):
        r.delete(k)

    vectorizer = DeterministicVectorizer()
    dim = vectorizer.dim

    hot_index = FaissHNSWHotIndex(dim=dim, index_dir=str(tmp_path), omp_threads=1)
    hot_storage = RedisHotStorage(r, prefix="dmr_test")
    cold = SQLiteColdStore(path=str(tmp_path / "cold.sqlite3"))

    tenant_id = "t"
    user_id = "u"
    user_key = f"{tenant_id}:{user_id}"
    now = time.time()

    for i in range(500):
        text = f"Human: key_{i}=value_{i}\nAI: ok"
        v = vectorizer.text_to_vector(text).astype(np.float32)
        hot_index.add(user_key, v, persist=False)
        hot_storage.put_turn(user_key, f"h{i}", text, f"sig{i:04d}", now + i)

    hot_index.persist(user_key)

    cold.put_many(
        [
            ColdRow(
                tenant_id,
                user_id,
                f"c{i}",
                f"csig{i:04d}",
                now + i,
                f"Human: alpha_{i} beta_{i}\nAI: ok",
            )
            for i in range(500)
        ]
    )

    policy = RetrievalPolicy(
        threshold=0.10,
        k_final=5,
        max_chars=600,
        k_hot_candidates=50,
        k_cold_candidates=50,
        budget_ms_hot=200.0,
        budget_ms_cold=200.0,
    )

    retriever = DeterministicRetriever(vectorizer, hot_index, hot_storage, cold, policy)
    ev = retriever.retrieve(tenant_id, user_id, "alpha")

    assert len(ev) <= policy.k_final
    assert sum(len(e.text) for e in ev) <= policy.max_chars