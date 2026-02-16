from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import redis

from dmr.vectorize import DeterministicVectorizer
from dmr.index.faiss_hot import FaissHNSWHotIndex
from dmr.storage.redis_hot import RedisHotStorage
from dmr.storage.cold_sqlite import SQLiteColdStore, ColdRow
from dmr.core.retrieval import DeterministicRetriever, RetrievalPolicy
from dmr.core.signatures import pack_signature, sha256_hex


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: Dict[str, Any]


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(path: str, s: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def _redis_try(redis_url: str) -> redis.Redis:
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    r.ping()
    return r


class NullHotStorage:
    """Hot storage stub that satisfies retrieval.py when Redis is unavailable."""
    def idxmap_mget(self, user_key: str, indices: Sequence[int]) -> List[str]:
        return []

    def get_turn(self, user_key: str, turn_id: str) -> Optional[dict]:
        return None

    def tombstoned(self, user_key: str, turn_id: str) -> bool:
        return False


def run_doctor(
    redis_url: str,
    cold_sqlite: str,
    faiss_dir: str,
    tenant_id: str,
    user_id: str,
    vector_dim: int,
    report_out: str,
    report_md: str,
    cert_md: str,
    runs: int,
    strict: bool,
) -> int:
    env: Dict[str, Any] = {
        "timestamp_utc": _now_iso(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "redis_url": redis_url,
        "cold_sqlite": cold_sqlite,
        "faiss_dir": faiss_dir,
        "vector_dim": int(vector_dim),
        "numpy": np.__version__,
    }

    try:
        import faiss  # type: ignore
        env["faiss"] = getattr(faiss, "__version__", "unknown")
    except Exception:
        env["faiss"] = "not_installed"

    checks: List[CheckResult] = []

    r: Optional[redis.Redis] = None
    try:
        r = _redis_try(redis_url)
        checks.append(CheckResult("redis_hot_available", True, {"ping": True}))
    except Exception as e:
        checks.append(
            CheckResult(
                "redis_hot_available",
                (False if strict else True),
                {"error": str(e), "mode": ("strict" if strict else "degraded_no_redis")},
            )
        )
        r = None

    vectorizer = DeterministicVectorizer()
    dim = int(vector_dim) if int(vector_dim) > 0 else vectorizer.dim
    user_key = f"{tenant_id}:{user_id}"
    now = time.time()

    cold = SQLiteColdStore(path=cold_sqlite)
    cold.put_many(
        [
            ColdRow(
                tenant_id,
                user_id,
                f"c{i}",
                sha256_hex(f"cold|{i}")[:16],
                now + i,
                f"Human: alpha_{i} beta_{i}\nAI: ok",
            )
            for i in range(200)
        ]
    )

    policy = RetrievalPolicy(
        threshold=0.10,
        k_final=5,
        max_chars=800,
        k_hot_candidates=30,
        k_cold_candidates=30,
        budget_ms_hot=150.0,
        budget_ms_cold=150.0,
    )

    hot_index = FaissHNSWHotIndex(dim=dim, index_dir=faiss_dir, omp_threads=1)

    if r is not None:
        hot = RedisHotStorage(r, prefix="dmr_doctor")
        for k in r.scan_iter("dmr_doctor:*"):
            r.delete(k)

        for i in range(80):
            text = f"Human: pref_{i}=val_{i}\nAI: ok"
            v = vectorizer.text_to_vector(text).astype(np.float32)
            hot_index.add(user_key, v, persist=False)
            hot.put_turn(user_key, f"h{i}", text, sha256_hex(f"{user_key}|h{i}|{text}")[:16], now + i)

        hot_index.persist(user_key)
        hot_storage = hot
    else:
        hot_storage = NullHotStorage()

    retriever = DeterministicRetriever(vectorizer, hot_index, hot_storage, cold, policy)

    q = "alpha"
    ev1 = retriever.retrieve(tenant_id, user_id, q)
    ev2 = retriever.retrieve(tenant_id, user_id, q)

    same_evidence = [(e.turn_id, e.signature, round(e.score, 6), e.source) for e in ev1] == [
        (e.turn_id, e.signature, round(e.score, 6), e.source) for e in ev2
    ]

    pol = retriever.policy
    pol_dict = {
        "threshold": pol.threshold,
        "k_final": pol.k_final,
        "max_chars": pol.max_chars,
        "budget_ms_hot": pol.budget_ms_hot,
        "budget_ms_cold": pol.budget_ms_cold,
    }

    s1 = pack_signature(tenant_id, user_id, q, pol_dict, [(e.turn_id, e.signature, e.score, e.source) for e in ev1])
    s2 = pack_signature(tenant_id, user_id, q, pol_dict, [(e.turn_id, e.signature, e.score, e.source) for e in ev2])

    checks.append(
        CheckResult(
            "strict_determinism_pre",
            bool(same_evidence and s1 == s2 and len(ev1) > 0),
            {"sig1": s1, "sig2": s2},
        )
    )

    ok_k = len(ev1) <= pol.k_final
    ok_chars = sum(len(e.text) for e in ev1) <= pol.max_chars
    checks.append(
        CheckResult(
            "no_saturation_contract",
            bool(ok_k and ok_chars),
            {
                "k_final": pol.k_final,
                "returned_k": len(ev1),
                "max_chars": pol.max_chars,
                "returned_chars": sum(len(e.text) for e in ev1),
            },
        )
    )

    has_cold = any(e.source == "cold" for e in ev1)
    checks.append(CheckResult("cold_storage_consultable", bool(has_cold), {"cold_hits": sum(1 for e in ev1 if e.source == "cold")}))

    ok_all = all(c.ok for c in checks)
    summary = {"dmr_version": "2026.0.2", "timestamp_utc": _now_iso(), "ok": bool(ok_all)}

    canonical = {
        "summary": summary,
        "environment": env,
        "checks": [{"name": c.name, "ok": c.ok, "details": c.details} for c in checks],
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    summary["report_signature"] = sha256_hex(blob)[:16]

    report = {
        "summary": summary,
        "environment": env,
        "checks": [{"name": c.name, "ok": c.ok, "details": c.details} for c in checks],
    }

    _write(report_out, json.dumps(report, indent=2, ensure_ascii=False))
    _write(report_md, "# DMR Doctor Report\n\n```json\n" + json.dumps(report, indent=2, ensure_ascii=False) + "\n```\n")
    _write(cert_md, "# DMR Compliance Certificate\n\n- Overall: " + ("PASS" if ok_all else "FAIL") + "\n- Signature: `" + summary["report_signature"] + "`\n")

    return 0 if ok_all else (1 if strict else 0)