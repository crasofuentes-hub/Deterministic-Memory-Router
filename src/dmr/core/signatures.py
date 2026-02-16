from __future__ import annotations
import hashlib
from typing import Iterable, Tuple

def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def pack_signature(
    tenant_id: str,
    user_id: str,
    query: str,
    policy: dict,
    evidence_items: Iterable[Tuple[str, str, float, str]],
) -> str:
    norm = []
    for turn_id, sig, score, source in evidence_items:
        norm.append((turn_id, sig, round(float(score), 6), source))
    norm.sort(key=lambda x: (x[3], x[0], x[1], x[2]))
    s = (
        f"t={tenant_id}|u={user_id}|q={query}|"
        f"thr={float(policy['threshold']):.6f}|k={int(policy['k_final'])}|mx={int(policy['max_chars'])}|"
        f"bh={float(policy['budget_ms_hot']):.3f}|bc={float(policy['budget_ms_cold']):.3f}|"
        f"ev={norm}"
    )
    return sha256_hex(s)[:16]