from __future__ import annotations

from pathlib import Path
import hashlib

from memory_router.core import MultiAgentMemorySystem, Budgets, Thresholds

def _h(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def test_multi_agent_faiss_deterministic(tmp_path: Path):
    store = tmp_path / "stores"

    sys1 = MultiAgentMemorySystem(
        store_dir=store,
        budgets=Budgets(topk_per_agent=3, max_agent_summary_tokens=30, max_recall_tokens=200),
        thresholds=Thresholds(similarity_gate=0.50),
    )

    now = 1_700_000_000
    for t in range(1, 51):
        txt = f"Turn {t}: prefiero Linux para dev. Detalle determinista."
        sys1.index_turn(stable_id=t, turn_index_1based=t, text=txt, ts_unix=now - t, meta={"turn": t})
    sys1.persist()

    out1 = sys1.query("Linux para dev", now_unix=now)
    h1 = _h(out1["fused_context"])
    assert out1["fused_tokens"] <= 200

    sys2 = MultiAgentMemorySystem(
        store_dir=store,
        budgets=Budgets(topk_per_agent=3, max_agent_summary_tokens=30, max_recall_tokens=200),
        thresholds=Thresholds(similarity_gate=0.50),
    )
    out2 = sys2.query("Linux para dev", now_unix=now)
    h2 = _h(out2["fused_context"])

    assert h1 == h2
    assert out1["fused_tokens"] == out2["fused_tokens"]