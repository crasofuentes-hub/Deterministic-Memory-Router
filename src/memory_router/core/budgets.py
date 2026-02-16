from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Budgets:
    # retrieval
    topk_per_agent: int = 3
    # summarization
    max_agent_summary_tokens: int = 30
    max_recall_tokens: int = 200


@dataclass(frozen=True)
class Thresholds:
    similarity_gate: float = 0.70
