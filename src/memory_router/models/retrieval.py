from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class RetrievedItem:
    agent: str
    stable_id: int
    text: str
    ts_unix: int

    match_query: float
    recency: float
    priority: float
    score_global: float

    meta: Dict[str, Any]