from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class MemoryItem:
    agent: str  # "preferences"|"code"|"conversation"
    stable_id: int  # incremental por agente (determinista)
    text: str
    ts_unix: int  # timestamp unix (segundos) para recency
    meta: Dict[str, Any]  # lineage, ids, etc.
