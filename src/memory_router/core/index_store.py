from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
from ..models.memory_item import MemoryItem


@dataclass
class InMemoryAgentIndex:
    agent: str
    _next_id: int = 1

    def __post_init__(self):
        self.items: List[MemoryItem] = []

    def add(
        self, text: str, ts_unix: int, meta: Dict[str, Any] | None = None
    ) -> MemoryItem:
        if meta is None:
            meta = {}
        item = MemoryItem(
            agent=self.agent,
            stable_id=self._next_id,
            text=text,
            ts_unix=int(ts_unix),
            meta=dict(meta),
        )
        self.items.append(item)
        self._next_id += 1
        return item


class IndexStore:
    def __init__(self):
        self.by_agent: Dict[str, InMemoryAgentIndex] = {}

    def agent(self, agent: str) -> InMemoryAgentIndex:
        if agent not in self.by_agent:
            self.by_agent[agent] = InMemoryAgentIndex(agent=agent)
        return self.by_agent[agent]
