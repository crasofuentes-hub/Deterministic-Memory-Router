from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import redis


@dataclass(frozen=True)
class TurnRecord:
    text: str
    signature: str
    ts: float


class RedisHotStorage:
    def __init__(
        self, url: str = "redis://localhost:6379/0", prefix: str = "dmr"
    ) -> None:
        self.r = redis.Redis.from_url(url, decode_responses=True)
        self.prefix = prefix

    def _turn_key(self, user_key: str, turn_id: str) -> str:
        return f"{self.prefix}:turn:{user_key}:{turn_id}"

    def _idxmap_key(self, user_key: str) -> str:
        return f"{self.prefix}:idxmap:{user_key}"

    def put_turn(
        self,
        user_key: str,
        turn_id: str,
        text: str,
        signature: str,
        ts: Optional[float] = None,
    ) -> None:
        if ts is None:
            ts = time.time()

        tkey = self._turn_key(user_key, turn_id)
        pipe = self.r.pipeline(transaction=True)
        pipe.hset(
            tkey, mapping={"text": text, "signature": signature, "ts": str(float(ts))}
        )
        pipe.execute()

    def get_turn(self, user_key: str, turn_id: str) -> Optional[Dict]:
        d = self.r.hgetall(self._turn_key(user_key, turn_id))
        if not d:
            return None

        try:
            ts = float(d.get("ts", "0") or 0.0)
        except Exception:
            ts = 0.0

        return {
            "text": d.get("text", ""),
            "signature": d.get("signature", ""),
            "ts": ts,
        }

    def put_index_map(self, user_key: str, values: Sequence[str]) -> None:
        key = self._idxmap_key(user_key)
        pipe = self.r.pipeline(transaction=True)
        pipe.delete(key)
        if values:
            pipe.rpush(key, *list(values))
        pipe.execute()

    def get_index_map(
        self, user_key: str, indices: Sequence[int]
    ) -> List[Optional[str]]:
        key = self._idxmap_key(user_key)
        pipe = self.r.pipeline(transaction=False)
        for i in indices:
            pipe.lindex(key, int(i))
        raw = pipe.execute()
        return [x if x is not None else None for x in raw]
