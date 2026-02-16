from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List, Optional, Dict
import redis

@dataclass
class RedisHotStorage:
    r: redis.Redis
    prefix: str = "dmr"

    def _turn_key(self, user_key: str, turn_id: str) -> str:
        return f"{self.prefix}:hot:{user_key}:turn:{turn_id}"

    def _idxmap_key(self, user_key: str) -> str:
        return f"{self.prefix}:hot:{user_key}:idxmap"

    def _tomb_key(self, user_key: str) -> str:
        return f"{self.prefix}:hot:{user_key}:tomb"

    def put_turn(self, user_key: str, turn_id: str, text: str, signature: str, ts: Optional[float] = None) -> None:
if ts is None:
    ts = time.time()
        tkey = self._turn_key(user_key, turn_id)
        pipe = self.r.pipeline(transaction=True)
        pipe.hset(tkey, mapping={"text": text, "signature": signature, "ts": str(float(ts))})
        pipe.rpush(self._idxmap_key(user_key), turn_id)
        pipe.execute()

    def get_turn(self, user_key: str, turn_id: str) -> Optional[Dict]:
        d = self.r.hgetall(self._turn_key(user_key, turn_id))
if not d:
    return None
try:
    ts = float(d.get("ts","0") or 0.0)
except Exception:
    ts = 0.0
        return {"text": d.get("text",""), "signature": d.get("signature",""), "ts": ts}

    def idxmap_mget(self, user_key: str, indices: List[int]) -> List[Optional[str]]:
        key = self._idxmap_key(user_key)
        pipe = self.r.pipeline(transaction=False)
for i in indices:
    pipe.lindex(key, int(i))
        raw = pipe.execute()
        out: List[Optional[str]] = []
        for v in raw:
            out.append(v if isinstance(v, str) and v else None)
        return out

    def tombstone(self, user_key: str, turn_id: str) -> bool:
        pipe = self.r.pipeline(transaction=True)
        pipe.sadd(self._tomb_key(user_key), turn_id)
        pipe.delete(self._turn_key(user_key, turn_id))
        pipe.execute()
        return True

    def is_tombstoned(self, user_key: str, turn_id: str) -> bool:
        return bool(self.r.sismember(self._tomb_key(user_key), turn_id))




