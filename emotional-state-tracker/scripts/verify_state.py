from __future__ import annotations

import hashlib
import json
from pathlib import Path

from emotional_state_tracker.tracker import EmotionalTracker


def _stable_digest(tracker: EmotionalTracker, user_id: str = "default") -> str:
    bucket = tracker.states_by_user.get(user_id, {})
    # Deterministic ordering by key
    items = [(k, bucket[k].tolist()) for k in sorted(bucket.keys())]
    blob = json.dumps(items, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def main() -> None:
    data_path = Path("data") / "golden_emotions.json"
    golden = json.loads(data_path.read_text(encoding="utf-8"))

    inputs = golden["inputs"]
    expected = golden["hash"]

    t = EmotionalTracker()
    for s in inputs:
        t.track(s)

    got = _stable_digest(t)
    if got != expected:
        raise SystemExit(f"Estado roto: expected={expected} got={got}")

    print("✓ Determinista emocional")


if __name__ == "__main__":
    main()