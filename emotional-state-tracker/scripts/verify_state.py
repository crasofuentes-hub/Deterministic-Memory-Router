from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    """
    Make monorepo package importable when running this script directly.
    Adds: <repo>/emotional-state-tracker/src to sys.path (at position 0).
    """
    this = Path(__file__).resolve()
    src = this.parents[1] / "src"
    if src.is_dir():
        s = str(src)
        if s not in sys.path:
            sys.path.insert(0, s)


def _stable_json_dumps(obj: object) -> str:
    # Deterministic JSON serialization for hashing.
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> int:
    _ensure_src_on_path()

    # Import AFTER sys.path injection (ruff: allow local import)
    from emotional_state_tracker.tracker import EmotionalTracker  # noqa: E402

    this = Path(__file__).resolve()
    data_path = this.parents[1] / "data" / "golden_emotions.json"

    golden = json.loads(data_path.read_text(encoding="utf-8-sig"))
    expected = str(golden.get("hash", "")).strip()

    # Compute current state deterministically from the golden input payload.
    # This assumes golden contains an "items" list used as input corpus.
    items = golden.get("items", [])
    tracker = EmotionalTracker()
    states = [tracker.track(it["text"]) for it in items] if items else []

    payload = {
        "schema": golden.get("schema", "v1"),
        "items": items,
        "states": states,
    }
    got = _sha256_hex(_stable_json_dumps(payload))

    if expected in ("", "REPLACE_ME_AFTER_FIRST_RUN"):
        print(f"Estado roto: expected={expected or 'MISSING'} got={got}")
        return 1

    if got != expected:
        print(f"Estado roto: expected={expected} got={got}")
        return 1

    print("OK: estado determinista verificado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
