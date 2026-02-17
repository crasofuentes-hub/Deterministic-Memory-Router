from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _force_determinism() -> None:
    # Hard determinism knobs (best-effort). Must run BEFORE importing/creating tracker internals.
    os.environ.setdefault("PYTHONHASHSEED", "0")

    # Python random
    try:
        import random  # noqa: F401
        random.seed(0)
    except Exception:
        pass

    # NumPy
    try:
        import numpy as np  # type: ignore
        np.random.seed(0)
    except Exception:
        pass

    # Torch (if used)
    try:
        import torch  # type: ignore

        torch.manual_seed(0)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(0)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass


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


def _to_jsonable(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, float):
        if x != x:
            return "NaN"
        if x == float("inf"):
            return "Infinity"
        if x == float("-inf"):
            return "-Infinity"
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if hasattr(x, "model_dump") and callable(getattr(x, "model_dump")):
        return _to_jsonable(x.model_dump())
    if is_dataclass(x):
        return _to_jsonable(asdict(x))
    if hasattr(x, "__dict__"):
        d = {k: v for k, v in vars(x).items() if not str(k).startswith("_")}
        return _to_jsonable(d)
    return str(x)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> int:
    _force_determinism()
    _ensure_src_on_path()

    from emotional_state_tracker.tracker import EmotionalTracker  # noqa: E402

    this = Path(__file__).resolve()
    data_path = this.parents[1] / "data" / "golden_emotions.json"

    golden = json.loads(data_path.read_text(encoding="utf-8-sig"))
    expected = str(golden.get("hash", "")).strip()

    items = golden.get("items", [])
    tracker = EmotionalTracker()

    states = [_to_jsonable(tracker.track(it["text"])) for it in items] if items else []

    payload = {
        "schema": golden.get("schema", "v1"),
        "items": items,
        "states": states,
    }
    got = _sha256_hex(_stable_json_dumps(payload))

    if expected in ("", "REPLACE_ME_AFTER_FIRST_RUN"):
        print(f"Estado sin fijar: expected={expected or 'MISSING'} got={got}")
        return 1

    if got != expected:
        print(f"Estado roto: expected={expected} got={got}")
        return 1

    print("OK: estado determinista verificado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())