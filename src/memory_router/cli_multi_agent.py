import argparse
import json
import time
from pathlib import Path

from memory_router.core.multi_agent import MultiAgentMemorySystem


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--mode", choices=["index", "query"], required=True)

    ap.add_argument("--text")
    ap.add_argument("--turn-index", type=int)
    ap.add_argument("--stable-id", type=int)

    ap.add_argument("--query-text")

    args = ap.parse_args()

    store = Path(args.store)

    system = MultiAgentMemorySystem(store_dir=store)

    if args.mode == "index":
        if args.text is None or args.turn_index is None or args.stable_id is None:
            raise ValueError("index mode requires --text --turn-index --stable-id")

        now = int(time.time())

        system.index_turn(
            stable_id=args.stable_id,
            turn_index_1based=args.turn_index,
            text=args.text,
            ts_unix=now,
            meta={"turn_end": args.turn_index},
        )
        system.persist()

        print("Indexed successfully.")
        return 0

    if args.mode == "query":
        if args.query_text is None:
            raise ValueError("query mode requires --query-text")

        now = int(time.time())
        out = system.query(args.query_text, now_unix=now)

        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
