import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

from ..utils.hashing import sha256_hex
from ..utils.schema_validator import SchemaRegistry, validate_payload
from ..core.classifier import classify_block

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _load_corpus_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows

def _route_labels(labels, conf_threshold: float) -> List[Dict[str, Any]]:
    picked = [{"name": l.name, "confidence": float(l.confidence)} for l in labels if l.confidence >= conf_threshold]
    if not picked:
        return [{"name": "conversation", "confidence": 1.0}]
    # orden estable
    order = {"preferences": 0, "code": 1, "conversation": 2, "other": 3}
    picked = sorted(picked, key=lambda x: (order.get(x["name"], 99), -x["confidence"]))
    return picked

def run_offline(
    corpus_path: Path,
    out_dir: Path,
    schemas_dir: Path,
    config_version: str,
    trace_id: str,
    mode: str = "generate",
    label_confidence: float = 0.65,
    code_ratio_threshold: float = 0.30,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = SchemaRegistry(schemas_dir)

    rows = _load_corpus_jsonl(corpus_path)

    items: List[Dict[str, Any]] = []
    for row in rows:
        text_block = str(row.get("text_block", ""))
        if not text_block.strip():
            continue

        res = classify_block(text_block, code_ratio_threshold=code_ratio_threshold)

        # split policy: si mixed -> 2 payloads (code_lines y non_code_lines)
        splits: List[Tuple[str, str]] = []
        if res.split == "mixed":
            if res.non_code_lines:
                splits.append(("non_code", "\n".join(res.non_code_lines)))
            if res.code_lines:
                splits.append(("code", "\n".join(res.code_lines)))
        else:
            splits.append(("full", text_block))

        for split_kind, split_text in splits:
            # re-clasifica cada split (determinista)
            rr = classify_block(split_text, code_ratio_threshold=code_ratio_threshold)
            routed = _route_labels(rr.labels, label_confidence)

            payload = {
                "schema_id": "block_classifier.v1",
                "config_version": config_version,
                "trace_id": trace_id,
                "labels": routed,
                "notes": f"classifier_v1 split={split_kind} code_ratio={rr.code_ratio:.3f}"
            }
            validate_payload(payload, "block_classifier.v1", registry)
            items.append(payload)

    manifest = {
        "config_version": config_version,
        "trace_id": trace_id,
        "timestamp": _utc_now_iso(),
        "items_count": len(items),
        "items_hash": sha256_hex(items),
        "policy": {
            "label_confidence": float(label_confidence),
            "code_ratio_threshold": float(code_ratio_threshold)
        }
    }

    manifest_path = out_dir / "run_manifest.json"
    if mode == "generate":
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    elif mode == "check":
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        for k in ("config_version", "trace_id", "items_count", "items_hash", "policy"):
            if expected.get(k) != manifest.get(k):
                raise AssertionError(f"Golden mismatch key={k}: expected={expected.get(k)} actual={manifest.get(k)}")
    else:
        raise ValueError("mode must be generate|check")

    return manifest

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--schemas", required=True)
    p.add_argument("--config-version", required=True)
    p.add_argument("--trace-id", default="trace_offline_0001")
    p.add_argument("--mode", default="generate")
    p.add_argument("--label-confidence", type=float, default=0.65)
    p.add_argument("--code-ratio-threshold", type=float, default=0.30)
    args = p.parse_args()

    run_offline(
        corpus_path=Path(args.corpus),
        out_dir=Path(args.out),
        schemas_dir=Path(args.schemas),
        config_version=args.config_version,
        trace_id=args.trace_id,
        mode=args.mode,
        label_confidence=args.label_confidence,
        code_ratio_threshold=args.code_ratio_threshold,
    )