from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..core.classifier import classify_block
from ..core.index_store import IndexStore
from ..core.mini_summarizer import make_mini_summary
from ..core.retrieval import retrieve_topk
from ..utils.hashing import sha256_hex
from ..utils.schema_validator import SchemaRegistry, validate_payload
from ..utils.json_canonical import canonical_dumps


def _load_corpus_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _utc_now_iso() -> str:
    # Determinista para tests: NO uses reloj real en golden
    return "1970-01-01T00:00:00Z"


def _as_label_dict(x: Any) -> Dict[str, Any]:
    # Soporta dict o dataclass/obj con atributos .name .confidence
    if isinstance(x, dict):
        return {"name": x.get("name"), "confidence": float(x.get("confidence", 0.0))}
    name = getattr(x, "name", None)
    conf = getattr(x, "confidence", 0.0)
    return {"name": name, "confidence": float(conf)}


def _route_labels(labels: List[Any], thr: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for it in labels:
        d = _as_label_dict(it)
        if d["name"] is None:
            continue
        if float(d["confidence"]) >= float(thr):
            out.append(d)
    # orden estable (input order)
    return out


def _call_retrieve_topk(items: List[Any], query: str, topk: int, now_unix: int) -> List[Any]:
    """
    Llama retrieve_topk sin asumir firma exacta.
    Soporta topk/top_k y now_unix/now.
    """
    sig = inspect.signature(retrieve_topk)
    kwargs: Dict[str, Any] = {}

    # Detecta nombres posibles
    names = [p.lower() for p in sig.parameters.keys()]

    # items + query: preferimos posicionales (más robusto)
    # topk:
    if "topk" in names:
        kwargs["topk"] = int(topk)
    elif "top_k" in names:
        kwargs["top_k"] = int(topk)

    # now_unix:
    if "now_unix" in names:
        kwargs["now_unix"] = int(now_unix)
    elif "now" in names:
        kwargs["now"] = int(now_unix)

    return retrieve_topk(query, items, **kwargs)


def _filter_by_agent(memory_items: List[Any], agent: str) -> List[Any]:
    def get_agent(mi: Any) -> Any:
        if isinstance(mi, dict):
            return mi.get("agent")
        return getattr(mi, "agent", None)

    return [mi for mi in memory_items if get_agent(mi) == agent]


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

    # Stage A: classifier payloads
    classifier_items: List[Dict[str, Any]] = []

    # Stage B: IndexStore por agente (determinista)
    store = IndexStore()
    now = 1_700_000_000  # fijo para recency determinista

    for i, row in enumerate(rows, start=1):
        block_id = str(row.get("block_id", f"b{i:03d}"))
        text_block = str(row.get("text_block", ""))
        if not text_block.strip():
            continue

        res = classify_block(text_block, code_ratio_threshold=code_ratio_threshold)

        splits: List[Tuple[str, str]] = []
        if res.split == "mixed":
            if res.non_code_lines:
                splits.append(("non_code", "\n".join(res.non_code_lines)))
            if res.code_lines:
                splits.append(("code", "\n".join(res.code_lines)))
        else:
            splits.append(("full", text_block))

        for split_kind, split_text in splits:
            rr = classify_block(split_text, code_ratio_threshold=code_ratio_threshold)
            routed = _route_labels(rr.labels, label_confidence)

            payload = {
                "schema_id": "block_classifier.v1",
                "config_version": config_version,
                "trace_id": trace_id,
                "labels": routed,
                "notes": f"classifier_v1 split={split_kind} code_ratio={rr.code_ratio:.3f}",
            }
            validate_payload(payload, "block_classifier.v1", registry)
            classifier_items.append(payload)

            ts = now - i
            meta = {"block_id": block_id, "split": split_kind}

            for lab in routed:
                name = lab["name"]
                if name in ("preferences", "code", "conversation"):
                    store.agent(name).add(split_text, ts, meta)

    # Stage C: retrieval determinista (por agente) + merge global simple
    all_items = (
        store.agent("preferences").items
        + store.agent("code").items
        + store.agent("conversation").items
    )

    topk_per_agent = 5
    topn_global = 8

    per_agent = []
    for agent in ("preferences", "code", "conversation"):
        filt = _filter_by_agent(all_items, agent)
        got = _call_retrieve_topk(filt, "offline_query", topk_per_agent, now)
        per_agent.extend(got)

    # Merge global determinista: conserva orden de aparición + corta topn_global
    global_ranked = per_agent[:topn_global]

    # Stage D: mini_summaries (schema mini_summary.v1.1)
    mini_summaries: List[Dict[str, Any]] = []
    for idx, it in enumerate(global_ranked, start=1):
        # best-effort para extraer campos
        agent = getattr(it, "agent", None) if not isinstance(it, dict) else it.get("agent", "conversation")
        stable_id = getattr(it, "stable_id", idx) if not isinstance(it, dict) else it.get("stable_id", idx)
        meta = getattr(it, "meta", {}) if not isinstance(it, dict) else it.get("meta", {})
        block_id = meta.get("block_id", f"b{idx:03d}")

        text = getattr(it, "text", "") if not isinstance(it, dict) else it.get("text", "")
        key_sentences = [text[:200]] if text else [""]

        scores = getattr(it, "scores", None) if not isinstance(it, dict) else it.get("scores")
        if not isinstance(scores, dict):
            scores = {"match_query": 0.0, "recency": 0.0, "priority": 0.0, "score_global": 0.0}

        ms = make_mini_summary(
            agent=str(agent or "conversation"),
            stable_id=int(stable_id),
            source_block_ids=[str(block_id)],
            key_sentences=key_sentences,
            scores=scores,
            config_version=config_version,
            trace_id=trace_id,
            registry=registry,
            max_tokens=40,
        )
        mini_summaries.append(ms)

    # Fuser (si no existe o falla, degradamos determinísticamente)
    fused_ok = False
    fused_payload: Dict[str, Any] = {}
    fused_text_fallback = "FUSER_NOT_IMPLEMENTED"

    # Manifest estable (sin timestamp real)
    manifest: Dict[str, Any] = {
        "config_version": config_version,
        "trace_id": trace_id,
        "timestamp": _utc_now_iso(),
        "counts": {
            "classifier_items": len(classifier_items),
            "mini_summaries": len(mini_summaries),
            "global_ranked": len(global_ranked),
        },
        "hashes": {
            "classifier_items": sha256_hex(classifier_items),
            "mini_summaries": sha256_hex(mini_summaries),
            "global_ranked": sha256_hex(global_ranked),
            "fused_payload": sha256_hex(fused_payload) if fused_ok else "FUSER_FAILED",
        },
        "fuser": {"ok": fused_ok},
    }

    manifest_path = out_dir / "run_manifest.json"

    if mode == "generate":
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (out_dir / "classifier_items.json").write_text(
            json.dumps(classifier_items, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "mini_summaries.json").write_text(
            json.dumps(mini_summaries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "global_ranked.json").write_text(
            canonical_dumps(global_ranked), encoding="utf-8"
        )
        if fused_ok:
            (out_dir / "fuser_output.json").write_text(
                json.dumps(fused_payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        else:
            (out_dir / "fuser_fallback.txt").write_text(fused_text_fallback, encoding="utf-8")

    elif mode == "check":
        expected = json.loads(manifest_path.read_text(encoding="utf-8"))
        for k in ("config_version", "trace_id"):
            if expected.get(k) != manifest.get(k):
                raise AssertionError(
                    f"Golden mismatch key={k}: expected={expected.get(k)} actual={manifest.get(k)}"
                )
        for k in ("counts", "hashes"):
            if expected.get(k) != manifest.get(k):
                raise AssertionError(f"Golden mismatch key={k}")

    else:
        raise ValueError("mode must be 'generate' or 'check'")

    # --- test-compat aliases (legacy keys expected by tests) ---
    manifest["items_count"] = int(manifest.get("counts", {}).get("global_ranked", 0))
    manifest["items_hash"]  = str(manifest.get("hashes", {}).get("global_ranked", ""))
    return manifest
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--schemas", required=True)
    ap.add_argument("--config-version", required=True)
    ap.add_argument("--trace-id", required=True)
    ap.add_argument("--mode", choices=["generate", "check"], default="generate")
    ap.add_argument("--label-confidence", type=float, default=0.65)
    ap.add_argument("--code-ratio-threshold", type=float, default=0.30)
    args = ap.parse_args()

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
