"""Compatibility shim (strict, no guessing silently).

Exports the API expected by batch.runner:
  - retrieve_by_agent
  - merge_global
  - build_mini_inputs

We resolve these symbols from core.retrieval with a small alias table.
If retrieval.py doesn't provide them (directly or via known aliases),
we raise an ImportError with an explicit list of expected names.
"""

from __future__ import annotations

from typing import Any, Callable

from . import retrieval as _r


def _resolve(name: str, candidates: list[str]) -> Callable[..., Any]:
    for c in candidates:
        fn = getattr(_r, c, None)
        if callable(fn):
            return fn
    raise ImportError(
        f"router_scoring: cannot resolve '{name}'. "
        f"Tried: {candidates}. "
        f"Available in core.retrieval: "
        f"{sorted([k for k in dir(_r) if not k.startswith('_')])}"
    )


# --- Resolve with alias tables (minimal, deterministic) ---
retrieve_by_agent = _resolve(
    "retrieve_by_agent",
    [
        "retrieve_by_agent",
        "retrieve_agent",
        "retrieve_per_agent",
        "retrieve_topk_by_agent",
        "retrieve_topk_per_agent",
        "retrieve",
    ],
)

merge_global = _resolve(
    "merge_global",
    [
        "merge_global",
        "merge",
        "merge_ranked",
        "merge_results",
        "global_merge",
    ],
)

build_mini_inputs = _resolve(
    "build_mini_inputs",
    [
        "build_mini_inputs",
        "build_inputs",
        "build_key_sentences",
        "build_extractor_inputs",
        "make_mini_inputs",
    ],
)

__all__ = ["retrieve_by_agent", "merge_global", "build_mini_inputs"]
