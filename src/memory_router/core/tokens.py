from __future__ import annotations

import re

_ws = re.compile(r"\s+")

def _simple_token_count(text: str) -> int:
    if not text:
        return 0
    return len([t for t in _ws.split(text.strip()) if t])

def _clip_to_tokens(text: str, max_tokens: int) -> str:
    if not text:
        return ""
    if max_tokens <= 0:
        return ""
    toks = [t for t in _ws.split(text.strip()) if t]
    if len(toks) <= max_tokens:
        return text.strip()
    return " ".join(toks[:max_tokens]).strip()