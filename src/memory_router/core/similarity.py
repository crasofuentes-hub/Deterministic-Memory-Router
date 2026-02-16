from __future__ import annotations
import math
import re
from typing import Dict, List

_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_]+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(text or "")]

def tf(tokens: List[str]) -> Dict[str, float]:
    m: Dict[str, float] = {}
    for t in tokens:
        m[t] = m.get(t, 0.0) + 1.0
    if not tokens:
        return m
    inv = 1.0 / float(len(tokens))
    for k in list(m.keys()):
        m[k] = m[k] * inv
    return m

def idf(docs: List[List[str]]) -> Dict[str, float]:
    df: Dict[str, int] = {}
    n = len(docs)
    for toks in docs:
        seen = set(toks)
        for t in seen:
            df[t] = df.get(t, 0) + 1
    out: Dict[str, float] = {}
    for t, c in df.items():
        # smooth idf: log((n+1)/(c+1)) + 1
        out[t] = math.log((n + 1.0) / (c + 1.0)) + 1.0
    return out

def tfidf_vec(tokens: List[str], idf_map: Dict[str, float]) -> Dict[str, float]:
    t = tf(tokens)
    v: Dict[str, float] = {}
    for k, val in t.items():
        v[k] = val * idf_map.get(k, 0.0)
    return v

def cos_sim(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # dot
    dot = 0.0
    # iterar sobre el más pequeño para determinismo y eficiencia
    if len(a) > len(b):
        a, b = b, a
    for k, va in a.items():
        vb = b.get(k)
        if vb is not None:
            dot += va * vb
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))