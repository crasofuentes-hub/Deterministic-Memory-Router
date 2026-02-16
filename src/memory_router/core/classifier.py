from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict

CODE_TOKENS = (
    "def ", "class ", "import ", "from ", "try:", "except", "raise ", "return ",
    "{", "}", ";", "=>", "function ", "const ", "let ", "var ", "public ", "private ",
)

PREF_PATTERNS = (
    r"\b(prefiero|prefiere|prefieres|me gusta|me encantan?|suelo usar|uso)\b",
    r"\b(from now on|i prefer|i use|i usually use|my preference)\b",
    r"\b(ya no uso|a partir de ahora|dej[eé] de|solo uso|he cambiado a)\b",
)

@dataclass(frozen=True)
class Label:
    name: str
    confidence: float

@dataclass(frozen=True)
class ClassifierResult:
    labels: List[Label]
    code_ratio: float
    split: str  # "none" | "code-heavy" | "mixed"
    code_lines: List[str]
    non_code_lines: List[str]
    notes: str

def _is_code_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if s.startswith("```") or s.startswith("#") or s.startswith("//"):
        return True
    for tok in CODE_TOKENS:
        if tok in s:
            return True
    return False

def _code_ratio(lines: List[str]) -> float:
    if not lines:
        return 0.0
    hits = sum(1 for ln in lines if _is_code_line(ln))
    return hits / max(1, len(lines))

def classify_block(text_block: str, code_ratio_threshold: float = 0.30) -> ClassifierResult:
    lines = text_block.splitlines()
    cr = _code_ratio(lines)

    code_lines = [ln for ln in lines if _is_code_line(ln)]
    non_code_lines = [ln for ln in lines if not _is_code_line(ln)]

    split = "none"
    if cr >= code_ratio_threshold and non_code_lines:
        split = "mixed"  # hay separación útil
    elif cr >= code_ratio_threshold:
        split = "code-heavy"

    labels: List[Label] = []

    # Heurística: code
    if cr >= code_ratio_threshold:
        labels.append(Label("code", min(1.0, 0.6 + cr * 0.4)))

    # Heurística: preferences
    pref_hit = any(re.search(p, text_block, re.IGNORECASE) for p in PREF_PATTERNS)
    if pref_hit:
        labels.append(Label("preferences", 0.85))

    # Default: conversation si nada fuerte
    if not labels:
        labels = [Label("conversation", 1.0)]
        notes = "default_conversation"
    else:
        # Siempre añade conversation con baja confianza si hay mezcla
        labels.append(Label("conversation", 0.55))
        # Orden determinista: preferences, code, conversation, other
        order = {"preferences": 0, "code": 1, "conversation": 2, "other": 3}
        labels = sorted(labels, key=lambda x: (order.get(x.name, 99), -x.confidence))
        notes = "heuristic_classifier_v1"

    # Dedup por name manteniendo max confidence, orden determinista
    seen: Dict[str, float] = {}
    for lb in labels:
        seen[lb.name] = max(seen.get(lb.name, 0.0), lb.confidence)
    labels2 = []
    for name in ("preferences","code","conversation","other"):
        if name in seen:
            labels2.append(Label(name, float(seen[name])))

    return ClassifierResult(
        labels=labels2,
        code_ratio=float(cr),
        split=split,
        code_lines=code_lines,
        non_code_lines=non_code_lines,
        notes=notes,
    )