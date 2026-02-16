from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class EmotionVector:
    scores: Dict[str, float]
    signature: str


class EmotionAnalyzer:
    WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9_]+")

    LEX = {
        "joy": {"feliz", "alegre", "contento", "genial", "bien"},
        "sadness": {"triste", "deprimido", "mal"},
        "anger": {"enojo", "enfadado", "molesto"},
        "fear": {"miedo", "asustado", "terror"},
    }

    INTENS = {"muy", "super", "re", "mega"}
    NEG = {"no", "nunca", "jamas", "jamás"}

    def analyze(self, text: str) -> EmotionVector:
        words = [w.lower() for w in self.WORD.findall(text or "")]
        scores: Dict[str, float] = {k: 0.0 for k in self.LEX.keys()}

        for i, w in enumerate(words):
            for emo, lex in self.LEX.items():
                if w in lex:
                    s = 1.0
                    if i > 0 and words[i - 1] in self.INTENS:
                        s *= 1.5
                    if i > 0 and words[i - 1] in self.NEG:
                        s *= 0.5
                    scores[emo] += s

        sig = hashlib.sha256(("emotion|" + (text or "")).encode("utf-8")).hexdigest()
        return EmotionVector(scores=scores, signature=sig)
