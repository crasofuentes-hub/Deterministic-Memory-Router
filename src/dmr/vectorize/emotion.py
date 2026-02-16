from __future__ import annotations
import re, hashlib
from dataclasses import dataclass
from typing import Dict

@dataclass
class EmotionResult:
    scores: Dict[str, float]
    dominant: str
    dominant_score: float
    arousal: float
    valence: float
    signature: str

class EmotionAnalyzer:
    LEX = {
        "joy": ["happy","great","awesome","excellent","good","genial","feliz","excelente","bien"],
        "sad": ["sad","depressed","cry","bad","triste","deprimido","llorar","mal"],
        "anxiety": ["anxious","nervous","worried","panic","ansioso","nervioso","preocupado","panico","pánico"],
        "anger": ["angry","furious","hate","annoyed","enfadado","furioso","odio","molesto"],
        "calm": ["calm","relaxed","peace","ok","tranquilo","relajado","paz"],
    }
    INTENS = ["very","super","ultra","muy","re","demasiado"]
    NEG = ["not","no","never","nunca","jamas","jamás"]

    def analyze(self, text: str) -> EmotionResult:
        t = text.lower()
        words = re.findall(r"[a-záéíóúñü]+", t)
        n = len(words) or 1
        exc = text.count("!") + text.count("¡")
        ell = text.count("...")
        scores = {k: 0.0 for k in self.LEX.keys()}
        for i, w in enumerate(words):
            for emo, lex in self.LEX.items():
                if w in lex:
                    s = 1.0
                    if i > 0 and words[i-1] in self.INTENS: s *= 1.5
                    if i > 0 and words[i-1] in self.NEG: s *= 0.5
                    scores[emo] += s
        for emo in scores:
            scores[emo] = min(round((scores[emo]/n)*10.0, 3), 1.0)
        if exc > 2:
            scores["joy"] = min(scores["joy"]*1.15, 1.0)
            scores["anger"] = min(scores["anger"]*1.15, 1.0)
        if ell > 1:
            scores["anxiety"] = min(scores["anxiety"]*1.2, 1.0)
        dominant = max(scores, key=scores.get)
        arousal = min(round((scores["anxiety"]+scores["anger"]+scores["joy"])/2.0, 3), 1.0)
        pos = scores["joy"] + scores["calm"]
        neg = scores["sad"] + scores["anxiety"] + scores["anger"]
        total = pos + neg or 1.0
        valence = round(pos/total, 3)
        sig = hashlib.sha256(str(sorted(scores.items())).encode("utf-8")).hexdigest()[:16]
        return EmotionResult(scores, dominant, scores[dominant], arousal, valence, sig)