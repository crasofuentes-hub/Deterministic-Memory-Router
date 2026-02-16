from __future__ import annotations
import numpy as np
from dmr.vectorize.emotion import EmotionAnalyzer

class DeterministicVectorizer:
    def __init__(self) -> None:
        self._emo = EmotionAnalyzer()

    @property
    def dim(self) -> int:
        return 20

    def text_to_vector(self, text: str) -> np.ndarray:
        words = text.split()
        wc = len(words) or 1
        len_score = min(len(text)/400.0, 1.0)
        speed = min(wc/12.0, 2.0)
        reps = (len(set(words))/wc) if wc else 1.0
        dots = min(text.count(".")+text.count("...")*2, 6.0)/6.0
        caps = sum(c.isupper() for c in text)/max(len(text), 1)
        emo = self._emo.analyze(text)
        emo5 = np.array([emo.scores["joy"],emo.scores["sad"],emo.scores["anxiety"],emo.scores["anger"],emo.scores["calm"]], dtype=np.float32)
        base = np.array([len_score, speed, reps, dots, caps], dtype=np.float32)
        meta = np.array([emo.arousal, emo.valence], dtype=np.float32)
        pad = np.zeros(8, dtype=np.float32)
        return np.concatenate([base, emo5, meta, pad]).astype(np.float32)