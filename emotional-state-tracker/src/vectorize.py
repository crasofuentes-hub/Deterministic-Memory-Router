from __future__ import annotations

import numpy as np


def text_to_vector(text: str) -> np.ndarray:
    """
    Deterministic, offline text->vector.
    Features (all in [0, ~]): [len_score, speed, reps, punct, caps]
    """
    t = text or ""
    words = t.split()
    n_words = len(words)

    len_score = len(t) / 100.0
    speed = n_words / 5.0  # assume ~5 seconds per turn (fixed heuristic)
    reps = (len(set(words)) / n_words) if n_words else 1.0
    punct = float(t.count(".") + t.count("!") + t.count("?") + t.count("..."))
    caps = (sum(1 for c in t if c.isupper()) / len(t)) if t else 0.0

    return np.array([len_score, speed, reps, punct, caps], dtype=np.float32)