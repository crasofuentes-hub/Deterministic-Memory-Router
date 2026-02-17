from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .vectorize import text_to_vector


@dataclass(frozen=True)
class TrackResult:
    state: str
    sha8: str


class EmotionalTracker:
    """
    Offline + deterministic.
    Stores per-user history: sha256(text) -> vector
    """

    def __init__(self) -> None:
        self.states_by_user: dict[str, dict[str, np.ndarray]] = {}

    def track(self, text: str, user_id: str = "default") -> TrackResult:
        vec = text_to_vector(text)
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        bucket = self.states_by_user.setdefault(user_id, {})
        bucket[key] = vec

        # Centroids are fixed constants (5D) aligned with vectorize() output.
        # Note: tune later; current values are simple, deterministic anchors.
        centroids: dict[str, np.ndarray] = {
            "calmo": np.array([0.10, 0.40, 0.90, 0.50, 0.05], dtype=np.float32),
            "ansioso": np.array([0.30, 1.20, 0.60, 2.00, 0.10], dtype=np.float32),
            "feliz": np.array([0.20, 0.80, 0.85, 1.50, 0.12], dtype=np.float32),
            "triste": np.array([0.25, 0.60, 0.70, 0.80, 0.02], dtype=np.float32),
            "neutral": np.array([0.15, 0.50, 0.80, 0.60, 0.06], dtype=np.float32),
        }

        dists = {name: float(np.linalg.norm(c - vec)) for name, c in centroids.items()}
        state = min(dists, key=dists.get)

        return TrackResult(state=state, sha8=key[:8])
