from __future__ import annotations

from scipy.spatial.distance import cosine

from .tracker import EmotionalTracker
from .vectorize import text_to_vector


def recall_emotion(
    tracker: EmotionalTracker,
    new_text: str,
    user_id: str = "default",
    threshold: float = 0.70,
) -> str:
    """
    Returns a deterministic advisory string based on cosine similarity.
    threshold in [0,1]. Higher => stricter match.
    """
    new_vec = text_to_vector(new_text)
    bucket = tracker.states_by_user.get(user_id, {})

    similar = 0
    for _key, vec in bucket.items():
        sim = 1.0 - float(cosine(new_vec, vec))
        if sim >= threshold:
            similar += 1

    if similar > 0:
        return f"Recuerdo: {similar} estados parecidos. Mantén el tono."
    return "Nuevo patrón. Sigo normal."
