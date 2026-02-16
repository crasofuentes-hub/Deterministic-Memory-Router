from __future__ import annotations

from emotional_state_tracker.recall import recall_emotion
from emotional_state_tracker.tracker import EmotionalTracker


def test_recall_detects_similarity():
    t = EmotionalTracker()
    t.track("Estoy ansioso...", user_id="u1")
    msg = recall_emotion(t, "Estoy ansioso...", user_id="u1", threshold=0.70)
    assert "Recuerdo:" in msg