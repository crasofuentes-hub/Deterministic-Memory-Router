from __future__ import annotations

from emotional_state_tracker.tracker import EmotionalTracker


def test_track_returns_state_and_hash8():
    t = EmotionalTracker()
    out = t.track("Hola. Todo bien.")
    assert out.state in {"calmo", "ansioso", "feliz", "triste", "neutral"}
    assert len(out.sha8) == 8


def test_track_stores_state():
    t = EmotionalTracker()
    t.track("Hola. Todo bien.", user_id="u1")
    assert "u1" in t.states_by_user
    assert len(t.states_by_user["u1"]) == 1