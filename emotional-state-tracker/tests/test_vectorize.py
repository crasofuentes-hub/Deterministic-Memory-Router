from __future__ import annotations

import numpy as np

from emotional_state_tracker.vectorize import text_to_vector


def test_text_to_vector_shape_dtype():
    v = text_to_vector("Hola. Todo bien.")
    assert isinstance(v, np.ndarray)
    assert v.shape == (5,)
    assert v.dtype == np.float32


def test_text_to_vector_deterministic():
    a = text_to_vector("Estoy ansioso...")
    b = text_to_vector("Estoy ansioso...")
    assert np.allclose(a, b)