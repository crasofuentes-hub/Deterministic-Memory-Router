from __future__ import annotations

import importlib.util
import pytest


def _has_sentence_transformers() -> bool:
    return importlib.util.find_spec("sentence_transformers") is not None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "embeddings: tests que requieren sentence-transformers/torch (instala con: pip install -e '.[embeddings]')",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if _has_sentence_transformers():
        return

    skip = pytest.mark.skip(
        reason="Falta 'sentence-transformers'. Instala: pip install -e '.[embeddings]'"
    )
    for item in items:
        if "embeddings" in item.keywords:
            item.add_marker(skip)
