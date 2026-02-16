from memory_router.core.classifier import classify_block

def test_classifier_default_conversation():
    r = classify_block("hola\ncomo estas")
    assert any(label.name == "conversation" for label in r.labels)
    assert r.code_ratio == 0.0

def test_classifier_detects_code():
    r = classify_block("def x():\n  return 1\n")
    assert any(label.name == "code" for label in r.labels)
    assert r.code_ratio > 0.0

def test_classifier_detects_preferences_es():
    r = classify_block("yo prefiero Linux para desarrollo")
    assert any(label.name == "preferences" for label in r.labels)

def test_classifier_mixed_split():
    r = classify_block("yo prefiero Linux\n\ndef x():\n  return 1\n", code_ratio_threshold=0.2)
    assert r.split in ("mixed","code-heavy")

