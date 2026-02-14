from pathlib import Path
from memory_router.utils.schema_validator import SchemaRegistry, validate_payload

def test_validate_block_classifier_ok():
    repo_root = Path(__file__).resolve().parents[2]
    schemas_dir = repo_root / "src" / "memory_router" / "config" / "schemas"
    reg = SchemaRegistry(schemas_dir)
    payload = {
        "schema_id": "block_classifier.v1",
        "config_version": "router_v1.0.0",
        "trace_id": "t1",
        "labels": [{"name": "conversation", "confidence": 1.0}],
        "notes": "ok"
    }
    validate_payload(payload, "block_classifier.v1", reg)