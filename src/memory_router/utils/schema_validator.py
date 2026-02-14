import json
from pathlib import Path
from typing import Any, Dict
from jsonschema import Draft202012Validator

class SchemaRegistry:
    def __init__(self, schemas_dir: Path):
        reg = json.loads((schemas_dir / "registry.json").read_text(encoding="utf-8"))
        self._map = {s["schema_id"]: (schemas_dir / s["path"]) for s in reg["schemas"]}

    def load_schema(self, schema_id: str) -> Dict[str, Any]:
        return json.loads(self._map[schema_id].read_text(encoding="utf-8"))

def validate_payload(payload: Dict[str, Any], schema_id: str, registry: SchemaRegistry) -> None:
    if payload.get("schema_id") != schema_id:
        raise ValueError("schema_id mismatch")
    schema = registry.load_schema(schema_id)
    Draft202012Validator(schema).validate(payload)