from __future__ import annotations
from typing import Dict, Any, Set
from ..utils.schema_validator import SchemaRegistry, validate_payload

def validate_fuser_output_semantic(payload: Dict[str, Any], allowed_snippet_refs: Set[str]) -> None:
    # schema
    # Nota: el caller valida schema_id correcto
    for it in payload.get("relevant_code", []):
        ref = it.get("snippet_ref", "")
        if ref and ref not in allowed_snippet_refs:
            raise ValueError(f"fuser created unknown snippet_ref: {ref}")

def validate_fuser_output(payload: Dict[str, Any], registry: SchemaRegistry, allowed_snippet_refs: Set[str]) -> None:
    validate_payload(payload, "fuser_output.v1", registry)
    validate_fuser_output_semantic(payload, allowed_snippet_refs)