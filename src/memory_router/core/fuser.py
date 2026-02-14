from __future__ import annotations
from typing import Dict, Any, List, Set
from ..utils.schema_validator import SchemaRegistry, validate_payload
from .validator import validate_fuser_output

def fuse(
    mini_summaries: List[Dict[str, Any]],
    *,
    config_version: str,
    trace_id: str,
    registry: SchemaRegistry,
) -> Dict[str, Any]:
    # Allowed snippet refs vienen de entradas code
    allowed_snippet_refs: Set[str] = set()
    for ms in mini_summaries:
        if ms.get("agent") == "code":
            allowed_snippet_refs.add(f"code_block_{ms.get('stable_id')}")

    active_preferences = []
    relevant_code = []
    conversation_facts = []
    notes = []
    pref_seen = set()

    # Orden determinista: ya vienen ordenadas upstream; respetamos
    for ms in mini_summaries:
        agent = ms.get("agent")
        sid = ms.get("stable_id")
        summ = (ms.get("summary") or "").strip()
        if not summ:
            continue

        if agent == "preferences":
            p = summ
            if p not in pref_seen:
                pref_seen.add(p)
                active_preferences.append({"pref": p[:120], "since": config_version})
        elif agent == "code":
            relevant_code.append({"summary": summ[:160], "snippet_ref": f"code_block_{sid}"})
        else:
            conversation_facts.append(summ[:180])

    out = {
        "schema_id": "fuser_output.v1",
        "config_version": config_version,
        "trace_id": trace_id,
        "active_preferences": active_preferences[:20],
        "preferences_history": [],  # M3.1 lo llenaremos con FSM real
        "relevant_code": relevant_code[:20],
        "conversation_facts": conversation_facts[:30],
        "notes": notes[:30],
    }

    validate_fuser_output(out, registry, allowed_snippet_refs)
    return out