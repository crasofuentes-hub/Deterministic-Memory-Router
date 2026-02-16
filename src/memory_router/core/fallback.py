from __future__ import annotations
from typing import Dict, Any, List

SECTION_ORDER = (
    "active_preferences",
    "preferences_history",
    "relevant_code",
    "conversation_facts",
    "notes",
)


def fallback_from_mini(mini_summaries: List[Dict[str, Any]]) -> str:
    prefs = []
    code = []
    conv = []

    for ms in mini_summaries:
        a = ms.get("agent")
        s = (ms.get("summary") or "").strip()
        if not s:
            continue
        if a == "preferences":
            prefs.append(s)
        elif a == "code":
            code.append(s)
        else:
            conv.append(s)

    lines = []
    lines.append("Active Preferences:")
    for p in prefs[:10]:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("Relevant Code:")
    for c in code[:10]:
        lines.append(f"- {c}")
    lines.append("")
    lines.append("Conversation Facts:")
    for x in conv[:10]:
        lines.append(f"- {x}")

    return "\n".join(lines).strip()
