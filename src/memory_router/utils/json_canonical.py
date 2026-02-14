from __future__ import annotations

import json
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any


def _to_jsonable(x: Any) -> Any:
    """
    Convierte objetos a estructuras JSON-serializables de forma determinista.
    - dataclasses -> asdict (recursivo)
    - Path -> str
    - objetos con __dict__ -> dict (sin privados)
    - dict/list/tuple/set -> recursivo (set -> lista ordenada)
    """
    if x is None or isinstance(x, (bool, int, float, str)):
        return x

    if isinstance(x, Path):
        return str(x)

    if is_dataclass(x):
        return _to_jsonable(asdict(x))

    if isinstance(x, dict):
        # claves a str para estabilidad
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    if isinstance(x, set):
        # orden determinista
        return sorted([_to_jsonable(v) for v in x], key=lambda z: json.dumps(z, sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    # objetos "simples" con atributos
    d = getattr(x, "__dict__", None)
    if isinstance(d, dict):
        clean = {k: v for k, v in d.items() if not str(k).startswith("_")}
        return _to_jsonable(clean)

    # último recurso: string (determinista)
    return str(x)


def canonical_dumps(obj: Any) -> str:
    """
    JSON canónico determinista:
      - conversión recursiva a JSON-friendly
      - sort_keys=True
      - separators sin espacios
      - ensure_ascii=False (UTF-8 estable)
    """
    j = _to_jsonable(obj)
    return json.dumps(j, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def canonical_json(obj: Any) -> str:
    # alias retro-compat usado por hashing.py
    return canonical_dumps(obj)


def canonical_bytes(obj: Any) -> bytes:
    return canonical_dumps(obj).encode("utf-8")