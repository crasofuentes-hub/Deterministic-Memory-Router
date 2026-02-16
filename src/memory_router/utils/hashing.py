import hashlib
from typing import Any
from .json_canonical import canonical_json


def sha256_hex(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()
