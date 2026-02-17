from __future__ import annotations

import sys
from pathlib import Path

# Add <repo>/emotional-state-tracker/src to sys.path so imports work in monorepo runs
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
