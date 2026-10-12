from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists():
    src_str = str(SRC)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
