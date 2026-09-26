from __future__ import annotations

import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from pipelines.phase1 import main


if __name__ == "__main__":
    main()
