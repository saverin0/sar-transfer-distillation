#!/usr/bin/env python
"""Convenience wrapper so `python sync.py` works without installing the package.

    python sync.py            pack src/ into every notebook
    python sync.py --check    exit 1 if a notebook holds stale code
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from sartransfer.sync import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
