#!/usr/bin/env python3
"""CLI wrapper: python data/synthetic/generate_dataset.py [-n 20000]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aavas.synthetic.generator import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
