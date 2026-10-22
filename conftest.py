"""Make the project root importable when pytest is run from the repository."""

from pathlib import Path

import sys

ROOT = str(Path(__file__).resolve().parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
