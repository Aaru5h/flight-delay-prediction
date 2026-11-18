"""Run the full suite in isolated classical/evaluation and neural processes."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    neural = ["tests/test_step6_neural.py", "tests/test_step7_sequence.py"]
    groups = [["-q", *[f"--ignore={p}" for p in neural]], ["-q", *neural]]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/private/tmp/flight-delay-matplotlib")}
    for args in groups:
        result = subprocess.run([sys.executable, "-m", "pytest", *args], cwd=root, env=env)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
