#!/usr/bin/env python
"""Wrapper to run evaluate_sac.py from the repo root while preserving imports."""
from pathlib import Path
import subprocess
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "evaluate_sac", *sys.argv[1:]]
    subprocess.run(cmd, check=True, cwd=repo_root)


if __name__ == "__main__":
    main()
