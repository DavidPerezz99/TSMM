"""
Run TSMM validation suite:
1) unit tests under tests/
2) source dataset disruption validator

Usage:
    python scripts/run_validation_suite.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml", help="Config path for source dataset validator")
    return p.parse_args()


def run(cmd: list[str], cwd: Path) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    return int(proc.returncode)


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    rc = run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], root)
    if rc != 0:
        print("\nValidation suite failed at unit tests.")
        sys.exit(rc)

    rc = run([sys.executable, "-B", "scripts/validate_source_dataset.py", "--config", args.config], root)
    if rc != 0:
        print("\nValidation suite failed at source dataset validator.")
        sys.exit(rc)

    print("\nValidation suite completed successfully.")


if __name__ == "__main__":
    main()
