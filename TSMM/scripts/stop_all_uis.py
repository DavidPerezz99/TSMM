"""Stop all TSMM UI apps in one command.

Stops:
- apps/dashboard.py
- apps/ui.py
- apps/validation_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "reports" / "runtime"
PID_FILE = RUNTIME_DIR / "ui_processes.json"


def main() -> int:
    targets = ["dashboard.py", "ui.py", "validation_dashboard.py"]
    killed = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if any(t in cmd for t in targets):
                proc.kill()
                killed.append(int(proc.pid))
        except Exception:
            continue

    try:
        PID_FILE.unlink(missing_ok=True)
    except Exception:
        pass

    print(json.dumps({"killed_pids": killed, "count": len(killed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
