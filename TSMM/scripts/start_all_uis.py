"""Start all TSMM UI apps in one command.

Starts:
- dashboard.py (8050)
- ui.py (8051)
- validation_dashboard.py (8052)

Usage:
  python scripts/start_all_uis.py
  python scripts/start_all_uis.py --force-restart
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, Dict, List

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "reports" / "runtime"
PID_FILE = RUNTIME_DIR / "ui_processes.json"

UI_SPECS = [
    {"name": "dashboard", "script": "dashboard.py", "port": 8050, "url": "http://127.0.0.1:8050"},
    {"name": "config_ui", "script": "ui.py", "port": 8051, "url": "http://127.0.0.1:8051"},
    {"name": "validation_dashboard", "script": "validation_dashboard.py", "port": 8052, "url": "http://127.0.0.1:8052"},
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start all TSMM UI apps")
    p.add_argument("--force-restart", action="store_true", help="Kill existing UI processes first")
    return p.parse_args()


def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", int(port))) == 0


def _kill_known_ui_processes() -> List[int]:
    killed: List[int] = []
    targets = ["dashboard.py", "ui.py", "validation_dashboard.py"]
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if any(t in cmd for t in targets):
                proc.kill()
                killed.append(int(proc.pid))
        except Exception:
            pass
    return killed


def _start_ui(script_name: str) -> int:
    script_path = ROOT / script_name
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    p = subprocess.Popen(
        [_PYW, str(script_path)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        env=os.environ.copy(),
    )
    return int(p.pid)


def main() -> int:
    args = parse_args()

    if args.force_restart:
        _kill_known_ui_processes()

    started: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for spec in UI_SPECS:
        if _is_port_open(int(spec["port"])):
            skipped.append({"name": spec["name"], "port": spec["port"], "reason": "port_already_open", "url": spec["url"]})
            continue
        pid = _start_ui(str(spec["script"]))
        started.append({"name": spec["name"], "port": spec["port"], "pid": pid, "url": spec["url"]})

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "python": sys.executable,
        "started": started,
        "skipped": skipped,
    }
    PID_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
