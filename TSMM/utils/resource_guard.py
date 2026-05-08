"""Runtime resource guard for CPU/RAM pressure mitigation.

If CPU or RAM stays above threshold for sustained duration, this guard
can temporarily shed non-critical services to provide recovery time.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Tuple

import psutil


def _runtime_dir(root: Path) -> Path:
    path = root / "reports" / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _append_event(root: Path, payload: Dict[str, Any]) -> None:
    p = _runtime_dir(root) / "resource_guard_events.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _read_state(root: Path) -> Dict[str, Any]:
    p = _runtime_dir(root) / "resource_guard_state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_state(root: Path, payload: Dict[str, Any]) -> None:
    p = _runtime_dir(root) / "resource_guard_state.json"
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _kill_endpoint_processes(root: Path) -> int:
    killed = 0
    pid_file = _runtime_dir(root) / "local_signal_endpoint_service.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
            if pid > 0:
                p = psutil.Process(pid)
                p.kill()
                killed += 1
        except Exception:
            pass
        try:
            pid_file.unlink(missing_ok=True)
        except Exception:
            pass

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if "local_signal_endpoint_service.py" in cmd:
                proc.kill()
                killed += 1
        except Exception:
            continue
    return killed


def _kill_ui_processes() -> int:
    killed = 0
    targets = {"dashboard.py", "ui.py", "validation_dashboard.py"}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(proc.info.get("cmdline") or [])
            if any(t in cmd for t in targets):
                proc.kill()
                killed += 1
        except Exception:
            continue
    return killed


def check_and_relieve(root: Path, trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = (trading_cfg.get("resource_guard") or {})
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "skipped": True, "reason": "disabled_in_config"}

    cpu_threshold = float(cfg.get("cpu_threshold_pct", 95.0) or 95.0)
    ram_threshold = float(cfg.get("ram_threshold_pct", 95.0) or 95.0)
    sustained_sec = int(cfg.get("sustained_seconds", 240) or 240)
    cooldown_sec = int(cfg.get("cooldown_seconds", 600) or 600)

    cpu = float(psutil.cpu_percent(interval=None))
    ram = float(psutil.virtual_memory().percent)
    now = int(time.time())
    state = _read_state(root)

    breached = (cpu >= cpu_threshold) or (ram >= ram_threshold)
    breach_since = int(state.get("breach_since", 0) or 0)
    relieved_at = int(state.get("last_relieved_at", 0) or 0)

    if breached:
        if breach_since <= 0:
            breach_since = now
            state["breach_since"] = breach_since
    else:
        state["breach_since"] = 0
        _write_state(root, state)
        return {
            "enabled": True,
            "breached": False,
            "cpu": cpu,
            "ram": ram,
            "breach_seconds": 0,
        }

    breach_seconds = max(now - breach_since, 0)
    should_relieve = breach_seconds >= sustained_sec and (now - relieved_at) >= cooldown_sec

    if should_relieve:
        kill_endpoint = bool(cfg.get("disable_endpoint_on_relief", True))
        kill_uis = bool(cfg.get("disable_uis_on_relief", True))

        endpoint_killed = _kill_endpoint_processes(root) if kill_endpoint else 0
        ui_killed = _kill_ui_processes() if kill_uis else 0

        state["last_relieved_at"] = now
        state["last_relieved_cpu"] = cpu
        state["last_relieved_ram"] = ram
        _write_state(root, state)

        event = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "action": "resource_relief",
            "cpu": cpu,
            "ram": ram,
            "breach_seconds": breach_seconds,
            "endpoint_killed": endpoint_killed,
            "ui_killed": ui_killed,
        }
        _append_event(root, event)
        return {
            "enabled": True,
            "breached": True,
            "relieved": True,
            "cpu": cpu,
            "ram": ram,
            "breach_seconds": breach_seconds,
            "endpoint_killed": endpoint_killed,
            "ui_killed": ui_killed,
        }

    _write_state(root, state)
    return {
        "enabled": True,
        "breached": True,
        "relieved": False,
        "cpu": cpu,
        "ram": ram,
        "breach_seconds": breach_seconds,
    }


def read_status(root: Path) -> Dict[str, Any]:
    cpu = float(psutil.cpu_percent(interval=None))
    ram = float(psutil.virtual_memory().percent)
    state = _read_state(root)
    return {
        "cpu": cpu,
        "ram": ram,
        "breach_since": int(state.get("breach_since", 0) or 0),
        "last_relieved_at": int(state.get("last_relieved_at", 0) or 0),
    }
