"""Continuous watchdog that keeps the local signal endpoint healthy.

Behavior:
- Polls GET /health on configured host/port.
- Restarts endpoint service after N consecutive health check failures.
- Enforces a single endpoint process by terminating duplicate workers.

This script is intended to run detached in the background.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import psutil
import requests
import yaml
_PYW = str(Path(sys.executable).with_name("pythonw.exe")) if os.name == "nt" and Path(sys.executable).with_name("pythonw.exe").exists() else sys.executable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime_scope import resolve_runtime_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSMM endpoint liveness watchdog")
    parser.add_argument(
        "--trading-config",
        default=os.environ.get("TRADING_CONFIG_PATH", str(ROOT / "config" / "trading_agent.yaml")),
        help="Trading config path used for endpoint host/port and runtime scoping",
    )
    parser.add_argument("--interval-sec", type=int, default=15, help="Health-check interval")
    parser.add_argument(
        "--failure-threshold",
        type=int,
        default=2,
        help="Consecutive failed probes before a restart attempt",
    )
    parser.add_argument(
        "--restart-cooldown-sec",
        type=int,
        default=45,
        help="Minimum seconds between restart attempts",
    )
    parser.add_argument(
        "--health-timeout-sec",
        type=float,
        default=5.0,
        help="HTTP timeout for /health checks",
    )
    parser.add_argument(
        "--log-path",
        default="",
        help="Optional explicit log path (defaults to runtime/endpoint_watchdog.log)",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _runtime_dir(trading_cfg: Dict[str, Any]) -> Path:
    runtime_root = resolve_runtime_dir(base_dir=ROOT, trading_cfg=trading_cfg)
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root


def _runtime_env(trading_cfg_path: Path, trading_cfg: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    env["TRADING_CONFIG_PATH"] = str(trading_cfg_path)
    env["TSMM_RUNTIME_DIR"] = str(_runtime_dir(trading_cfg))
    return env


def _endpoint_env(trading_cfg: Dict[str, Any]) -> Dict[str, str]:
    endpoint_cfg = dict(trading_cfg.get("endpoint_lifecycle") or {})
    env = os.environ.copy()
    env["TSMM_SIGNAL_HOST"] = str(endpoint_cfg.get("host", "127.0.0.1"))
    env["TSMM_SIGNAL_PORT"] = str(int(endpoint_cfg.get("port", 8000) or 8000))
    return env


def _now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _log_line(log_path: Path, message: str) -> None:
    line = f"[{_now_utc()}] {message}"
    print(line, flush=True)
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _cmdline_parts(proc: psutil.Process) -> List[str]:
    try:
        return [str(part or "") for part in (proc.info.get("cmdline") or [])]
    except Exception:
        return []


def _endpoint_processes(service_script: Path, port: int = 8000) -> List[Tuple[int, float]]:
    target_full = str(service_script.resolve()).lower()
    target_name = service_script.name.lower()
    matched: List[Tuple[int, float]] = []

    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            name = str(proc.info.get("name") or "").lower()
            if name not in {"python", "python.exe"}:
                continue
            cmdline = _cmdline_parts(proc)
            if not cmdline:
                continue
            joined = " ".join(cmdline).lower()
            if target_full in joined or target_name in joined:
                matched.append((int(proc.info.get("pid") or 0), float(proc.info.get("create_time") or 0.0)))
        except Exception:
            continue

    matched = [(pid, created) for pid, created in matched if pid > 0]

    # Fallback: if psutil returned nothing, check which PID owns the listening port
    # (handles "Not Responding" processes that psutil can't read)
    if not matched:
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if conn.status == "LISTEN" and conn.laddr.port == port and conn.pid:
                    matched.append((int(conn.pid), 0.0))
        except Exception:
            pass

    matched.sort(key=lambda item: item[1])
    return matched


def _health_ok(host: str, port: int, timeout_seconds: float) -> bool:
    url = f"http://{host}:{port}/health"
    try:
        response = requests.get(url, timeout=max(timeout_seconds, 1.0))
        return response.status_code == 200
    except Exception:
        return False


def _launch_endpoint(
    service_script: Path,
    trading_cfg_path: Path,
    trading_cfg: Dict[str, Any],
    stdout_log: Path,
) -> int:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    env = {**_runtime_env(trading_cfg_path, trading_cfg), **_endpoint_env(trading_cfg)}
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    file_handle = open(stdout_log, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            [_PYW, str(service_script.resolve())],
            cwd=str(ROOT),
            env=env,
            stdout=file_handle,
            stderr=file_handle,
            creationflags=creationflags,
        )
        return int(proc.pid)
    finally:
        file_handle.close()


def _terminate_pid(pid: int) -> None:
    if pid <= 0:
        return
    try:
        psutil.Process(pid).terminate()
    except Exception:
        return


def main() -> int:
    args = _parse_args()
    trading_cfg_path = Path(str(args.trading_config or "").strip())
    if not trading_cfg_path.is_absolute():
        trading_cfg_path = (ROOT / trading_cfg_path).resolve()

    trading_cfg = _load_yaml(trading_cfg_path)
    endpoint_cfg = dict(trading_cfg.get("endpoint_lifecycle") or {})
    host = str(endpoint_cfg.get("host", "127.0.0.1"))
    port = int(endpoint_cfg.get("port", 8000) or 8000)
    service_script_raw = str(endpoint_cfg.get("service_script") or "scripts/local_signal_endpoint_service.py").strip()
    service_script = Path(service_script_raw)
    if not service_script.is_absolute():
        service_script = (ROOT / service_script).resolve()

    runtime_dir = _runtime_dir(trading_cfg)
    log_path = Path(str(args.log_path or "").strip())
    if not str(log_path):
        log_path = runtime_dir / "endpoint_watchdog.log"
    elif not log_path.is_absolute():
        log_path = (ROOT / log_path).resolve()

    interval_sec = max(int(args.interval_sec or 0), 5)
    failure_threshold = max(int(args.failure_threshold or 0), 1)
    restart_cooldown_sec = max(int(args.restart_cooldown_sec or 0), 10)
    health_timeout_sec = max(float(args.health_timeout_sec or 0.0), 1.0)

    _log_line(
        log_path,
        (
            "endpoint watchdog started "
            f"host={host} port={port} interval={interval_sec}s threshold={failure_threshold} "
            f"cooldown={restart_cooldown_sec}s config={trading_cfg_path}"
        ),
    )

    consecutive_failures = 0
    last_restart_ts = 0.0

    while True:
        healthy = _health_ok(host=host, port=port, timeout_seconds=health_timeout_sec)
        endpoint_pids = _endpoint_processes(service_script, port=port)

        # Keep a single endpoint worker to avoid port contention after repeated recoveries.
        if len(endpoint_pids) > 1:
            for pid, _created in endpoint_pids[:-1]:
                _terminate_pid(pid)
                _log_line(log_path, f"terminated duplicate endpoint pid={pid}")
            endpoint_pids = endpoint_pids[-1:]

        if healthy:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            _log_line(
                log_path,
                f"health check failed failures={consecutive_failures}/{failure_threshold} endpoint_pids={[pid for pid, _ in endpoint_pids]}",
            )

        should_restart = consecutive_failures >= failure_threshold
        cooldown_ok = (time.time() - last_restart_ts) >= float(restart_cooldown_sec)

        if should_restart and cooldown_ok:
            for pid, _created in endpoint_pids:
                _terminate_pid(pid)
                _log_line(log_path, f"terminated unhealthy endpoint pid={pid}")

            try:
                new_pid = _launch_endpoint(
                    service_script=service_script,
                    trading_cfg_path=trading_cfg_path,
                    trading_cfg=trading_cfg,
                    stdout_log=runtime_dir / "endpoint_recovery.log",
                )
                last_restart_ts = time.time()
                consecutive_failures = 0
                _log_line(log_path, f"started endpoint pid={new_pid}")
            except Exception as exc:
                _log_line(log_path, f"restart failed error={type(exc).__name__}:{exc}")

        time.sleep(float(interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
