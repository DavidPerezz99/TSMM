"""Recover core TSMM services after a Windows reboot.

Starts the local signal endpoint, restarts the Telegram listener, and resumes
active trading jobs from persisted runtime state when those processes are not
already running.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.runtime_scope import resolve_runtime_dir

from utils.notification_telegram import send_telegram_broadcast
_PYW = str(Path(sys.executable).with_name("pythonw.exe")) if os.name == "nt" and Path(sys.executable).with_name("pythonw.exe").exists() else sys.executable

TRADING_CFG_PATH = Path(os.environ.get("TRADING_CONFIG_PATH", str(ROOT / "config" / "trading_agent.yaml")))

ACTIVE_JOB_STATUSES = {
    "agent_b_running",
    "running",
    "started",
    "waiting_market_open",
    "agent_a_completed",
    "pending_approval",
}


def _runtime_dir(trading_cfg: Dict[str, Any]) -> Path:
    runtime_root = resolve_runtime_dir(base_dir=ROOT, trading_cfg=trading_cfg)
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root


def _runtime_env(trading_cfg: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    env["TSMM_RUNTIME_DIR"] = str(_runtime_dir(trading_cfg))
    env["TRADING_CONFIG_PATH"] = str(TRADING_CFG_PATH)
    return env


def _telegram_subscribers_path() -> Path:
    return _runtime_dir(_load_yaml(TRADING_CFG_PATH)) / "telegram_subscribers.json"


def _account_profile_label(trading_cfg: Dict[str, Any]) -> str:
    runtime_cfg = dict(trading_cfg.get("runtime") or {}) if isinstance(trading_cfg, dict) else {}
    label = str(runtime_cfg.get("profile_label") or runtime_cfg.get("job_id_prefix") or "").strip()
    if label:
        return label

    listener_cfg = dict(trading_cfg.get("telegram_listener") or {}) if isinstance(trading_cfg, dict) else {}
    command_prefix = str(listener_cfg.get("command_prefix") or "").strip().lstrip("/")
    if command_prefix and command_prefix.lower() != "tsmm":
        return command_prefix.upper()
    return "TSMM"


def _reboot_recovery_alert_message(payload: Dict[str, Any], trading_cfg: Dict[str, Any] | None = None) -> str:
    actions = payload.get("actions") or []
    summarized = []
    for action in actions:
        kind = str((action or {}).get("kind") or "unknown").strip()
        status = str((action or {}).get("status") or "unknown").strip()
        job_id = str((action or {}).get("job_id") or "").strip()
        pid = int((action or {}).get("pid", 0) or 0)
        details = f"{kind}={status}"
        if job_id:
            details += f"({job_id})"
        if pid > 0:
            details += f" pid={pid}"
        summarized.append(details)

    summary = "; ".join(summarized) if summarized else "no recovery actions were required"
    account_label = _account_profile_label(trading_cfg or {})
    return (
        f"[{account_label}] TSMM reboot recovery completed\n"
        f"checked_at_utc={payload.get('checked_at', 'n/a')}\n"
        f"enabled={payload.get('enabled', False)} dry_run={payload.get('dry_run', False)}\n"
        f"actions: {summary}"
    )


def _send_console_recovery_alert(message: str, timeout_seconds: int = 20) -> Dict[str, Any]:
    if os.name != "nt":
        return {"ok": False, "skipped": True, "reason": "non_windows"}

    safe_lines = [str(line).replace("'", "''") for line in str(message or "TSMM reboot recovery completed").splitlines()]
    script_lines = [
        "$Host.UI.RawUI.WindowTitle = 'TSMM Reboot Recovery'",
        "$raw = @(",
    ]
    script_lines.extend([f"'{line}'" for line in safe_lines])
    script_lines.extend([
        ")",
        "$raw | ForEach-Object { Write-Host $_ -ForegroundColor Yellow }",
        f"Start-Sleep -Seconds {max(int(timeout_seconds or 20), 5)}",
    ])
    command = "; ".join(script_lines)

    creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0  # type: ignore[attr-defined]
    try:
        proc = subprocess.Popen(
            ["powershell.exe", "-NoLogo", "-NoExit", "-Command", command],
            cwd=str(ROOT),
            creationflags=creationflags,
        )
        return {"ok": True, "pid": int(proc.pid)}
    except Exception:
        try:
            user32 = ctypes.windll.user32
            MB_OK = 0x00000000
            MB_ICONINFORMATION = 0x00000040
            user32.MessageBoxW(None, str(message), "TSMM Reboot Recovery", MB_OK | MB_ICONINFORMATION)
            return {"ok": True, "fallback": "message_box"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def _send_recovery_alerts(trading_cfg: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    recovery_cfg = dict(trading_cfg.get("boot_recovery") or {})
    message = _reboot_recovery_alert_message(payload, trading_cfg)
    alerts: Dict[str, Any] = {"message": message}

    if bool(recovery_cfg.get("send_telegram_alert", True)):
        alerts["telegram"] = send_telegram_broadcast(
            trading_cfg.get("telegram_notifications") or {},
            message,
            subscribers_path=str(_telegram_subscribers_path()),
        )

    if bool(recovery_cfg.get("send_console_alert", True)):
        alerts["console"] = _send_console_recovery_alert(
            message,
            timeout_seconds=int(recovery_cfg.get("console_alert_timeout_seconds", 20) or 20),
        )

    return alerts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover TSMM runtime services after reboot")
    parser.add_argument("--dry-run", action="store_true", help="Report planned recovery actions without launching processes")
    return parser.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _cmdline_contains_all(markers: List[str]) -> bool:
    wanted = [str(m).lower() for m in markers if str(m).strip()]
    if not wanted:
        return False
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
        except Exception:
            continue
        if cmdline and all(marker in cmdline for marker in wanted):
            return True
    return False


def _parity_enforcer_running() -> bool:
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
        except Exception:
            continue
        if "enforce_broker_parity.py" in cmdline:
            return True
    return False


def _endpoint_watchdog_running(trading_cfg_path: Path) -> bool:
    target_path = str(trading_cfg_path.resolve()).lower()
    target_name = trading_cfg_path.name.lower()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            raw_cmdline = [str(part or "") for part in (proc.info.get("cmdline") or [])]
        except Exception:
            continue
        cmdline = [part.lower() for part in raw_cmdline]
        cmdline_text = " ".join(cmdline)
        if "endpoint_liveness_watchdog.py" not in cmdline_text:
            continue
        if "--trading-config" in cmdline:
            idx = cmdline.index("--trading-config")
            if idx + 1 < len(raw_cmdline):
                candidate = _resolve_listener_config_arg(raw_cmdline[idx + 1]).resolve()
                candidate_text = str(candidate).lower()
                if candidate_text == target_path or candidate.name.lower() == target_name:
                    return True
            continue
        # If no explicit config argument is present, assume default-profile watchdog.
        default_target = str((ROOT / "config" / "trading_agent.yaml").resolve()).lower()
        if target_path == default_target:
            return True
    return False


def _listener_process_running(trading_cfg_path: Path) -> bool:
    resolved_target = str(trading_cfg_path.resolve()).lower()
    default_target = str((ROOT / "config" / "trading_agent.yaml").resolve()).lower()
    target_name = trading_cfg_path.name.lower()

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            raw_cmdline = [str(part or "") for part in (proc.info.get("cmdline") or [])]
        except Exception:
            continue
        cmdline = [part.lower() for part in raw_cmdline]
        cmdline_text = " ".join(cmdline)
        if "telegram_command_listener.py" not in cmdline_text:
            continue
        if "--trading-config" in cmdline:
            idx = cmdline.index("--trading-config")
            if idx + 1 < len(raw_cmdline):
                candidate = _resolve_listener_config_arg(raw_cmdline[idx + 1])
                if _listener_covers_target(candidate, trading_cfg_path):
                    return True
                if candidate.name.lower() == target_name and target_name:
                    return True
            continue
        if resolved_target == default_target or _listener_covers_target(ROOT / "config" / "trading_agent.yaml", trading_cfg_path):
            return True
    return False


def _resolve_listener_config_arg(raw_value: str) -> Path:
    raw_path = Path(str(raw_value or "").strip())
    return raw_path if raw_path.is_absolute() else (ROOT / raw_path)


def _listener_covers_target(running_cfg_path: Path, target_cfg_path: Path) -> bool:
    running_resolved = _resolve_listener_config_arg(str(running_cfg_path)).resolve()
    target_resolved = _resolve_listener_config_arg(str(target_cfg_path)).resolve()
    if str(running_resolved).lower() == str(target_resolved).lower():
        return True

    running_cfg = _load_yaml(running_resolved)
    mirror_cfg = dict(running_cfg.get("account_mirror") or {})
    peer_rel = str(mirror_cfg.get("peer_trading_config_path") or "").strip()
    if not peer_rel:
        return False
    peer_resolved = _resolve_listener_config_arg(peer_rel).resolve()
    return str(peer_resolved).lower() == str(target_resolved).lower()


def _launch_detached(args: List[str], *, env: Dict[str, str] | None = None, stdout_path: Path | None = None) -> Dict[str, Any]:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    stdout_handle = subprocess.DEVNULL
    stderr_handle = subprocess.DEVNULL
    file_handle = None
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        file_handle = open(stdout_path, "a", encoding="utf-8")
        stdout_handle = file_handle
        stderr_handle = file_handle

    try:
        proc = subprocess.Popen(
            args,
            cwd=str(ROOT),
            env=env or os.environ.copy(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            creationflags=creationflags,
        )
    finally:
        if file_handle is not None:
            file_handle.close()

    return {"ok": True, "pid": int(proc.pid), "args": args}


def _endpoint_env(trading_cfg: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    cfg = dict(trading_cfg.get("endpoint_lifecycle") or {})
    env["TSMM_SIGNAL_HOST"] = str(cfg.get("host", "127.0.0.1"))
    env["TSMM_SIGNAL_PORT"] = str(int(cfg.get("port", 8000) or 8000))
    return env


def _registry_payload() -> Dict[str, Any]:
    trading_cfg = _load_yaml(TRADING_CFG_PATH)
    return _load_json(_runtime_dir(trading_cfg) / "trading_job_registry.json")


def _state_matches_job(job_id: str, state_path: Path, state: Dict[str, Any]) -> bool:
    if not isinstance(state, dict) or not state:
        return False
    state_job_id = str(state.get("job_id") or "").strip()
    if state_job_id != str(job_id):
        return False
    declared_state_path = str(state.get("state_path") or "").strip()
    if declared_state_path and Path(declared_state_path).resolve() != state_path.resolve():
        return False
    return True


def _state_for_job(job_id: str, registry: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict((registry.get("jobs") or {}).get(job_id) or {})
    state_path = str(meta.get("state_path") or "").strip()
    if state_path:
        resolved_path = Path(state_path)
        state = _load_json(resolved_path)
        if _state_matches_job(job_id, resolved_path, state):
            return state
    return {}


def _resumable_job_ids(registry: Dict[str, Any]) -> List[str]:
    candidate_ids: List[str] = []
    for raw in (registry.get("active_job_ids") or []):
        job_id = str(raw or "").strip()
        if job_id and job_id not in candidate_ids:
            candidate_ids.append(job_id)
    latest = str(registry.get("latest_job_id") or "").strip()
    if latest and latest not in candidate_ids:
        candidate_ids.append(latest)

    resumable: List[str] = []
    for job_id in candidate_ids:
        state = _state_for_job(job_id, registry)
        status = str((state or {}).get("status") or "").strip().lower()
        stage = str((state or {}).get("stage") or "").strip().lower()
        approved = bool((state or {}).get("agent_a_approved"))
        has_order = bool((state or {}).get("order"))
        has_position = bool((state or {}).get("position"))
        if status not in ACTIVE_JOB_STATUSES:
            continue
        if stage == "agent_b":
            resumable.append(job_id)
            continue
        if status == "waiting_market_open" and stage == "agent_a" and approved:
            resumable.append(job_id)
            continue
        if stage == "agent_a" and approved and (has_order or has_position):
            resumable.append(job_id)
    return resumable


def main() -> int:
    args = parse_args()
    trading_cfg = _load_yaml(TRADING_CFG_PATH)
    runtime_dir = _runtime_dir(trading_cfg)
    recovery_cfg = dict(trading_cfg.get("boot_recovery") or {})
    log_path_raw = str(recovery_cfg.get("log_path") or "").strip()
    log_path = ((ROOT / log_path_raw).resolve() if log_path_raw and not os.path.isabs(log_path_raw) else Path(log_path_raw)) if log_path_raw else (runtime_dir / "reboot_recovery_last.json")

    payload: Dict[str, Any] = {
        "checked_at": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": bool(args.dry_run),
        "enabled": bool(recovery_cfg.get("enabled", True)),
        "actions": [],
    }

    if not bool(recovery_cfg.get("enabled", True)):
        payload["message"] = "boot recovery disabled"
        _write_json(log_path, payload)
        print(json.dumps(payload, indent=2))
        return 0

    if bool(recovery_cfg.get("restart_endpoint", True)):
        if _cmdline_contains_all(["local_signal_endpoint_service.py"]):
            payload["actions"].append({"kind": "endpoint", "status": "already_running"})
        else:
            action = {"kind": "endpoint", "status": "planned" if args.dry_run else "started"}
            if not args.dry_run:
                out = _launch_detached(
                    [_PYW, str((ROOT / "scripts" / "local_signal_endpoint_service.py").resolve())],
                    env={**_endpoint_env(trading_cfg), **_runtime_env(trading_cfg)},
                    stdout_path=runtime_dir / "endpoint_recovery.log",
                )
                action.update(out)
            payload["actions"].append(action)

    if bool(recovery_cfg.get("restart_endpoint_watchdog", False)):
        if _endpoint_watchdog_running(TRADING_CFG_PATH):
            payload["actions"].append({"kind": "endpoint_watchdog", "status": "already_running"})
        else:
            action = {"kind": "endpoint_watchdog", "status": "planned" if args.dry_run else "started"}
            if not args.dry_run:
                interval_seconds = max(int(recovery_cfg.get("endpoint_watchdog_interval_seconds", 15) or 15), 5)
                failure_threshold = max(int(recovery_cfg.get("endpoint_watchdog_failure_threshold", 2) or 2), 1)
                cooldown_seconds = max(int(recovery_cfg.get("endpoint_watchdog_cooldown_seconds", 45) or 45), 10)
                health_timeout_seconds = max(float(recovery_cfg.get("endpoint_watchdog_health_timeout_seconds", 5.0) or 5.0), 1.0)
                out = _launch_detached(
                    [
                        _PYW,
                        str((ROOT / "scripts" / "endpoint_liveness_watchdog.py").resolve()),
                        "--trading-config",
                        str(TRADING_CFG_PATH.resolve()),
                        "--interval-sec",
                        str(interval_seconds),
                        "--failure-threshold",
                        str(failure_threshold),
                        "--restart-cooldown-sec",
                        str(cooldown_seconds),
                        "--health-timeout-sec",
                        str(health_timeout_seconds),
                    ],
                    env={**_runtime_env(trading_cfg), **_endpoint_env(trading_cfg)},
                    stdout_path=runtime_dir / "endpoint_watchdog.log",
                )
                action.update(out)
            payload["actions"].append(action)

    if bool(recovery_cfg.get("restart_listener", True)):
        if _listener_process_running(TRADING_CFG_PATH):
            payload["actions"].append({"kind": "listener", "status": "already_running"})
        else:
            action = {"kind": "listener", "status": "planned" if args.dry_run else "started"}
            if not args.dry_run:
                out = _launch_detached(
                    [
                        sys.executable,
                        str((ROOT / "scripts" / "telegram_command_listener.py").resolve()),
                        "--trading-config",
                        str(TRADING_CFG_PATH.resolve()),
                    ],
                    env=_runtime_env(trading_cfg),
                    stdout_path=runtime_dir / "listener_recovery.log",
                )
                action.update(out)
            payload["actions"].append(action)

    if bool(recovery_cfg.get("restart_parity_enforcer", True)):
        if _parity_enforcer_running():
            payload["actions"].append({"kind": "parity_enforcer", "status": "already_running"})
        else:
            action = {"kind": "parity_enforcer", "status": "planned" if args.dry_run else "started"}
            if not args.dry_run:
                source_cfg = str(recovery_cfg.get("parity_source_config") or "config/trading_agent.yaml").strip()
                target_cfg = str(recovery_cfg.get("parity_target_config") or "config/trading_agent_ftmo.yaml").strip()
                interval_seconds = max(int(recovery_cfg.get("parity_interval_seconds", 20) or 20), 5)
                out = _launch_detached(
                    [
                        sys.executable,
                        "-u",
                        str((ROOT / "scripts" / "enforce_broker_parity.py").resolve()),
                        "--source-config",
                        source_cfg,
                        "--target-config",
                        target_cfg,
                        "--interval",
                        str(interval_seconds),
                    ],
                    env=_runtime_env(trading_cfg),
                    stdout_path=runtime_dir / "parity_enforcer.log",
                )
                action.update(out)
            payload["actions"].append(action)

    if bool(recovery_cfg.get("resume_active_jobs", True)):
        registry = _registry_payload()
        for job_id in _resumable_job_ids(registry):
            if _cmdline_contains_all(["app.py", "trading-job", "--job-id", job_id]):
                payload["actions"].append({"kind": "trading_resume", "job_id": job_id, "status": "already_running"})
                continue
            action = {"kind": "trading_resume", "job_id": job_id, "status": "planned" if args.dry_run else "started"}
            if not args.dry_run:
                out = _launch_detached(
                    [sys.executable, str((ROOT / "app.py").resolve()), "trading-job", "resume", "--job-id", job_id],
                    env=_runtime_env(trading_cfg),
                    stdout_path=runtime_dir / f"resume_{job_id}.log",
                )
                action.update(out)
            payload["actions"].append(action)

    if not args.dry_run:
        payload["alerts"] = _send_recovery_alerts(trading_cfg, payload)

    _write_json(log_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())