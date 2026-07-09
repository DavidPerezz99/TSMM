"""Always-on Telegram command listener for TSMM backend control.

Supported commands (prefix configurable, default /tsmm):
- /tsmm help|commands|?
- /tsmm status
- /tsmm deploy [--refresh|--no-refresh] [--dry-run] [--no-start-job]
- /tsmm deploy stop
- /tsmm trading start [--plan-model MODEL]
- /tsmm trading resume
- /tsmm trading status
- /tsmm trading stop
- /tsmm endpoint restart
- /tsmm ui start|stop
- /tsmm resource status|relieve

Natural-language chat is also supported (for example: "start trading", "please stop deploy", "how is trading doing?").
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import psutil
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple
from zoneinfo import ZoneInfo

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.notification_telegram import _resolve_secret, send_telegram_notification  # noqa: E402
from utils.resource_guard import check_and_relieve, read_status as read_resource_status  # noqa: E402
from utils.llm_connector import load_llm_providers_config, call_llm  # noqa: E402
from utils.copilot_bridge import queue_copilot_request  # noqa: E402
from utils.market_db import query_ohlc  # noqa: E402
from utils.investing_agent import MT5Adapter, _collect_all_model_assessment_signals, _signal_interpretation_mode  # noqa: E402
from utils.runtime_scope import resolve_job_id_prefix, resolve_runtime_dir  # noqa: E402
from utils.trading_job import _should_auto_request_followup_agent_a  # noqa: E402


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _listener_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(trading_cfg.get("telegram_listener") or {})


def _runtime_root() -> Path:
    return resolve_runtime_dir(base_dir=ROOT)


def _set_runtime_scope_env(trading_cfg: Dict[str, Any]) -> Path:
    runtime_root = resolve_runtime_dir(base_dir=ROOT, trading_cfg=trading_cfg)
    os.environ["TSMM_RUNTIME_DIR"] = str(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    return runtime_root


def _scheduled_refresh_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict((_listener_cfg(trading_cfg).get("scheduled_model_refresh") or {}))


def _weekend_utc_quiet_mode_active(trading_cfg: Dict[str, Any], now_utc: datetime | None = None) -> bool:
    cfg = _scheduled_refresh_cfg(trading_cfg)
    if not bool(cfg.get("weekend_utc_quiet_mode", True)):
        return False

    current_utc = now_utc or datetime.utcnow().replace(tzinfo=timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)

    tz_name = str(cfg.get("weekend_quiet_timezone") or _agent_timezone_name(trading_cfg)).strip() or _agent_timezone_name(trading_cfg)
    start_parts = _parse_hhmm(cfg.get("weekend_quiet_start_time") or "17:00")
    end_parts = _parse_hhmm(cfg.get("weekend_quiet_end_time") or "17:00")
    start_day = int(cfg.get("weekend_quiet_start_day", 4) or 4)
    end_day = int(cfg.get("weekend_quiet_end_day", 6) or 6)
    if start_parts is None or end_parts is None:
        return current_utc.weekday() >= 5
    if start_day < 0 or start_day > 6 or end_day < 0 or end_day > 6:
        return current_utc.weekday() >= 5

    local_now = current_utc.astimezone(ZoneInfo(tz_name))
    days_since_start = (local_now.weekday() - start_day) % 7
    start_date = local_now.date() - timedelta(days=days_since_start)
    start_local = datetime(
        start_date.year,
        start_date.month,
        start_date.day,
        start_parts[0],
        start_parts[1],
        tzinfo=ZoneInfo(tz_name),
    )

    days_until_end = (end_day - start_day) % 7
    if days_until_end == 0 and end_parts <= start_parts:
        days_until_end = 7
    end_date = start_date + timedelta(days=days_until_end)
    end_local = datetime(
        end_date.year,
        end_date.month,
        end_date.day,
        end_parts[0],
        end_parts[1],
        tzinfo=ZoneInfo(tz_name),
    )
    return start_local <= local_now < end_local


def _tg_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(trading_cfg.get("telegram_notifications") or {})


def _ai_agent_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict((_listener_cfg(trading_cfg).get("ai_agent") or {}))


def _external_ops_agent_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict((_ai_agent_cfg(trading_cfg).get("external_service") or {}))


def _agent_timezone_name(trading_cfg: Dict[str, Any]) -> str:
    return str(((trading_cfg.get("agent") or {}).get("timezone") or "UTC")).strip() or "UTC"


def _format_utc_for_agent_timezone(raw_value: Any, trading_cfg: Dict[str, Any]) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return "n/a"
    try:
        dt_utc = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        tz_name = _agent_timezone_name(trading_cfg)
        local_dt = dt_utc.astimezone(ZoneInfo(tz_name))
        return f"{local_dt.strftime('%Y-%m-%d %H:%M:%S')} {tz_name} ({dt_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC)"
    except Exception:
        return raw


def _token_and_chat_ids(trading_cfg: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    tcfg = _tg_cfg(trading_cfg)
    lcfg = _listener_cfg(trading_cfg)

    token = _resolve_secret(str(tcfg.get("bot_token") or "")).strip()
    chat_id_default = _resolve_secret(str(tcfg.get("chat_id") or "")).strip()

    allowed = [str(c).strip() for c in (lcfg.get("allowed_chat_ids") or []) if str(c).strip()]

    return token, chat_id_default, allowed


def _subscriber_path() -> Path:
    p = _runtime_root() / "telegram_subscribers.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _subscriber_chat_ids(trading_cfg: Dict[str, Any], default_chat_id: str = "") -> List[str]:
    payload = _read_json(_subscriber_path())
    configured_allowed = [str(c).strip() for c in (_listener_cfg(trading_cfg).get("allowed_chat_ids") or []) if str(c).strip()]
    subscribers = [str(c).strip() for c in (payload.get("chat_ids") or []) if str(c).strip()]
    merged: List[str] = []
    for chat_id in ([default_chat_id] if default_chat_id else []) + configured_allowed + subscribers:
        if not chat_id or chat_id in merged:
            continue
        merged.append(chat_id)
    return merged


def _register_subscriber(chat_id: str) -> None:
    clean_chat_id = str(chat_id or "").strip()
    if not clean_chat_id:
        return
    payload = _read_json(_subscriber_path())
    chat_ids = [str(c).strip() for c in (payload.get("chat_ids") or []) if str(c).strip()]
    if clean_chat_id not in chat_ids:
        chat_ids.append(clean_chat_id)
    payload["chat_ids"] = chat_ids
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(_subscriber_path(), payload)


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


def _telegram_message_with_account(trading_cfg: Dict[str, Any], message: str) -> str:
    account_label = _account_profile_label(trading_cfg)
    raw_message = str(message or "").strip()
    if not raw_message:
        return account_label
    account_prefix = f"[{account_label}] "
    if raw_message.startswith(account_prefix):
        return raw_message
    return f"{account_prefix}{raw_message}"


def _send_to_chat_ids(
    trading_cfg: Dict[str, Any],
    chat_ids: List[str],
    message: str,
    allow_agent_mode: bool = False,
) -> None:
    scoped_message = _telegram_message_with_account(trading_cfg, message)
    for chat_id in [str(c).strip() for c in chat_ids if str(c).strip()]:
        if (not allow_agent_mode) and _chat_mode(chat_id) == "agent":
            continue
        tcfg = dict(_tg_cfg(trading_cfg))
        tcfg["chat_id"] = chat_id
        send_telegram_notification(tcfg, scoped_message[:3500])


def _resolve_trading_config_path(path_like: Path | str) -> Path:
    raw_path = Path(path_like)
    return raw_path if raw_path.is_absolute() else (ROOT / raw_path)


def _listener_profile_entries(base_trading_config_path: Path, trading_cfg: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    pending: List[Tuple[Path, Dict[str, Any]]] = [(_resolve_trading_config_path(base_trading_config_path), dict(trading_cfg or {}))]
    seen: set[str] = set()

    while pending:
        config_path, cfg = pending.pop(0)
        resolved_path = config_path.resolve()
        resolved_key = str(resolved_path).lower()
        if resolved_key in seen:
            continue
        seen.add(resolved_key)

        current_cfg = dict(cfg or _load_yaml(resolved_path))
        if not current_cfg:
            continue

        prefix = str((_listener_cfg(current_cfg).get("command_prefix") or "/tsmm")).strip() or "/tsmm"
        entries.append(
            {
                "config_path": resolved_path,
                "trading_cfg": current_cfg,
                "prefix": prefix,
            }
        )

        mirror_cfg = dict(current_cfg.get("account_mirror") or {})
        peer_rel = str(mirror_cfg.get("peer_trading_config_path") or "").strip()
        if not peer_rel:
            continue
        peer_path = _resolve_trading_config_path(peer_rel)
        if not peer_path.exists():
            continue
        pending.append((peer_path, _load_yaml(peer_path)))

    return entries


def _select_listener_profile(text: str, profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_text = str(text or "").strip().lower()
    for profile in profiles:
        prefix = str(profile.get("prefix") or "").strip().lower()
        if prefix and raw_text.startswith(prefix):
            return profile
    return profiles[0] if profiles else {"config_path": ROOT / "config" / "trading_agent.yaml", "trading_cfg": {}, "prefix": "/tsmm"}


_telegram_session = requests.Session()
_telegram_session.headers.update({"Connection": "keep-alive"})

def _api_get(token: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    base = f"https://api.telegram.org/bot{token}/{method}"
    try:
        r = _telegram_session.get(base, params=params, timeout=30)
        if r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        return {"ok": False, "raw": r.text[:500]}
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "timeout"}
    except requests.exceptions.ConnectionError as e:
        return {"ok": False, "error": f"connection_error:{e}"}


def _run_cmd(args: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        creationflags=creationflags,
    )
    return {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "args": list(args),
        "stdout": (proc.stdout or "")[-3000:],
        "stderr": (proc.stderr or "")[-3000:],
    }


def _run_cmd_async(args: List[str], env: Dict[str, str]) -> Dict[str, Any]:
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    p = subprocess.Popen(
        args,
        cwd=str(ROOT),
        env=env,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "ok": True,
        "returncode": 0,
        "args": list(args),
        "pid": int(p.pid),
        "stdout": "",
        "stderr": "",
    }


def _stop_local_endpoint_service(script_rel: str) -> List[int]:
    pid_file = ROOT / "reports" / "runtime" / "local_signal_endpoint_service.pid"
    stopped_pids: List[int] = []
    candidate_pids: List[int] = []

    if pid_file.exists():
        try:
            pid = int((pid_file.read_text(encoding="utf-8") or "0").strip() or "0")
        except Exception:
            pid = 0
        if pid > 0:
            candidate_pids.append(pid)

    script_name = Path(script_rel).name.lower()
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
        except Exception:
            continue
        if script_name and script_name in cmdline and proc.pid not in candidate_pids:
            candidate_pids.append(int(proc.pid))

    for pid in candidate_pids:
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            stopped_pids.append(pid)
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
        except Exception:
            continue

    if pid_file.exists():
        pid_file.unlink(missing_ok=True)

    return stopped_pids


def _audit_path() -> Path:
    p = _runtime_root() / "telegram_command_audit.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _conversation_path() -> Path:
    p = _runtime_root() / "telegram_conversation_log.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _llm_session_memory_root() -> Path:
    p = _runtime_root() / "llm_session_memories"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _llm_session_dir(chat_id: str) -> Path:
    return _llm_session_memory_root() / f"chat_{str(chat_id or '').strip()}"


def _chat_mode_state_path() -> Path:
    p = _runtime_root() / "telegram_chat_mode_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _approval_request_path() -> Path:
    return _runtime_root() / "trading_approval_request.json"


def _approval_response_path() -> Path:
    return _runtime_root() / "trading_approval_response.json"


def _trading_state_path() -> Path:
    return _runtime_root() / "trading_job_state.json"


def _job_registry_path() -> Path:
    return _runtime_root() / "trading_job_registry.json"


def _job_root_dir() -> Path:
    return _runtime_root() / "trading_jobs"


def _new_job_id(trading_cfg: Dict[str, Any] | None = None) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    prefix = resolve_job_id_prefix(trading_cfg)
    if prefix:
        return f"job_{prefix}_{stamp}"
    return f"job_{stamp}"


def _job_state_path(job_id: str) -> Path:
    return _job_root_dir() / str(job_id) / _trading_state_path().name


def _job_approval_request_path(job_id: str) -> Path:
    return _job_root_dir() / str(job_id) / _approval_request_path().name


def _job_approval_response_path(job_id: str) -> Path:
    return _job_root_dir() / str(job_id) / _approval_response_path().name


def _registry_payload() -> Dict[str, Any]:
    return _read_json(_job_registry_path())


def _latest_job_id() -> str:
    registry = _registry_payload()
    active_ids = [str(x).strip() for x in (registry.get("active_job_ids") or []) if str(x).strip()]
    if active_ids:
        return active_ids[-1]
    return str(registry.get("latest_job_id") or "").strip()


def _job_stop_flag_path(job_id: str) -> Path:
    return _job_root_dir() / str(job_id) / "trading_job_stop.flag"


def _scheduled_refresh_state_path() -> Path:
    return _runtime_root() / "scheduled_model_refresh_state.json"


def _all_disk_job_ids() -> List[str]:
    root = _job_root_dir()
    if not root.exists():
        return []

    job_ids: List[str] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        job_id = str(child.name or "").strip()
        if not job_id:
            continue
        if (child / _trading_state_path().name).exists():
            job_ids.append(job_id)
    return sorted(job_ids)


def _active_job_statuses() -> set[str]:
    return {"agent_b_running", "running", "started", "waiting_market_open", "agent_a_completed", "pending_approval"}


def _is_active_trading_state(state: Dict[str, Any]) -> bool:
    job_id = str((state or {}).get("job_id") or "").strip()
    if job_id and _pending_approval_for_job(job_id):
        return True

    status = str((state or {}).get("status") or "").strip().lower()
    if status not in _active_job_statuses():
        return False
    if status == "agent_b_running":
        return _job_ticket(state) > 0
    if status == "agent_a_completed":
        stage = str((state or {}).get("stage") or "").strip().lower()
        ended_at = str((state or {}).get("ended_at") or "").strip()
        closed_reason = str((state or {}).get("closed_reason") or "").strip().lower()
        return stage == "agent_a" and not ended_at and not closed_reason
    return True


def _job_display_status(state: Dict[str, Any]) -> str:
    position = (state.get("position") or {}) if isinstance(state, dict) else {}
    order = (state.get("order") or {}) if isinstance(state, dict) else {}
    job_id = str((state or {}).get("job_id") or "").strip()
    raw_status = str((state or {}).get("status") or "n/a").strip().lower() or "n/a"
    if job_id and _pending_approval_for_job(job_id):
        return "pending_approval"
    if isinstance(position, dict) and position:
        return "running_position"
    if isinstance(order, dict) and order:
        return "programmed_order"
    return raw_status


def _active_job_ids(trading_cfg: Dict[str, Any] | None = None) -> List[str]:
    default_cfg_path = Path(os.environ.get("TRADING_CONFIG_PATH", str(ROOT / "config" / "trading_agent.yaml")))
    resolved_trading_cfg = trading_cfg or _load_yaml(default_cfg_path)
    if resolved_trading_cfg:
        _set_runtime_scope_env(resolved_trading_cfg)

    def _job_id_matches_profile(job_id: str) -> bool:
        clean_job_id = str(job_id or "").strip()
        if not clean_job_id:
            return False

        active_prefix = str(resolve_job_id_prefix(resolved_trading_cfg) or "").strip()
        if active_prefix:
            return clean_job_id.startswith(f"job_{active_prefix}_")

        mirror_cfg = dict((resolved_trading_cfg.get("account_mirror") or {})) if isinstance(resolved_trading_cfg, dict) else {}
        peer_profile = str(mirror_cfg.get("peer_profile") or "").strip()
        if peer_profile and clean_job_id.startswith(f"job_{peer_profile}_"):
            return False
        return True

    registry = _registry_payload()
    candidate_ids: List[str] = []
    for raw_id in (registry.get("active_job_ids") or []):
        clean_id = str(raw_id).strip()
        if clean_id and _job_id_matches_profile(clean_id) and clean_id not in candidate_ids:
            candidate_ids.append(clean_id)
    for raw_id in (registry.get("jobs") or {}).keys():
        clean_id = str(raw_id).strip()
        if clean_id and _job_id_matches_profile(clean_id) and clean_id not in candidate_ids:
            candidate_ids.append(clean_id)
    for raw_id in _all_disk_job_ids():
        clean_id = str(raw_id).strip()
        if clean_id and _job_id_matches_profile(clean_id) and clean_id not in candidate_ids:
            candidate_ids.append(clean_id)

    mt5_cfg = (((resolved_trading_cfg.get("broker") or {}).get("mt5") or {})) if isinstance(resolved_trading_cfg, dict) else {}
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, _msg_conn = adapter.connect()
    validated_ids: List[str] = []
    try:
        for job_id in candidate_ids:
            state = _read_trading_state(job_id)
            if not state:
                continue
            if ok_conn:
                refreshed_state = _refresh_job_state_from_mt5(resolved_trading_cfg, state, adapter)
                if refreshed_state != state:
                    state_path_raw = str(refreshed_state.get("state_path") or _job_state_path(job_id))
                    _persist_job_state(Path(state_path_raw), refreshed_state)
                state = refreshed_state
            if _is_active_trading_state(state):
                validated_ids.append(job_id)
        if validated_ids:
            return validated_ids

        latest_state = _read_json(_trading_state_path())
        latest_job_id = str(latest_state.get("job_id") or "").strip()
        if latest_state and ok_conn:
            refreshed_latest = _refresh_job_state_from_mt5(resolved_trading_cfg, latest_state, adapter)
            if refreshed_latest != latest_state:
                latest_path_raw = str(refreshed_latest.get("state_path") or _trading_state_path())
                _persist_job_state(Path(latest_path_raw), refreshed_latest)
                latest_state = refreshed_latest
                latest_job_id = str(latest_state.get("job_id") or latest_job_id).strip()
        if latest_job_id and _is_active_trading_state(latest_state):
            return [latest_job_id]
        return []
    finally:
        if ok_conn:
            adapter.shutdown()


def _autonomy_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(trading_cfg.get("autonomous_trading") or {})


def _parse_hhmm(raw_value: Any) -> Tuple[int, int] | None:
    raw = str(raw_value or "").strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour, minute


def _parse_state_datetime_utc(raw_value: Any) -> datetime | None:
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _current_autonomous_session(trading_cfg: Dict[str, Any], now_utc: datetime | None = None) -> Dict[str, Any]:
    cfg = _autonomy_cfg(trading_cfg)
    if not bool(cfg.get("enabled", False)):
        return {}

    session_windows = cfg.get("session_windows") or []
    if not session_windows:
        return {}

    tz_name = str(cfg.get("timezone") or _agent_timezone_name(trading_cfg)).strip() or _agent_timezone_name(trading_cfg)
    current_utc = now_utc or datetime.utcnow().replace(tzinfo=timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    local_now = current_utc.astimezone(ZoneInfo(tz_name))

    for day_offset in (-1, 0, 1):
        base_date = local_now.date() + timedelta(days=day_offset)
        for raw_window in session_windows:
            window = dict(raw_window or {})
            start_parts = _parse_hhmm(window.get("start"))
            end_parts = _parse_hhmm(window.get("end"))
            session_name = str(window.get("name") or "session").strip().lower() or "session"
            if start_parts is None or end_parts is None:
                continue
            start_local = datetime(
                base_date.year,
                base_date.month,
                base_date.day,
                start_parts[0],
                start_parts[1],
                tzinfo=ZoneInfo(tz_name),
            )
            end_local = datetime(
                base_date.year,
                base_date.month,
                base_date.day,
                end_parts[0],
                end_parts[1],
                tzinfo=ZoneInfo(tz_name),
            )
            if end_local <= start_local:
                end_local = end_local + timedelta(days=1)
            if start_local <= local_now < end_local:
                start_utc = start_local.astimezone(timezone.utc)
                end_utc = end_local.astimezone(timezone.utc)
                return {
                    "name": session_name,
                    "timezone": tz_name,
                    "session_key": f"{session_name}:{start_utc.strftime('%Y%m%d%H%M')}",
                    "start_utc": start_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_utc": end_utc.strftime("%Y-%m-%d %H:%M:%S"),
                    "start_local": start_local.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_local": end_local.strftime("%Y-%m-%d %H:%M:%S"),
                }
    return {}


def _all_known_job_ids() -> List[str]:
    registry = _registry_payload()
    known_ids: List[str] = []
    for raw_id in (registry.get("active_job_ids") or []):
        clean_id = str(raw_id).strip()
        if clean_id and clean_id not in known_ids:
            known_ids.append(clean_id)
    for raw_id in (registry.get("jobs") or {}).keys():
        clean_id = str(raw_id).strip()
        if clean_id and clean_id not in known_ids:
            known_ids.append(clean_id)
    latest_job_id = str(registry.get("latest_job_id") or "").strip()
    if latest_job_id and latest_job_id not in known_ids:
        known_ids.append(latest_job_id)
    return known_ids


def _session_job_states(session_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    start_utc = _parse_state_datetime_utc(session_info.get("start_utc"))
    end_utc = _parse_state_datetime_utc(session_info.get("end_utc"))
    if start_utc is None or end_utc is None:
        return []

    states: List[Dict[str, Any]] = []
    for job_id in _all_known_job_ids():
        state = _read_trading_state(job_id)
        if not state:
            continue
        started_at = _parse_state_datetime_utc(state.get("started_at"))
        if started_at is None or started_at < start_utc or started_at >= end_utc:
            continue
        states.append(state)
    return states


def _autonomous_capacity_limit(trading_cfg: Dict[str, Any]) -> int:
    autonomy = _autonomy_cfg(trading_cfg)
    configured = int(autonomy.get("max_jobs_per_session", 0) or 0)
    if configured > 0:
        return configured
    risk_cfg = (trading_cfg.get("risk") or {})
    return max(int(risk_cfg.get("max_open_positions", 3) or 3), 1)


def _session_has_mandatory_coverage(session_states: List[Dict[str, Any]]) -> bool:
    for state in session_states:
        job_id = str(state.get("job_id") or "").strip()
        status = str(state.get("status") or "").strip().lower()
        if _pending_approval_for_job(job_id):
            return True
        if _job_ticket(state) > 0:
            return True
        if isinstance(state.get("order"), dict) and state.get("order"):
            return True
        if isinstance(state.get("position"), dict) and state.get("position"):
            return True
        if status in {"waiting_market_open", "agent_b_running", "running", "started"}:
            return True
    return False


def _session_active_job_count(session_states: List[Dict[str, Any]]) -> int:
    count = 0
    for state in session_states:
        job_id = str(state.get("job_id") or "").strip()
        if _pending_approval_for_job(job_id):
            count += 1
            continue
        if _is_active_trading_state(state):
            count += 1
    return count


def _state_autonomous_trigger(state: Dict[str, Any]) -> str:
    request_ctx = dict((state or {}).get("request_context") or {})
    return str(request_ctx.get("autonomous_trigger") or "").strip().lower()


def _session_autonomous_stats(session_states: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "mandatory_launches": 0,
        "followup_launches": 0,
        "filtered_followups": 0,
        "last_mandatory_started_at": None,
        "last_followup_started_at": None,
    }
    for state in session_states:
        started_at = _parse_state_datetime_utc(state.get("started_at"))
        trigger = _state_autonomous_trigger(state)
        closed_reason = str(state.get("closed_reason") or "").strip().lower()
        if trigger == "mandatory_session":
            stats["mandatory_launches"] = int(stats.get("mandatory_launches") or 0) + 1
            previous = stats.get("last_mandatory_started_at")
            if started_at is not None and (previous is None or started_at > previous):
                stats["last_mandatory_started_at"] = started_at
        elif trigger == "autonomous_followup":
            stats["followup_launches"] = int(stats.get("followup_launches") or 0) + 1
            previous = stats.get("last_followup_started_at")
            if started_at is not None and (previous is None or started_at > previous):
                stats["last_followup_started_at"] = started_at
            if closed_reason == "autonomous_followup_filtered":
                stats["filtered_followups"] = int(stats.get("filtered_followups") or 0) + 1
    return stats


def _state_has_manual_or_external_close(state: Dict[str, Any]) -> bool:
    if bool((state or {}).get("manual_or_external_close_detected", False)):
        return True

    closed_reason = str((state or {}).get("closed_reason") or "").strip().lower()
    if closed_reason in {"manual_stop", "manual_close_via_telegram", "position_not_found_assumed_closed"}:
        return True

    close_outcome = dict((state or {}).get("close_outcome") or {})
    reason_label = str(close_outcome.get("reason_label") or "").strip().lower()
    if reason_label in {"client", "mobile", "web", "manual"}:
        return True

    comment = str(close_outcome.get("comment") or "").strip().lower()
    if "manual" in comment and reason_label not in {"expert", "sl", "tp", "so"}:
        return True
    return False


def _session_manual_or_external_close_marker(session_states: List[Dict[str, Any]]) -> Dict[str, Any]:
    latest: Dict[str, Any] = {}
    latest_started: datetime | None = None

    for state in session_states:
        if not _state_has_manual_or_external_close(state):
            continue
        started_at = _parse_state_datetime_utc(state.get("started_at"))
        if latest_started is None or (started_at is not None and started_at >= latest_started):
            latest_started = started_at
            latest = dict(state or {})

    if not latest:
        return {"blocked": False}

    return {
        "blocked": True,
        "job_id": str(latest.get("job_id") or "").strip(),
        "closed_reason": str(latest.get("closed_reason") or "").strip(),
        "close_outcome_reason": str((dict(latest.get("close_outcome") or {})).get("reason_label") or "").strip(),
    }


def _seconds_since(moment: datetime | None, now_utc: datetime) -> float:
    if moment is None:
        return float("inf")
    return max((now_utc - moment).total_seconds(), 0.0)


def _mode_b_supervision_active(trading_cfg: Dict[str, Any]) -> bool:
    mb_cfg = dict(trading_cfg.get("mode_b") or {})
    return bool(mb_cfg.get("enabled", False)) and bool(mb_cfg.get("manage_existing_positions", True))


def _launch_autonomous_trading_start(
    trading_cfg: Dict[str, Any],
    trading_config_path: Path,
    submission_mode: str,
    autonomous_trigger: str,
) -> Dict[str, Any]:
    job_id = _new_job_id(trading_cfg)
    env = os.environ.copy()
    env["TRADING_CONFIG_PATH"] = str(trading_config_path)
    args = [
        sys.executable,
        "app.py",
        "trading-job",
        "start",
        "--job-id",
        job_id,
        "--submission-mode",
        submission_mode,
        "--autonomous-trigger",
        autonomous_trigger,
    ]
    out = _run_cmd_async(args, env)
    out["job_id"] = job_id
    out["submission_mode"] = submission_mode
    out["autonomous_trigger"] = autonomous_trigger
    return out


def _fmt_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    raw = str(value).strip()
    return raw or "n/a"


def _format_utc_local_short(raw_value: Any, trading_cfg: Dict[str, Any]) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return "n/a"
    try:
        dt_utc = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        tz_name = _agent_timezone_name(trading_cfg)
        local_dt = dt_utc.astimezone(ZoneInfo(tz_name))
        return local_dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return raw


def _job_ticket(state: Dict[str, Any]) -> int:
    position = (state.get("position") or {}) if isinstance(state, dict) else {}
    order = (state.get("order") or {}) if isinstance(state, dict) else {}
    return int(position.get("ticket") or order.get("order_ticket") or order.get("ticket") or 0)


def _refresh_job_state_from_mt5(trading_cfg: Dict[str, Any], state: Dict[str, Any], adapter: MT5Adapter) -> Dict[str, Any]:
    if not isinstance(state, dict) or not state:
        return state

    refreshed = dict(state)
    plan = (refreshed.get("plan") or {}) if isinstance(refreshed.get("plan"), dict) else {}
    position = (refreshed.get("position") or {}) if isinstance(refreshed.get("position"), dict) else {}
    order = (refreshed.get("order") or {}) if isinstance(refreshed.get("order"), dict) else {}
    stage = str(refreshed.get("stage") or "").strip().lower()
    status = str(refreshed.get("status") or "").strip().lower()
    ended_at = str(refreshed.get("ended_at") or "").strip()
    closed_reason = str(refreshed.get("closed_reason") or "").strip().lower()

    # Do not auto-reactivate jobs that were explicitly terminated.
    if status in {"completed", "closed", "failed", "stopped", "killed"}:
        return refreshed
    if ended_at or closed_reason:
        return refreshed

    submission_mode = str(refreshed.get("order_submission_mode") or plan.get("order_submission_mode") or "programmed").strip().lower()
    agent_a_approved = bool(refreshed.get("agent_a_approved"))

    exec_cfg = (trading_cfg.get("execution") or {})
    symbol = str(exec_cfg.get("symbol") or plan.get("symbol") or "").strip()
    volume = float(plan.get("volume", exec_cfg.get("default_volume", 0.0)) or 0.0)
    entry = float(plan.get("entry", 0.0) or 0.0)
    stop_loss = float(plan.get("stop_loss", 0.0) or 0.0)
    take_profit = float(plan.get("take_profit", 0.0) or 0.0)

    live_position = None
    pos_ticket = int(position.get("ticket", 0) or 0)
    order_ticket = int(order.get("order_ticket", 0) or order.get("ticket", 0) or 0)
    if pos_ticket > 0:
        pos_lookup = adapter.get_position_by_ticket(pos_ticket)
        if pos_lookup.get("ok"):
            live_position = pos_lookup.get("position")
    if not live_position and order_ticket > 0:
        pos_lookup = adapter.find_position_by_order(order_ticket)
        if pos_lookup.get("ok"):
            live_position = pos_lookup.get("position")
    if not live_position and symbol and volume > 0.0 and entry > 0.0:
        pos_lookup = adapter.find_live_position_by_plan(
            symbol=symbol,
            volume=volume,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if pos_lookup.get("ok"):
            live_position = pos_lookup.get("position")

    if isinstance(live_position, dict) and live_position:
        refreshed["position"] = live_position
        refreshed["stage"] = "agent_b"
        refreshed["mode"] = "mode_b"
        refreshed["status"] = "agent_b_running"
        refreshed.pop("ended_at", None)
        refreshed.pop("closed_reason", None)
        return refreshed

    live_order = None
    if order_ticket > 0:
        order_lookup = adapter.get_pending_order_by_ticket(order_ticket)
        if order_lookup.get("ok"):
            live_order = order_lookup.get("order")
    if not live_order and stage == "agent_a" and submission_mode == "programmed" and agent_a_approved and symbol and volume > 0.0 and entry > 0.0:
        order_lookup = adapter.find_pending_order_by_plan(
            symbol=symbol,
            volume=volume,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        if order_lookup.get("ok"):
            live_order = order_lookup.get("order")

    if isinstance(live_order, dict) and live_order:
        refreshed["order"] = live_order
        if status not in {"pending_approval", "waiting_market_open"}:
            refreshed["status"] = "agent_a_completed"
        refreshed.pop("ended_at", None)
        refreshed.pop("closed_reason", None)
    elif order and stage == "agent_a" and submission_mode == "programmed" and status not in {"pending_approval", "waiting_market_open"}:
        expiration_raw = refreshed.get("programmed_order_expiration_utc") or order.get("expiration_utc")
        expiration_dt = _parse_state_datetime_utc(expiration_raw)
        missing_reason = "programmed_order_expired" if expiration_dt and expiration_dt <= datetime.utcnow().replace(tzinfo=timezone.utc) else "programmed_order_missing_in_mt5"
        refreshed.pop("order", None)
        refreshed["status"] = "closed"
        refreshed["ended_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        refreshed["closed_reason"] = missing_reason

    return refreshed


def _pending_order_maintenance_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(((_autonomy_cfg(trading_cfg).get("pending_order_maintenance") or {})))
    default_threshold = float((((trading_cfg.get("mode_b") or {}).get("close_consensus_threshold", 0.25)) or 0.25))
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "assessment_timeout_seconds": max(float(cfg.get("assessment_timeout_seconds", 3.0) or 3.0), 0.5),
        "cancel_opposed_consensus_threshold": abs(float(cfg.get("cancel_opposed_consensus_threshold", default_threshold) or default_threshold)),
        "entry_cross_tolerance_abs": max(float(cfg.get("entry_cross_tolerance_abs", 0.05) or 0.05), 0.0),
    }


def _market_reference_price_for_order(adapter: MT5Adapter, symbol: str, side: str) -> float | None:
    mt5 = getattr(adapter, "_mt5", None)
    if mt5 is None:
        return None
    try:
        tick = mt5.symbol_info_tick(symbol)
    except Exception:
        tick = None
    if tick is None:
        return None
    try:
        if str(side or "").strip().lower() == "buy":
            return float(getattr(tick, "ask", 0.0) or 0.0)
        if str(side or "").strip().lower() == "sell":
            return float(getattr(tick, "bid", 0.0) or 0.0)
    except Exception:
        return None
    return None


def _pending_order_entry_crossed(order: Dict[str, Any], market_price: float | None, tolerance_abs: float) -> bool:
    if market_price is None or market_price <= 0.0:
        return False
    try:
        entry = float((order or {}).get("price_open", 0.0) or 0.0)
    except Exception:
        entry = 0.0
    if entry <= 0.0:
        return False

    order_type = int((order or {}).get("type", -99) or -99)
    tol = max(float(tolerance_abs or 0.0), 0.0)
    if order_type in {2, 6}:  # buy limit / buy stop limit
        return market_price <= entry - tol
    if order_type == 4:  # buy stop
        return market_price >= entry + tol
    if order_type in {3, 7}:  # sell limit / sell stop limit
        return market_price >= entry + tol
    if order_type == 5:  # sell stop
        return market_price <= entry - tol
    return False


def _programmed_order_maintenance_decision(
    state: Dict[str, Any],
    order: Dict[str, Any],
    assessment: Dict[str, Any],
    market_price: float | None,
    trading_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = _pending_order_maintenance_cfg(trading_cfg)
    side = _infer_order_decision(order) or str(((state.get("plan") or {}).get("decision") or "")).strip().lower()
    consensus = str((assessment or {}).get("consensus") or "hold").strip().lower() or "hold"
    score = float((assessment or {}).get("consensus_score", 0.0) or 0.0)
    threshold = float(cfg.get("cancel_opposed_consensus_threshold", 0.25) or 0.25)
    configured_interpretation = _signal_interpretation_mode(trading_cfg)
    plan_interpretation = str((((state.get("plan") or {}).get("signal_interpretation") or "momentum"))).strip().lower() or "momentum"

    if plan_interpretation != configured_interpretation:
        return {
            "cancel": True,
            "reason": "programmed_order_strategy_mismatch",
            "side": side,
            "consensus": consensus,
            "consensus_score": score,
            "market_price": market_price,
        }

    if _pending_order_entry_crossed(order, market_price, float(cfg.get("entry_cross_tolerance_abs", 0.05) or 0.05)):
        return {
            "cancel": True,
            "reason": "programmed_order_entry_crossed",
            "side": side,
            "consensus": consensus,
            "consensus_score": score,
            "market_price": market_price,
        }

    if side == "buy" and consensus == "sell" and score <= -abs(threshold):
        return {
            "cancel": True,
            "reason": "programmed_order_consensus_invalidated",
            "side": side,
            "consensus": consensus,
            "consensus_score": score,
            "market_price": market_price,
        }
    if side == "sell" and consensus == "buy" and score >= abs(threshold):
        return {
            "cancel": True,
            "reason": "programmed_order_consensus_invalidated",
            "side": side,
            "consensus": consensus,
            "consensus_score": score,
            "market_price": market_price,
        }

    return {
        "cancel": False,
        "reason": "keep_programmed_order",
        "side": side,
        "consensus": consensus,
        "consensus_score": score,
        "market_price": market_price,
    }


def _programmed_order_dedupe_key(state: Dict[str, Any]) -> tuple[Any, ...]:
    order = (state.get("order") or {}) if isinstance(state, dict) else {}
    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    return (
        str(order.get("symbol") or "").strip().upper(),
        int(order.get("type", -1) or -1),
        round(float(order.get("price_open", 0.0) or 0.0), 5),
        round(float(order.get("sl", 0.0) or 0.0), 5),
        round(float(order.get("tp", 0.0) or 0.0), 5),
        round(float(order.get("volume_current", order.get("volume_initial", plan.get("volume", 0.0))) or 0.0), 8),
        str(plan.get("decision") or "").strip().lower(),
        str(plan.get("model") or "").strip().lower(),
    )


def _maintain_programmed_orders(trading_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = _pending_order_maintenance_cfg(trading_cfg)
    if not bool(cfg.get("enabled", True)):
        return []

    candidate_states: List[Dict[str, Any]] = []
    for job_id in _active_job_ids(trading_cfg):
        state = _read_trading_state(job_id)
        if not state:
            continue
        if str((state.get("stage") or "")).strip().lower() != "agent_a":
            continue
        if str((state.get("order_submission_mode") or "")).strip().lower() != "programmed":
            continue
        if not isinstance(state.get("order"), dict) or not state.get("order"):
            continue
        candidate_states.append(state)

    if not candidate_states:
        return []

    candidate_states.sort(
        key=lambda item: (
            str((item.get("started_at") or "")).strip(),
            str((item.get("job_id") or "")).strip(),
        ),
        reverse=True,
    )

    assessment = _collect_all_model_assessment_signals(
        trading_cfg=trading_cfg,
        timeout_sec=float(cfg.get("assessment_timeout_seconds", 3.0) or 3.0),
    )

    adapter = MT5Adapter((((trading_cfg.get("broker") or {}).get("mt5") or {})))
    ok_conn, _msg_conn = adapter.connect()
    if not ok_conn:
        return []

    events: List[Dict[str, Any]] = []
    seen_programmed_orders: set[tuple[Any, ...]] = set()
    try:
        for state in candidate_states:
            job_id = str(state.get("job_id") or "").strip()
            state_path = Path(str(state.get("state_path") or _job_state_path(job_id)))
            refreshed_state = _refresh_job_state_from_mt5(trading_cfg, state, adapter)
            if refreshed_state != state:
                _persist_job_state(state_path, refreshed_state)
            state = refreshed_state
            order = (state.get("order") or {}) if isinstance(state.get("order"), dict) else {}
            if not _is_active_trading_state(state) or not order:
                continue

            dedupe_key = _programmed_order_dedupe_key(state)
            if dedupe_key in seen_programmed_orders:
                decision = {
                    "cancel": True,
                    "reason": "programmed_order_duplicate",
                    "consensus": assessment.get("consensus"),
                    "consensus_score": float(assessment.get("consensus_score", 0.0) or 0.0),
                    "market_price": None,
                }
            else:
                seen_programmed_orders.add(dedupe_key)

                side = _infer_order_decision(order) or str(((state.get("plan") or {}).get("decision") or "")).strip().lower()
                market_price = _market_reference_price_for_order(adapter, str(order.get("symbol") or ""), side)
                decision = _programmed_order_maintenance_decision(state, order, assessment, market_price, trading_cfg)
            if not bool(decision.get("cancel", False)):
                continue

            order_ticket = int(order.get("order_ticket", 0) or order.get("ticket", 0) or 0)
            cancel_res = adapter.cancel_pending_order(order_ticket) if order_ticket > 0 else {"ok": False, "reason": "missing_order_ticket"}

            post_cancel_state = _refresh_job_state_from_mt5(trading_cfg, state, adapter)
            post_cancel_order = (post_cancel_state.get("order") or {}) if isinstance(post_cancel_state.get("order"), dict) else {}
            post_cancel_position = (post_cancel_state.get("position") or {}) if isinstance(post_cancel_state.get("position"), dict) else {}
            if isinstance(post_cancel_position, dict) and post_cancel_position:
                if post_cancel_state != state:
                    _persist_job_state(state_path, post_cancel_state)
                continue

            if bool(cancel_res.get("ok", False)) or str(cancel_res.get("reason") or "") == "order_not_pending" or not post_cancel_order:
                closed_state = dict(post_cancel_state)
                closed_state.pop("order", None)
                closed_state["status"] = "closed"
                closed_state["closed_reason"] = str(decision.get("reason") or "programmed_order_maintained_closed")
                closed_state["ended_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                closed_state["pending_order_maintenance"] = {
                    "evaluated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                    "decision": decision,
                    "assessment": {
                        "consensus": assessment.get("consensus"),
                        "consensus_score": assessment.get("consensus_score"),
                        "source": assessment.get("source"),
                    },
                    "cancel_result": cancel_res,
                }
                _persist_job_state(state_path, closed_state)
                events.append(
                    {
                        "ok": True,
                        "job_id": job_id,
                        "order_ticket": order_ticket,
                        "action": "cancelled",
                        "reason": closed_state.get("closed_reason"),
                        "consensus": decision.get("consensus"),
                        "consensus_score": decision.get("consensus_score"),
                        "market_price": decision.get("market_price"),
                        "entry": order.get("price_open"),
                    }
                )
                continue

            events.append(
                {
                    "ok": False,
                    "job_id": job_id,
                    "order_ticket": order_ticket,
                    "action": "cancel_failed",
                    "reason": str(decision.get("reason") or "programmed_order_cancel_failed"),
                    "cancel_result": cancel_res,
                }
            )
    finally:
        adapter.shutdown()

    return events


def _job_table_row(trading_cfg: Dict[str, Any], state: Dict[str, Any]) -> List[str]:
    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    position = (state.get("position") or {}) if isinstance(state, dict) else {}
    order = (state.get("order") or {}) if isinstance(state, dict) else {}
    agent_b_plan = (state.get("agent_b_plan") or {}) if isinstance(state, dict) else {}
    signal = str(plan.get("decision") or "n/a")
    model = str(plan.get("model") or "n/a")
    signal_label = f"{signal}/{model}"
    profit = position.get("profit")
    status = _job_display_status(state)
    entry_value = position.get("price_open") or order.get("price_open") or plan.get("entry")
    stop_loss_value = position.get("sl") or order.get("sl") or plan.get("stop_loss")
    take_profit_value = position.get("tp") or order.get("tp") or plan.get("take_profit")
    expiration_value = state.get("programmed_order_expiration_utc") or order.get("expiration_utc")
    if status == "running_position":
        handler = "Agent B"
    elif status == "pending_approval":
        handler = "Awaiting approval"
    elif status == "programmed_order":
        handler = "Agent A pending fill"
    else:
        handler = "Agent A"
    next_review_value = agent_b_plan.get("next_review_utc") if handler == "Agent B" else None
    return [
        str(state.get("job_id") or "n/a"),
        status,
        handler,
        signal_label,
        _fmt_value(entry_value),
        _fmt_value(stop_loss_value),
        _fmt_value(take_profit_value),
        _format_utc_local_short(state.get("started_at"), trading_cfg),
        _format_utc_local_short(expiration_value, trading_cfg) if expiration_value else "n/a",
        _format_utc_local_short(next_review_value, trading_cfg) if next_review_value else "n/a",
        _fmt_value(profit),
        str(_job_ticket(state) or order.get("order_ticket") or "n/a"),
    ]


def _is_tsmm_broker_item(payload: Dict[str, Any]) -> bool:
    comment = str((payload or {}).get("comment") or "")
    magic = int((payload or {}).get("magic", 0) or 0)
    return "TSMM" in comment or magic in {7070001, 7070002}


def _infer_order_decision(order: Dict[str, Any]) -> str:
    order_type = int((order or {}).get("type", -99) or -99)
    if order_type in {0, 2, 4, 6}:
        return "buy"
    if order_type in {1, 3, 5, 7}:
        return "sell"
    return _infer_position_decision(order)


def _state_broker_key(state: Dict[str, Any]) -> str:
    position = (state.get("position") or {}) if isinstance(state, dict) else {}
    order = (state.get("order") or {}) if isinstance(state, dict) else {}
    position_ticket = int(position.get("ticket", 0) or 0)
    if position_ticket > 0:
        return f"broker_ticket:{position_ticket}"
    order_ticket = int(order.get("order_ticket", 0) or order.get("ticket", 0) or 0)
    if order_ticket > 0:
        # Use the same key namespace as positions so a filled order does not
        # show twice (tracked state + synthetic broker position) in active jobs.
        return f"broker_ticket:{order_ticket}"
    return f"job:{str((state or {}).get('job_id') or '').strip()}"


def _synthetic_display_state_from_position(position: Dict[str, Any], trading_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ticket = int((position or {}).get("ticket", 0) or 0)
    pos_time = int((position or {}).get("time", 0) or 0)
    started_at = datetime.utcfromtimestamp(pos_time).strftime("%Y-%m-%d %H:%M:%S") if pos_time > 0 else datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    decision = _infer_position_decision(position)
    return {
        "job_id": _synthetic_orphan_job_id(ticket, trading_cfg),
        "status": "agent_b_running",
        "stage": "agent_b",
        "mode": "mode_b",
        "started_at": started_at,
        "position": position,
        "plan": {
            "decision": decision or "n/a",
            "model": "mt5_live",
            "entry": position.get("price_open"),
            "stop_loss": position.get("sl"),
            "take_profit": position.get("tp"),
            "volume": position.get("volume"),
            "symbol": position.get("symbol"),
        },
        "order_submission_mode": "market",
        "recovery_method": "broker_display_only",
    }


def _synthetic_display_state_from_order(order: Dict[str, Any], trading_cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ticket = int((order or {}).get("order_ticket", 0) or (order or {}).get("ticket", 0) or 0)
    order_time = int((order or {}).get("time", 0) or 0)
    started_at = datetime.utcfromtimestamp(order_time).strftime("%Y-%m-%d %H:%M:%S") if order_time > 0 else datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    decision = _infer_order_decision(order)
    return {
        "job_id": _synthetic_orphan_order_job_id(ticket, trading_cfg),
        "status": "agent_a_completed",
        "stage": "agent_a",
        "mode": "mode_a",
        "started_at": started_at,
        "order": order,
        "plan": {
            "decision": decision or "n/a",
            "model": "mt5_pending",
            "entry": order.get("price_open"),
            "stop_loss": order.get("sl"),
            "take_profit": order.get("tp"),
            "volume": order.get("volume"),
            "symbol": order.get("symbol"),
        },
        "order_submission_mode": "programmed",
        "programmed_order_expiration_utc": order.get("expiration_utc"),
        "recovery_method": "broker_display_only",
    }


def _active_job_display_states(trading_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    _set_runtime_scope_env(trading_cfg)
    active_ids = _active_job_ids(trading_cfg)
    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, _msg_conn = adapter.connect()
    display_states: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()

    def _append_state(state: Dict[str, Any]) -> None:
        if not state:
            return
        state_key = _state_broker_key(state)
        if state_key in seen_keys:
            return
        seen_keys.add(state_key)
        display_states.append(state)

    try:
        for job_id in active_ids:
            state = _read_trading_state(job_id)
            if not state:
                continue
            if ok_conn:
                refreshed_state = _refresh_job_state_from_mt5(trading_cfg, state, adapter)
                if refreshed_state != state:
                    state_path_raw = str(refreshed_state.get("state_path") or _job_state_path(job_id))
                    _persist_job_state(Path(state_path_raw), refreshed_state)
                state = refreshed_state
            _append_state(state)

        if ok_conn:
            listed_positions = adapter.list_open_positions()
            if bool(listed_positions.get("ok", False)):
                for position in (listed_positions.get("positions") or []):
                    if not isinstance(position, dict) or not _is_tsmm_broker_item(position):
                        continue
                    _append_state(_synthetic_display_state_from_position(position, trading_cfg))

            mt5 = getattr(adapter, "_mt5", None)
            raw_orders = mt5.orders_get() or [] if mt5 is not None else []
            for raw_order in raw_orders:
                if isinstance(raw_order, dict):
                    order = raw_order
                elif hasattr(adapter, "_serialize_order"):
                    order = adapter._serialize_order(raw_order)
                else:
                    continue
                if not isinstance(order, dict) or not _is_tsmm_broker_item(order):
                    continue
                _append_state(_synthetic_display_state_from_order(order, trading_cfg))
    finally:
        if ok_conn:
            adapter.shutdown()

    return display_states


def _format_active_jobs_digest(trading_cfg: Dict[str, Any]) -> str:
    display_states = _active_job_display_states(trading_cfg)
    if not display_states:
        return _telegram_message_with_account(trading_cfg, "TSMM active jobs\nNo active trading jobs are currently being managed.")

    lines = [f"TSMM active jobs ({len(display_states)})"]
    lines.append(f"updated_at: {_format_utc_for_agent_timezone(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'), trading_cfg)}")
    headers = ["Job ID", "Market State", "Handled By", "Signal", "Entry", "SL", "TP", "Order Date", "Expiry", "Next Review", "P/L", "MT5 Ticket"]
    rows: List[List[str]] = []
    for state in display_states:
        rows.append(_job_table_row(trading_cfg, state))

    if not rows:
        lines.append("No active trading jobs are currently being managed.")
        return "\n".join(lines)[:3500]

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _render_row(values: List[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    lines.append("")
    lines.append(_render_row(headers))
    lines.append(separator)
    for row in rows:
        lines.append(_render_row(row))
    return _telegram_message_with_account(trading_cfg, "\n".join(lines))[:3500]


def _job_registry_summary(payload: Dict[str, Any], state_path: Path) -> Dict[str, Any]:
    position = (payload.get("position") or {}) if isinstance(payload, dict) else {}
    order = (payload.get("order") or {}) if isinstance(payload, dict) else {}
    return {
        "job_id": str(payload.get("job_id") or "").strip(),
        "state_path": str(state_path),
        "status": payload.get("status"),
        "stage": payload.get("stage"),
        "mode": payload.get("mode"),
        "started_at": payload.get("started_at"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "closed_reason": payload.get("closed_reason"),
        "position_ticket": position.get("ticket"),
        "order_submission_mode": payload.get("order_submission_mode"),
        "programmed_order_expiration_utc": payload.get("programmed_order_expiration_utc") or order.get("expiration_utc"),
    }


def _latest_state_alias_payload(payload: Dict[str, Any], state_path: Path) -> Dict[str, Any]:
    summary = _job_registry_summary(payload, state_path)
    if "plan" in payload:
        summary["plan"] = payload.get("plan")
    if "order" in payload:
        summary["order"] = payload.get("order")
    if "position" in payload:
        summary["position"] = payload.get("position")
    if "ended_at" in payload:
        summary["ended_at"] = payload.get("ended_at")
    if "market_reopen_at_utc" in payload:
        summary["market_reopen_at_utc"] = payload.get("market_reopen_at_utc")
    return summary


def _cleanup_closed_jobs_enabled() -> bool:
    cfg = _load_yaml(Path(os.environ.get("TRADING_CONFIG_PATH", str(ROOT / "config" / "trading_agent.yaml"))))
    return bool(((cfg.get("trading_job") or {}).get("cleanup_closed_jobs", True)))


def _should_cleanup_job_state(payload: Dict[str, Any]) -> bool:
    if not _cleanup_closed_jobs_enabled():
        return False
    return str(payload.get("status") or "").strip().lower() in {"closed", "stopped"}


def _choose_latest_job_id(jobs: Dict[str, Dict[str, Any]]) -> str:
    if not jobs:
        return ""
    ranked = sorted(
        jobs.items(),
        key=lambda item: (
            str((item[1] or {}).get("updated_at") or ""),
            str((item[1] or {}).get("started_at") or ""),
            str(item[0] or ""),
        ),
    )
    return str(ranked[-1][0] or "") if ranked else ""


def _refresh_latest_state_alias() -> None:
    registry = _registry_payload()
    jobs = dict(registry.get("jobs") or {})
    active_ids = [str(x).strip() for x in (registry.get("active_job_ids") or []) if str(x).strip()]
    preferred_job_id = active_ids[-1] if active_ids else _choose_latest_job_id(jobs)
    alias_path = _trading_state_path()
    if not preferred_job_id:
        if alias_path.exists():
            alias_path.unlink()
        return

    meta = dict(jobs.get(preferred_job_id) or {})
    state_path_raw = str(meta.get("state_path") or "").strip()
    payload = _read_json(Path(state_path_raw)) if state_path_raw else {}
    if payload and state_path_raw:
        _write_json(alias_path, _latest_state_alias_payload(payload, Path(state_path_raw)))
        return
    if meta:
        _write_json(alias_path, meta)
        return
    if alias_path.exists():
        alias_path.unlink()


def _cleanup_job_runtime(state_path: Path, payload: Dict[str, Any]) -> None:
    job_id = str(payload.get("job_id") or "").strip()
    registry = _registry_payload()
    jobs = dict(registry.get("jobs") or {})
    if job_id:
        jobs.pop(job_id, None)
    active_statuses = _active_job_statuses()
    registry["jobs"] = jobs
    registry["active_job_ids"] = [
        jid for jid, meta in jobs.items()
        if str((meta or {}).get("status") or "").strip().lower() in active_statuses
    ]
    registry["latest_job_id"] = registry["active_job_ids"][-1] if registry["active_job_ids"] else _choose_latest_job_id(jobs)
    registry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(_job_registry_path(), registry)

    if state_path.parent.exists():
        shutil.rmtree(state_path.parent, ignore_errors=True)

    _refresh_latest_state_alias()


def _update_job_registry_from_state(state_path: Path, payload: Dict[str, Any]) -> None:
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        return

    registry = _registry_payload()
    jobs = dict(registry.get("jobs") or {})
    jobs[job_id] = _job_registry_summary(payload, state_path)
    active_statuses = _active_job_statuses()
    registry["jobs"] = jobs
    registry["latest_job_id"] = job_id
    registry["active_job_ids"] = [
        jid for jid, meta in jobs.items()
        if str((meta or {}).get("status") or "").strip().lower() in active_statuses
    ]
    registry["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(_job_registry_path(), registry)


def _persist_job_state(state_path: Path, payload: Dict[str, Any]) -> None:
    payload["state_path"] = str(state_path)
    _write_json(state_path, payload)
    if _should_cleanup_job_state(payload):
        _cleanup_job_runtime(state_path, payload)
        return
    _write_json(_trading_state_path(), _latest_state_alias_payload(payload, state_path))
    _update_job_registry_from_state(state_path, payload)


def _find_job_for_ticket(ticket: int) -> Tuple[str, Path, Dict[str, Any]]:
    for job_id in _active_job_ids():
        state = _read_trading_state(job_id)
        if _job_ticket(state) == int(ticket):
            return job_id, _job_state_path(job_id), state
    return "", Path(), {}


def _infer_position_decision(position: Dict[str, Any]) -> str:
    pos_type = int((position or {}).get("type", -99) or -99)
    if pos_type == 0:
        return "buy"
    if pos_type == 1:
        return "sell"

    try:
        entry = float((position or {}).get("price_open", 0.0) or 0.0)
        stop_loss = float((position or {}).get("sl", 0.0) or 0.0)
        take_profit = float((position or {}).get("tp", 0.0) or 0.0)
    except Exception:
        return ""

    if entry > 0.0 and take_profit > entry:
        return "buy"
    if entry > 0.0 and take_profit > 0.0 and take_profit < entry:
        return "sell"
    if entry > 0.0 and stop_loss > entry:
        return "sell"
    if entry > 0.0 and stop_loss > 0.0 and stop_loss < entry:
        return "buy"
    return ""


def _normalize_trade_side_token(raw_side: Any) -> str:
    token = str(raw_side or "").strip().lower()
    if token in {"buy", "long", "bull", "bullish"}:
        return "buy"
    if token in {"sell", "short", "bear", "bearish"}:
        return "sell"
    return ""


def _tracked_position_tickets() -> set[int]:
    tickets: set[int] = set()
    for job_id in _all_disk_job_ids():
        state = _read_json(_job_state_path(job_id))
        ticket = _job_ticket(state)
        if ticket > 0:
            tickets.add(ticket)
    latest_state = _read_json(_trading_state_path())
    latest_ticket = _job_ticket(latest_state)
    if latest_ticket > 0:
        tickets.add(latest_ticket)
    return tickets


def _listed_tsmm_positions_by_ticket(trading_cfg: Dict[str, Any], adapter: MT5Adapter) -> Dict[int, Dict[str, Any]]:
    listed = adapter.list_open_positions()
    if not bool(listed.get("ok", False)):
        return {}

    target_symbol = str(((trading_cfg.get("execution") or {}).get("symbol") or "")).strip()
    positions_by_ticket: Dict[int, Dict[str, Any]] = {}
    for position in (listed.get("positions") or []):
        if not isinstance(position, dict) or not _is_tsmm_broker_item(position):
            continue
        ticket = int(position.get("ticket", 0) or 0)
        if ticket <= 0:
            continue
        if target_symbol and str(position.get("symbol") or "").strip() != target_symbol:
            continue
        positions_by_ticket[ticket] = position
    return positions_by_ticket


def _job_for_ticket_on_disk(ticket: int) -> Tuple[str, Path, Dict[str, Any]]:
    target_ticket = int(ticket or 0)
    if target_ticket <= 0:
        return "", Path(), {}

    for job_id in _all_disk_job_ids():
        state_path = _job_state_path(job_id)
        state = _read_json(state_path)
        if _job_ticket(state) == target_ticket:
            return str(job_id or "").strip(), state_path, state

    latest_state = _read_json(_trading_state_path())
    if _job_ticket(latest_state) == target_ticket:
        latest_job_id = str(latest_state.get("job_id") or "").strip()
        if latest_job_id:
            return latest_job_id, _job_state_path(latest_job_id), latest_state

    return "", Path(), {}


def _synthetic_orphan_job_id(position_ticket: int, trading_cfg: Dict[str, Any] | None = None) -> str:
    prefix = resolve_job_id_prefix(trading_cfg)
    base = f"mt5_orphan_pos_{int(position_ticket)}"
    if prefix:
        return f"job_{prefix}_{base}"
    return f"job_{base}"


def _synthetic_orphan_order_job_id(order_ticket: int, trading_cfg: Dict[str, Any] | None = None) -> str:
    prefix = resolve_job_id_prefix(trading_cfg)
    base = f"mt5_orphan_ord_{int(order_ticket)}"
    if prefix:
        return f"job_{prefix}_{base}"
    return f"job_{base}"


def _protective_sltp_for_position(
    trading_cfg: Dict[str, Any],
    position: Dict[str, Any],
    decision: str,
) -> Dict[str, Any]:
    side = str(decision or "").strip().lower()
    if side not in {"buy", "sell"}:
        return {"ok": False, "reason": "unsupported_side"}

    try:
        entry = float(position.get("price_open", 0.0) or 0.0)
    except Exception:
        entry = 0.0
    if entry <= 0.0:
        return {"ok": False, "reason": "missing_entry_price"}

    risk_cfg = (trading_cfg.get("risk") or {})
    sl_pct = max(float(risk_cfg.get("stop_loss_pct", 0.8) or 0.8), 0.01) / 100.0
    tp_pct = max(float(risk_cfg.get("take_profit_pct", 1.6) or 1.6), 0.01) / 100.0

    if side == "buy":
        stop_loss = entry * (1.0 - sl_pct)
        take_profit = entry * (1.0 + tp_pct)
    else:
        stop_loss = entry * (1.0 + sl_pct)
        take_profit = entry * (1.0 - tp_pct)

    return {
        "ok": True,
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
    }


def _agent_b_rebind_payload(
    state: Dict[str, Any],
    managed_position: Dict[str, Any],
    decision: str,
    adoption_kind: str,
) -> Dict[str, Any]:
    payload = dict(state or {})
    position = dict(managed_position or {})
    now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    payload["status"] = "agent_b_running"
    payload["stage"] = "agent_b"
    payload["mode"] = "mode_b"
    payload["position"] = position
    payload["agent_b_started_at"] = now_str
    payload.setdefault("started_at", now_str)

    plan = (payload.get("plan") or {}) if isinstance(payload.get("plan"), dict) else {}
    plan["decision"] = str(decision or "").strip().lower()
    plan["entry"] = position.get("price_open")
    plan["stop_loss"] = position.get("sl")
    plan["take_profit"] = position.get("tp")
    plan["volume"] = position.get("volume")
    plan["symbol"] = position.get("symbol")
    if adoption_kind == "manual":
        plan["model"] = "manual_entry_auto_adopt"
    elif adoption_kind == "external":
        plan["model"] = "any_live_position_auto_adopt"
    else:
        plan["model"] = "mt5_orphan_recovery"
    payload["plan"] = plan

    if adoption_kind == "manual":
        payload["recovery_method"] = "listener_manual_mt5_position"
    elif adoption_kind == "external":
        payload["recovery_method"] = "listener_any_live_mt5_position"
    else:
        payload["recovery_method"] = "listener_orphan_mt5_position"
    payload["adoption_kind"] = adoption_kind

    payload.pop("ended_at", None)
    payload.pop("closed_reason", None)
    payload.pop("close_outcome", None)
    payload.pop("pending_close_reason", None)
    payload.pop("pending_close_status", None)
    payload.pop("manual_or_external_close_detected", None)
    return payload


def _adopt_untracked_live_agent_b_positions(
    trading_cfg: Dict[str, Any],
    trading_config_path: Path,
    job_cooldowns: Dict[str, float],
    default_chat_id: str,
    last_chat_id: str,
    adapter: MT5Adapter,
) -> List[Dict[str, Any]]:
    env = os.environ.copy()
    env["TRADING_CONFIG_PATH"] = str(trading_config_path)
    env.setdefault("CONFIG_PATH", "config/config.yaml")
    target_chat_ids = _subscriber_chat_ids(trading_cfg, default_chat_id or last_chat_id)
    listener_cfg = _listener_cfg(trading_cfg)
    adopt_manual_entries = bool(listener_cfg.get("adopt_manual_entries", False))
    adopt_any_running_positions = bool(listener_cfg.get("adopt_any_running_positions", False))
    rebind_existing_non_agent_b = bool(listener_cfg.get("rebind_existing_non_agent_b", True))
    seed_missing_sltp_on_adopt = bool(listener_cfg.get("seed_missing_sltp_on_adopt", True))
    now_ts = time.time()
    events: List[Dict[str, Any]] = []

    listed = adapter.list_open_positions()
    if not bool(listed.get("ok", False)):
        return [{"ok": False, "reason": listed.get("message", "list_open_positions_failed")}]

    tracked_tickets = _tracked_position_tickets()
    target_symbol = str(((trading_cfg.get("execution") or {}).get("symbol") or "")).strip()

    for position in (listed.get("positions") or []):
        if not isinstance(position, dict):
            continue
        ticket = int(position.get("ticket", 0) or 0)
        if ticket <= 0:
            continue
        if (not adopt_any_running_positions) and target_symbol and str(position.get("symbol") or "").strip() != target_symbol:
            continue

        comment = str(position.get("comment") or "")
        magic = int(position.get("magic", 0) or 0)
        is_tsmm_tagged = "TSMM" in comment or magic in {7070001, 7070002}
        is_manual_candidate = magic == 0
        if adopt_any_running_positions:
            if is_tsmm_tagged:
                adoption_kind = "tsmm"
            elif is_manual_candidate:
                adoption_kind = "manual"
            else:
                adoption_kind = "external"
        else:
            if not is_tsmm_tagged and not (adopt_manual_entries and is_manual_candidate):
                continue
            adoption_kind = "manual" if (not is_tsmm_tagged and is_manual_candidate) else "tsmm"

        decision = _infer_position_decision(position)
        if decision not in {"buy", "sell"}:
            events.append({"ok": False, "ticket": ticket, "reason": "unable_to_infer_position_side"})
            continue

        managed_position = dict(position)
        sl_now = float(managed_position.get("sl", 0.0) or 0.0)
        tp_now = float(managed_position.get("tp", 0.0) or 0.0)
        protection_result: Dict[str, Any] | None = None
        if seed_missing_sltp_on_adopt and (sl_now <= 0.0 or tp_now <= 0.0):
            desired = _protective_sltp_for_position(trading_cfg, managed_position, decision)
            if bool(desired.get("ok", False)):
                protection_result = adapter.modify_position_risk(
                    ticket,
                    stop_loss=float(desired.get("stop_loss", 0.0) or 0.0),
                    take_profit=float(desired.get("take_profit", 0.0) or 0.0),
                )
                if bool((protection_result or {}).get("ok", False)) and isinstance((protection_result or {}).get("position"), dict):
                    managed_position = dict((protection_result or {}).get("position") or managed_position)
                else:
                    events.append(
                        {
                            "ok": False,
                            "ticket": ticket,
                            "reason": "manual_position_protection_update_failed",
                            "details": protection_result,
                        }
                    )

        existing_job_id, existing_state_path, existing_state = _job_for_ticket_on_disk(ticket)
        if existing_job_id:
            tracked_tickets.add(ticket)
            if not existing_state:
                if not adopt_any_running_positions:
                    continue
            else:
                state_path = existing_state_path or _job_state_path(existing_job_id)
                _sync_state_runner_pid_from_process(existing_job_id, existing_state, state_path)
                should_reconcile = _should_auto_reconcile_agent_b_job(trading_cfg, existing_state)
                should_rebind = bool(rebind_existing_non_agent_b) and not should_reconcile
                if not should_reconcile and not should_rebind:
                    continue

                if float(job_cooldowns.get(existing_job_id) or 0.0) > now_ts:
                    continue

                _stop_job_runner(existing_state)
                if should_rebind:
                    existing_state = _agent_b_rebind_payload(existing_state, managed_position, decision, adoption_kind)
                    _persist_job_state(state_path, existing_state)

                out = _run_cmd_async([sys.executable, "app.py", "trading-job", "resume", "--job-id", existing_job_id], env)
                if not bool(out.get("ok", False)):
                    job_cooldowns[existing_job_id] = now_ts + 60.0
                    events.append({"ok": False, "job_id": existing_job_id, "ticket": ticket, "reason": out.get("error", "resume_launch_failed")})
                    continue

                pid = int(out.get("pid") or 0)
                existing_state["runner_pid"] = pid
                existing_state["runner_started_at"] = _runner_started_at_value()
                existing_state["position"] = managed_position
                if should_rebind:
                    existing_state["status"] = "agent_b_running"
                    existing_state["stage"] = "agent_b"
                    existing_state["mode"] = "mode_b"
                _persist_job_state(state_path, existing_state)
                job_cooldowns[existing_job_id] = now_ts + 60.0
                action = "rebind_existing_live_position_to_agent_b" if should_rebind else "resume_tracked_mt5_position"
                events.append({"ok": True, "job_id": existing_job_id, "action": action, "pid": pid, "ticket": ticket, "adoption_kind": adoption_kind, "protection_result": protection_result})
                if target_chat_ids:
                    descriptor = "manual" if adoption_kind == "manual" else ("external" if adoption_kind == "external" else "TSMM")
                    _send_to_chat_ids(
                        trading_cfg,
                        target_chat_ids,
                        f"Resumed Agent B supervision for an existing {descriptor} MT5 live position: job_id={existing_job_id}; pid={pid}; mt5_ticket={ticket}",
                        allow_agent_mode=True,
                    )
                continue

        if ticket in tracked_tickets and not adopt_any_running_positions:
            continue

        synthetic_job_id = _synthetic_orphan_job_id(ticket, trading_cfg)
        if float(job_cooldowns.get(synthetic_job_id) or 0.0) > now_ts:
            continue

        pos_time = int(position.get("time", 0) or 0)
        started_at = datetime.utcfromtimestamp(pos_time).strftime("%Y-%m-%d %H:%M:%S") if pos_time > 0 else datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        state_path = _job_state_path(synthetic_job_id)
        synthetic_state = {
            "job_id": synthetic_job_id,
            "job_type": "trading_job",
            "state_path": str(state_path),
            "status": "agent_b_running",
            "stage": "agent_b",
            "mode": "mode_b",
            "started_at": started_at,
            "agent_b_started_at": started_at,
            "position": managed_position,
            "plan": {
                "decision": decision,
                "model": "manual_entry_auto_adopt" if adoption_kind == "manual" else ("any_live_position_auto_adopt" if adoption_kind == "external" else "mt5_orphan_recovery"),
                "entry": managed_position.get("price_open"),
                "stop_loss": managed_position.get("sl"),
                "take_profit": managed_position.get("tp"),
                "volume": managed_position.get("volume"),
                "symbol": managed_position.get("symbol"),
            },
            "order_submission_mode": "market",
            "runner_pid": 0,
            "runner_started_at": "",
            "recovery_method": "listener_manual_mt5_position" if adoption_kind == "manual" else ("listener_any_live_mt5_position" if adoption_kind == "external" else "listener_orphan_mt5_position"),
            "adoption_kind": adoption_kind,
        }
        _persist_job_state(state_path, synthetic_state)

        out = _run_cmd_async([sys.executable, "app.py", "trading-job", "resume", "--job-id", synthetic_job_id], env)
        if not bool(out.get("ok", False)):
            job_cooldowns[synthetic_job_id] = now_ts + 60.0
            events.append({"ok": False, "job_id": synthetic_job_id, "ticket": ticket, "reason": out.get("error", "resume_launch_failed")})
            continue

        pid = int(out.get("pid") or 0)
        synthetic_state["runner_pid"] = pid
        synthetic_state["runner_started_at"] = _runner_started_at_value()
        _persist_job_state(state_path, synthetic_state)
        tracked_tickets.add(ticket)
        job_cooldowns[synthetic_job_id] = now_ts + 60.0
        event = {
            "ok": True,
            "job_id": synthetic_job_id,
            "action": "adopt_orphan_mt5_position",
            "pid": pid,
            "ticket": ticket,
            "adoption_kind": adoption_kind,
            "protection_result": protection_result,
        }
        events.append(event)
        if target_chat_ids:
            descriptor = "manual" if adoption_kind == "manual" else ("external" if adoption_kind == "external" else "TSMM")
            _send_to_chat_ids(
                trading_cfg,
                target_chat_ids,
                f"Recovered an orphan {descriptor} MT5 live position into Agent B supervision: job_id={synthetic_job_id}; pid={pid}; mt5_ticket={ticket}",
                allow_agent_mode=True,
            )

    return events


def _read_trading_state(job_id: str = "") -> Dict[str, Any]:
    requested_job_id = str(job_id or "").strip()
    if requested_job_id:
        # When a specific job id is requested, never fall back to another job state.
        return _read_json(_job_state_path(requested_job_id))

    latest_job_id = _latest_job_id()
    if latest_job_id:
        payload = _read_json(_job_state_path(latest_job_id))
        if payload:
            return payload
    return _read_json(_trading_state_path())


def _await_job_state(job_id: str, timeout_seconds: float, poll_seconds: float = 0.25) -> Dict[str, Any]:
    target_job_id = str(job_id or "").strip()
    if not target_job_id:
        return {}

    timeout_seconds = max(float(timeout_seconds or 0.0), 0.0)
    if timeout_seconds <= 0.0:
        return _read_json(_job_state_path(target_job_id))

    deadline = time.time() + timeout_seconds
    while True:
        payload = _read_json(_job_state_path(target_job_id))
        if payload and str(payload.get("job_id") or "").strip() == target_job_id:
            return payload
        if time.time() >= deadline:
            return {}
        time.sleep(max(float(poll_seconds or 0.25), 0.05))


def _format_trading_launch_message(
    action: str,
    pid: Any,
    job_id: str,
    startup_state: Dict[str, Any],
) -> str:
    resolved_action = str(action or "start").strip().lower() or "start"
    resolved_job_id = str(job_id or "latest").strip() or "latest"
    pid_text = str(pid if pid is not None else "n/a")

    if startup_state:
        status = _job_display_status(startup_state)
        stage = str(startup_state.get("stage") or "n/a")
        return (
            f"trading {resolved_action} started pid={pid_text}; job_id={resolved_job_id}; "
            f"status={status}; stage={stage}"
        )

    return (
        f"trading {resolved_action} spawned pid={pid_text}; job_id={resolved_job_id}; "
        "state=pending_initialization (startup sync and model analysis can take a few minutes before runtime state appears)"
    )


def _approval_request_is_live(req: Dict[str, Any], job_id: str = "") -> bool:
    if str((req or {}).get("status") or "").strip().lower() != "pending":
        return False

    deadline_raw = str((req or {}).get("deadline_utc") or "").strip()
    if deadline_raw:
        deadline = _parse_state_datetime_utc(deadline_raw)
        if deadline is not None and deadline < datetime.utcnow().replace(tzinfo=timezone.utc):
            return False

    resolved_job_id = str(job_id or (req or {}).get("job_id") or "").strip()
    if not resolved_job_id:
        return True

    state = _read_json(_job_state_path(resolved_job_id))
    if not state:
        return False

    status = str(state.get("status") or "").strip().lower()
    stage = str(state.get("stage") or "").strip().lower()
    ended_at = str(state.get("ended_at") or "").strip()
    closed_reason = str(state.get("closed_reason") or "").strip().lower()
    if ended_at or closed_reason or status in {"completed", "closed", "failed", "stopped", "killed"}:
        return False
    if stage != "agent_a":
        return False
    return True


def _pending_approval_candidates() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    jobs_root = _job_root_dir()
    if jobs_root.exists():
        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue
            req_path = _job_approval_request_path(job_dir.name)
            req = _read_json(req_path)
            if not _approval_request_is_live(req, job_dir.name):
                continue
            req["request_path"] = str(req_path)
            req.setdefault("response_path", str(_job_approval_response_path(job_dir.name)))
            req.setdefault("job_id", job_dir.name)
            out.append(req)
    if not out:
        req = _read_json(_approval_request_path())
        if _approval_request_is_live(req):
            req.setdefault("request_path", str(_approval_request_path()))
            req.setdefault("response_path", str(_approval_response_path()))
            out.append(req)
    return sorted(out, key=lambda x: str(x.get("created_at_utc") or ""))


def _latest_pending_approval() -> Dict[str, Any]:
    candidates = _pending_approval_candidates()
    return candidates[-1] if candidates else {}


def _pending_approval_for_job(job_id: str) -> Dict[str, Any]:
    clean_job_id = str(job_id or "").strip()
    if not clean_job_id:
        return {}
    req = _read_json(_job_approval_request_path(clean_job_id))
    if _approval_request_is_live(req, clean_job_id):
        return req
    return {}


def _refresh_latest_approval_alias() -> None:
    latest = _latest_pending_approval()
    alias_path = _approval_request_path()
    if latest:
        _write_json(alias_path, latest)
    elif alias_path.exists():
        alias_path.unlink()


def _write_audit(entry: Dict[str, Any]) -> None:
    with _audit_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _write_conversation(entry: Dict[str, Any]) -> None:
    with _conversation_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _llm_session_file(chat_id: str) -> Path | None:
    clean_chat_id = str(chat_id or "").strip()
    if not clean_chat_id:
        return None
    payload = _chat_mode_payload()
    chats = payload.get("chats") or {}
    entry = chats.get(clean_chat_id) or {}
    session_id = str(entry.get("session_id") or "").strip()
    if not session_id:
        return None
    return _llm_session_dir(clean_chat_id) / f"{session_id}.jsonl"


def _append_llm_session_memory(chat_id: str, entry: Dict[str, Any]) -> None:
    session_file = _llm_session_file(chat_id)
    if session_file is None:
        return
    session_file.parent.mkdir(parents=True, exist_ok=True)
    with session_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _write_llm_session_metadata(chat_id: str, extra: Dict[str, Any] | None = None) -> None:
    clean_chat_id = str(chat_id or "").strip()
    if not clean_chat_id:
        return
    payload = _chat_mode_payload()
    chats = payload.get("chats") or {}
    entry = chats.get(clean_chat_id) or {}
    session_id = str(entry.get("session_id") or "").strip()
    if not session_id:
        return
    session_dir = _llm_session_dir(clean_chat_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    meta_path = session_dir / f"{session_id}.meta.json"
    metadata = {
        "chat_id": clean_chat_id,
        "session_id": session_id,
        "mode": entry.get("mode"),
        "started_at": entry.get("started_at"),
        "updated_at": entry.get("updated_at"),
        "source_text": entry.get("source_text"),
    }
    if extra:
        metadata.update(extra)
    _write_json(meta_path, metadata)


def _record_llm_session_message(chat_id: str, direction: str, text: str, **extra: Any) -> None:
    clean_chat_id = str(chat_id or "").strip()
    if not clean_chat_id or _chat_mode(clean_chat_id) != "agent":
        return
    payload = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "direction": str(direction or "").strip() or "unknown",
        "chat_id": clean_chat_id,
        "text": str(text or ""),
    }
    if extra:
        payload.update(extra)
    _append_llm_session_memory(clean_chat_id, payload)


def _console_trace(message: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {message}", flush=True)


def _allowed_roots(trading_cfg: Dict[str, Any]) -> set[str]:
    lcfg = _listener_cfg(trading_cfg)
    roots = [str(x).strip().lower() for x in (lcfg.get("allowed_commands") or ["status", "deploy", "trading", "endpoint", "ui", "resource", "history", "analysis"]) if str(x).strip()]
    return set(roots)


def _help_message(prefix: str) -> str:
    p = str(prefix or "/tsmm").strip() or "/tsmm"
    return (
        "TSMM Bot Commands\n"
        f"- {p} help: Show this command list.\n"
        f"- {p} status: Show trading state, endpoint status, and active LLM provider.\n"
        f"- {p} deploy [--refresh|--no-refresh|--dry-run|--no-start-job]: Run deployment pipeline.\n"
        f"- {p} deploy stop: Request an active deployment pipeline to stop safely.\n"
        f"- {p} trading start [--plan-model MODEL] [--submission-mode programmed|market] [--job-id JOB]: Start trading job. Defaults to programmed for the mandatory session order.\n"
        f"- {p} trading resume [--job-id JOB]: Resume Agent B trading loop.\n"
        f"- {p} trading jobs: Show all active TSMM trading jobs in a readable list.\n"
        f"- {p} trading close [--job-id JOB|--ticket TICKET|--side buy|sell]: Close a specific live MT5 position and end its TSMM job.\n"
        f"- {p} trading close-restart: Close the current live MT5 position, then launch a new trading job.\n"
        f"- {p} trading status [--job-id JOB]: Show trading job status and decision.\n"
        f"- {p} trading stop [--job-id JOB]: Stop all trading jobs, or only the specified job when --job-id is provided.\n"
        f"- {p} trading kill [--job-id JOB]: Hard-kill all trading jobs, or only the specified job when --job-id is provided.\n"
        f"- {p} trading approve: Approve the latest pending Agent A or Agent B request.\n"
        f"- {p} trading reject: Reject the latest pending Agent A or Agent B request.\n"
        f"- {p} endpoint restart: Restart local signal endpoint service.\n"
        f"- {p} ui start: Start UI apps.\n"
        f"- {p} ui stop: Stop UI apps.\n"
        f"- {p} resource status: Show CPU/RAM guard status.\n"
        f"- {p} resource relieve: Run immediate resource-relief action.\n"
        "In Telegram agent chat mode, say 'say copilot <prompt>' to queue a request for the active VS Code Copilot session and have its reply sent back to this Telegram chat.\n"
        "Latest async request status is posted automatically every 2 minutes and once when completed.\n"
        "A readable active-jobs summary is posted automatically every 10 minutes.\n"
        "You can also send natural language requests (for example: 'start trading', 'deploy with refresh', 'show trading status')."
    )


def _contains_any(text: str, phrases: List[str]) -> bool:
    t = str(text or "").strip().lower()
    return any(p in t for p in phrases)


def _looks_like_deploy_request(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return False

    patterns = [
        r"^deploy\b",
        r"\b(can|could|would|please|run|start|launch|trigger)\s+(you\s+)?deploy\b",
        r"\bdeploy\s+(with|without|using|no|--)",
        r"\b(run|start|launch|trigger)\s+deployment\b",
        r"\bdeployment\s+(with|without)\b",
    ]
    return any(re.search(pattern, t) for pattern in patterns)


def _is_agent_mode_enter_request(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False

    enter_markers = {
        "transfer to agent",
        "switch to agent",
        "agent mode",
        "chat with agent",
        "talk to agent",
    }
    if normalized in enter_markers:
        return True

    tokens = set(re.findall(r"[a-z0-9]+", normalized))
    if "agent" not in tokens:
        return False
    if "mode" in tokens:
        return True
    return bool(tokens & {"transfer", "switch", "chat", "talk", "speak", "connect"})


def _is_agent_mode_exit_request(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False

    exit_markers = {
        "exit agent",
        "leave agent",
        "back to tsmm",
        "return to tsmm",
        "exit chat mode",
    }
    if normalized in exit_markers:
        return True

    if re.search(r"\b(exit|leave)\s+agent(?:\s+mode)?\b", normalized):
        return True
    if re.search(r"\b(back|return)\s+to\s+tsmm\b", normalized):
        return True
    return False


def _extract_copilot_bridge_prompt(text: str) -> str | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    match = re.match(r"^say\s+copilot\b[\s:,-]*(.*)$", raw, flags=re.IGNORECASE)
    if not match:
        return None
    return str(match.group(1) or "").strip()


def _handle_copilot_bridge_request(text: str, trading_cfg: Dict[str, Any], source_chat_id: str = "") -> Dict[str, Any] | None:
    prompt = _extract_copilot_bridge_prompt(text)
    if prompt is None:
        return None

    if not prompt:
        return {
            "handled": True,
            "ok": False,
            "message": "Usage: say copilot <your request>. Example: say copilot search the web for today's gold macro headlines.",
            "parsed_from_natural_language": True,
            "agent_chat_mode": _chat_mode(source_chat_id) == "agent",
        }

    queued = queue_copilot_request(
        ROOT,
        trading_cfg,
        prompt=prompt,
        chat_id=source_chat_id,
        metadata={
            "source_text": str(text or "").strip(),
            "chat_mode": _chat_mode(source_chat_id),
            "target_session": "current_vscode_copilot_session",
        },
    )
    if not bool(queued.get("ok", False)):
        return {
            "handled": True,
            "ok": False,
            "message": f"Copilot handoff could not be queued: {queued.get('error', 'unknown_error')}",
            "parsed_from_natural_language": True,
            "agent_chat_mode": _chat_mode(source_chat_id) == "agent",
        }

    request = dict(queued.get("request") or {})
    return {
        "handled": True,
        "ok": True,
        "message": (
            f"Copilot handoff queued as {request.get('request_id', 'n/a')}. "
            "This request is waiting for the active VS Code Copilot session to read it and send the reply back here in Telegram."
        ),
        "parsed_from_natural_language": True,
        "agent_chat_mode": _chat_mode(source_chat_id) == "agent",
        "copilot_bridge_request_id": request.get("request_id"),
    }


def _infer_natural_language_tail(text: str) -> Tuple[str | None, str | None]:
    t = str(text or "").strip().lower()
    if not t:
        return None, None

    ops_chat_terms = [
        "order",
        "position",
        "market",
        "agent a",
        "agent b",
        "approval",
        "recommend",
        "signal",
        "trade idea",
        "live position",
    ]

    if _contains_any(t, ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        return None, "I am online and ready. Tell me what you want in plain language, or type '/tsmm help'."

    if _contains_any(t, ["help", "what can you do", "commands", "how do i", "usage", "manual"]):
        return "help", None

    ticket_match = re.search(r"ticket(?:\s+id)?\s*[:=]?\s*(\d+)", t)
    job_id_match = re.search(r"job\s+id\s*[:=]?\s*([a-z0-9_\-]+)", t)
    if not job_id_match:
        job_id_match = re.search(r"\b(job_[a-z0-9_\-]+)\b", t)
    close_target_tail = ""
    close_side = ""
    if _contains_any(t, ["buying job", "buy job", "buy position", "buy trade", "long job", "long position", "long trade"]):
        close_side = "buy"
    elif _contains_any(t, ["selling job", "sell job", "sell position", "sell trade", "short job", "short position", "short trade"]):
        close_side = "sell"
    if ticket_match:
        close_target_tail = f" --ticket {ticket_match.group(1)}"
    elif job_id_match:
        close_target_tail = f" --job-id {job_id_match.group(1)}"
    elif close_side:
        close_target_tail = f" --side {close_side}"

    if _contains_any(t, ["deploy stop", "deployment stop", "stop deploy", "stop deployment", "cancel deploy", "cancel deployment", "abort deploy", "abort deployment", "halt deploy", "halt deployment"]):
        return "deploy stop", None

    if _looks_like_deploy_request(t):
        flags: List[str] = []
        if _contains_any(t, ["dry run", "dry-run", "simulate"]):
            flags.append("--dry-run")
        if _contains_any(t, ["no refresh", "without refresh", "skip refresh"]):
            flags.append("--no-refresh")
        elif _contains_any(t, ["refresh", "force refresh", "update data"]):
            flags.append("--refresh")
        if _contains_any(t, ["no start", "don't start", "dont start", "without trading", "no-start-job"]):
            flags.append("--no-start-job")
        tail = "deploy" + (" " + " ".join(flags) if flags else "")
        return tail, None

    if _contains_any(t, ["trading status", "status trading", "how is trading", "trading doing", "trading progress"]):
        if job_id_match:
            return f"trading status --job-id {job_id_match.group(1)}", None
        return "trading status", None

    if _contains_any(t, ["active jobs", "trading jobs", "managed jobs", "running jobs", "list jobs"]):
        return "trading jobs", None

    if _contains_any(t, [
        "finish buying job",
        "finish buy job",
        "finish buying trade",
        "finish buy trade",
        "finish long job",
        "finish long trade",
    ]):
        return "trading close --side buy", None

    if _contains_any(t, [
        "finish selling job",
        "finish sell job",
        "finish selling trade",
        "finish sell trade",
        "finish short job",
        "finish short trade",
    ]):
        return "trading close --side sell", None

    if _contains_any(t, ["finish current job", "finish this job", "finish current trade", "finish this trade"]):
        return f"trading close{close_target_tail}", None

    if _contains_any(t, ["close current position and restart trading", "close current positions and restart trading", "close current position and re start trading", "close current positions and re start trading", "close position and restart trading", "close positions and restart trading", "close position and re start trading", "close positions and re start trading", "close that position and restart trading", "close that position and re start trading", "close current position and start trading", "close current positions and start trading", "close that position and start trading", "close trade and restart trading", "close trade and re start trading"]):
        return "trading close-restart", None

    if _contains_any(t, ["close current position", "close current positions", "close that position", "close that trade", "close live position", "close live order", "close trade", "close the position"]):
        return f"trading close{close_target_tail}", None

    if _contains_any(t, ["start trading", "begin trading", "run trading", "launch trading"]):
        return "trading start", None

    if _contains_any(t, ["resume trading", "continue trading", "restart trading loop"]):
        return "trading resume", None

    if _contains_any(t, ["approve", "approved", "approve trade", "approve trading", "approve agent a", "approve agent b", "approve order"]):
        return "trading approve", None

    if _contains_any(t, ["reject", "rejected", "deny", "decline", "reject trade", "reject trading", "reject agent a", "reject agent b", "reject order"]):
        return "trading reject", None

    if _contains_any(t, ["stop trading", "halt trading", "end trading"]):
        return "trading stop", None

    if _contains_any(t, ["restart endpoint", "endpoint restart", "reset endpoint", "reboot endpoint"]):
        return "endpoint restart", None

    if _contains_any(t, ["start ui", "open ui", "launch ui", "start dashboard"]):
        return "ui start", None

    if _contains_any(t, ["stop ui", "close ui", "shutdown ui", "hide ui"]):
        return "ui stop", None

    if _contains_any(t, ["resource status", "cpu", "ram", "memory status", "resource usage"]):
        return "resource status", None

    if _contains_any(t, ["resource relieve", "relieve resources", "free resources", "relieve pressure"]):
        return "resource relieve", None

    if _contains_any(t, ["status", "health", "are you online", "system status"]):
        if _contains_any(t, ops_chat_terms):
            return None, None
        return "status", None

    return None, (
        "I understood that as chat, but not as an action yet. "
        "Try saying things like 'start trading', 'deploy with refresh', 'show trading status', or '/tsmm help'."
    )


def _is_pid_alive(pid: int) -> bool:
    try:
        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        return False


def _command_process_running(command_fragment: str) -> bool:
    fragment = str(command_fragment or "").strip().lower()
    if not fragment:
        return False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
        except Exception:
            continue
        if fragment in cmdline:
            return True
    return False


def _latest_request_blocks_autonomy(request_info: Dict[str, Any]) -> bool:
    if not request_info:
        return False
    if bool(request_info.get("done_notified", False)):
        return False
    pid = int(request_info.get("pid") or 0)
    if pid <= 0 or not _is_pid_alive(pid):
        return False
    req_type = str(request_info.get("type") or "").strip().lower()
    return req_type in {"trading start", "trading resume", "trading close-restart"} or req_type.startswith("autonomous ")


def _scheduled_refresh_target(trading_cfg: Dict[str, Any], now_utc: datetime | None = None) -> Dict[str, Any]:
    cfg = _scheduled_refresh_cfg(trading_cfg)
    if not bool(cfg.get("enabled", False)):
        return {}

    tz_name = str(cfg.get("market_close_timezone") or "America/New_York").strip() or "America/New_York"
    close_parts = _parse_hhmm(cfg.get("market_close_time") or "17:00")
    if close_parts is None:
        return {}
    catch_up_enabled = bool(cfg.get("catch_up_missed_market_closes", True))

    current_utc = now_utc or datetime.utcnow().replace(tzinfo=timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)
    local_now = current_utc.astimezone(ZoneInfo(tz_name))
    scheduled_local = datetime(
        local_now.year,
        local_now.month,
        local_now.day,
        close_parts[0],
        close_parts[1],
        tzinfo=ZoneInfo(tz_name),
    )
    if catch_up_enabled and local_now < scheduled_local:
        scheduled_local -= timedelta(days=1)
    if catch_up_enabled:
        while scheduled_local.weekday() >= 5:
            scheduled_local -= timedelta(days=1)
    scheduled_utc = scheduled_local.astimezone(timezone.utc)
    return {
        "timezone": tz_name,
        "target_date": scheduled_local.strftime("%Y-%m-%d"),
        "scheduled_local": scheduled_local.strftime("%Y-%m-%d %H:%M:%S"),
        "scheduled_utc": scheduled_utc.strftime("%Y-%m-%d %H:%M:%S"),
        "due": current_utc >= scheduled_utc,
        "catch_up_enabled": catch_up_enabled,
    }


def _terminate_processes_by_command_fragments(command_fragments: List[str]) -> List[int]:
    fragments = [str(fragment or "").strip().lower() for fragment in (command_fragments or []) if str(fragment or "").strip()]
    if not fragments:
        return []

    stopped_pids: List[int] = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
        except Exception:
            continue
        if not cmdline or not any(fragment in cmdline for fragment in fragments):
            continue
        try:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except psutil.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            stopped_pids.append(int(proc.pid))
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
        except Exception:
            continue
    return stopped_pids


def _enforce_weekend_utc_quiet_mode(trading_cfg: Dict[str, Any], now_utc: datetime | None = None) -> Dict[str, Any]:
    current_utc = now_utc or datetime.utcnow().replace(tzinfo=timezone.utc)
    active = _weekend_utc_quiet_mode_active(trading_cfg, now_utc=current_utc)
    result: Dict[str, Any] = {
        "active": active,
        "trading_kill": {},
        "stopped_trading_pids": [],
        "stopped_endpoint_pids": [],
    }
    if not active:
        return result

    refresh_cfg = _scheduled_refresh_cfg(trading_cfg)
    trading_fragments = ["app.py trading-job resume", "app.py trading-job start"]
    if any(_command_process_running(fragment) for fragment in trading_fragments):
        result["trading_kill"] = _run_cmd([sys.executable, "app.py", "trading-job", "kill"], os.environ.copy())
        result["stopped_trading_pids"] = _terminate_processes_by_command_fragments(trading_fragments)

    endpoint_script = str(((trading_cfg.get("endpoint_lifecycle") or {}).get("service_script") or "scripts/local_signal_endpoint_service.py"))
    if bool(refresh_cfg.get("weekend_stop_endpoint", True)) and not _command_process_running("deploy_agent_pipeline.py"):
        if _command_process_running(Path(endpoint_script).name):
            result["stopped_endpoint_pids"] = _stop_local_endpoint_service(endpoint_script)
    return result


def _handle_weekend_quiet_mode_exit(trading_cfg: Dict[str, Any], default_chat_id: str, last_chat_id: str) -> Dict[str, Any]:
    refresh_cfg = _scheduled_refresh_cfg(trading_cfg)
    result: Dict[str, Any] = {
        "endpoint_restart": {},
        "digest": "",
        "notified": False,
    }

    if bool(refresh_cfg.get("weekend_stop_endpoint", True)):
        result["endpoint_restart"] = _restart_endpoint_service(trading_cfg)

    digest = _format_active_jobs_digest(trading_cfg)
    result["digest"] = digest
    target_chat_ids = _subscriber_chat_ids(trading_cfg, default_chat_id or last_chat_id)
    message = "weekend quiet mode lifted. MT5 positions and pending orders are being resumed for supervision.\n\n" + digest
    if target_chat_ids:
        _send_to_chat_ids(trading_cfg, target_chat_ids, message)
        result["notified"] = True
    _console_trace(message)
    return result


def _deploy_refresh_requested(trading_cfg: Dict[str, Any], extra_args: List[str]) -> bool:
    normalized = [str(arg).strip().lower() for arg in (extra_args or []) if str(arg).strip()]
    if "--refresh" in normalized:
        return True
    if "--no-refresh" in normalized:
        return False
    lcfg = _listener_cfg(trading_cfg)
    cfg_path = ROOT / str(lcfg.get("default_pipeline_config") or "config/agent_pipeline.yaml")
    pipeline_cfg = _load_yaml(cfg_path)
    return bool(((pipeline_cfg.get("refresh") or {}).get("enabled", True)))


def _launch_deploy_pipeline(trading_cfg: Dict[str, Any], extra_args: List[str] | None = None) -> Dict[str, Any]:
    lcfg = _listener_cfg(trading_cfg)
    script = str(lcfg.get("run_deploy_script", "scripts/deploy_agent_pipeline.py"))
    cfg = str(lcfg.get("default_pipeline_config", "config/agent_pipeline.yaml"))
    env = os.environ.copy()
    args = [sys.executable, script, "--pipeline-config", cfg]

    allowed = {"--refresh", "--no-refresh", "--dry-run", "--no-start-job"}
    clean_args = [str(tok).strip() for tok in (extra_args or []) if str(tok).strip() in allowed]
    args.extend(clean_args)

    out = _run_cmd_async(args, env)
    out["refresh_requested"] = _deploy_refresh_requested(trading_cfg, clean_args)
    return out


def _scheduled_refresh_start_message(target: Dict[str, Any]) -> str:
    return (
        "scheduled model refresh started: "
        f"target_date={target.get('target_date', 'n/a')}; "
        f"market_close_local={target.get('scheduled_local', 'n/a')} {target.get('timezone', 'n/a')}; "
        "scope=all_models; trading_jobs_continue=yes"
    )


def _scheduled_refresh_runtime_message(state: Dict[str, Any]) -> str:
    req = {
        "type": "deploy",
        "pid": int(state.get("pid") or 0),
    }
    return "scheduled model refresh status: " + _latest_request_status_message(req)


def _job_runner_pid(state: Dict[str, Any]) -> int:
    try:
        return int((state or {}).get("runner_pid") or 0)
    except Exception:
        return 0


def _runner_started_at_value() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _find_live_job_runner_pid(job_id: str) -> int:
    desired_job_id = str(job_id or "").strip().lower()
    if not desired_job_id:
        return 0

    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline_parts = [str(part or "") for part in (proc.info.get("cmdline") or [])]
        except Exception:
            continue
        if not cmdline_parts:
            continue

        cmdline_lower_parts = [part.lower() for part in cmdline_parts]
        cmdline_text = " ".join(cmdline_lower_parts)
        if "app.py" not in cmdline_text or "trading-job" not in cmdline_text:
            continue

        try:
            idx = cmdline_lower_parts.index("--job-id")
        except ValueError:
            continue
        if idx + 1 >= len(cmdline_lower_parts):
            continue
        if str(cmdline_lower_parts[idx + 1] or "").strip() != desired_job_id:
            continue

        try:
            return int(proc.info.get("pid") or 0)
        except Exception:
            continue
    return 0


def _resolve_live_job_runner_pid(job_id: str, state: Dict[str, Any]) -> int:
    runner_pid = _job_runner_pid(state)
    if runner_pid > 0 and _is_pid_alive(runner_pid):
        return runner_pid
    return _find_live_job_runner_pid(job_id)


def _sync_state_runner_pid_from_process(job_id: str, state: Dict[str, Any], state_path: Path) -> int:
    live_runner_pid = _resolve_live_job_runner_pid(job_id, state)
    if live_runner_pid <= 0:
        return 0
    if _job_runner_pid(state) == live_runner_pid:
        return live_runner_pid

    state["runner_pid"] = live_runner_pid
    state["runner_started_at"] = _runner_started_at_value()
    _persist_job_state(state_path, state)
    return live_runner_pid


def _should_auto_resume_waiting_job(job_id: str, state: Dict[str, Any]) -> bool:
    if not state:
        return False
    if str(state.get("stage") or "").strip().lower() != "agent_a":
        return False
    if not bool(state.get("agent_a_approved")):
        return False
    status = str(state.get("status") or "").strip().lower()
    if status not in {"waiting_market_open", "agent_a_completed"}:
        return False
    return _resolve_live_job_runner_pid(job_id, state) <= 0


def _agent_b_heartbeat_stale(trading_cfg: Dict[str, Any], state: Dict[str, Any], now_utc: datetime | None = None) -> bool:
    if not state:
        return True

    mode_b_cfg = dict(trading_cfg.get("mode_b") or {})
    listener_cfg = _listener_cfg(trading_cfg)
    poll_seconds = max(int(mode_b_cfg.get("poll_seconds", 300) or 300), 1)
    stale_after_seconds = max(
        int(listener_cfg.get("agent_b_stale_after_seconds", (poll_seconds * 2) + 90) or ((poll_seconds * 2) + 90)),
        poll_seconds + 30,
    )

    current_utc = now_utc or datetime.utcnow().replace(tzinfo=timezone.utc)
    if current_utc.tzinfo is None:
        current_utc = current_utc.replace(tzinfo=timezone.utc)

    last_tick = _parse_state_datetime_utc(state.get("last_mode_b_tick"))
    runner_started_at = _parse_state_datetime_utc(state.get("runner_started_at"))
    started_at = _parse_state_datetime_utc(state.get("agent_b_started_at") or state.get("started_at"))
    anchor = last_tick or runner_started_at or started_at
    if anchor is None:
        return True
    return (current_utc - anchor).total_seconds() > float(stale_after_seconds)


def _stop_job_runner(state: Dict[str, Any]) -> int:
    runner_pid = _job_runner_pid(state)
    if runner_pid <= 0:
        return 0
    try:
        proc = psutil.Process(runner_pid)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        return runner_pid
    except (psutil.NoSuchProcess, ProcessLookupError):
        return 0
    except Exception:
        return 0


def _auto_resume_waiting_jobs(
    trading_cfg: Dict[str, Any],
    trading_config_path: Path,
    job_cooldowns: Dict[str, float],
    default_chat_id: str,
    last_chat_id: str,
) -> List[Dict[str, Any]]:
    interval_sec = max(int((_listener_cfg(trading_cfg).get("auto_resume_waiting_jobs_interval_seconds", 60) or 60)), 15)
    env = os.environ.copy()
    env["TRADING_CONFIG_PATH"] = str(trading_config_path)
    launched: List[Dict[str, Any]] = []
    now_ts = time.time()

    for job_id in _active_job_ids():
        if float(job_cooldowns.get(job_id) or 0.0) > now_ts:
            continue

        state = _read_trading_state(job_id)
        state_path = _job_state_path(job_id)

        # Guard against stale state runner_pid values after reboot/recovery.
        if _sync_state_runner_pid_from_process(job_id, state, state_path) > 0:
            job_cooldowns[job_id] = now_ts + float(interval_sec)
            continue

        if not _should_auto_resume_waiting_job(job_id, state):
            job_cooldowns.pop(job_id, None)
            continue

        args = [sys.executable, "app.py", "trading-job", "resume", "--job-id", job_id]
        out = _run_cmd_async(args, env)
        if not bool(out.get("ok", False)):
            job_cooldowns[job_id] = now_ts + float(interval_sec)
            continue

        pid = int(out.get("pid") or 0)
        job_cooldowns[job_id] = now_ts + float(interval_sec)
        launched.append({"job_id": job_id, "pid": pid})

        state_path_raw = str(state.get("state_path") or state_path)
        state_path = Path(state_path_raw)
        state["runner_pid"] = pid
        state["runner_started_at"] = _runner_started_at_value()
        _persist_job_state(state_path, state)

        target_chat_ids = _subscriber_chat_ids(trading_cfg, default_chat_id or last_chat_id)
        if target_chat_ids:
            resume_status = str(state.get("status") or "unknown")
            reopen_at = str(state.get("market_reopen_at_utc") or "n/a")
            _send_to_chat_ids(
                trading_cfg,
                target_chat_ids,
                (
                    f"trading auto-resume started: job_id={job_id}; pid={pid}; "
                    f"status={resume_status}; market_reopen_at_utc={reopen_at}"
                ),
            )

    return launched


def _should_auto_reconcile_agent_b_job(trading_cfg: Dict[str, Any], state: Dict[str, Any]) -> bool:
    if not state:
        return False
    if str(state.get("status") or "").strip().lower() != "agent_b_running":
        return False
    if str(state.get("stage") or "").strip().lower() != "agent_b":
        return False
    if _job_ticket(state) <= 0:
        return False
    job_id = str(state.get("job_id") or "").strip()
    if _resolve_live_job_runner_pid(job_id, state) <= 0:
        return True
    return _agent_b_heartbeat_stale(trading_cfg, state)


def _reconcile_orphaned_agent_b_jobs(
    trading_cfg: Dict[str, Any],
    trading_config_path: Path,
    job_cooldowns: Dict[str, float],
    default_chat_id: str,
    last_chat_id: str,
) -> List[Dict[str, Any]]:
    interval_sec = max(int((_listener_cfg(trading_cfg).get("auto_resume_waiting_jobs_interval_seconds", 60) or 60)), 15)
    env = os.environ.copy()
    env["TRADING_CONFIG_PATH"] = str(trading_config_path)
    env.setdefault("CONFIG_PATH", "config/config.yaml")
    target_chat_ids = _subscriber_chat_ids(trading_cfg, default_chat_id or last_chat_id)
    now_ts = time.time()
    events: List[Dict[str, Any]] = []

    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, msg_conn = adapter.connect()
    if not ok_conn:
        return [{"ok": False, "reason": f"mt5_connect_failed:{msg_conn}"}]

    try:
        live_positions_by_ticket = _listed_tsmm_positions_by_ticket(trading_cfg, adapter)
        for job_id in _active_job_ids():
            if float(job_cooldowns.get(job_id) or 0.0) > now_ts:
                continue

            state = _read_trading_state(job_id)
            state_path = _job_state_path(job_id)
            _sync_state_runner_pid_from_process(job_id, state, state_path)
            if not _should_auto_reconcile_agent_b_job(trading_cfg, state):
                job_cooldowns.pop(job_id, None)
                continue

            ticket = _job_ticket(state)
            pos = adapter.get_position_by_ticket(ticket)
            live_position = (pos.get("position") or {}) if bool(pos.get("ok", False)) and isinstance(pos.get("position"), dict) else {}
            if not live_position:
                live_position = dict(live_positions_by_ticket.get(ticket) or {})

            if live_position:
                _stop_job_runner(state)
                args = [sys.executable, "app.py", "trading-job", "resume", "--job-id", job_id]
                out = _run_cmd_async(args, env)
                if not bool(out.get("ok", False)):
                    job_cooldowns[job_id] = now_ts + float(interval_sec)
                    continue

                pid = int(out.get("pid") or 0)
                state["runner_pid"] = pid
                state["runner_started_at"] = _runner_started_at_value()
                state["position"] = live_position
                _persist_job_state(state_path, state)
                job_cooldowns[job_id] = now_ts + float(interval_sec)
                event = {"ok": True, "job_id": job_id, "action": "resume_agent_b", "pid": pid, "ticket": ticket}
                events.append(event)
                if target_chat_ids:
                    _send_to_chat_ids(
                        trading_cfg,
                        target_chat_ids,
                        f"trading auto-resume started: job_id={job_id}; pid={pid}; stage=agent_b; mt5_ticket={ticket}",
                        allow_agent_mode=True,
                    )
                continue

            close_outcome = adapter.get_position_close_outcome(ticket)
            if bool(close_outcome.get("ok", False)):
                state["close_outcome"] = close_outcome
            state["status"] = "closed"
            state["closed_reason"] = "position_not_found_assumed_closed"
            state["manual_or_external_close_detected"] = True
            state["ended_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            if _should_auto_request_followup_agent_a(state, trading_cfg):
                new_job_id = _new_job_id(trading_cfg)
                env["TSMM_AGENT_A_AUTO_CREATED"] = "1"
                followup = _run_cmd_async([sys.executable, "app.py", "trading-job", "start", "--job-id", new_job_id], env)
                state["followup_agent_a"] = {
                    "ok": bool(followup.get("ok", False)),
                    "job_id": new_job_id,
                    "pid": int(followup.get("pid") or 0),
                    "trigger_closed_reason": state.get("closed_reason"),
                }

            _persist_job_state(state_path, state)
            job_cooldowns.pop(job_id, None)
            event = {
                "ok": True,
                "job_id": job_id,
                "action": "close_stale_agent_b",
                "ticket": ticket,
                "close_outcome": close_outcome,
                "followup_agent_a": state.get("followup_agent_a"),
            }
            events.append(event)
            if target_chat_ids:
                message = f"job_id: {job_id}\n\n{_job_finished_message(state, state.get('close_result') or {}, state.get('position') or {})}"[:3500]
                _send_to_chat_ids(trading_cfg, target_chat_ids, message, allow_agent_mode=True)
                followup = state.get("followup_agent_a") if isinstance(state.get("followup_agent_a"), dict) else {}
                if followup.get("ok"):
                    _send_to_chat_ids(
                        trading_cfg,
                        target_chat_ids,
                        (
                            "Previous trading job finished with an auto-reentry eligible close outcome. "
                            f"Starting a new Agent A entry search now. new_job_id={followup.get('job_id')}; pid={followup.get('pid')}"
                        )[:3500],
                        allow_agent_mode=True,
                    )

        events.extend(
            _adopt_untracked_live_agent_b_positions(
                trading_cfg=trading_cfg,
                trading_config_path=trading_config_path,
                job_cooldowns=job_cooldowns,
                default_chat_id=default_chat_id,
                last_chat_id=last_chat_id,
                adapter=adapter,
            )
        )
    finally:
        adapter.shutdown()

    return events


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _chat_mode_payload() -> Dict[str, Any]:
    payload = _read_json(_chat_mode_state_path())
    chats = payload.get("chats")
    if not isinstance(chats, dict):
        payload["chats"] = {}
    return payload


def _chat_mode(chat_id: str) -> str:
    payload = _chat_mode_payload()
    chats = payload.get("chats") or {}
    entry = chats.get(str(chat_id).strip()) or {}
    return str(entry.get("mode") or "default").strip().lower() or "default"


def _set_chat_mode(chat_id: str, mode: str, source_text: str = "") -> None:
    clean_chat_id = str(chat_id or "").strip()
    if not clean_chat_id:
        return
    payload = _chat_mode_payload()
    chats = payload.get("chats") or {}
    normalized_mode = str(mode or "default").strip().lower() or "default"
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")
    session_id = ""
    if normalized_mode == "agent":
        session_id = datetime.utcnow().strftime("llm_session_%Y%m%d_%H%M%S_%f")
    chats[clean_chat_id] = {
        "mode": normalized_mode,
        "session_id": session_id,
        "started_at": started_at if normalized_mode == "agent" else "",
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_text": str(source_text or "").strip(),
    }
    payload["chats"] = chats
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_json(_chat_mode_state_path(), payload)
    if normalized_mode == "agent":
        _write_llm_session_metadata(clean_chat_id)
        _append_llm_session_memory(
            clean_chat_id,
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "direction": "system",
                "chat_id": clean_chat_id,
                "event": "agent_session_started",
                "text": str(source_text or "").strip(),
            },
        )


def _clear_chat_mode(chat_id: str) -> None:
    clean_chat_id = str(chat_id or "").strip()
    if not clean_chat_id:
        return
    payload = _chat_mode_payload()
    chats = payload.get("chats") or {}
    if clean_chat_id in chats:
        prior_entry = dict(chats.get(clean_chat_id) or {})
        if str(prior_entry.get("mode") or "").strip().lower() == "agent":
            _append_llm_session_memory(
                clean_chat_id,
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "direction": "system",
                    "chat_id": clean_chat_id,
                    "event": "agent_session_ended",
                    "text": "Agent chat mode disabled.",
                },
            )
            _write_llm_session_metadata(clean_chat_id, {"ended_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        chats.pop(clean_chat_id, None)
        payload["chats"] = chats
        payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_json(_chat_mode_state_path(), payload)


def _chat_mode_switch_response(text: str, source_chat_id: str = "") -> Dict[str, Any] | None:
    if _is_agent_mode_enter_request(text):
        _set_chat_mode(source_chat_id, "agent", source_text=text)
        return {
            "handled": True,
            "ok": True,
            "message": (
                "Agent chat mode is now active for this Telegram chat. "
                "Your next non-command messages will go directly to the LLM chat path, and TSMM background notifications are muted for this chat until you exit agent mode. "
                "Use 'back to tsmm' or 'exit agent' to leave this mode."
            ),
            "parsed_from_natural_language": True,
        }

    if _is_agent_mode_exit_request(text):
        _clear_chat_mode(source_chat_id)
        return {
            "handled": True,
            "ok": True,
            "message": "Agent chat mode is now off. Normal TSMM routing is active again.",
            "parsed_from_natural_language": True,
        }

    return None


def _load_app_cfg() -> Dict[str, Any]:
    return _load_yaml(ROOT / "config" / "config.yaml")


def _summarize_market_snapshot(app_cfg: Dict[str, Any]) -> Dict[str, Any]:
    db_path = ROOT / str(app_cfg.get("data_path") or "data/market_data.sqlite")
    data_symbol = str(app_cfg.get("data_symbol") or app_cfg.get("sql_symbol") or "XAUUSD").strip()
    if not db_path.exists():
        return {"ok": False, "reason": f"db_missing:{db_path}"}

    out: Dict[str, Any] = {"ok": True}
    try:
        df_1h = query_ohlc(str(db_path), timeframe_minutes=60, latest_records=3, symbol=data_symbol)
        if not df_1h.empty and len(df_1h) >= 1:
            last = df_1h.iloc[-1]
            prev = df_1h.iloc[-2] if len(df_1h) >= 2 else None
            out["1h"] = {
                "date": str(last.get("DATE")),
                "open": float(last.get("OPEN", 0.0) or 0.0),
                "high": float(last.get("HIGH", 0.0) or 0.0),
                "low": float(last.get("LOW", 0.0) or 0.0),
                "close": float(last.get("CLOSE", 0.0) or 0.0),
                "delta_close": None if prev is None else float(last.get("CLOSE", 0.0) or 0.0) - float(prev.get("CLOSE", 0.0) or 0.0),
            }
        df_7h = query_ohlc(str(db_path), timeframe_minutes=420, latest_records=2, symbol=data_symbol)
        if not df_7h.empty and len(df_7h) >= 1:
            last = df_7h.iloc[-1]
            prev = df_7h.iloc[-2] if len(df_7h) >= 2 else None
            out["7h"] = {
                "date": str(last.get("DATE")),
                "open": float(last.get("OPEN", 0.0) or 0.0),
                "high": float(last.get("HIGH", 0.0) or 0.0),
                "low": float(last.get("LOW", 0.0) or 0.0),
                "close": float(last.get("CLOSE", 0.0) or 0.0),
                "delta_close": None if prev is None else float(last.get("CLOSE", 0.0) or 0.0) - float(prev.get("CLOSE", 0.0) or 0.0),
            }
    except Exception as e:
        return {"ok": False, "reason": f"market_snapshot_failed:{e}"}
    return out


def _mt5_live_snapshot(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok, msg = adapter.connect()
    if not ok:
        return {"ok": False, "reason": msg}

    try:
        mt5 = adapter._mt5
        if mt5 is None:
            return {"ok": False, "reason": "mt5_not_connected"}
        positions = mt5.positions_get() or []
        orders = mt5.orders_get() or []
        return {
            "ok": True,
            "positions": [
                {
                    "ticket": int(getattr(p, "ticket", 0) or 0),
                    "symbol": str(getattr(p, "symbol", "") or ""),
                    "volume": float(getattr(p, "volume", 0.0) or 0.0),
                    "price_open": float(getattr(p, "price_open", 0.0) or 0.0),
                    "sl": float(getattr(p, "sl", 0.0) or 0.0),
                    "tp": float(getattr(p, "tp", 0.0) or 0.0),
                    "comment": str(getattr(p, "comment", "") or ""),
                }
                for p in positions[:5]
            ],
            "orders": [
                {
                    "ticket": int(getattr(o, "ticket", 0) or 0),
                    "symbol": str(getattr(o, "symbol", "") or ""),
                    "type": int(getattr(o, "type", -1) or -1),
                    "price_open": float(getattr(o, "price_open", 0.0) or 0.0),
                    "sl": float(getattr(o, "sl", 0.0) or 0.0),
                    "tp": float(getattr(o, "tp", 0.0) or 0.0),
                    "comment": str(getattr(o, "comment", "") or ""),
                }
                for o in orders[:5]
            ],
        }
    except Exception as e:
        return {"ok": False, "reason": f"mt5_snapshot_failed:{e}"}
    finally:
        adapter.shutdown()


def _build_ops_context(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    state = _read_trading_state()
    approval = _latest_pending_approval()
    app_cfg = _load_app_cfg()
    market = _summarize_market_snapshot(app_cfg)
    mt5 = _mt5_live_snapshot(trading_cfg)
    return {
        "trading_state": state,
        "pending_approval": approval if approval else None,
        "market_snapshot": market,
        "mt5": mt5,
    }


def _compact_ops_context(context: Dict[str, Any]) -> Dict[str, Any]:
    state = context.get("trading_state") or {}
    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    mode_b = (state.get("mode_b") or {}) if isinstance(state, dict) else {}
    agent_b_plan = (state.get("agent_b_plan") or {}) if isinstance(state, dict) else {}
    approval = context.get("pending_approval") or {}
    market = context.get("market_snapshot") or {}
    mt5 = context.get("mt5") or {}

    compact_plan = {
        "decision": plan.get("decision"),
        "model": plan.get("model"),
        "entry": plan.get("entry"),
        "stop_loss": plan.get("stop_loss"),
        "take_profit": plan.get("take_profit"),
        "volume": plan.get("volume"),
        "confidence": plan.get("confidence"),
        "cm_accuracy": plan.get("cm_accuracy"),
        "signal_score": plan.get("signal_score"),
        "input_fooling_risk": plan.get("input_fooling_risk"),
        "rationale": plan.get("rationale"),
        "risk_notes": (plan.get("risk_notes") or [])[:6],
    }
    compact_mode_b = {
        "consensus": mode_b.get("consensus"),
        "consensus_score": mode_b.get("consensus_score"),
        "last_tick_utc": mode_b.get("last_tick_utc"),
        "next_review_utc": mode_b.get("next_review_utc"),
    }
    compact_agent_b = {
        "recommendation": agent_b_plan.get("recommendation"),
        "consensus": agent_b_plan.get("consensus"),
        "consensus_score": agent_b_plan.get("consensus_score"),
        "reason": agent_b_plan.get("reason"),
        "next_review_utc": agent_b_plan.get("next_review_utc"),
        "actions": (agent_b_plan.get("actions") or [])[:5],
    }
    compact_approval = None
    if approval:
        compact_approval = {
            "approval_id": approval.get("approval_id"),
            "title": approval.get("title"),
            "status": approval.get("status"),
            "deadline_utc": approval.get("deadline_utc"),
        }

    compact_mt5 = {
        "ok": mt5.get("ok"),
        "reason": mt5.get("reason"),
        "positions": (mt5.get("positions") or [])[:2],
        "orders": (mt5.get("orders") or [])[:2],
    }
    compact_market = {
        "ok": market.get("ok"),
        "reason": market.get("reason"),
        "1h": market.get("1h"),
        "7h": market.get("7h"),
    }

    return {
        "trading_state": {
            "job_type": state.get("job_type"),
            "status": state.get("status"),
            "stage": state.get("stage"),
            "mode": state.get("mode"),
            "started_at": state.get("started_at"),
            "closed_reason": state.get("closed_reason"),
            "plan": compact_plan,
            "mode_b": compact_mode_b,
            "agent_b_plan": compact_agent_b,
        },
        "pending_approval": compact_approval,
        "market_snapshot": compact_market,
        "mt5": compact_mt5,
    }


def _resolve_live_position_for_close(trading_cfg: Dict[str, Any], side: str = "") -> Dict[str, Any]:
    requested_side = _normalize_trade_side_token(side)
    state = _read_trading_state()
    desired_ticket = int(
        (((state.get("position") or {}).get("ticket") or 0))
        or (((state.get("agent_b_plan") or {}).get("position_ticket") or 0))
        or 0
    )
    symbol_hint = str((((trading_cfg.get("execution") or {}).get("symbol")) or "")).strip()

    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok, msg = adapter.connect()
    if not ok:
        return {"ok": False, "message": msg}

    try:
        mt5 = adapter._mt5
        if mt5 is None:
            return {"ok": False, "message": "mt5_not_connected"}

        if desired_ticket:
            pos = adapter.get_position_by_ticket(desired_ticket)
            if pos.get("ok") and pos.get("position"):
                resolved_position = dict(pos.get("position") or {})
                if requested_side:
                    resolved_side = _normalize_trade_side_token(resolved_position.get("side"))
                    if resolved_side and resolved_side != requested_side:
                        resolved_position = {}
                if resolved_position:
                    return {"ok": True, "ticket": int(desired_ticket), "position": resolved_position}

        positions = mt5.positions_get() or []
        serialized_positions = [
            {
                "ticket": int(getattr(p, "ticket", 0) or 0),
                "symbol": str(getattr(p, "symbol", "") or ""),
                "volume": float(getattr(p, "volume", 0.0) or 0.0),
                "price_open": float(getattr(p, "price_open", 0.0) or 0.0),
                "side": _normalize_trade_side_token(adapter._position_side_from_type(getattr(p, "type", -1))),
            }
            for p in positions
        ]

        candidate_positions = list(serialized_positions)
        if symbol_hint:
            symbol_filtered = [p for p in candidate_positions if str(p.get("symbol") or "") == symbol_hint]
            if symbol_filtered:
                candidate_positions = symbol_filtered

        if requested_side:
            side_filtered = [p for p in candidate_positions if str(p.get("side") or "") == requested_side]
            if len(side_filtered) == 1:
                pos = side_filtered[0]
                return {
                    "ok": True,
                    "ticket": int(pos.get("ticket", 0) or 0),
                    "position": pos,
                }
            if len(side_filtered) > 1:
                return {
                    "ok": False,
                    "message": f"multiple_live_positions_for_side:{requested_side}",
                    "open_positions": side_filtered[:5],
                }

        if len(candidate_positions) == 1:
            p = candidate_positions[0]
            return {
                "ok": True,
                "ticket": int(p.get("ticket", 0) or 0),
                "position": p,
            }

        return {
            "ok": False,
            "message": "could_not_resolve_single_live_position",
            "open_positions": [
                {
                    "ticket": int((p or {}).get("ticket", 0) or 0),
                    "symbol": str((p or {}).get("symbol", "") or ""),
                    "volume": float((p or {}).get("volume", 0.0) or 0.0),
                    "side": str((p or {}).get("side", "") or ""),
                }
                for p in serialized_positions[:5]
            ],
        }
    finally:
        adapter.shutdown()


def _resolve_close_target(trading_cfg: Dict[str, Any], job_id: str = "", ticket: int = 0, side: str = "") -> Dict[str, Any]:
    desired_job_id = str(job_id or "").strip()
    desired_ticket = int(ticket or 0)
    desired_side = _normalize_trade_side_token(side)

    if desired_ticket > 0:
        matched_job_id, state_path, state = _find_job_for_ticket(desired_ticket)
        return {
            "ok": True,
            "ticket": desired_ticket,
            "job_id": matched_job_id,
            "state_path": str(state_path) if state_path else "",
            "state": state,
            "position": ((state.get("position") or {}) if state else {}),
        }

    if desired_job_id:
        state = _read_trading_state(desired_job_id)
        if not state:
            return {"ok": False, "message": f"job_not_found:{desired_job_id}"}
        position = (state.get("position") or {}) if isinstance(state, dict) else {}
        agent_b_plan = (state.get("agent_b_plan") or {}) if isinstance(state, dict) else {}
        resolved_ticket = int(position.get("ticket") or agent_b_plan.get("position_ticket") or 0)
        if resolved_ticket <= 0:
            return {"ok": False, "message": f"job_has_no_live_position_ticket:{desired_job_id}"}
        if desired_side:
            plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
            resolved_side = _normalize_trade_side_token(position.get("side") or plan.get("decision"))
            if resolved_side and resolved_side != desired_side:
                return {"ok": False, "message": f"job_position_side_mismatch:{desired_job_id}:{resolved_side}"}
        return {
            "ok": True,
            "ticket": resolved_ticket,
            "job_id": desired_job_id,
            "state_path": str(_job_state_path(desired_job_id)),
            "state": state,
            "position": position,
        }

    return _resolve_live_position_for_close(trading_cfg, side=desired_side)


def _close_live_position(trading_cfg: Dict[str, Any], job_id: str = "", ticket: int = 0, side: str = "") -> Dict[str, Any]:
    resolved = _resolve_close_target(trading_cfg, job_id=job_id, ticket=ticket, side=side)
    if not bool(resolved.get("ok", False)):
        return resolved

    ticket = int(resolved.get("ticket") or 0)
    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok, msg = adapter.connect()
    if not ok:
        return {"ok": False, "message": msg}

    try:
        close_res = adapter.close_position_by_ticket(ticket)
    finally:
        adapter.shutdown()

    if not bool(close_res.get("ok", False)):
        return close_res

    state = resolved.get("state") if isinstance(resolved.get("state"), dict) else {}
    if not state:
        state = _read_json(_trading_state_path())
    if not state:
        state = _read_trading_state(str(resolved.get("job_id") or ""))
    if state:
        state["status"] = "closed"
        state["closed_reason"] = "manual_close_via_telegram"
        state["ended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        state["close_result"] = close_res
        resolved_job_id = str(resolved.get("job_id") or state.get("job_id") or "").strip()
        if resolved_job_id:
            state["job_id"] = resolved_job_id
        state_path_raw = str(resolved.get("state_path") or state.get("state_path") or "").strip()
        state_path = Path(state_path_raw) if state_path_raw else _trading_state_path()
        _persist_job_state(state_path, state)

    return {
        "ok": True,
        "ticket": ticket,
        "job_id": str(resolved.get("job_id") or state.get("job_id") or "").strip(),
        "position": resolved.get("position") or {},
        "close_result": close_res,
    }


def _format_ops_fallback(user_text: str, context: Dict[str, Any]) -> str:
    trading_cfg = context.get("trading_cfg") or {}
    state = context.get("trading_state") or {}
    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    mode_b = (state.get("mode_b") or {}) if isinstance(state, dict) else {}
    agent_b_plan = (state.get("agent_b_plan") or {}) if isinstance(state, dict) else {}
    mt5 = context.get("mt5") or {}
    market = context.get("market_snapshot") or {}
    approval = context.get("pending_approval") or {}
    submission_mode = str(state.get("order_submission_mode") or plan.get("order_submission_mode") or "programmed")
    programmed_expiration = str(state.get("programmed_order_expiration_utc") or (state.get("order") or {}).get("expiration_utc") or "").strip()
    text = str(user_text or "").strip().lower()
    positions = mt5.get("positions") or []
    orders = mt5.get("orders") or []
    first_position = positions[0] if positions else {}
    first_order = orders[0] if orders else {}

    if _contains_any(text, ["close current position", "close current positions", "close position", "close positions", "close current order", "close trade"]):
        lines = ["TSMM ops agent summary"]
        if first_position:
            lines.append(
                f"Yes. The current live position is ticket={first_position.get('ticket', 'n/a')} symbol={first_position.get('symbol', 'n/a')} "
                f"volume={first_position.get('volume', 'n/a')} entry={first_position.get('price_open', 'n/a')}."
            )
        else:
            lines.append("No current open MT5 position was found to close.")
        if _contains_any(text, ["restart", "re start", "start again", "new trading job"]):
            lines.append("To execute that explicitly through chat, send: /tsmm trading close-restart")
        else:
            lines.append("To execute that explicitly through chat, send: /tsmm trading close")
        lines.append("I do not auto-close a live position from a general question. An explicit trading command is required.")
        return "\n".join(lines)[:3500]

    if _contains_any(text, ["agent b", "recommend", "live position", "manage", "management"]):
        lines = ["TSMM ops agent summary"]
        if agent_b_plan:
            lines.append(
                f"Agent B recommends {agent_b_plan.get('recommendation', 'n/a')} for the live position. "
                f"Consensus={agent_b_plan.get('consensus', 'n/a')} score={agent_b_plan.get('consensus_score', 'n/a')} "
                f"next_review={_format_utc_for_agent_timezone(agent_b_plan.get('next_review_utc'), trading_cfg)}"
            )
        elif mode_b:
            lines.append(
                f"Mode B is active with consensus={mode_b.get('consensus', 'n/a')} "
                f"score={mode_b.get('consensus_score', 'n/a')} next_review={_format_utc_for_agent_timezone(mode_b.get('next_review_utc'), trading_cfg)}"
            )
        else:
            lines.append("No Agent B management plan is available in the current trading state.")
        if first_position:
            lines.append(
                f"Live MT5 position: ticket={first_position.get('ticket', 'n/a')} symbol={first_position.get('symbol', 'n/a')} "
                f"volume={first_position.get('volume', 'n/a')} entry={first_position.get('price_open', 'n/a')} "
                f"sl={first_position.get('sl', 'n/a')} tp={first_position.get('tp', 'n/a')}"
            )
        elif first_order:
            lines.append(
                f"No open position is visible, but there is a pending order: ticket={first_order.get('order_ticket', first_order.get('ticket', 'n/a'))} "
                f"symbol={first_order.get('symbol', 'n/a')} price={first_order.get('price_open', 'n/a')}"
            )
        lines.append(f"Trading status: {state.get('status', 'unknown')} | mode: {state.get('mode', 'unknown')} | stage: {state.get('stage', 'unknown')}")
        return "\n".join(lines)[:3500]

    if _contains_any(text, ["order", "position", "current trade", "open trade", "current order", "live order"]):
        lines = ["TSMM ops agent summary"]
        if first_position:
            lines.append(
                f"Current live order status: open position ticket={first_position.get('ticket', 'n/a')} "
                f"symbol={first_position.get('symbol', 'n/a')} volume={first_position.get('volume', 'n/a')} "
                f"entry={first_position.get('price_open', 'n/a')} sl={first_position.get('sl', 'n/a')} tp={first_position.get('tp', 'n/a')}"
            )
        elif first_order:
            lines.append(
                f"Current live order status: pending order ticket={first_order.get('order_ticket', first_order.get('ticket', 'n/a'))} "
                f"symbol={first_order.get('symbol', 'n/a')} price={first_order.get('price_open', 'n/a')} "
                f"sl={first_order.get('sl', 'n/a')} tp={first_order.get('tp', 'n/a')}"
            )
        else:
            lines.append("Current live order status: no open MT5 position and no pending MT5 order were found.")
        if agent_b_plan:
            lines.append(
                f"Agent B recommendation: {agent_b_plan.get('recommendation', 'n/a')} "
                f"with consensus={agent_b_plan.get('consensus', 'n/a')} score={agent_b_plan.get('consensus_score', 'n/a')}"
            )
        if approval:
            lines.append(f"Pending approval: title={approval.get('title', 'n/a')} deadline={approval.get('deadline_utc', 'n/a')}")
        lines.append(f"Submission mode: {submission_mode}")
        if programmed_expiration:
            lines.append(f"Programmed order expiration: {programmed_expiration} UTC")
        lines.append(f"Trading status: {state.get('status', 'unknown')} | mode: {state.get('mode', 'unknown')} | stage: {state.get('stage', 'unknown')}")
        return "\n".join(lines)[:3500]

    lines = ["TSMM ops agent summary"]
    lines.append(f"Request: {user_text}")
    lines.append(f"Trading status: {state.get('status', 'unknown')} | mode: {state.get('mode', 'unknown')} | stage: {state.get('stage', 'unknown')}")
    if plan:
        lines.append(f"Agent A plan: decision={plan.get('decision', 'n/a')} model={plan.get('model', 'n/a')} entry={plan.get('entry', 'n/a')} sl={plan.get('stop_loss', 'n/a')} tp={plan.get('take_profit', 'n/a')} submission_mode={submission_mode}")
    if programmed_expiration:
        lines.append(f"Programmed order expiration: {programmed_expiration} UTC")
    if agent_b_plan:
        lines.append(f"Agent B plan: recommendation={agent_b_plan.get('recommendation', 'n/a')} consensus={agent_b_plan.get('consensus', 'n/a')} score={agent_b_plan.get('consensus_score', 'n/a')} next_review={_format_utc_for_agent_timezone(agent_b_plan.get('next_review_utc'), trading_cfg)}")
    elif mode_b:
        lines.append(f"Mode B consensus: {mode_b.get('consensus', 'n/a')} score={mode_b.get('consensus_score', 'n/a')} next_review={_format_utc_for_agent_timezone(mode_b.get('next_review_utc'), trading_cfg)}")
    if approval:
        lines.append(f"Pending approval: title={approval.get('title', 'n/a')} deadline={approval.get('deadline_utc', 'n/a')}")
    if mt5.get('ok'):
        lines.append(f"MT5: open_positions={len(mt5.get('positions') or [])} pending_orders={len(mt5.get('orders') or [])}")
    if market.get('ok'):
        h1 = market.get('1h') or {}
        h7 = market.get('7h') or {}
        if h1:
            lines.append(f"Market 1h: close={h1.get('close')} delta_close={h1.get('delta_close')} at {h1.get('date')}")
        if h7:
            lines.append(f"Market 7h: close={h7.get('close')} delta_close={h7.get('delta_close')} at {h7.get('date')}")
    lines.append("Actions I can execute through chat: start trading, resume trading, stop trading, approve/reject pending approvals, deploy, endpoint restart, UI control, resource status.")
    return "\n".join(lines)[:3500]


def _is_lightweight_ops_chat(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False

    trading_markers = [
        "trade",
        "trading",
        "status",
        "position",
        "order",
        "deploy",
        "endpoint",
        "resource",
        "approval",
        "mt5",
        "market",
        "signal",
        "agent",
        "job",
        "resume",
        "stop",
        "kill",
        "close",
        "profit",
        "loss",
        "price",
    ]
    if _contains_any(raw, trading_markers):
        return False

    lightweight_messages = {
        "hi",
        "hello",
        "hey",
        "yo",
        "hola",
        "good morning",
        "good afternoon",
        "good evening",
        "how are you",
        "what can you do",
        "who are you",
        "help",
        "thanks",
        "thank you",
    }
    if raw in lightweight_messages:
        return True

    return len(raw.split()) <= 4 and raw.endswith(("?", "!", ".")) and raw[:-1].strip() in lightweight_messages


def _lightweight_ops_fallback() -> str:
    return (
        "Hello. TSMM chat is online on the local Ollama path. "
        "You can talk normally, or ask for trading status, active jobs, deploy state, endpoint state, or resource status."
    )


def _agent_chat_fallback() -> str:
    return (
        "The LLM chat session is active, but I could not generate a conversational reply just now. "
        "I did not route your message into TSMM ops mode. Please try again, or say 'back to tsmm' if you want command routing again."
    )


def _agent_chat_requests_tsmm_context(text: str) -> bool:
    raw = str(text or "").strip().lower()
    if not raw:
        return False
    terms = [
        "tsmm",
        "mt5",
        "resource",
        "cpu",
        "ram",
        "memory",
        "current job",
        "current jobs",
        "running job",
        "running jobs",
        "trading job",
        "trading jobs",
        "signal",
        "signals",
        "analysis",
        "analysis summary",
        "summary file",
        "summary files",
        "position",
        "positions",
        "order",
        "orders",
        "plan",
        "consensus",
        "timeframe",
        "inference",
        "output vector",
        "forecast",
        "model",
        "app",
    ]
    return _contains_any(raw, terms)


def _builtin_agent_chat_response(text: str, trading_cfg: Dict[str, Any]) -> Dict[str, Any] | None:
    raw = str(text or "").strip().lower()
    if not raw:
        return None

    if (
        _contains_any(raw, ["are you there", "still there", "can you hear me", "how are you"])
        or bool(re.search(r"\b(hello|hi|hey)\b", raw))
    ):
        return {
            "handled": True,
            "ok": True,
            "message": "I am here and the agent chat session is active. You can keep talking normally.",
            "agent_chat_mode": True,
            "parsed_from_natural_language": True,
            "builtin_agent_reply": True,
        }

    if _contains_any(raw, ["what are you doing", "what are you up to", "what do you do", "what can you do"]):
        return {
            "handled": True,
            "ok": True,
            "message": (
                "I am in Telegram agent-chat mode right now. I can chat normally, answer from TSMM runtime context when you ask for app state, "
                "and hand a request to this VS Code Copilot session when you start it with 'say copilot ...'."
            ),
            "agent_chat_mode": True,
            "parsed_from_natural_language": True,
            "builtin_agent_reply": True,
        }

    copilot_terms = ["copilot", "github copilot"]
    copilot_request_terms = [
        "bring up",
        "tell",
        "ask",
        "need",
        "connect",
        "pass",
        "send",
        "grab",
        "check",
        "here",
    ]
    if _contains_any(raw, copilot_terms) and _contains_any(raw, copilot_request_terms):
        return {
            "handled": True,
            "ok": True,
            "message": (
                "To hand this over to your live VS Code Copilot session, start the message with 'say copilot ...'. "
                "Example: 'say copilot grab the latest chat history'."
            ),
            "agent_chat_mode": True,
            "parsed_from_natural_language": True,
            "builtin_agent_reply": True,
        }

    if _contains_any(raw, ["resource state", "resource status", "cpu", "ram", "memory"]):
        status = read_resource_status(ROOT)
        return {
            "handled": True,
            "ok": True,
            "message": (
                f"Current TSMM resource state: cpu={status.get('cpu'):.1f}% ram={status.get('ram'):.1f}% "
                f"breach_since={status.get('breach_since')} last_relieved_at={status.get('last_relieved_at')}"
            ),
            "agent_chat_mode": True,
            "parsed_from_natural_language": True,
            "builtin_agent_reply": True,
        }

    if _contains_any(raw, ["search the web", "search web", "browse the web", "internet access", "web access"]):
        return {
            "handled": True,
            "ok": True,
            "message": (
                "Not through this Telegram agent-chat path. I can use the TSMM runtime context available here, "
                "but web search is not exposed as a direct Telegram chat capability. To route that request to your live Copilot session, say 'say copilot search the web for ...'."
            ),
            "agent_chat_mode": True,
            "parsed_from_natural_language": True,
            "builtin_agent_reply": True,
        }

    if _contains_any(raw, ["console commands", "terminal commands", "shell commands", "execute commands", "run commands"]):
        return {
            "handled": True,
            "ok": True,
            "message": (
                "Not from free-form Telegram agent chat. This chat can discuss TSMM state in read-only mode, "
                "but arbitrary console execution is not exposed here as a normal chat capability. If you want to hand a terminal task to your live Copilot session, say 'say copilot run this terminal task: ...'."
            ),
            "agent_chat_mode": True,
            "parsed_from_natural_language": True,
            "builtin_agent_reply": True,
        }

    if _contains_any(raw, ["what time is now", "what time is it", "current time", "time now"]):
        tz_name = _agent_timezone_name(trading_cfg)
        now_local = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(ZoneInfo(tz_name))
        return {
            "handled": True,
            "ok": True,
            "message": f"Current time is {now_local.strftime('%Y-%m-%d %H:%M:%S')} {tz_name}.",
            "agent_chat_mode": True,
            "parsed_from_natural_language": True,
            "builtin_agent_reply": True,
        }

    return None


def _active_job_summaries_for_agent_chat(trading_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for state in _active_job_display_states(trading_cfg)[:5]:
        plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
        position = (state.get("position") or {}) if isinstance(state, dict) else {}
        order = (state.get("order") or {}) if isinstance(state, dict) else {}
        agent_b_plan = (state.get("agent_b_plan") or {}) if isinstance(state, dict) else {}
        entry_value = position.get("price_open") or order.get("price_open") or plan.get("entry")
        stop_loss_value = position.get("sl") or order.get("sl") or plan.get("stop_loss")
        take_profit_value = position.get("tp") or order.get("tp") or plan.get("take_profit")
        summaries.append(
            {
                "job_id": state.get("job_id"),
                "market_state": _job_display_status(state),
                "handler": "Agent B" if _job_display_status(state) == "running_position" else "Agent A",
                "decision": plan.get("decision"),
                "model": plan.get("model"),
                "entry": entry_value,
                "stop_loss": stop_loss_value,
                "take_profit": take_profit_value,
                "mt5_ticket": _job_ticket(state) or order.get("order_ticket") or order.get("ticket"),
                "agent_a_rationale": plan.get("rationale"),
                "agent_a_risk_notes": (plan.get("risk_notes") or [])[:4],
                "agent_b_recommendation": agent_b_plan.get("recommendation"),
                "agent_b_consensus": agent_b_plan.get("consensus"),
                "agent_b_score": agent_b_plan.get("consensus_score"),
                "agent_b_reason": agent_b_plan.get("reason"),
                "next_review_utc": agent_b_plan.get("next_review_utc"),
            }
        )
    return summaries


def _agent_tsmm_context_fallback(text: str, trading_cfg: Dict[str, Any], compact_context: Dict[str, Any]) -> str:
    active_jobs = compact_context.get("active_jobs") or []
    raw = str(text or "").strip().lower()
    lines = ["I can inspect the current TSMM state in read-only mode from the running app context."]

    if active_jobs:
        lines.append(f"TSMM currently reports {len(active_jobs)} active job(s).")
        for job in active_jobs[:3]:
            lines.append(
                f"- {job.get('job_id', 'n/a')}: state={job.get('market_state', 'n/a')} signal={job.get('decision', 'n/a')}/{job.get('model', 'n/a')} "
                f"entry={job.get('entry', 'n/a')} sl={job.get('stop_loss', 'n/a')} tp={job.get('take_profit', 'n/a')} ticket={job.get('mt5_ticket', 'n/a')}"
            )
            if job.get("agent_a_rationale"):
                lines.append(f"  Agent A rationale: {job.get('agent_a_rationale')}")
            if job.get("agent_b_recommendation"):
                lines.append(
                    f"  Agent B: recommendation={job.get('agent_b_recommendation')} consensus={job.get('agent_b_consensus', 'n/a')} "
                    f"score={job.get('agent_b_score', 'n/a')}"
                )
            if job.get("agent_b_reason"):
                lines.append(f"  Agent B reason: {job.get('agent_b_reason')}")
    else:
        lines.append("I do not see any active TSMM jobs in the current runtime context.")

    if _contains_any(raw, ["inference", "output vector", "timeframe"]):
        lines.append(
            "I do not have a chat execution path wired to run a fresh timeframe inference or produce a new output vector from agent mode. "
            "I can only discuss the live TSMM state and any stored plan/consensus data already exposed here."
        )
    else:
        lines.append(
            "If you want more detail on a specific job, ask for its job id, signal rationale, Agent B recommendation, or MT5 ticket and I will use the current TSMM context."
        )

    return "\n".join(lines)[:3500]


def _recent_llm_session_messages(chat_id: str, limit: int = 8) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    clean_chat_id = str(chat_id or "").strip()
    if not clean_chat_id:
        return messages

    session_files: List[Path] = []
    current_session_file = _llm_session_file(clean_chat_id)
    if current_session_file is not None and current_session_file.exists():
        session_files.append(current_session_file)

    session_dir = _llm_session_dir(clean_chat_id)
    if session_dir.exists():
        previous_files = sorted(session_dir.glob("llm_session_*.jsonl"), reverse=True)
        for session_file in previous_files:
            if current_session_file is not None and session_file == current_session_file:
                continue
            session_files.append(session_file)
            if len(session_files) >= 3:
                break

    for session_file in session_files:
        try:
            lines = session_file.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue

        batch: List[Dict[str, str]] = []
        for raw_line in lines[-max(limit * 3, limit):]:
            try:
                entry = json.loads(raw_line)
            except Exception:
                continue
            direction = str(entry.get("direction") or "").strip().lower()
            if direction not in {"inbound", "outbound"}:
                continue
            text = str(entry.get("text") or "").strip()
            if not text:
                continue
            role = "user" if direction == "inbound" else "assistant"
            batch.append({"role": role, "text": text})
        if batch:
            messages = batch + messages
        if len(messages) >= limit:
            break
    return messages[-limit:]


def _handle_agent_chat(text: str, trading_cfg: Dict[str, Any], source_chat_id: str = "") -> Dict[str, Any]:
    ai_cfg = _ai_agent_cfg(trading_cfg)
    builtin_reply = _builtin_agent_chat_response(text, trading_cfg)
    if builtin_reply is not None:
        return builtin_reply

    if not bool(ai_cfg.get("enabled", True)):
        return {
            "handled": True,
            "ok": True,
            "message": _agent_chat_fallback(),
            "agent_chat_mode": True,
            "parsed_from_natural_language": True,
        }

    providers_path = str((trading_cfg.get("llm") or {}).get("providers_config_path", "config/llm_providers.yaml"))
    providers_cfg = load_llm_providers_config(str(ROOT / providers_path) if not os.path.isabs(providers_path) else providers_path)
    timeout_sec = int(ai_cfg.get("timeout_seconds", (trading_cfg.get("llm") or {}).get("timeout_seconds", 45)) or 45)
    recent_messages = _recent_llm_session_messages(source_chat_id, limit=8)
    history_lines = [f"{msg['role'].upper()}: {msg['text']}" for msg in recent_messages if msg.get("text")]
    history_block = "\n".join(history_lines[-8:])
    wants_tsmm_context = _agent_chat_requests_tsmm_context(text)
    compact_context: Dict[str, Any] = {}
    if wants_tsmm_context:
        context = _build_ops_context(trading_cfg)
        compact_context = _compact_ops_context({**context, "trading_cfg": trading_cfg})
        compact_context["active_jobs"] = _active_job_summaries_for_agent_chat(trading_cfg)

    prompt = (
        "You are the TSMM Telegram assistant in dedicated LLM chat mode. "
        "The user explicitly switched away from TSMM operational routing. "
        "Reply conversationally and directly to the user's message. "
        "Do not dump trading status, market context, or TSMM ops summaries unless the user explicitly asks for TSMM operational details. "
        "When TSMM context is provided below, you do have read-only access to the running app state and should use it instead of claiming you have no access. "
        "Do not trigger or imply command execution. "
        "If the user asks for a fresh inference, new output vector, or file access that is not already present in the provided context, say so plainly instead of pretending it already happened. "
        "If asked about self-awareness, answer honestly that you are an AI assistant and not self-aware. "
        "Keep replies concise, natural, and limited to 1 to 4 short sentences.\n\n"
        f"RECENT_SESSION_TRANSCRIPT:\n{history_block or 'n/a'}\n\n"
        f"LIVE_TSMM_CONTEXT_JSON:\n{json.dumps(compact_context, default=str)[:5000] if compact_context else 'n/a'}\n\n"
        f"CURRENT_USER_MESSAGE:\n{text}\n"
    )

    for provider_name in _select_llm_providers_for_chat(trading_cfg, providers_cfg):
        out = call_llm(provider_name=provider_name, prompt=prompt, providers_cfg=providers_cfg, timeout_sec=timeout_sec)
        if bool(out.get("ok", False)) and str(out.get("text") or "").strip():
            return {
                "handled": True,
                "ok": True,
                "message": str(out.get("text") or "").strip()[:3500],
                "llm_provider": provider_name,
                "agent_chat_mode": True,
                "parsed_from_natural_language": True,
            }

    return {
        "handled": True,
        "ok": True,
        "message": _agent_tsmm_context_fallback(text, trading_cfg, compact_context) if wants_tsmm_context else _agent_chat_fallback(),
        "agent_chat_mode": True,
        "parsed_from_natural_language": True,
    }

def _job_finished_message(state: Dict[str, Any], close_result: Dict[str, Any], position: Dict[str, Any]) -> str:
    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    pos = dict(position or {})
    if isinstance(state.get("position"), dict):
        pos = dict(state.get("position") or {}) | pos
    parts = ["Trading job finished."]
    parts.append(f"status={state.get('status', 'n/a')}")
    parts.append(f"closed_reason={state.get('closed_reason', 'n/a')}")
    parts.append(f"decision={plan.get('decision', 'n/a')}")
    parts.append(f"entry={plan.get('entry', 'n/a')}")
    ticket = pos.get("ticket") or close_result.get("ticket") or close_result.get("position_ticket") or "n/a"
    parts.append(f"mt5_ticket={ticket}")
    if pos.get("symbol"):
        parts.append(f"symbol={pos.get('symbol')}")
    if pos.get("volume") is not None:
        parts.append(f"volume={pos.get('volume')}")
    if pos.get("price_open") is not None:
        parts.append(f"price_open={pos.get('price_open')}")
    if pos.get("price_current") is not None:
        parts.append(f"last_price={pos.get('price_current')}")
    if pos.get("profit") is not None:
        parts.append(f"profit={pos.get('profit')}")
    if state.get("started_at"):
        parts.append(f"started_at={state.get('started_at')}")
    if state.get("ended_at"):
        parts.append(f"ended_at={state.get('ended_at')}")
    return "; ".join(parts)


def _format_trading_status_summary(
    trading_cfg: Dict[str, Any],
    state: Dict[str, Any],
    approval: Dict[str, Any] | None = None,
    endpoint_pid_exists: bool | None = None,
    llm_provider: str | None = None,
) -> str:
    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    mode_b = (state.get("mode_b") or {}) if isinstance(state, dict) else {}
    agent_b_plan = (state.get("agent_b_plan") or {}) if isinstance(state, dict) else {}
    ticket = _job_ticket(state) if isinstance(state, dict) else 0
    submission_mode = str(state.get("order_submission_mode") or plan.get("order_submission_mode") or "programmed") if isinstance(state, dict) else "programmed"
    programmed_expiration = str(state.get("programmed_order_expiration_utc") or ((state.get("order") or {}).get("expiration_utc") if isinstance(state, dict) else "") or "").strip()

    parts = [
        f"job_id={state.get('job_id', 'n/a')}",
        f"status={state.get('status', 'unknown')}",
        f"mode={state.get('mode', 'unknown')}",
        f"stage={state.get('stage', 'unknown')}",
    ]
    if ticket:
        parts.append(f"mt5_ticket={ticket}")
    parts.append(f"submission_mode={submission_mode}")
    if programmed_expiration:
        parts.append(f"programmed_order_expiration_utc={programmed_expiration}")
    if endpoint_pid_exists is not None:
        parts.append(f"endpoint_pid_file={'yes' if endpoint_pid_exists else 'no'}")
    if llm_provider:
        parts.append(f"llm_provider={llm_provider}")
    if plan:
        parts.append(f"decision={plan.get('decision', 'n/a')}")
        parts.append(f"model={plan.get('model', 'n/a')}")
        next_review = agent_b_plan.get('next_review_utc')
        if next_review:
            parts.append(f"next_review={_format_utc_for_agent_timezone(next_review, trading_cfg)}")
    elif mode_b:
        parts.append(f"mode_b_consensus={mode_b.get('consensus', 'n/a')}")
        parts.append(f"mode_b_score={mode_b.get('consensus_score', 'n/a')}")
        parts.append(f"next_review={_format_utc_for_agent_timezone(mode_b.get('next_review_utc'), trading_cfg)}")
    if approval:
        parts.append(f"pending_approval={approval.get('title', 'n/a')}")
        parts.append(f"approval_deadline={approval.get('deadline_utc', 'n/a')}")
    parts.append(f"closed_reason={state.get('closed_reason', 'n/a')}")
    return "; ".join(parts)


def _select_llm_providers_for_chat(trading_cfg: Dict[str, Any], providers_cfg: Dict[str, Any]) -> List[str]:
    ai_cfg = _ai_agent_cfg(trading_cfg)
    preferred = str(ai_cfg.get("provider") or "").strip()
    llm_cfg = (trading_cfg.get("llm") or {})
    configured = str(llm_cfg.get("provider") or providers_cfg.get("default_provider", "")).strip()
    candidates = [preferred, configured, "local_ollama", "github_models", "huggingface", "openai_compatible", "anthropic", "local_transformers"]
    ordered: List[str] = []
    seen = set()
    providers = (providers_cfg.get("providers") or {})
    for name in candidates:
        if not name or name in seen:
            continue
        provider_cfg = providers.get(name) or {}
        if not bool(provider_cfg.get("enabled", False)):
            continue
        if str(provider_cfg.get("type") or "").strip().lower() == "local_transformers":
            continue
        ordered.append(name)
        seen.add(name)
    return ordered


def _call_external_ops_agent(text: str, trading_cfg: Dict[str, Any], source_chat_id: str = "", compact_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = _external_ops_agent_cfg(trading_cfg)
    if not bool(cfg.get("enabled", False)):
        return {}

    base_url = str(cfg.get("base_url") or "").strip().rstrip("/")
    if not base_url:
        return {}

    endpoint = str(cfg.get("chat_endpoint") or "/chat").strip()
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    url = f"{base_url}{endpoint}"
    timeout_sec = int(cfg.get("timeout_seconds", 90) or 90)
    payload = {
        "message": text,
        "chat_id": source_chat_id or "telegram",
        "source": "telegram",
        "auto_execute": bool(cfg.get("auto_execute", True)),
        "tsmm_context": compact_context or {},
    }
    headers = {"Content-Type": "application/json"}
    api_key = _resolve_secret(cfg.get("api_key") or cfg.get("auth_token") or "")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout_sec)
        if response.status_code >= 400:
            return {}
        data = response.json() if response.headers.get("content-type", "").lower().startswith("application/json") else {}
    except Exception:
        return {}

    message = str((data or {}).get("reply") or (data or {}).get("message") or "").strip()
    if not message:
        return {}

    return {
        "handled": True,
        "ok": True,
        "message": message[:3500],
        "llm_provider": str((data or {}).get("provider") or "aorq_agentic_layers").strip() or "aorq_agentic_layers",
        "parsed_from_natural_language": True,
        "agentic_service": True,
        "agentic_tool_results": (data or {}).get("tool_results") or [],
    }


def _handle_ops_chat(text: str, trading_cfg: Dict[str, Any], source_chat_id: str = "") -> Dict[str, Any]:
    ai_cfg = _ai_agent_cfg(trading_cfg)
    if not bool(ai_cfg.get("enabled", True)):
        context = _build_ops_context(trading_cfg)
        return {"handled": True, "ok": True, "message": _format_ops_fallback(text, context), "ops_agent_fallback": True}

    providers_path = str((trading_cfg.get("llm") or {}).get("providers_config_path", "config/llm_providers.yaml"))
    providers_cfg = load_llm_providers_config(str(ROOT / providers_path) if not os.path.isabs(providers_path) else providers_path)
    timeout_sec = int(ai_cfg.get("timeout_seconds", (trading_cfg.get("llm") or {}).get("timeout_seconds", 45)) or 45)

    if _is_lightweight_ops_chat(text):
        prompt = (
            "You are TSMM Telegram Ops Agent running locally. "
            "The user sent a lightweight conversational opener. "
            "Reply naturally in 1 to 3 short sentences. "
            "Mention that you can help with TSMM trading status, jobs, deploys, endpoint health, and resource checks, "
            "but do not dump live trading context unless the user asks for it.\n\n"
            f"USER_MESSAGE:\n{text}\n"
        )
        for provider_name in _select_llm_providers_for_chat(trading_cfg, providers_cfg):
            out = call_llm(provider_name=provider_name, prompt=prompt, providers_cfg=providers_cfg, timeout_sec=timeout_sec)
            if bool(out.get("ok", False)) and str(out.get("text") or "").strip():
                return {
                    "handled": True,
                    "ok": True,
                    "message": str(out.get("text") or "").strip()[:3500],
                    "llm_provider": provider_name,
                    "parsed_from_natural_language": True,
                }
        return {
            "handled": True,
            "ok": True,
            "message": _lightweight_ops_fallback(),
            "ops_agent_fallback": True,
            "parsed_from_natural_language": True,
        }

    context = _build_ops_context(trading_cfg)

    compact_context = _compact_ops_context({**context, "trading_cfg": trading_cfg})
    external_reply = _call_external_ops_agent(text, trading_cfg, source_chat_id=source_chat_id, compact_context=compact_context)
    if external_reply.get("message"):
        return external_reply

    prompt = (
        "You are TSMM Telegram Ops Agent. Answer using the provided live trading and market context. "
        "Be concise, factual, and operational. Do not invent fills, approvals, or positions. "
        "Prefer short plain-English sentences over JSON. "
        "If the user asks for an action that requires execution, mention the concrete chat command or approval step. "
        "If approval is pending, mention it explicitly.\n\n"
        f"USER_MESSAGE:\n{text}\n\n"
        f"LIVE_CONTEXT_JSON:\n{json.dumps(compact_context, default=str)[:4000]}\n"
    )

    for provider_name in _select_llm_providers_for_chat(trading_cfg, providers_cfg):
        out = call_llm(provider_name=provider_name, prompt=prompt, providers_cfg=providers_cfg, timeout_sec=timeout_sec)
        if bool(out.get("ok", False)) and str(out.get("text") or "").strip():
            return {
                "handled": True,
                "ok": True,
                "message": str(out.get("text") or "").strip()[:3500],
                "llm_provider": provider_name,
                "parsed_from_natural_language": True,
            }

    return {
        "handled": True,
        "ok": True,
        "message": _format_ops_fallback(text, context),
        "ops_agent_fallback": True,
        "parsed_from_natural_language": True,
    }


def _read_last_jsonl(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            return {}
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return {}
        return json.loads(lines[-1])
    except Exception:
        return {}


def _latest_request_status_message(request_info: Dict[str, Any]) -> str:
    req_type = str(request_info.get("type") or "request")
    pid = int(request_info.get("pid") or 0)
    job_id = str(request_info.get("job_id") or "").strip()
    alive = _is_pid_alive(pid)
    base = f"request_status: type={req_type}; pid={pid}; running={'yes' if alive else 'no'}"
    if job_id:
        base += f"; job_id={job_id}"

    if req_type == "deploy":
        stage_tail = _read_last_jsonl(ROOT / "reports" / "runtime" / "deployment_pipeline_stage_log.jsonl")
        stage = str(stage_tail.get("stage") or "n/a")
        stage_status = str(stage_tail.get("status") or "n/a")
        return f"{base}; stage={stage}; stage_status={stage_status}"

    if req_type in {"trading start", "trading resume"}:
        state = _read_trading_state(job_id)
        plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
        approval = _latest_pending_approval()
        base = (
            f"{base}; status={state.get('status', 'unknown')}; stage={state.get('stage', 'unknown')}; "
            f"decision={plan.get('decision', 'n/a')}; closed_reason={state.get('closed_reason', 'n/a')}"
        )
        if approval and str(approval.get("job_id") or "").strip() == job_id:
            base += (
                f"; pending_approval={approval.get('title', 'n/a')}; "
                f"approval_deadline={approval.get('deadline_utc', 'n/a')}; "
                f"action=/tsmm trading approve --job-id {job_id}"
            )
        return base

    return base


def _latest_request_done_message(request_info: Dict[str, Any]) -> str:
    req_type = str(request_info.get("type") or "request")
    pid = int(request_info.get("pid") or 0)
    job_id = str(request_info.get("job_id") or "").strip()
    base = f"request_done: type={req_type}; pid={pid}; running=no"
    if job_id:
        base += f"; job_id={job_id}"

    if req_type == "deploy":
        summary = _read_json(ROOT / "reports" / "runtime" / "deployment_pipeline_last.json")
        return (
            f"{base}; llm_provider={((summary.get('llm') or {}).get('chosen_provider', 'n/a'))}; "
            f"endpoint_ok={((summary.get('endpoint') or {}).get('ok', 'n/a'))}; "
            f"trading_started={((summary.get('trading') or {}).get('started', False))}"
        )

    if req_type in {"trading start", "trading resume"}:
        state = _read_trading_state(job_id)
        plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
        approval = _latest_pending_approval()
        base = (
            f"{base}; status={state.get('status', 'unknown')}; stage={state.get('stage', 'unknown')}; "
            f"decision={plan.get('decision', 'n/a')}; closed_reason={state.get('closed_reason', 'n/a')}"
        )
        if approval and str(approval.get("job_id") or "").strip() == job_id:
            base += (
                f"; pending_approval={approval.get('title', 'n/a')}; "
                f"approval_deadline={approval.get('deadline_utc', 'n/a')}; "
                f"action=/tsmm trading approve --job-id {job_id}"
            )
        return base

    return base


def _restart_endpoint_service(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    lcfg = _listener_cfg(trading_cfg)
    p_cfg = _load_yaml(ROOT / str(lcfg.get("default_pipeline_config") or "config/agent_pipeline.yaml"))
    ep_cfg = (p_cfg.get("endpoints") or {})

    script = str(ep_cfg.get("service_script", "scripts/local_signal_endpoint_service.py"))
    host = str(ep_cfg.get("host", "127.0.0.1"))
    port = int(ep_cfg.get("port", 8000) or 8000)
    stopped_pids = _stop_local_endpoint_service(script)

    env = os.environ.copy()
    env["TSMM_SIGNAL_HOST"] = host
    env["TSMM_SIGNAL_PORT"] = str(port)

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    p = subprocess.Popen(
        [sys.executable, script],
        cwd=str(ROOT),
        env=env,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    health_url = f"http://{host}:{port}/health"
    ok = False
    payload: Dict[str, Any] = {}
    for _ in range(20):
        time.sleep(1)
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                payload = r.json()
                ok = True
                break
        except Exception:
            pass

    return {
        "ok": ok,
        "pid": int(p.pid),
        "stopped_pids": stopped_pids,
        "health_url": health_url,
        "health": payload,
    }


def _handle_command(
    text: str,
    trading_cfg: Dict[str, Any],
    source_chat_id: str = "",
    trading_config_path: Path | None = None,
) -> Dict[str, Any]:
    lcfg = _listener_cfg(trading_cfg)
    prefix = str(lcfg.get("command_prefix", "/tsmm")).strip() or "/tsmm"
    body = text.strip()
    if not body.lower().startswith(prefix.lower()):
        switch_out = _chat_mode_switch_response(body, source_chat_id=source_chat_id)
        if switch_out is not None:
            return switch_out

        copilot_handoff = _handle_copilot_bridge_request(body, trading_cfg, source_chat_id=source_chat_id)
        if copilot_handoff is not None:
            return copilot_handoff

        if _chat_mode(source_chat_id) == "agent":
            routed = _handle_agent_chat(body, trading_cfg, source_chat_id=source_chat_id)
            if routed.get("message"):
                routed["agent_chat_mode"] = True
                return routed

        if bool(lcfg.get("allow_natural_language", True)):
            tail, chat_msg = _infer_natural_language_tail(body)
            if tail:
                routed = _handle_command(
                    f"{prefix} {tail}",
                    trading_cfg,
                    source_chat_id=source_chat_id,
                    trading_config_path=trading_config_path,
                )
                if routed.get("handled", False):
                    routed["message"] = (
                        f"I interpreted your request as: '{tail}'.\n"
                        f"{str(routed.get('message') or '')}"
                    )
                    routed["parsed_from_natural_language"] = True
                return routed
            if chat_msg:
                ops_reply = _handle_ops_chat(body, trading_cfg, source_chat_id=source_chat_id)
                if ops_reply.get("message"):
                    return ops_reply
                return {
                    "handled": True,
                    "ok": True,
                    "message": chat_msg,
                    "parsed_from_natural_language": True,
                }

        if bool(lcfg.get("reply_on_non_command", True)):
            return _handle_ops_chat(body, trading_cfg, source_chat_id=source_chat_id)
        return {"handled": False, "reason": "wrong_prefix"}

    tail = body[len(prefix) :].strip()
    if not tail:
        return {
            "handled": True,
            "ok": True,
            "message": _help_message(prefix),
        }

    parts = tail.split()
    cmd = parts[0].lower()
    rest = parts[1:]

    if cmd in {"help", "commands", "?"}:
        return {
            "handled": True,
            "ok": True,
            "message": _help_message(prefix),
        }

    if cmd not in _allowed_roots(trading_cfg):
        return {"handled": True, "ok": False, "message": f"command_not_allowed:{cmd}"}

    env = os.environ.copy()
    if trading_config_path is not None:
        env["TRADING_CONFIG_PATH"] = str(trading_config_path)

    if cmd == "status":
        endpoint_path = ROOT / "reports" / "runtime" / "local_signal_endpoint_service.pid"
        last_summary = ROOT / "reports" / "runtime" / "deployment_pipeline_last.json"
        state = _read_trading_state()

        summary = {}
        if last_summary.exists():
            try:
                summary = json.loads(last_summary.read_text(encoding="utf-8"))
            except Exception:
                summary = {}

        approval = _latest_pending_approval()

        return {
            "handled": True,
            "ok": True,
            "message": "status: "
            + _format_trading_status_summary(
                trading_cfg,
                state or {},
                approval=approval if approval else None,
                endpoint_pid_exists=endpoint_path.exists(),
                llm_provider=str((summary.get('llm') or {}).get('chosen_provider', 'n/a')),
            ),
        }

    if cmd == "deploy":
        if rest and rest[0].lower() == "stop":
            script = str(lcfg.get("run_deploy_script", "scripts/deploy_agent_pipeline.py"))
            args = [sys.executable, script, "--stop"]
            out = _run_cmd(args, env)
            return {
                "handled": True,
                "ok": out.get("ok", False),
                "message": f"deploy stop rc={out.get('returncode')}",
                "exec_args": out.get("args"),
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "returncode": out.get("returncode"),
            }

        # Deploy can run for several minutes; run detached to keep listener responsive.
        out = _launch_deploy_pipeline(trading_cfg, extra_args=rest)
        return {
            "handled": True,
            "ok": out.get("ok", False),
            "message": f"deploy started pid={out.get('pid')}",
            "track_request": {
                "type": "deploy",
                "pid": out.get("pid"),
                "restart_endpoint_after_done": bool(out.get("refresh_requested", False)),
            },
            "exec_args": out.get("args"),
            "stdout": out.get("stdout", ""),
            "stderr": out.get("stderr", ""),
            "returncode": out.get("returncode"),
        }

    if cmd == "trading":
        action = rest[0].lower() if rest else ""
        requested_job_id = ""
        requested_ticket = 0
        requested_side = ""
        requested_plan_model = ""
        requested_submission_mode = ""
        idx = 1
        while idx < len(rest):
            token = rest[idx]
            if token == "--plan-model" and idx + 1 < len(rest):
                requested_plan_model = str(rest[idx + 1]).strip()
                idx += 2
                continue
            if token == "--submission-mode" and idx + 1 < len(rest):
                requested_submission_mode = str(rest[idx + 1]).strip().lower()
                idx += 2
                continue
            if token == "--job-id" and idx + 1 < len(rest):
                requested_job_id = str(rest[idx + 1]).strip()
                idx += 2
                continue
            if token == "--ticket" and idx + 1 < len(rest):
                try:
                    requested_ticket = int(rest[idx + 1])
                except Exception:
                    requested_ticket = 0
                idx += 2
                continue
            if token == "--side" and idx + 1 < len(rest):
                requested_side = _normalize_trade_side_token(rest[idx + 1])
                idx += 2
                continue
            idx += 1

        if action == "status":
            state = _read_trading_state(requested_job_id)
            approval = _latest_pending_approval()
            return {
                "handled": True,
                "ok": True,
                "message": "trading_status: " + _format_trading_status_summary(trading_cfg, state, approval=approval if approval else None),
            }

        if action == "jobs":
            return {
                "handled": True,
                "ok": True,
                "message": _format_active_jobs_digest(trading_cfg),
            }

        if action in {"approve", "reject"}:
            req = _latest_pending_approval()
            if not req or str(req.get("status") or "").strip().lower() != "pending":
                return {"handled": True, "ok": False, "message": "no_pending_approval_request"}
            decision = "approve" if action == "approve" else "reject"
            response_path = Path(str(req.get("response_path") or _approval_response_path()))
            request_path = Path(str(req.get("request_path") or _approval_request_path()))
            _write_json(
                response_path,
                {
                    "approval_id": req.get("approval_id"),
                    "decision": decision,
                    "chat_id": str(source_chat_id or ""),
                    "received_at_utc": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            req["status"] = decision
            req["resolved_at_utc"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _write_json(request_path, req)
            _refresh_latest_approval_alias()
            return {
                "handled": True,
                "ok": True,
                "message": f"trading approval recorded: decision={decision}; title={req.get('title', 'n/a')}; job_id={req.get('job_id', 'n/a')}",
            }

        if action == "close":
            close_res = _close_live_position(trading_cfg, job_id=requested_job_id, ticket=requested_ticket, side=requested_side)
            if not bool(close_res.get("ok", False)):
                return {
                    "handled": True,
                    "ok": False,
                    "message": f"trading close failed: {close_res.get('message', close_res.get('error', 'unknown'))}",
                }
            pos = close_res.get("position") or {}
            return {
                "handled": True,
                "ok": True,
                "message": (
                    f"trading close executed: job_id={close_res.get('job_id', requested_job_id or 'n/a')}; ticket={close_res.get('ticket', 'n/a')}; "
                    f"symbol={pos.get('symbol', 'n/a')}; volume={pos.get('volume', 'n/a')}"
                ),
            }

        if action == "close-restart":
            close_res = _close_live_position(trading_cfg)
            if not bool(close_res.get("ok", False)):
                return {
                    "handled": True,
                    "ok": False,
                    "message": f"trading close-restart failed during close: {close_res.get('message', close_res.get('error', 'unknown'))}",
                }
            new_job_id = _new_job_id(trading_cfg)
            args = [sys.executable, "app.py", "trading-job", "start", "--job-id", new_job_id]
            out = _run_cmd_async(args, env)
            return {
                "handled": True,
                "ok": out.get("ok", False),
                "message": f"trading close-restart executed: closed ticket={close_res.get('ticket', 'n/a')}; new trading job pid={out.get('pid')}; job_id={new_job_id}",
                "track_request": {
                    "type": "trading close-restart",
                    "pid": out.get("pid"),
                    "job_id": new_job_id,
                },
                "exec_args": out.get("args"),
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "returncode": out.get("returncode"),
            }

        if action not in {"start", "resume", "stop", "kill"}:
            return {"handled": True, "ok": False, "message": "usage: trading start|resume|stop|kill|jobs|close|close-restart|status|approve|reject [--submission-mode programmed|market] [--job-id JOB] [--ticket TICKET] [--side buy|sell]"}

        args = [sys.executable, "app.py", "trading-job", action]
        if action == "start" and requested_plan_model:
            args.extend(["--plan-model", requested_plan_model])
        if action == "start":
            submission_mode = requested_submission_mode if requested_submission_mode in {"programmed", "market"} else "programmed"
            args.extend(["--submission-mode", submission_mode])

        if action == "start":
            requested_job_id = requested_job_id or _new_job_id(trading_cfg)
        if requested_job_id:
            args.extend(["--job-id", requested_job_id])

        # trading start/resume can block; launch detached so listener keeps polling.
        if action in {"start", "resume"}:
            out = _run_cmd_async(args, env)
            startup_wait_seconds = max(float(lcfg.get("trading_start_state_wait_seconds", 2.0) or 2.0), 0.0)
            startup_wait_seconds = min(startup_wait_seconds, 10.0)
            startup_state = _await_job_state(requested_job_id, timeout_seconds=startup_wait_seconds) if requested_job_id else {}
            return {
                "handled": True,
                "ok": out.get("ok", False),
                "message": _format_trading_launch_message(
                    action=action,
                    pid=out.get("pid"),
                    job_id=requested_job_id or "latest",
                    startup_state=startup_state,
                ),
                "track_request": {
                    "type": f"trading {action}",
                    "pid": out.get("pid"),
                    "job_id": requested_job_id or "",
                    "startup_state_detected": bool(startup_state),
                },
                "exec_args": out.get("args"),
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "returncode": out.get("returncode"),
            }

        out = _run_cmd(args, env)
        message = f"trading {action} rc={out.get('returncode')}"
        stdout_text = str(out.get("stdout") or "").strip()
        if action in {"stop", "kill"} and stdout_text:
            message = stdout_text
        return {
            "handled": True,
            "ok": out.get("ok", False),
            "message": message,
            "exec_args": out.get("args"),
            "stdout": out.get("stdout", ""),
            "stderr": out.get("stderr", ""),
            "returncode": out.get("returncode"),
        }

    if cmd == "endpoint" and rest and rest[0].lower() == "restart":
        out = _restart_endpoint_service(trading_cfg)
        return {
            "handled": True,
            "ok": out.get("ok", False),
            "message": f"endpoint restart ok={out.get('ok')} pid={out.get('pid')}",
        }

    if cmd == "ui":
        action = rest[0].lower() if rest else ""
        if action == "start":
            out = _run_cmd([sys.executable, "scripts/start_all_uis.py"], env)
            return {
                "handled": True,
                "ok": out.get("ok", False),
                "message": f"ui start rc={out.get('returncode')}",
                "exec_args": out.get("args"),
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "returncode": out.get("returncode"),
            }
        if action == "stop":
            out = _run_cmd([sys.executable, "scripts/stop_all_uis.py"], env)
            return {
                "handled": True,
                "ok": out.get("ok", False),
                "message": f"ui stop rc={out.get('returncode')}",
                "exec_args": out.get("args"),
                "stdout": out.get("stdout", ""),
                "stderr": out.get("stderr", ""),
                "returncode": out.get("returncode"),
            }
        return {"handled": True, "ok": False, "message": "usage: ui start|stop"}

    if cmd == "resource":
        action = rest[0].lower() if rest else "status"
        if action == "status":
            st = read_resource_status(ROOT)
            return {
                "handled": True,
                "ok": True,
                "message": (
                    f"resource_status: cpu={st.get('cpu'):.1f}% ram={st.get('ram'):.1f}% "
                    f"breach_since={st.get('breach_since')} last_relieved_at={st.get('last_relieved_at')}"
                ),
            }
        if action == "relieve":
            out = check_and_relieve(ROOT, trading_cfg)
            return {
                "handled": True,
                "ok": True,
                "message": f"resource_relief: {json.dumps(out, default=str)}",
            }
        return {"handled": True, "ok": False, "message": "usage: resource status|relieve"}

    if cmd == "history":
        try:
            from utils.trade_memory import recent_analysis as _recent
            _r = _recent(15)
            if _r.get("n_trades", 0) == 0:
                return {"handled": True, "ok": True, "message": "No trade history yet."}
            _lines = [
                f"Summary (last {_r['n_trades']} trades):",
                f"Win/Loss: {_r['wins']}W / {_r['losses']}L",
                f"Win rate: {_r['win_rate']*100:.1f}%",
                f"Total P&L: ${_r['total_pnl']:.2f}",
                f"Avg/trade: ${_r['avg_pnl_per_trade']:.2f}",
                f"Avg confidence: {_r['avg_confidence']:.3f}",
                f"Best trade: ${_r['best_trade'] or 0:.2f}",
                f"Worst trade: ${_r['worst_trade'] or 0:.2f}",
            ]
            return {"handled": True, "ok": True, "message": chr(10).join(_lines)}
        except Exception as e:
            return {"handled": True, "ok": False, "message": f"history unavailable: {e}"}

    if cmd == "analysis":
        sub = rest[0].lower() if rest else "week"
        try:
            from utils.trade_memory import weekly_analysis as _weekly
            _a = _weekly()
            if _a.get("total", 0) == 0:
                return {"handled": True, "ok": True, "message": "Not enough data for weekly analysis yet."}
            _lines = [
                f"Weekly Analysis ({_a['period']}):",
                f"Total: {_a['total']} trades ({_a['wins']}W / {_a['losses']}L)",
                f"Win rate: {_a['win_rate']*100:.1f}%",
                f"Total P&L: ${_a['total_pnl']:.2f}",
                f"Avg hold: {_a['avg_hold_hours']:.1f}h",
            ]
            if _a.get('best_setups'):
                _lines.append("")
                _lines.append("Best trades:")
                for _s in _a['best_setups']:
                    _lines.append(f"  {_s['decision']} +${_s['pnl']:.2f} (conf={_s['confidence']:.3f}, {_s['reason']})")
            if _a.get('worst_setups'):
                _lines.append("")
                _lines.append("Worst trades:")
                for _s in _a['worst_setups']:
                    _lines.append(f"  {_s['decision']} ${_s['pnl']:.2f} (conf={_s['confidence']:.3f}, {_s['reason']})")
            return {"handled": True, "ok": True, "message": chr(10).join(_lines)}
        except Exception as e:
            return {"handled": True, "ok": False, "message": f"analysis unavailable: {e}"}

    return {"handled": True, "ok": False, "message": "unknown command"}


def run_listener(trading_config_path: Path) -> int:
    trading_cfg = _load_yaml(trading_config_path)
    runtime_root = _set_runtime_scope_env(trading_cfg)
    os.environ["TRADING_CONFIG_PATH"] = str(trading_config_path)
    profile_entries = _listener_profile_entries(trading_config_path, trading_cfg)
    lcfg = _listener_cfg(trading_cfg)
    if not bool(lcfg.get("enabled", False)):
        print("telegram listener disabled in config")
        return 1

    token, default_chat_id, allowed_chat_ids = _token_and_chat_ids(trading_cfg)
    if not token:
        print("telegram bot token missing")
        return 2

    poll_seconds = max(int(lcfg.get("poll_seconds", 3) or 3), 1)
    progress_interval_sec = max(int(lcfg.get("latest_request_status_interval_seconds", 120) or 120), 30)
    active_jobs_interval_sec = max(int(lcfg.get("active_jobs_status_interval_seconds", 600) or 600), 60)
    auto_resume_enabled = bool(lcfg.get("auto_resume_waiting_jobs_enabled", True))
    auto_resume_interval_sec = max(int(lcfg.get("auto_resume_waiting_jobs_interval_seconds", 60) or 60), 15)
    autonomy_cfg = _autonomy_cfg(trading_cfg)
    autonomy_enabled = bool(autonomy_cfg.get("enabled", False))
    autonomy_interval_sec = max(int(autonomy_cfg.get("scan_interval_seconds", 600) or 600), 60)
    offset = 0
    latest_request: Dict[str, Any] = {}
    last_chat_id = default_chat_id
    auto_resume_cooldowns_by_profile: Dict[str, Dict[str, float]] = {}
    agent_b_reconcile_cooldowns_by_profile: Dict[str, Dict[str, float]] = {}
    next_active_jobs_at = time.time() + float(active_jobs_interval_sec)
    next_auto_resume_at = time.time()
    next_autonomy_at = time.time()
    next_scheduled_refresh_at = time.time()
    next_inference_at = time.time()
    previous_weekend_quiet_mode = _weekend_utc_quiet_mode_active(trading_cfg, now_utc=datetime.utcnow().replace(tzinfo=timezone.utc))

    print(f"telegram listener started: cfg={trading_config_path} prefix={lcfg.get('command_prefix', '/tsmm')} runtime={runtime_root}")

    while True:
        try:
            trading_cfg = _load_yaml(trading_config_path)
            runtime_root = _set_runtime_scope_env(trading_cfg)
            os.environ["TRADING_CONFIG_PATH"] = str(trading_config_path)
            profile_entries = _listener_profile_entries(trading_config_path, trading_cfg)
            lcfg = _listener_cfg(trading_cfg)
            auto_resume_enabled = bool(lcfg.get("auto_resume_waiting_jobs_enabled", True))
            auto_resume_interval_sec = max(int(lcfg.get("auto_resume_waiting_jobs_interval_seconds", 60) or 60), 15)
            progress_interval_sec = max(int(lcfg.get("latest_request_status_interval_seconds", 120) or 120), 30)
            active_jobs_interval_sec = max(int(lcfg.get("active_jobs_status_interval_seconds", 600) or 600), 60)
            autonomy_cfg = _autonomy_cfg(trading_cfg)
            autonomy_enabled = bool(autonomy_cfg.get("enabled", False))
            autonomy_interval_sec = max(int(autonomy_cfg.get("scan_interval_seconds", 600) or 600), 60)

            refresh_cfg = _scheduled_refresh_cfg(trading_cfg)
            refresh_check_interval_sec = max(int(refresh_cfg.get("check_interval_seconds", 60) or 60), 30)
            current_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
            weekend_quiet_mode = _weekend_utc_quiet_mode_active(trading_cfg, now_utc=current_utc)

            if previous_weekend_quiet_mode and not weekend_quiet_mode:
                reopen_out = _handle_weekend_quiet_mode_exit(trading_cfg, default_chat_id, last_chat_id)
                _console_trace(f"weekend quiet mode lifted: {reopen_out}")

            if bool(refresh_cfg.get("enabled", False)) and time.time() >= next_scheduled_refresh_at:
                next_scheduled_refresh_at = time.time() + float(refresh_check_interval_sec)
                refresh_state = _read_json(_scheduled_refresh_state_path())
                refresh_chat_ids = [str(c).strip() for c in (refresh_state.get("chat_ids") or []) if str(c).strip()]
                if not refresh_chat_ids:
                    refresh_chat_ids = _subscriber_chat_ids(trading_cfg, default_chat_id or last_chat_id)

                refresh_pid = int(refresh_state.get("pid") or 0)
                refresh_alive = refresh_pid > 0 and _is_pid_alive(refresh_pid)
                if refresh_alive and refresh_chat_ids and time.time() >= float(refresh_state.get("next_status_at") or 0):
                    _send_to_chat_ids(trading_cfg, refresh_chat_ids, _scheduled_refresh_runtime_message(refresh_state))
                    refresh_state["next_status_at"] = time.time() + float(progress_interval_sec)
                    _write_json(_scheduled_refresh_state_path(), refresh_state)

                if refresh_pid > 0 and (not refresh_alive) and not bool(refresh_state.get("done_notified", False)):
                    restart_out: Dict[str, Any] = {}
                    if bool(refresh_state.get("restart_endpoint_after_done", False)) and not weekend_quiet_mode:
                        restart_out = _restart_endpoint_service(trading_cfg)
                    done_req = {
                        "type": "deploy",
                        "pid": refresh_pid,
                        "endpoint_restart_after_done": restart_out,
                    }
                    if refresh_chat_ids:
                        _send_to_chat_ids(trading_cfg, refresh_chat_ids, "scheduled model refresh completed: " + _latest_request_done_message(done_req))
                    refresh_state["done_notified"] = True
                    refresh_state["pid"] = 0
                    refresh_state["last_completed_target_date"] = str(refresh_state.get("target_date") or "")
                    refresh_state["last_completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    refresh_state["endpoint_restart"] = restart_out
                    _write_json(_scheduled_refresh_state_path(), refresh_state)

                target = _scheduled_refresh_target(trading_cfg, now_utc=current_utc)
                target_date = str(target.get("target_date") or "")
                scheduled_due = bool(target.get("due", False))
                already_completed = target_date and str(refresh_state.get("last_completed_target_date") or "") == target_date
                already_started = target_date and str(refresh_state.get("target_date") or "") == target_date and refresh_alive
                deploy_running = _command_process_running("deploy_agent_pipeline.py")
                if scheduled_due and target_date and not already_completed and not already_started and not deploy_running:
                    launch_out = _launch_deploy_pipeline(trading_cfg, extra_args=["--refresh", "--no-start-job"])
                    if launch_out.get("ok") and launch_out.get("pid"):
                        refresh_state = {
                            "target_date": target_date,
                            "scheduled_local": target.get("scheduled_local"),
                            "scheduled_utc": target.get("scheduled_utc"),
                            "timezone": target.get("timezone"),
                            "pid": int(launch_out.get("pid") or 0),
                            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "done_notified": False,
                            "next_status_at": time.time() + float(progress_interval_sec),
                            "chat_ids": refresh_chat_ids,
                            "restart_endpoint_after_done": bool(refresh_cfg.get("restart_endpoint_after_done", True)),
                        }
                        _write_json(_scheduled_refresh_state_path(), refresh_state)
                        if refresh_chat_ids:
                            _send_to_chat_ids(trading_cfg, refresh_chat_ids, _scheduled_refresh_start_message(target))

            if weekend_quiet_mode:
                quiet_out = _enforce_weekend_utc_quiet_mode(trading_cfg, now_utc=current_utc)
                if quiet_out.get("stopped_trading_pids") or quiet_out.get("stopped_endpoint_pids"):
                    _console_trace(f"weekend quiet mode enforced: {quiet_out}")

            if (not weekend_quiet_mode) and auto_resume_enabled and time.time() >= next_auto_resume_at:
                for profile in profile_entries:
                    profile_cfg = dict(profile.get("trading_cfg") or {})
                    profile_path = _resolve_trading_config_path(profile.get("config_path") or trading_config_path)
                    profile_key = str(profile_path.resolve()).lower()
                    profile_label = str((profile_cfg.get("runtime") or {}).get("profile_label") or profile_path.stem)

                    _set_runtime_scope_env(profile_cfg)
                    os.environ["TRADING_CONFIG_PATH"] = str(profile_path)

                    profile_auto_resume_cooldowns = auto_resume_cooldowns_by_profile.setdefault(profile_key, {})
                    launched = _auto_resume_waiting_jobs(
                        trading_cfg=profile_cfg,
                        trading_config_path=profile_path,
                        job_cooldowns=profile_auto_resume_cooldowns,
                        default_chat_id=default_chat_id,
                        last_chat_id=last_chat_id,
                    )
                    if launched:
                        last_launch = launched[-1]
                        latest_request = {
                            "chat_id": default_chat_id or last_chat_id,
                            "subscriber_chat_ids": _subscriber_chat_ids(profile_cfg, default_chat_id),
                            "type": f"trading resume ({profile_label})",
                            "pid": int(last_launch.get("pid") or 0),
                            "job_id": str(last_launch.get("job_id") or ""),
                            "next_status_at": time.time() + float(progress_interval_sec),
                            "done_notified": False,
                        }

                    profile_reconcile_cooldowns = agent_b_reconcile_cooldowns_by_profile.setdefault(profile_key, {})
                    reconciled = _reconcile_orphaned_agent_b_jobs(
                        trading_cfg=profile_cfg,
                        trading_config_path=profile_path,
                        job_cooldowns=profile_reconcile_cooldowns,
                        default_chat_id=default_chat_id,
                        last_chat_id=last_chat_id,
                    )
                    if reconciled:
                        _console_trace(f"agent_b reconciliation events ({profile_label})={reconciled}")

                runtime_root = _set_runtime_scope_env(trading_cfg)
                os.environ["TRADING_CONFIG_PATH"] = str(trading_config_path)
                next_auto_resume_at = time.time() + float(auto_resume_interval_sec)

            if (not weekend_quiet_mode) and autonomy_enabled and time.time() >= next_autonomy_at:
                next_autonomy_at = time.time() + float(autonomy_interval_sec)
                maintenance_events = _maintain_programmed_orders(trading_cfg)
                if maintenance_events:
                    target_chat_ids = _subscriber_chat_ids(trading_cfg, default_chat_id or last_chat_id)
                    for event in maintenance_events:
                        if bool(event.get("ok", False)):
                            msg = (
                                "programmed order maintenance updated a pending order: "
                                f"job_id={event.get('job_id')}; action={event.get('action')}; reason={event.get('reason')}; "
                                f"entry={event.get('entry')}; market_price={event.get('market_price')}; "
                                f"consensus={event.get('consensus')}; score={event.get('consensus_score')}"
                            )
                        else:
                            msg = (
                                "programmed order maintenance could not update a pending order: "
                                f"job_id={event.get('job_id')}; action={event.get('action')}; reason={event.get('reason')}; "
                                f"details={event.get('cancel_result')}"
                            )
                        if target_chat_ids:
                            _send_to_chat_ids(trading_cfg, target_chat_ids, msg)
                        _console_trace(msg)
                current_session = _current_autonomous_session(trading_cfg)
                request_in_flight = _latest_request_blocks_autonomy(latest_request)
                pending_approval = _latest_pending_approval()
                if current_session and not request_in_flight:
                    session_states = _session_job_states(current_session)
                    session_stats = _session_autonomous_stats(session_states)
                    session_capacity = _autonomous_capacity_limit(trading_cfg)
                    session_active_jobs = _session_active_job_count(session_states)
                    mandatory_cooldown_seconds = max(int(autonomy_cfg.get("mandatory_session_cooldown_seconds", autonomy_interval_sec) or autonomy_interval_sec), autonomy_interval_sec)
                    followup_cooldown_seconds = max(int(autonomy_cfg.get("followup_cooldown_seconds", autonomy_interval_sec) or autonomy_interval_sec), autonomy_interval_sec)
                    max_followup_launches = max(int(autonomy_cfg.get("max_followup_launches_per_session", session_capacity) or session_capacity), 1)
                    max_filtered_followups = max(int(autonomy_cfg.get("max_filtered_followups_per_session", max_followup_launches) or max_followup_launches), 1)
                    mandatory_ready = _seconds_since(session_stats.get("last_mandatory_started_at"), current_utc) >= float(mandatory_cooldown_seconds)
                    followup_ready = _seconds_since(session_stats.get("last_followup_started_at"), current_utc) >= float(followup_cooldown_seconds)
                    target_chat_ids = _subscriber_chat_ids(trading_cfg, default_chat_id or last_chat_id)
                    launch_out: Dict[str, Any] = {}
                    manual_close_block = _session_manual_or_external_close_marker(session_states)
                    autonomy_blocked_by_manual_close = bool(autonomy_cfg.get("block_after_manual_or_external_close", True)) and bool(manual_close_block.get("blocked", False))

                    if autonomy_blocked_by_manual_close:
                        _console_trace(
                            "autonomous launch blocked after manual/external close: "
                            f"job_id={manual_close_block.get('job_id')}; "
                            f"closed_reason={manual_close_block.get('closed_reason')}; "
                            f"close_outcome_reason={manual_close_block.get('close_outcome_reason')}"
                        )

                    if (
                        not autonomy_blocked_by_manual_close
                        and not pending_approval
                        and not _session_has_mandatory_coverage(session_states)
                        and mandatory_ready
                    ):
                        launch_out = _launch_autonomous_trading_start(
                            trading_cfg=trading_cfg,
                            trading_config_path=trading_config_path,
                            submission_mode="programmed",
                            autonomous_trigger="mandatory_session",
                        )
                        if launch_out.get("ok"):
                            msg = (
                                "autonomous session entry scan started: "
                                f"session={current_session.get('name')}; job_id={launch_out.get('job_id')}; "
                                f"submission_mode=programmed; session_start_local={current_session.get('start_local')} {current_session.get('timezone')}"
                            )
                            if target_chat_ids:
                                _send_to_chat_ids(trading_cfg, target_chat_ids, msg)
                            _console_trace(msg)
                    elif (
                        not autonomy_blocked_by_manual_close
                        and bool(autonomy_cfg.get("followup_enabled", True))
                        and not pending_approval
                        and session_active_jobs < session_capacity
                        and _mode_b_supervision_active(trading_cfg)
                        and followup_ready
                        and int(session_stats.get("followup_launches") or 0) < max_followup_launches
                        and int(session_stats.get("filtered_followups") or 0) < max_filtered_followups
                    ):
                        followup_mode = str(autonomy_cfg.get("followup_submission_mode") or "market").strip().lower()
                        if followup_mode not in {"programmed", "market"}:
                            followup_mode = "market"
                        launch_out = _launch_autonomous_trading_start(
                            trading_cfg=trading_cfg,
                            trading_config_path=trading_config_path,
                            submission_mode=followup_mode,
                            autonomous_trigger="autonomous_followup",
                        )
                        if launch_out.get("ok"):
                            msg = (
                                "autonomous follow-up scan started: "
                                f"session={current_session.get('name')}; job_id={launch_out.get('job_id')}; "
                                f"submission_mode={followup_mode}; active_session_jobs={session_active_jobs}/{session_capacity}"
                            )
                            if target_chat_ids:
                                _send_to_chat_ids(trading_cfg, target_chat_ids, msg)
                            _console_trace(msg)

                    if launch_out.get("ok") and launch_out.get("pid"):
                        latest_request = {
                            "chat_id": default_chat_id or last_chat_id,
                            "subscriber_chat_ids": _subscriber_chat_ids(trading_cfg, default_chat_id),
                            "type": f"autonomous {launch_out.get('autonomous_trigger')}",
                            "pid": int(launch_out.get("pid") or 0),
                            "job_id": str(launch_out.get("job_id") or ""),
                            "next_status_at": time.time() + float(progress_interval_sec),
                            "done_notified": False,
                        }

            res = _api_get(token, "getUpdates", {"timeout": 25, "offset": offset})
            updates = res.get("result") or []
            for upd in updates:
                try:
                    uid = int(upd.get("update_id", 0))
                    offset = max(offset, uid + 1)
                    msg = upd.get("message") or upd.get("channel_post") or {}
                    chat = msg.get("chat") or {}
                    chat_id = str(chat.get("id", "")).strip()
                    text = str(msg.get("text") or "").strip()
                    if not chat_id or not text:
                        continue
                    if allowed_chat_ids and chat_id not in allowed_chat_ids:
                        continue
                    last_chat_id = chat_id or last_chat_id
                    _register_subscriber(chat_id)

                    if bool(_listener_cfg(trading_cfg).get("log_conversations", True)):
                        _write_conversation(
                            {
                                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "direction": "inbound",
                                "chat_id": chat_id,
                                "text": text,
                            }
                        )
                    _record_llm_session_message(chat_id, "inbound", text)

                    # Reload cfg each command so changes are applied without restart.
                    trading_cfg = _load_yaml(trading_config_path)
                    lcfg = _listener_cfg(trading_cfg)
                    auto_resume_enabled = bool(lcfg.get("auto_resume_waiting_jobs_enabled", True))
                    auto_resume_interval_sec = max(int(lcfg.get("auto_resume_waiting_jobs_interval_seconds", 60) or 60), 15)
                    raw_text = text
                    _console_trace(f"inbound chat_id={chat_id} text={raw_text}")
                    selected_profile = _select_listener_profile(raw_text, profile_entries)
                    selected_trading_cfg = dict(selected_profile.get("trading_cfg") or trading_cfg)
                    selected_trading_config_path = Path(selected_profile.get("config_path") or trading_config_path)
                    _set_runtime_scope_env(selected_trading_cfg)
                    os.environ["TRADING_CONFIG_PATH"] = str(selected_trading_config_path)
                    if bool(lcfg.get("require_secret", False)):
                        secret_env = str(lcfg.get("secret_env", "TSMM_TELEGRAM_COMMAND_SECRET")).strip() or "TSMM_TELEGRAM_COMMAND_SECRET"
                        secret = os.environ.get(secret_env, "").strip()
                        marker = f"#{secret}" if secret else ""
                        if not secret or marker not in raw_text:
                            out = {"handled": True, "ok": False, "message": "authentication_failed"}
                        else:
                            text = raw_text.replace(marker, "").strip()
                            out = _handle_command(
                                text,
                                selected_trading_cfg,
                                source_chat_id=str(chat_id),
                                trading_config_path=selected_trading_config_path,
                            )
                    else:
                        out = _handle_command(
                            text,
                            selected_trading_cfg,
                            source_chat_id=str(chat_id),
                            trading_config_path=selected_trading_config_path,
                        )
                    if not out.get("handled", False):
                        _console_trace("command ignored: not handled")
                        continue

                    if out.get("exec_args"):
                        _console_trace(f"exec args={out.get('exec_args')}")
                    if out.get("returncode") is not None:
                        _console_trace(f"exec returncode={out.get('returncode')}")
                    _console_trace(f"command result ok={bool(out.get('ok', False))} message={out.get('message')}")

                    tr = out.get("track_request") if isinstance(out, dict) else None
                    if isinstance(tr, dict) and tr.get("pid"):
                        latest_request = {
                            "chat_id": chat_id,
                            "subscriber_chat_ids": _subscriber_chat_ids(selected_trading_cfg, default_chat_id),
                            "type": str(tr.get("type") or "request"),
                            "pid": int(tr.get("pid") or 0),
                            "next_status_at": time.time() + float(progress_interval_sec),
                            "done_notified": False,
                        }
                        if tr.get("restart_endpoint_after_done") is not None:
                            latest_request["restart_endpoint_after_done"] = bool(tr.get("restart_endpoint_after_done"))

                    _write_audit(
                        {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "chat_id": chat_id,
                            "text": raw_text,
                            "parsed_text": text,
                            "ok": bool(out.get("ok", False)),
                            "message": out.get("message"),
                            "exec_args": out.get("exec_args"),
                            "returncode": out.get("returncode"),
                            "stdout": out.get("stdout"),
                            "stderr": out.get("stderr"),
                        }
                    )

                    body = str(out.get("message") or "")
                    if out.get("stdout"):
                        body += "\nstdout:\n" + str(out.get("stdout"))
                    if out.get("stderr"):
                        body += "\nstderr:\n" + str(out.get("stderr"))

                    tcfg = _tg_cfg(selected_trading_cfg)
                    tcfg = dict(tcfg)
                    tcfg["chat_id"] = chat_id or default_chat_id
                    send_res = send_telegram_notification(tcfg, body[:3500])
                    _console_trace(
                        "telegram send "
                        f"ok={bool(send_res.get('ok', False))} "
                        f"status_code={send_res.get('status_code')} "
                        f"message_id={send_res.get('message_id')}"
                    )

                    _write_audit(
                        {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "chat_id": chat_id,
                            "event": "telegram_send_result",
                            "ok": bool(send_res.get("ok", False)),
                            "status_code": send_res.get("status_code"),
                            "error": send_res.get("error"),
                            "message_id": send_res.get("message_id"),
                        }
                    )

                    if bool(_listener_cfg(trading_cfg).get("log_conversations", True)):
                        _write_conversation(
                            {
                                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "direction": "outbound",
                                "chat_id": chat_id,
                                "text": body[:3500],
                            }
                        )
                    _record_llm_session_message(
                        chat_id,
                        "outbound",
                        body[:3500],
                        llm_provider=out.get("llm_provider"),
                        agent_chat_mode=bool(out.get("agent_chat_mode", False)),
                    )
                    _set_runtime_scope_env(trading_cfg)
                    os.environ["TRADING_CONFIG_PATH"] = str(trading_config_path)
                except Exception:
                    continue

            # Resource guard runs continuously while listener is live.
            try:
                check_and_relieve(ROOT, trading_cfg)
            except Exception:
                pass

            if latest_request and latest_request.get("chat_id"):
                req_pid = int(latest_request.get("pid") or 0)
                req_chat_id = str(latest_request.get("chat_id") or "")
                req_chat_ids = [str(c).strip() for c in (latest_request.get("subscriber_chat_ids") or []) if str(c).strip()] or [req_chat_id]
                req_done_notified = bool(latest_request.get("done_notified", False))
                req_alive = _is_pid_alive(req_pid)

                if req_alive and time.time() >= float(latest_request.get("next_status_at") or 0):
                    status_msg = _latest_request_status_message(latest_request)
                    _send_to_chat_ids(trading_cfg, req_chat_ids, status_msg)
                    latest_request["next_status_at"] = time.time() + float(progress_interval_sec)

                if (not req_alive) and (not req_done_notified):
                    if str(latest_request.get("type") or "").strip().lower() == "deploy" and bool(latest_request.get("restart_endpoint_after_done", False)):
                        latest_request["endpoint_restart_after_done"] = _restart_endpoint_service(trading_cfg)
                    done_msg = _latest_request_done_message(latest_request)
                    _send_to_chat_ids(trading_cfg, req_chat_ids, done_msg)
                    latest_request["done_notified"] = True

            if time.time() >= next_active_jobs_at:
                target_chat_ids = _subscriber_chat_ids(trading_cfg, default_chat_id or last_chat_id)
                if target_chat_ids:
                    digest_msg = _format_active_jobs_digest(trading_cfg)
                    _send_to_chat_ids(trading_cfg, target_chat_ids, digest_msg, allow_agent_mode=True)
                next_active_jobs_at = time.time() + float(active_jobs_interval_sec)

            # Periodic inference refresh (~5 min) when no active position
            if time.time() >= next_inference_at and not weekend_quiet_mode:
                next_inference_at = time.time() + 300
                try:
                    _run_cmd_async(
                        [sys.executable or 'python', 'scripts/full_horizon_report.py'],
                        env=dict(os.environ),
                    )
                    _console_trace("periodic inference refresh launched")
                except Exception:
                    pass

            previous_weekend_quiet_mode = weekend_quiet_mode
            # Robust sleep - ensures minimum wait time even if interrupted
            _sleep_deadline = time.time() + poll_seconds
            while time.time() < _sleep_deadline:
                _remaining = _sleep_deadline - time.time()
                if _remaining <= 0:
                    break
                time.sleep(min(_remaining, 0.5))
        except KeyboardInterrupt:
            print("telegram listener interrupted")
            return 0
        except Exception:
            _console_trace(f"listener loop error: sleeping {poll_seconds}s before retry")
            _sleep_deadline = time.time() + poll_seconds
            while time.time() < _sleep_deadline:
                _remaining = _sleep_deadline - time.time()
                if _remaining <= 0:
                    break
                time.sleep(min(_remaining, 0.5))


def main() -> int:
    parser = argparse.ArgumentParser(description="TSMM Telegram command listener")
    parser.add_argument("--trading-config", default="config/trading_agent.yaml", help="Path to trading config")
    args = parser.parse_args()

    return run_listener(ROOT / args.trading_config)


if __name__ == "__main__":
    raise SystemExit(main())
