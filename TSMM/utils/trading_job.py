"""
Trading job orchestration for Agent A -> Agent B lifecycle.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import psutil
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from .investing_agent import (
    MT5Adapter,
    _collect_all_model_assessment_signals,
    _collect_mode_b_signals,
    load_trading_config,
    run_investing_agent,
)
from .llm_connector import load_llm_providers_config, call_llm
from .market_sentiment import aggregate_market_sentiment
from .agent_memory import AgentMemoryStore
from .agent_channel import publish_channel_message
from .live_data import bootstrap_master_on_backend_start, sync_dataset_source_from_master, resolve_tiingo_token_candidates
from .notification_email import send_email_notification
from .notification_telegram import send_telegram_broadcast
from .operation_feedback_store import (
    log_agent_a_plan_feedback,
    log_agent_b_sample_feedback,
    log_notification_feedback,
    log_state_transition_feedback,
)
from .runtime_scope import resolve_job_id_prefix, resolve_runtime_dir, resolve_runtime_file
from .trading_signal_policy import evaluate_joint_ohlc_policy


def _now_utc() -> datetime:
    return datetime.utcnow()


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _current_trading_config_path(default: str = "config/trading_agent.yaml") -> str:
    return str(os.environ.get("TRADING_CONFIG_PATH", default) or default).strip() or default


def _truthy_env(var_name: str) -> bool:
    return str(os.environ.get(var_name, "") or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _account_profile_label(trading_cfg: Optional[Dict[str, Any]]) -> str:
    cfg = trading_cfg or {}
    runtime_cfg = (cfg.get("runtime") or {}) if isinstance(cfg, dict) else {}
    label = str(runtime_cfg.get("profile_label") or runtime_cfg.get("job_id_prefix") or "").strip()
    if label:
        return label

    listener_cfg = (cfg.get("telegram_listener") or {}) if isinstance(cfg, dict) else {}
    command_prefix = str(listener_cfg.get("command_prefix") or "").strip().lstrip("/")
    if command_prefix and command_prefix.lower() != "tsmm":
        return command_prefix.upper()

    trading_cfg_path = _current_trading_config_path().lower()
    if "ftmo" in trading_cfg_path:
        return "FTMO"
    return "DEFAULT"


def _account_mirror_cfg(trading_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = trading_cfg or {}
    return dict((cfg.get("account_mirror") or {}) if isinstance(cfg, dict) else {})


def _incoming_account_mirror_context() -> Dict[str, Any]:
    source_job_id = str(os.environ.get("TSMM_ACCOUNT_MIRROR_SOURCE_JOB_ID", "") or "").strip()
    source_cfg_path = str(os.environ.get("TSMM_ACCOUNT_MIRROR_SOURCE_CONFIG_PATH", "") or "").strip()
    if not source_job_id or not source_cfg_path:
        return {}

    source_profile = str(os.environ.get("TSMM_ACCOUNT_MIRROR_SOURCE_PROFILE", "") or "").strip() or "SOURCE"
    return {
        "enabled": True,
        "role": "mirror",
        "source_profile": source_profile,
        "source_job_id": source_job_id,
        "peer_profile": source_profile,
        "peer_job_id": source_job_id,
        "peer_trading_config_path": source_cfg_path,
        "launched_by_account_mirror": True,
    }


def _launch_account_mirror_start(
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    logger,
    selected_model: Optional[str],
    source_job_id: str,
    request_context: Optional[Dict[str, Any]] = None,
    source_plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mirror_cfg = _account_mirror_cfg(trading_cfg)
    if not bool(mirror_cfg.get("enabled", False)):
        return {"ok": False, "skipped": True, "reason": "mirror_disabled"}
    if str(os.environ.get("TSMM_ACCOUNT_MIRROR_SUPPRESS", "") or "").strip().lower() in {"1", "true", "yes", "y"}:
        return {"ok": False, "skipped": True, "reason": "mirror_suppressed"}
    if not bool(mirror_cfg.get("mirror_job_starts", True)):
        return {"ok": False, "skipped": True, "reason": "mirror_start_disabled"}

    peer_cfg_path = str(mirror_cfg.get("peer_trading_config_path") or "").strip()
    if not peer_cfg_path:
        return {"ok": False, "skipped": True, "reason": "missing_peer_trading_config_path"}

    current_cfg_path = _current_trading_config_path()
    if os.path.normcase(peer_cfg_path) == os.path.normcase(current_cfg_path):
        return {"ok": False, "skipped": True, "reason": "peer_trading_config_matches_current"}

    try:
        peer_trading_cfg = load_trading_config(peer_cfg_path)
    except Exception as exc:
        if logger is not None:
            logger.exception("Failed to load peer trading config for account mirror")
        return {
            "ok": False,
            "error": str(exc),
            "peer_trading_config_path": peer_cfg_path,
        }

    peer_job_id = _new_job_id(peer_trading_cfg)
    peer_profile = str(mirror_cfg.get("peer_profile") or _account_profile_label(peer_trading_cfg)).strip() or "PEER"
    submission_mode = str(((request_context or {}).get("effective_submission_mode") or "")).strip().lower()
    autonomous_trigger = str(((request_context or {}).get("autonomous_trigger") or "")).strip().lower()

    env = os.environ.copy()
    env["CONFIG_PATH"] = str(os.environ.get("CONFIG_PATH", "config/config.yaml") or "config/config.yaml")
    env["TRADING_CONFIG_PATH"] = peer_cfg_path
    env.pop("TSMM_RUNTIME_DIR", None)
    env["TSMM_ACCOUNT_MIRROR_SUPPRESS"] = "1"
    env["TSMM_ACCOUNT_MIRROR_SOURCE_JOB_ID"] = source_job_id
    env["TSMM_ACCOUNT_MIRROR_SOURCE_CONFIG_PATH"] = current_cfg_path
    env["TSMM_ACCOUNT_MIRROR_SOURCE_PROFILE"] = _account_profile_label(trading_cfg)

    # When a source plan is provided, force the mirror to copy the exact trade
    # instead of running independent Agent A analysis.
    if isinstance(source_plan, dict) and source_plan:
        # Only copy the fields that define the trade decision — not internal metadata.
        copy_keys = (
            "decision", "model", "entry", "stop_loss", "take_profit", "volume",
            "confidence", "cm_accuracy", "signal_score", "success_probability",
            "input_fooling_risk", "order_submission_mode",
            "analysis_grounding_timeframe", "analysis_grounding_timeframe_minutes",
        )
        forced_plan = {k: v for k, v in source_plan.items() if k in copy_keys and v is not None}
        forced_plan["source"] = "account_mirror_exact_copy"
        env["TSMM_FORCE_AGENT_A_PLAN_JSON"] = json.dumps(forced_plan, ensure_ascii=True, separators=(",", ":"), default=str)

    command = [sys.executable, "app.py", "trading-job", "start", "--job-id", peer_job_id]
    if submission_mode:
        command.extend(["--submission-mode", submission_mode])
    if selected_model:
        command.extend(["--plan-model", str(selected_model)])
    if autonomous_trigger:
        command.extend(["--autonomous-trigger", autonomous_trigger])

    cmds = list(command)
    if os.name == "nt" and len(cmds) > 0 and "python" in str(cmds[0]).lower():
        pass  # CREATE_NO_WINDOW handled by sitecustomize
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    try:
        proc = subprocess.Popen(
            cmds,
            cwd=str(_project_root()),
            env=env,
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        if logger is not None:
            logger.exception("Failed to launch mirrored peer trading job")
        return {
            "ok": False,
            "error": str(exc),
            "peer_profile": peer_profile,
            "peer_job_id": peer_job_id,
            "peer_trading_config_path": peer_cfg_path,
        }

    return {
        "ok": True,
        "enabled": True,
        "role": "source",
        "peer_profile": peer_profile,
        "peer_job_id": peer_job_id,
        "peer_trading_config_path": peer_cfg_path,
        "source_profile": _account_profile_label(trading_cfg),
        "source_job_id": source_job_id,
        "pid": int(proc.pid),
    }


def _mirror_agent_a_entry_on_peer_preflight_failure(
    app_config: Dict[str, Any],
    source_trading_cfg: Dict[str, Any],
    output_dir: str,
    source_state: Dict[str, Any],
) -> Dict[str, Any]:
    mirror = dict((source_state.get("mirror") or {}) if isinstance(source_state, dict) else {})
    peer_job_id = str(mirror.get("peer_job_id") or "").strip()
    peer_cfg_path = str(mirror.get("peer_trading_config_path") or "").strip()
    if not peer_job_id or not peer_cfg_path:
        return {"ok": False, "skipped": True, "reason": "missing_peer_mirror_metadata"}

    try:
        peer_trading_cfg = load_trading_config(peer_cfg_path)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "peer_job_id": peer_job_id,
            "peer_trading_config_path": peer_cfg_path,
        }

    peer_profile = str(mirror.get("peer_profile") or _account_profile_label(peer_trading_cfg)).strip() or "PEER"
    peer_state_path = _state_path(output_dir, peer_trading_cfg, peer_job_id)
    peer_state = _load_state(peer_state_path)
    if not peer_state:
        return {
            "ok": False,
            "skipped": True,
            "reason": "peer_state_not_found",
            "peer_job_id": peer_job_id,
            "peer_profile": peer_profile,
            "peer_trading_config_path": peer_cfg_path,
        }

    peer_status = str(peer_state.get("status") or "").strip().lower()
    peer_stage = str(peer_state.get("stage") or "").strip().lower()
    peer_closed_reason = str(peer_state.get("closed_reason") or "").strip().lower()
    if peer_status != "failed" or peer_stage != "preflight" or "data_sync_failed" not in peer_closed_reason:
        return {
            "ok": False,
            "skipped": True,
            "reason": "peer_not_data_sync_preflight_failure",
            "peer_job_id": peer_job_id,
            "peer_profile": peer_profile,
            "peer_trading_config_path": peer_cfg_path,
            "peer_status": peer_status,
            "peer_stage": peer_stage,
            "peer_closed_reason": str(peer_state.get("closed_reason") or ""),
        }

    plan = dict((source_state.get("plan") or {}) if isinstance(source_state.get("plan"), dict) else {})
    decision = str(plan.get("decision") or "").strip().lower()
    if decision not in {"buy", "sell"}:
        return {
            "ok": False,
            "skipped": True,
            "reason": "source_plan_not_tradeable",
            "peer_job_id": peer_job_id,
            "peer_profile": peer_profile,
        }

    submission_mode = str(source_state.get("order_submission_mode") or _agent_a_order_submission_mode(plan, source_trading_cfg)).strip().lower()
    if submission_mode != "market":
        return {
            "ok": False,
            "skipped": True,
            "reason": "mirror_fallback_supports_market_only",
            "peer_job_id": peer_job_id,
            "peer_profile": peer_profile,
            "submission_mode": submission_mode,
        }

    exec_cfg = (peer_trading_cfg.get("execution") or {})
    symbol = str(exec_cfg.get("symbol") or app_config.get("symbol") or "XAUUSD")
    volume = _safe_float(plan.get("volume", exec_cfg.get("default_volume", 0.01)), 0.01)
    entry_ref = _safe_float(plan.get("entry"), 0.0)
    if entry_ref <= 0.0:
        source_position = (source_state.get("position") or {}) if isinstance(source_state.get("position"), dict) else {}
        entry_ref = _safe_float(source_position.get("price_open"), 0.0)

    mt5_cfg = (((peer_trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, msg_conn = adapter.connect()
    if not ok_conn:
        return {
            "ok": False,
            "error": f"peer_mt5_connect_failed:{msg_conn}",
            "peer_job_id": peer_job_id,
            "peer_profile": peer_profile,
            "peer_trading_config_path": peer_cfg_path,
        }

    try:
        if entry_ref > 0.0 and volume > 0.0:
            tj_peer = (peer_trading_cfg.get("trading_job") or {})
            dedup = _find_similar_mt5_exposure(
                adapter=adapter,
                symbol=symbol,
                side=decision,
                entry=entry_ref,
                volume=volume,
                entry_tolerance=_safe_float(tj_peer.get("duplicate_entry_tolerance", 0.15), 0.15),
                volume_tolerance=_safe_float(tj_peer.get("duplicate_volume_tolerance", 1e-9), 1e-9),
                tsmm_only=bool(tj_peer.get("duplicate_tsmm_only", True)),
            )
            if (dedup.get("pending_orders") or dedup.get("open_positions")):
                return {
                    "ok": True,
                    "skipped": True,
                    "reason": "peer_existing_exposure",
                    "peer_job_id": peer_job_id,
                    "peer_profile": peer_profile,
                    "peer_trading_config_path": peer_cfg_path,
                    "dedup": dedup,
                }

        order_res = adapter.place_market_order(
            symbol=symbol,
            side=decision,
            volume=volume,
            stop_loss=float(plan.get("stop_loss", 0.0) or 0.0),
            take_profit=float(plan.get("take_profit", 0.0) or 0.0),
        )
        if not bool(order_res.get("ok", False)):
            return {
                "ok": False,
                "error": str(order_res.get("message") or "peer_market_order_failed"),
                "peer_job_id": peer_job_id,
                "peer_profile": peer_profile,
                "peer_trading_config_path": peer_cfg_path,
                "order": order_res,
            }

        position = order_res.get("position") if isinstance(order_res.get("position"), dict) else None
        if not position:
            order_ticket = int(order_res.get("order_ticket", 0) or 0)
            if order_ticket > 0:
                find_res = adapter.find_position_by_order(order_ticket)
                if bool(find_res.get("ok", False)) and isinstance(find_res.get("position"), dict):
                    position = find_res.get("position")

        if not isinstance(position, dict) or int(position.get("ticket", 0) or 0) <= 0:
            return {
                "ok": False,
                "error": "peer_market_position_not_found",
                "peer_job_id": peer_job_id,
                "peer_profile": peer_profile,
                "peer_trading_config_path": peer_cfg_path,
                "order": order_res,
            }

        now_iso = _iso(_now_utc())
        source_profile = str(mirror.get("source_profile") or _account_profile_label(source_trading_cfg)).strip() or "SOURCE"
        source_job_id = str(source_state.get("job_id") or mirror.get("source_job_id") or "").strip()
        source_cfg_path = _current_trading_config_path()

        peer_state_out = dict(peer_state)
        peer_state_out["job_id"] = peer_job_id
        peer_state_out["job_type"] = "trading_job"
        peer_state_out.setdefault("started_at", now_iso)
        peer_state_out["runner_pid"] = int(peer_state_out.get("runner_pid", 0) or 0)
        peer_state_out["stage"] = "agent_b"
        peer_state_out["mode"] = "mode_b"
        peer_state_out["status"] = "agent_b_running"
        peer_state_out["agent_b_started_at"] = now_iso
        peer_state_out["plan"] = plan
        peer_state_out["report_path"] = source_state.get("report_path")
        peer_state_out["signal_json_path"] = source_state.get("signal_json_path")
        peer_state_out["order_submission_mode"] = "market"
        peer_state_out["order"] = order_res
        peer_state_out["position"] = position
        peer_state_out["closed_reason"] = None
        peer_state_out.pop("ended_at", None)

        peer_state_out["mirror"] = {
            "enabled": True,
            "role": "mirror",
            "source_profile": source_profile,
            "source_job_id": source_job_id,
            "peer_profile": source_profile,
            "peer_job_id": source_job_id,
            "peer_trading_config_path": source_cfg_path,
            "launched_by_account_mirror": True,
        }
        peer_state_out["mirror_entry_fallback"] = {
            "timestamp_utc": now_iso,
            "reason": str(peer_state.get("closed_reason") or ""),
            "source_job_id": source_job_id,
            "source_profile": source_profile,
            "source_account": _account_profile_label(source_trading_cfg),
            "mode": "market",
            "result": order_res,
        }
        peer_state_out.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=peer_trading_cfg,
                channel="agent_a",
                kind="order_filled",
                message=(
                    f"Agent A mirrored linked-account market entry from {source_profile} after peer preflight data-sync failure. "
                    "Agent B supervision metadata was restored for continued linked-account management."
                ),
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={
                    "mirrored_from_account": source_profile,
                    "mirrored_from_job_id": source_job_id,
                    "order": order_res,
                    "position": position,
                    "fallback_reason": str(peer_state.get("closed_reason") or ""),
                },
                force_telegram=True,
                job_id=peer_job_id,
            )
        )
        _save_job_state(output_dir, peer_trading_cfg, peer_state_path, peer_state_out)

        return {
            "ok": True,
            "peer_job_id": peer_job_id,
            "peer_profile": peer_profile,
            "peer_trading_config_path": peer_cfg_path,
            "peer_state_path": peer_state_path,
            "fallback_reason": str(peer_state.get("closed_reason") or ""),
            "order": order_res,
            "position": position,
        }
    finally:
        adapter.shutdown()


def _propagate_mirror_job_action(
    action: str,
    output_dir: str,
    state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = state or {}
    mirror = dict((payload.get("mirror") or {}) if isinstance(payload, dict) else {})
    peer_job_id = str(mirror.get("peer_job_id") or "").strip()
    peer_cfg_path = str(mirror.get("peer_trading_config_path") or "").strip()
    if not peer_job_id or not peer_cfg_path:
        return {"ok": False, "skipped": True, "reason": "missing_peer_mirror_metadata"}

    try:
        peer_trading_cfg = load_trading_config(peer_cfg_path)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "peer_job_id": peer_job_id,
            "peer_trading_config_path": peer_cfg_path,
        }

    if action == "stop":
        result = stop_trading_job(output_dir, peer_trading_cfg, job_id=peer_job_id, propagate_mirror=False)
    elif action == "kill":
        result = kill_trading_job(output_dir, peer_trading_cfg, job_id=peer_job_id, propagate_mirror=False)
    else:
        return {"ok": False, "skipped": True, "reason": f"unsupported_action:{action}"}

    return {
        "ok": bool(result.get("ok", False)),
        "action": action,
        "peer_job_id": peer_job_id,
        "peer_profile": str(mirror.get("peer_profile") or "").strip() or "PEER",
        "peer_trading_config_path": peer_cfg_path,
        "result": result,
    }


def _mirror_agent_b_position_action(
    action: str,
    output_dir: str,
    source_trading_cfg: Dict[str, Any],
    source_state: Dict[str, Any],
    source_job_id: str,
    risk_adjustment: Optional[Dict[str, Any]] = None,
    closed_reason: str = "",
    final_status: str = "closed",
) -> Dict[str, Any]:
    mirror_cfg = _account_mirror_cfg(source_trading_cfg)
    if not bool(mirror_cfg.get("enabled", False)):
        return {"ok": False, "skipped": True, "reason": "mirror_disabled"}

    if action == "risk_update" and not bool(mirror_cfg.get("mirror_agent_b_risk_updates", True)):
        return {"ok": False, "skipped": True, "reason": "mirror_agent_b_risk_updates_disabled"}
    if action == "close" and not bool(mirror_cfg.get("mirror_agent_b_close_actions", True)):
        return {"ok": False, "skipped": True, "reason": "mirror_agent_b_close_actions_disabled"}

    mirror = dict((source_state.get("mirror") or {}) if isinstance(source_state, dict) else {})
    peer_job_id = str(mirror.get("peer_job_id") or "").strip()
    peer_cfg_path = str(mirror.get("peer_trading_config_path") or "").strip()
    if not peer_job_id or not peer_cfg_path:
        return {"ok": False, "skipped": True, "reason": "missing_peer_mirror_metadata"}

    try:
        peer_trading_cfg = load_trading_config(peer_cfg_path)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "peer_job_id": peer_job_id,
            "peer_trading_config_path": peer_cfg_path,
        }

    peer_profile = str(mirror.get("peer_profile") or _account_profile_label(peer_trading_cfg)).strip() or "PEER"
    peer_state_path = _state_path(output_dir, peer_trading_cfg, peer_job_id)
    peer_state = _load_state(peer_state_path)
    if not peer_state:
        return {
            "ok": False,
            "skipped": True,
            "reason": "peer_state_not_found",
            "peer_job_id": peer_job_id,
            "peer_trading_config_path": peer_cfg_path,
        }

    if action == "risk_update" and (str(peer_state.get("ended_at") or "").strip() or str(peer_state.get("closed_reason") or "").strip()):
        return {
            "ok": False,
            "skipped": True,
            "reason": "peer_job_already_closed",
            "peer_job_id": peer_job_id,
            "peer_profile": peer_profile,
        }

    position_block = (peer_state.get("position") or {}) if isinstance(peer_state.get("position"), dict) else {}
    position_ticket = int(position_block.get("ticket", 0) or 0)
    if str(peer_state.get("stage") or "").strip().lower() != "agent_b" or position_ticket <= 0:
        recovered = _recover_agent_b_state_from_live_position(peer_state, peer_trading_cfg, output_dir)
        if not recovered.get("ok"):
            return {
                "ok": False,
                "error": str(recovered.get("reason") or "peer_position_recovery_failed"),
                "peer_job_id": peer_job_id,
                "peer_profile": peer_profile,
                "peer_trading_config_path": peer_cfg_path,
            }
        peer_state = dict(recovered.get("state") or {})
        _save_job_state(output_dir, peer_trading_cfg, peer_state_path, peer_state)
        position_block = (peer_state.get("position") or {}) if isinstance(peer_state.get("position"), dict) else {}
        position_ticket = int(position_block.get("ticket", 0) or 0)

    if position_ticket <= 0:
        return {
            "ok": False,
            "error": "peer_position_ticket_missing",
            "peer_job_id": peer_job_id,
            "peer_profile": peer_profile,
            "peer_trading_config_path": peer_cfg_path,
        }

    mt5_cfg = (((peer_trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, msg_conn = adapter.connect()
    if not ok_conn:
        return {
            "ok": False,
            "error": f"peer_mt5_connect_failed:{msg_conn}",
            "peer_job_id": peer_job_id,
            "peer_profile": peer_profile,
            "peer_trading_config_path": peer_cfg_path,
        }

    try:
        if action == "risk_update":
            mirrored_result = adapter.modify_position_risk(
                position_ticket,
                stop_loss=(risk_adjustment or {}).get("stop_loss"),
                take_profit=(risk_adjustment or {}).get("take_profit"),
            )
            action_signature = json.dumps(
                {
                    "action": (risk_adjustment or {}).get("action"),
                    "stop_loss": (risk_adjustment or {}).get("stop_loss"),
                    "take_profit": (risk_adjustment or {}).get("take_profit"),
                },
                sort_keys=True,
                default=str,
            )
            peer_state["last_risk_adjustment"] = {
                "timestamp_utc": _iso(_now_utc()),
                "signature": action_signature,
                "result": mirrored_result,
                "mirrored_from": {
                    "account": _account_profile_label(source_trading_cfg),
                    "job_id": source_job_id,
                },
            }
            if isinstance(mirrored_result.get("position"), dict):
                peer_state["position"] = mirrored_result.get("position")
            if mirrored_result.get("ok") and not mirrored_result.get("skipped"):
                peer_state.setdefault("notifications", []).append(
                    _notify(
                        output_dir=output_dir,
                        trading_cfg=peer_trading_cfg,
                        channel="agent_b",
                        kind="risk_update",
                        message=(
                            f"Agent B mirrored linked-account risk action from {_account_profile_label(source_trading_cfg)}. "
                            f"action={(risk_adjustment or {}).get('action')}, "
                            f"sl={(risk_adjustment or {}).get('stop_loss', 'unchanged')}, "
                            f"tp={(risk_adjustment or {}).get('take_profit', 'unchanged')}."
                        ),
                        requires_approval=False,
                        emergency=False,
                        approval_deadline_utc=None,
                        metadata={
                            "mirrored_from_account": _account_profile_label(source_trading_cfg),
                            "mirrored_from_job_id": source_job_id,
                            "risk_adjustment": risk_adjustment,
                            "result": mirrored_result,
                        },
                        force_telegram=True,
                        job_id=peer_job_id,
                    )
                )
            _save_job_state(output_dir, peer_trading_cfg, peer_state_path, peer_state)
        elif action == "close":
            close_ok = _attempt_agent_b_close(
                adapter=adapter,
                state=peer_state,
                pos_ticket=position_ticket,
                output_dir=output_dir,
                trading_cfg=peer_trading_cfg,
                job_id=peer_job_id,
                closed_reason=closed_reason,
                final_status=final_status,
                failure_message=(
                    f"Agent B mirrored a linked-account close request from {_account_profile_label(source_trading_cfg)}, "
                    "but MT5 rejected the peer close. The peer position remains under supervision until MT5 confirms closure."
                ),
                propagate_mirror_action=False,
            )
            mirrored_result = {
                "ok": bool(close_ok),
                "result": peer_state.get("close_result"),
            }
            peer_state["last_mirrored_close"] = {
                "timestamp_utc": _iso(_now_utc()),
                "mirrored_from": {
                    "account": _account_profile_label(source_trading_cfg),
                    "job_id": source_job_id,
                },
                "closed_reason": closed_reason,
                "result": peer_state.get("close_result"),
            }
            _save_job_state(output_dir, peer_trading_cfg, peer_state_path, peer_state)
        else:
            return {"ok": False, "skipped": True, "reason": f"unsupported_action:{action}"}
    finally:
        adapter.shutdown()

    return {
        "ok": bool(mirrored_result.get("ok", False)),
        "action": action,
        "peer_job_id": peer_job_id,
        "peer_profile": peer_profile,
        "peer_trading_config_path": peer_cfg_path,
        "result": mirrored_result,
    }


def _telegram_subscribers_path(output_dir: str, trading_cfg: Optional[Dict[str, Any]] = None) -> str:
    return str(_runtime_dir(output_dir, trading_cfg) / "telegram_subscribers.json")


def _agent_timezone_name(trading_cfg: Dict[str, Any]) -> str:
    return str(((trading_cfg.get("agent") or {}).get("timezone") or "UTC")).strip() or "UTC"


def _parse_hhmm(raw_value: Any) -> Optional[tuple[int, int]]:
    raw = str(raw_value or "").strip()
    m = re.match(r"^(\d{1,2}):(\d{2})$", raw)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return hour, minute


def _next_market_open_utc(trading_cfg: Dict[str, Any], symbol: str, now_utc: Optional[datetime] = None) -> Optional[datetime]:
    exec_cfg = (trading_cfg.get("execution") or {})
    schedule_cfg = (exec_cfg.get("market_schedule") or {})
    windows_cfg = schedule_cfg.get("closed_windows") or {}
    if isinstance(windows_cfg, dict):
        windows = windows_cfg.get(str(symbol or "").upper()) or windows_cfg.get("default") or []
    elif isinstance(windows_cfg, list):
        windows = windows_cfg
    else:
        windows = []
    if not windows:
        return None

    tz_name = str(schedule_cfg.get("timezone") or _agent_timezone_name(trading_cfg)).strip() or "UTC"
    tz = ZoneInfo(tz_name)
    current_utc = now_utc or _now_utc()
    local_now = current_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(tz)

    for day_offset in (0, 1):
        day = (local_now + timedelta(days=day_offset)).date()
        for window in windows:
            start_parts = _parse_hhmm((window or {}).get("start")) if isinstance(window, dict) else None
            end_parts = _parse_hhmm((window or {}).get("end")) if isinstance(window, dict) else None
            if not start_parts or not end_parts:
                continue

            start_local = datetime(day.year, day.month, day.day, start_parts[0], start_parts[1], tzinfo=tz)
            end_local = datetime(day.year, day.month, day.day, end_parts[0], end_parts[1], tzinfo=tz)
            if end_local <= start_local:
                end_local += timedelta(days=1)

            if start_local <= local_now < end_local:
                return end_local.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    return None


def _is_market_closed_retcode(order_res: Dict[str, Any]) -> bool:
    retcode = int(order_res.get("retcode", 0) or 0)
    message = str(order_res.get("message") or "").lower()
    return retcode == 10018 or "market closed" in message


def _runtime_dir(output_dir: str, trading_cfg: Optional[Dict[str, Any]] = None) -> Path:
    return resolve_runtime_dir(output_dir=output_dir, trading_cfg=trading_cfg)


def _job_registry_path(output_dir: str, trading_cfg: Optional[Dict[str, Any]] = None) -> Path:
    return _runtime_dir(output_dir, trading_cfg) / "trading_job_registry.json"


def _job_root_dir(output_dir: str, trading_cfg: Optional[Dict[str, Any]] = None) -> Path:
    return _runtime_dir(output_dir, trading_cfg) / "trading_jobs"


def _job_dir(output_dir: str, trading_cfg: Dict[str, Any], job_id: str) -> Path:
    return _job_root_dir(output_dir, trading_cfg) / str(job_id)


def _job_file_path(output_dir: str, trading_cfg: Dict[str, Any], job_id: str, configured_path: str, fallback_name: str) -> str:
    configured = str(configured_path or "").strip()
    filename = Path(configured).name if configured else fallback_name
    return str(_job_dir(output_dir, trading_cfg, job_id) / filename)


def _state_path(output_dir: str, trading_cfg: Dict[str, Any], job_id: Optional[str] = None) -> str:
    tj = (trading_cfg.get("trading_job") or {})
    configured = str(tj.get("state_path") or "")
    if job_id:
        return _job_file_path(output_dir, trading_cfg, job_id, configured, "trading_job_state.json")
    return str(resolve_runtime_file(configured_path=configured, fallback_name="trading_job_state.json", output_dir=output_dir, trading_cfg=trading_cfg))


def _stop_flag_path(output_dir: str, trading_cfg: Dict[str, Any], job_id: Optional[str] = None) -> str:
    tj = (trading_cfg.get("trading_job") or {})
    configured = str(tj.get("stop_flag_path") or "")
    if job_id:
        return _job_file_path(output_dir, trading_cfg, job_id, configured, "trading_job_stop.flag")
    return str(resolve_runtime_file(configured_path=configured, fallback_name="trading_job_stop.flag", output_dir=output_dir, trading_cfg=trading_cfg))


def _approval_request_path(output_dir: str, trading_cfg: Dict[str, Any], job_id: Optional[str] = None) -> str:
    tj = (trading_cfg.get("trading_job") or {})
    configured = str(tj.get("approval_request_path") or "")
    if job_id:
        return _job_file_path(output_dir, trading_cfg, job_id, configured, "trading_approval_request.json")
    return str(resolve_runtime_file(configured_path=configured, fallback_name="trading_approval_request.json", output_dir=output_dir, trading_cfg=trading_cfg))


def _approval_response_path(output_dir: str, trading_cfg: Dict[str, Any], job_id: Optional[str] = None) -> str:
    tj = (trading_cfg.get("trading_job") or {})
    configured = str(tj.get("approval_response_path") or "")
    if job_id:
        return _job_file_path(output_dir, trading_cfg, job_id, configured, "trading_approval_response.json")
    return str(resolve_runtime_file(configured_path=configured, fallback_name="trading_approval_response.json", output_dir=output_dir, trading_cfg=trading_cfg))


def _save_state(path: str, payload: Dict[str, Any]) -> None:
    _write_json_atomic(path, payload)


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=os.path.dirname(path),
            prefix=".state_",
            suffix=".tmp",
            delete=False,
        ) as f:
            temp_path = f.name
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def _load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
        except Exception:
            return {}

        # Reboots can leave null-byte tails in JSON files; recover first valid object.
        for candidate in (raw, raw.replace("\x00", "")):
            candidate = str(candidate or "")
            if not candidate.strip():
                continue
            try:
                obj, _ = json.JSONDecoder().raw_decode(candidate.lstrip("\ufeff \t\r\n"))
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
        return {}


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    _write_json_atomic(path, payload)


def _new_job_id(trading_cfg: Optional[Dict[str, Any]] = None) -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    prefix = resolve_job_id_prefix(trading_cfg)
    if prefix:
        return f"job_{prefix}_{stamp}"
    return f"job_{stamp}"


def _runner_pid() -> int:
    try:
        return int(os.getpid())
    except Exception:
        return 0


def _job_state_summary(payload: Dict[str, Any], state_path: str) -> Dict[str, Any]:
    order = (payload.get("order") or {}) if isinstance(payload, dict) else {}
    return {
        "job_id": str(payload.get("job_id") or "").strip(),
        "state_path": str(state_path),
        "status": payload.get("status"),
        "stage": payload.get("stage"),
        "mode": payload.get("mode"),
        "started_at": payload.get("started_at"),
        "updated_at": _iso(_now_utc()),
        "closed_reason": payload.get("closed_reason"),
        "runner_pid": int(payload.get("runner_pid", 0) or 0),
        "position_ticket": (((payload.get("position") or {}).get("ticket")) if isinstance(payload, dict) else None),
        "order_submission_mode": payload.get("order_submission_mode"),
        "programmed_order_expiration_utc": payload.get("programmed_order_expiration_utc") or order.get("expiration_utc"),
        "ended_at": payload.get("ended_at"),
    }


def _is_active_job_summary(meta: Dict[str, Any]) -> bool:
    status = str((meta or {}).get("status") or "").strip().lower()
    if status in {"agent_b_running", "running", "started", "waiting_market_open", "pending_approval"}:
        return True
    if status != "agent_a_completed":
        return False
    stage = str((meta or {}).get("stage") or "").strip().lower()
    ended_at = str((meta or {}).get("ended_at") or "").strip()
    closed_reason = str((meta or {}).get("closed_reason") or "").strip().lower()
    return stage == "agent_a" and not ended_at and not closed_reason


def _latest_state_alias_payload(payload: Dict[str, Any], state_path: str) -> Dict[str, Any]:
    summary = _job_state_summary(payload, state_path)
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


def _cleanup_closed_jobs_enabled(trading_cfg: Dict[str, Any]) -> bool:
    tj = (trading_cfg.get("trading_job") or {})
    return bool(tj.get("cleanup_closed_jobs", True))


def _should_cleanup_job_state(trading_cfg: Dict[str, Any], payload: Dict[str, Any]) -> bool:
    if not _cleanup_closed_jobs_enabled(trading_cfg):
        return False
    status = str(payload.get("status") or "").strip().lower()
    return status in {"closed", "stopped"}


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


def _refresh_latest_state_alias(output_dir: str, trading_cfg: Dict[str, Any]) -> None:
    alias_path = _state_path(output_dir, trading_cfg)
    registry = _load_state(str(_job_registry_path(output_dir, trading_cfg)))
    jobs = dict(registry.get("jobs") or {})
    active_ids = [str(x).strip() for x in (registry.get("active_job_ids") or []) if str(x).strip()]
    preferred_job_id = active_ids[-1] if active_ids else _choose_latest_job_id(jobs)

    if not preferred_job_id:
        if os.path.exists(alias_path):
            os.remove(alias_path)
        return

    meta = dict(jobs.get(preferred_job_id) or {})
    state_path = str(meta.get("state_path") or "").strip()
    payload = _load_state(state_path) if state_path else {}
    if payload and state_path:
        _write_json(alias_path, _latest_state_alias_payload(payload, state_path))
        return
    if meta:
        _write_json(alias_path, meta)
        return
    if os.path.exists(alias_path):
        os.remove(alias_path)


def _cleanup_job_runtime(output_dir: str, trading_cfg: Dict[str, Any], payload: Dict[str, Any], state_path: str) -> None:
    job_id = str(payload.get("job_id") or "").strip()
    registry_path = _job_registry_path(output_dir, trading_cfg)
    registry = _load_state(str(registry_path))
    jobs = dict(registry.get("jobs") or {})
    if job_id:
        jobs.pop(job_id, None)

    active_job_ids = [
        jid for jid, meta in jobs.items()
        if _is_active_job_summary(meta)
    ]
    registry["jobs"] = jobs
    registry["active_job_ids"] = active_job_ids
    registry["latest_job_id"] = active_job_ids[-1] if active_job_ids else _choose_latest_job_id(jobs)
    registry["updated_at"] = _iso(_now_utc())
    _write_json(str(registry_path), registry)

    job_dir = Path(state_path).parent
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    _refresh_latest_state_alias(output_dir, trading_cfg)


def _update_job_registry(output_dir: str, trading_cfg: Dict[str, Any], payload: Dict[str, Any], state_path: str) -> None:
    job_id = str(payload.get("job_id") or "").strip()
    if not job_id:
        return

    registry_path = _job_registry_path(output_dir, trading_cfg)
    registry = _load_state(str(registry_path))
    jobs = dict(registry.get("jobs") or {})
    jobs[job_id] = _job_state_summary(payload, state_path)

    active_job_ids = [
        jid for jid, meta in jobs.items()
        if _is_active_job_summary(meta)
    ]

    registry["jobs"] = jobs
    registry["active_job_ids"] = active_job_ids
    registry["latest_job_id"] = active_job_ids[-1] if active_job_ids else _choose_latest_job_id(jobs)
    registry["updated_at"] = _iso(_now_utc())
    _write_json(str(registry_path), registry)


def _write_latest_state_alias(output_dir: str, trading_cfg: Dict[str, Any], payload: Dict[str, Any]) -> None:
    alias_path = _state_path(output_dir, trading_cfg)
    state_path = str(payload.get("state_path") or _state_path(output_dir, trading_cfg, str(payload.get("job_id") or "").strip()))
    _write_json(alias_path, _latest_state_alias_payload(payload, state_path))


def _rebuild_latest_approval_alias(output_dir: str, trading_cfg: Dict[str, Any]) -> None:
    latest_pending: Dict[str, Any] = {}
    latest_created = ""
    jobs_root = _job_root_dir(output_dir, trading_cfg)
    if jobs_root.exists():
        for job_dir in jobs_root.iterdir():
            if not job_dir.is_dir():
                continue
            req_path = Path(_approval_request_path(output_dir, trading_cfg, job_dir.name))
            payload = _load_state(str(req_path))
            if str(payload.get("status") or "").strip().lower() != "pending":
                continue
            created = str(payload.get("created_at_utc") or "")
            if created >= latest_created:
                latest_created = created
                latest_pending = payload

    alias_path = _approval_request_path(output_dir, trading_cfg)
    if latest_pending:
        _write_json(alias_path, latest_pending)
    elif os.path.exists(alias_path):
        os.remove(alias_path)


def _resolve_job_id(output_dir: str, trading_cfg: Dict[str, Any], job_id: Optional[str] = None) -> str:
    requested = str(job_id or "").strip()
    if requested:
        return requested
    registry = _load_state(str(_job_registry_path(output_dir, trading_cfg)))
    active_ids = [str(x).strip() for x in (registry.get("active_job_ids") or []) if str(x).strip()]
    if active_ids:
        return active_ids[-1]
    return str(registry.get("latest_job_id") or "").strip()


def _is_stoppable_job_state(payload: Dict[str, Any]) -> bool:
    status = str((payload or {}).get("status") or "").strip().lower()
    if status not in {"agent_b_running", "running", "started", "waiting_market_open", "agent_a_completed"}:
        return False
    if status == "agent_a_completed":
        stage = str((payload or {}).get("stage") or "").strip().lower()
        ended_at = str((payload or {}).get("ended_at") or "").strip()
        closed_reason = str((payload or {}).get("closed_reason") or "").strip().lower()
        return stage == "agent_a" and not ended_at and not closed_reason
    return True


def _stoppable_job_ids(output_dir: str, trading_cfg: Dict[str, Any]) -> list[str]:
    registry = _load_state(str(_job_registry_path(output_dir, trading_cfg)))
    candidate_ids: list[str] = []
    for raw_id in (registry.get("active_job_ids") or []):
        clean_id = str(raw_id).strip()
        if clean_id and clean_id not in candidate_ids:
            candidate_ids.append(clean_id)
    for raw_id in (registry.get("jobs") or {}).keys():
        clean_id = str(raw_id).strip()
        if clean_id and clean_id not in candidate_ids:
            candidate_ids.append(clean_id)

    out: list[str] = []
    for current_job_id in candidate_ids:
        state = _load_state(_state_path(output_dir, trading_cfg, current_job_id))
        if state and _is_stoppable_job_state(state):
            out.append(current_job_id)

    if out:
        return out

    latest_state = _load_state(_state_path(output_dir, trading_cfg))
    latest_job_id = str((latest_state or {}).get("job_id") or "").strip()
    if latest_job_id and _is_stoppable_job_state(latest_state):
        return [latest_job_id]
    return []


def _save_job_state(output_dir: str, trading_cfg: Dict[str, Any], state_path: str, payload: Dict[str, Any]) -> None:
    previous_state = _load_state(state_path)
    payload["state_path"] = state_path
    _save_state(state_path, payload)

    try:
        log_state_transition_feedback(
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            previous_state=previous_state,
            current_state=payload,
        )
    except Exception:
        pass

    if _should_cleanup_job_state(trading_cfg, payload):
        _cleanup_job_runtime(output_dir, trading_cfg, payload, state_path)
        return
    _update_job_registry(output_dir, trading_cfg, payload, state_path)
    _write_latest_state_alias(output_dir, trading_cfg, payload)


def _ensure_assessment_data_fresh(
    trading_cfg: Dict[str, Any],
    logger,
    app_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    dash_cfg = (trading_cfg.get("dashboard") or {})
    if not bool(dash_cfg.get("startup_sync_enabled", True)):
        return {"ok": True, "skipped": True, "reason": "startup_sync_disabled"}

    master_path = str(dash_cfg.get("master_table_path") or dash_cfg.get("raw_data_path") or "").strip()
    if not master_path:
        return {"ok": False, "error": "missing_master_path"}

    token_env = str(dash_cfg.get("tiingo_token_env", "TIINGO_API_TOKEN")).strip() or "TIINGO_API_TOKEN"
    token_envs_cfg = dash_cfg.get("tiingo_token_envs")
    token_rotation_state_path = str(dash_cfg.get("tiingo_token_rotation_state_path") or "").strip() or None
    token = os.environ.get(token_env, "")
    token_candidates = resolve_tiingo_token_candidates(
        token_env=token_env,
        token_envs=token_envs_cfg,
        token=token,
    )
    if not token_candidates:
        configured_envs = [token_env]
        if isinstance(token_envs_cfg, str):
            configured_envs.extend([x.strip() for x in token_envs_cfg.replace(";", ",").split(",") if str(x).strip()])
        elif isinstance(token_envs_cfg, (list, tuple, set)):
            configured_envs.extend([str(x).strip() for x in token_envs_cfg if str(x).strip()])
        configured_envs = list(dict.fromkeys(configured_envs))
        return {"ok": False, "error": f"missing_token_envs:{','.join(configured_envs)}"}

    symbol = str(dash_cfg.get("tiingo_symbol", "xauusd")).strip().lower() or "xauusd"
    rate = str(dash_cfg.get("tiingo_rate", "1min")).strip() or "1min"
    max_pulls = int(dash_cfg.get("startup_max_pulls", 2) or 2)
    freshness_lag_minutes = int(dash_cfg.get("startup_freshness_lag_minutes", 20) or 20)

    sync_result = bootstrap_master_on_backend_start(
        master_table_path=master_path,
        rate=rate,
        symbol=symbol,
        token=token,
        max_pulls=max_pulls,
        freshness_lag_minutes=freshness_lag_minutes,
        token_env=token_env,
        token_envs=token_envs_cfg,
        token_rotation_state_path=token_rotation_state_path,
    )

    active_sync_result = None
    active_cfg = app_config or {}
    active_sql_symbol = str(
        active_cfg.get("sql_symbol")
        or active_cfg.get("symbol")
        or dash_cfg.get("sql_symbol")
        or symbol
    ).strip()
    active_data_path = str(active_cfg.get("data_path") or "").strip()
    if active_data_path and not active_data_path.lower().endswith((".db", ".sqlite")):
        active_sync_result = sync_dataset_source_from_master(
            master_table_path=master_path,
            output_path=active_data_path,
            timeframe_minutes=active_cfg.get("data_timeframe_minutes"),
            records=int(active_cfg.get("records", 5000) or 5000),
            rolling_windows=list(active_cfg.get("rolling_windows") or [2, 7, 30, 60]),
            n_steps=int(active_cfg.get("n_steps", 1) or 1),
            horizon=int(active_cfg.get("horizon", 1) or 1),
            symbol=active_sql_symbol,
            logger=logger,
        )

    out = {
        "ok": bool(sync_result.get("ok", False)),
        "master_path": master_path,
        "token_env": token_env,
        "configured_token_envs": [item.get("env") for item in token_candidates],
        "freshness_lag_minutes": freshness_lag_minutes,
        "sync_result": sync_result,
        "active_config_sync_result": active_sync_result,
    }
    if not out["ok"]:
        out["error"] = str(sync_result.get("error") or sync_result.get("latest_date") or "master_sync_not_aligned")
    return out


def _clear_file(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)


def _windows_yes_no(title: str, message: str) -> Optional[bool]:
    try:
        user32 = ctypes.windll.user32
        MB_YESNO = 0x00000004
        MB_ICONQUESTION = 0x00000020
        IDYES = 6
        out = user32.MessageBoxW(None, message, title, MB_YESNO | MB_ICONQUESTION)
        return bool(out == IDYES)
    except Exception:
        return None


def _input_with_timeout(prompt: str, timeout_sec: int) -> Optional[str]:
    holder = {"value": None}

    def _reader():
        try:
            holder["value"] = input(prompt)
        except Exception:
            holder["value"] = None

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=max(int(timeout_sec), 1))
    if t.is_alive():
        return None
    return holder.get("value")


def request_approval(
    title: str,
    message: str,
    timeout_sec: int,
    channels: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    trading_cfg: Optional[Dict[str, Any]] = None,
    approval_metadata: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
) -> bool:
    channels = [str(c).strip().lower() for c in (channels or ["popup", "terminal"])]

    deadline = _now_utc() + timedelta(seconds=max(int(timeout_sec), 1))
    approval_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    request_path = ""
    response_path = ""
    if output_dir and trading_cfg:
        resolved_job_id = _resolve_job_id(output_dir, trading_cfg, job_id)
        request_path = _approval_request_path(output_dir, trading_cfg, resolved_job_id)
        response_path = _approval_response_path(output_dir, trading_cfg, resolved_job_id)
        _clear_file(response_path)
        payload = {
            "approval_id": approval_id,
            "job_id": resolved_job_id,
            "title": title,
            "message": message,
            "channels": channels,
            "created_at_utc": _iso(_now_utc()),
            "deadline_utc": _iso(deadline),
            "metadata": {**(approval_metadata or {}), "job_id": resolved_job_id},
            "status": "pending",
            "request_path": request_path,
            "response_path": response_path,
        }
        _write_json(request_path, payload)
        _rebuild_latest_approval_alias(output_dir, trading_cfg)

    def _check_telegram_response() -> Optional[bool]:
        if not response_path or not os.path.exists(response_path):
            return None
        payload = _load_state(response_path)
        if str(payload.get("approval_id") or "").strip() != approval_id:
            return None
        decision = str(payload.get("decision") or "").strip().lower()
        if decision in {"approve", "approved", "yes", "y"}:
            return True
        if decision in {"reject", "rejected", "no", "n"}:
            return False
        return None

    if "telegram" in channels and response_path:
        wait_until = time.time() + max(int(timeout_sec), 1)
        while time.time() <= wait_until:
            tg_decision = _check_telegram_response()
            if tg_decision is not None:
                _clear_file(request_path)
                _clear_file(response_path)
                _rebuild_latest_approval_alias(output_dir, trading_cfg)
                return tg_decision
            time.sleep(1)

    if "popup" in channels:
        res = _windows_yes_no(title, message)
        if res is True:
            _clear_file(request_path)
            _clear_file(response_path)
            if output_dir and trading_cfg:
                _rebuild_latest_approval_alias(output_dir, trading_cfg)
            return True
        if res is False and "terminal" not in channels:
            _clear_file(request_path)
            _clear_file(response_path)
            if output_dir and trading_cfg:
                _rebuild_latest_approval_alias(output_dir, trading_cfg)
            return False

    if "terminal" in channels:
        ans = _input_with_timeout(f"{message} [yes/no]: ", timeout_sec=timeout_sec)
        if ans is None:
            _clear_file(request_path)
            _clear_file(response_path)
            if output_dir and trading_cfg:
                _rebuild_latest_approval_alias(output_dir, trading_cfg)
            return False
        approved = str(ans).strip().lower() in {"y", "yes", "ok", "approve"}
        _clear_file(request_path)
        _clear_file(response_path)
        if output_dir and trading_cfg:
            _rebuild_latest_approval_alias(output_dir, trading_cfg)
        return approved

    _clear_file(request_path)
    _clear_file(response_path)
    if output_dir and trading_cfg:
        _rebuild_latest_approval_alias(output_dir, trading_cfg)

    return False


def stop_trading_job(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    job_id: Optional[str] = None,
    propagate_mirror: bool = True,
) -> Dict[str, Any]:
    requested_job_id = str(job_id or "").strip()
    target_job_ids = [requested_job_id] if requested_job_id else _stoppable_job_ids(output_dir, trading_cfg)
    if not target_job_ids:
        return {
            "ok": False,
            "job_ids": [],
            "stop_paths": [],
            "message": "No active trading jobs found to stop.",
        }

    stop_paths: list[str] = []
    mirror_results: list[Dict[str, Any]] = []
    timestamp = _iso(_now_utc())
    for target_job_id in target_job_ids:
        stop_path = _stop_flag_path(output_dir, trading_cfg, target_job_id)
        os.makedirs(os.path.dirname(stop_path), exist_ok=True)
        with open(stop_path, "w", encoding="utf-8") as f:
            f.write(timestamp)
        stop_paths.append(stop_path)

        if propagate_mirror and bool(_account_mirror_cfg(trading_cfg).get("mirror_stop_requests", True)):
            state_path = _state_path(output_dir, trading_cfg, target_job_id)
            mirror_results.append(_propagate_mirror_job_action("stop", output_dir, _load_state(state_path)))

    message = (
        f"Trading job stop flags written for {len(target_job_ids)} job(s): {', '.join(target_job_ids)}"
        if len(target_job_ids) > 1
        else f"Trading job stop flag written: {stop_paths[0]}"
    )
    return {
        "ok": True,
        "job_ids": target_job_ids,
        "stop_paths": stop_paths,
        "mirror_results": mirror_results,
        "message": message,
    }


def _kill_process_tree(pid: int) -> Dict[str, Any]:
    target_pid = int(pid or 0)
    if target_pid <= 0:
        return {"ok": False, "killed": False, "reason": "missing_pid", "pid": target_pid}

    try:
        proc = psutil.Process(target_pid)
    except psutil.NoSuchProcess:
        return {"ok": True, "killed": False, "reason": "pid_not_found", "pid": target_pid}
    except Exception as exc:
        return {"ok": False, "killed": False, "reason": f"process_lookup_failed:{exc}", "pid": target_pid}

    children = []
    try:
        children = proc.children(recursive=True)
    except Exception:
        children = []

    for child in children:
        try:
            child.kill()
        except Exception:
            pass
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        pass
    except Exception as exc:
        return {"ok": False, "killed": False, "reason": f"kill_failed:{exc}", "pid": target_pid}

    gone = []
    alive = []
    try:
        gone, alive = psutil.wait_procs(children + [proc], timeout=5)
    except Exception:
        alive = []
    return {
        "ok": True,
        "killed": True,
        "pid": target_pid,
        "terminated": [int(p.pid) for p in gone],
        "alive_after_wait": [int(p.pid) for p in alive],
    }


def kill_trading_job(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    job_id: Optional[str] = None,
    propagate_mirror: bool = True,
) -> Dict[str, Any]:
    requested_job_id = str(job_id or "").strip()
    target_job_ids = [requested_job_id] if requested_job_id else _stoppable_job_ids(output_dir, trading_cfg)
    if not target_job_ids:
        return {
            "ok": False,
            "job_ids": [],
            "message": "No active trading jobs found to kill.",
            "results": [],
        }

    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, _msg_conn = adapter.connect()
    timestamp = _iso(_now_utc())
    results = []
    mirror_results: list[Dict[str, Any]] = []

    try:
        for target_job_id in target_job_ids:
            state_path = _state_path(output_dir, trading_cfg, target_job_id)
            state = _load_state(state_path)
            if not state:
                results.append({"job_id": target_job_id, "ok": False, "reason": "state_not_found"})
                continue

            stop_path = _stop_flag_path(output_dir, trading_cfg, target_job_id)
            os.makedirs(os.path.dirname(stop_path), exist_ok=True)
            with open(stop_path, "w", encoding="utf-8") as f:
                f.write(timestamp)

            kill_res = _kill_process_tree(int(state.get("runner_pid", 0) or 0))
            cancel_res: Dict[str, Any] = {"ok": False, "skipped": True, "reason": "no_pending_order_cancelled"}
            close_res: Dict[str, Any] = {"ok": False, "skipped": True, "reason": "no_live_position_close_attempted"}
            order = (state.get("order") or {}) if isinstance(state.get("order"), dict) else {}
            position = (state.get("position") or {}) if isinstance(state.get("position"), dict) else {}
            order_ticket = int(order.get("order_ticket", 0) or 0)
            position_ticket = int(position.get("ticket", 0) or 0)
            if ok_conn and position_ticket > 0:
                close_res = adapter.close_position_by_ticket(position_ticket)
            if ok_conn and order_ticket > 0 and position_ticket <= 0:
                cancel_res = adapter.cancel_pending_order(order_ticket)

            state["runner_pid"] = 0
            state["status"] = "killed"
            state["ended_at"] = timestamp
            state["closed_reason"] = "manual_kill"
            state["kill_requested_at"] = timestamp
            state["kill_result"] = kill_res
            state["stop_flag_path"] = stop_path
            if position_ticket > 0:
                state["close_result"] = close_res
            if order_ticket > 0 and position_ticket <= 0:
                state["cancel_result"] = cancel_res
            _save_job_state(output_dir, trading_cfg, state_path, state)

            action_ok = bool(kill_res.get("ok", False))
            if position_ticket > 0:
                action_ok = action_ok or bool(close_res.get("ok", False))
            if order_ticket > 0 and position_ticket <= 0:
                action_ok = action_ok or bool(cancel_res.get("ok", False))

            results.append(
                {
                    "job_id": target_job_id,
                    "ok": action_ok,
                    "pid": int((kill_res or {}).get("pid", 0) or 0),
                    "kill_result": kill_res,
                    "close_result": close_res,
                    "cancel_result": cancel_res,
                }
            )

            if propagate_mirror and bool(_account_mirror_cfg(trading_cfg).get("mirror_kill_requests", True)):
                mirror_results.append(_propagate_mirror_job_action("kill", output_dir, state))
    finally:
        if ok_conn:
            adapter.shutdown()

    success_job_ids = [
        str(item.get("job_id") or "").strip()
        for item in results
        if bool(item.get("ok", False)) and str(item.get("job_id") or "").strip()
    ]
    attempted_job_ids = [str(item.get("job_id") or "").strip() for item in results if str(item.get("job_id") or "").strip()]
    return {
        "ok": any(bool(item.get("ok", False)) for item in results),
        "job_ids": success_job_ids,
        "results": results,
        "mirror_results": mirror_results,
        "message": (
            f"Trading job kill executed for {len(success_job_ids)} job(s): {', '.join(success_job_ids)}"
            if success_job_ids
            else (
                "Trading job kill request completed, but no process or broker exposure was terminated"
                + (f" for: {', '.join(attempted_job_ids)}" if attempted_job_ids else ".")
            )
        ),
    }


def _clear_stop_flag(output_dir: str, trading_cfg: Dict[str, Any], job_id: Optional[str] = None) -> None:
    stop_path = _stop_flag_path(output_dir, trading_cfg, job_id)
    if os.path.exists(stop_path):
        os.remove(stop_path)


def _should_stop(output_dir: str, trading_cfg: Dict[str, Any], job_id: Optional[str] = None) -> bool:
    return os.path.exists(_stop_flag_path(output_dir, trading_cfg, job_id))


def _delayed_stop_loss_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    exec_cfg = (trading_cfg.get("execution") or {}) if isinstance(trading_cfg, dict) else {}
    cfg = dict(exec_cfg.get("delayed_stop_loss") or {})
    risk_cfg = (trading_cfg.get("risk") or {}) if isinstance(trading_cfg, dict) else {}
    legacy_cfg = risk_cfg.get("delayed_stop_loss")
    if isinstance(legacy_cfg, dict):
        merged = dict(legacy_cfg)
        merged.update(cfg)
        cfg = merged
    return cfg


def _fallback_stop_loss_for_plan(plan: Dict[str, Any], trading_cfg: Dict[str, Any]) -> float:
    decision = str((plan or {}).get("decision") or "").strip().lower()
    entry = _safe_float((plan or {}).get("entry"), 0.0)
    risk_cfg = (trading_cfg.get("risk") or {}) if isinstance(trading_cfg, dict) else {}
    sl_pct = max(_safe_float(risk_cfg.get("stop_loss_pct", 0.8), 0.8), 0.01) / 100.0
    if decision == "buy" and entry > 0.0:
        return round(entry * (1.0 - sl_pct), 6)
    if decision == "sell" and entry > 0.0:
        return round(entry * (1.0 + sl_pct), 6)
    return 0.0


def _plan_requests_delayed_stop_loss(plan: Dict[str, Any], trading_cfg: Dict[str, Any]) -> bool:
    cfg = _delayed_stop_loss_cfg(trading_cfg)
    if not bool(cfg.get("enabled", False)):
        return False

    decision = str((plan or {}).get("decision") or "").strip().lower()
    if decision not in {"buy", "sell"}:
        return False

    conviction = (plan.get("conviction") or {}) if isinstance(plan, dict) else {}
    risk_mode = str(conviction.get("risk_mode") or "").strip().lower()
    allowed_modes = cfg.get("allowed_conviction_modes") or ["no_sl"]
    allowed = {str(item).strip().lower() for item in allowed_modes if str(item).strip()}
    conviction_score = _safe_float(conviction.get("conviction"), 0.0)
    min_conviction = min(max(_safe_float(cfg.get("min_conviction", 0.8), 0.8), 0.0), 1.0)
    plan_sl = (plan or {}).get("stop_loss")
    return (
        risk_mode in allowed
        and conviction_score >= min_conviction
        and (plan_sl is None or _safe_float(plan_sl, 0.0) <= 0.0)
    )


def _order_volume_for_plan(plan: Dict[str, Any], trading_cfg: Dict[str, Any]) -> float:
    exec_cfg = (trading_cfg.get("execution") or {}) if isinstance(trading_cfg, dict) else {}
    default_volume = max(_safe_float(exec_cfg.get("default_volume", 0.01), 0.01), 0.0)
    requested_volume = max(_safe_float((plan or {}).get("volume", default_volume), default_volume), 0.0)
    if not _plan_requests_delayed_stop_loss(plan, trading_cfg):
        return requested_volume

    cfg = _delayed_stop_loss_cfg(trading_cfg)
    max_multiplier = max(_safe_float(cfg.get("max_volume_multiplier", 1.0), 1.0), 0.0)
    return min(requested_volume, default_volume * max_multiplier)


def _order_risk_levels_for_plan(plan: Dict[str, Any], trading_cfg: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
    delayed = _plan_requests_delayed_stop_loss(plan, trading_cfg)
    stop_loss = _safe_float((plan or {}).get("stop_loss"), 0.0)
    take_profit = _safe_float((plan or {}).get("take_profit"), 0.0)
    fallback_stop = _fallback_stop_loss_for_plan(plan, trading_cfg)

    if delayed:
        return 0.0, take_profit, {
            "mode": "delayed_protection",
            "initial_broker_stop_loss": 0.0,
            "planned_stop_loss": stop_loss if stop_loss > 0.0 else fallback_stop,
            "take_profit": take_profit,
            "reason": "delayed_stop_loss_enabled",
        }

    if stop_loss <= 0.0 and fallback_stop > 0.0:
        return fallback_stop, take_profit, {
            "mode": "standard",
            "initial_broker_stop_loss": fallback_stop,
            "planned_stop_loss": fallback_stop,
            "take_profit": take_profit,
            "reason": "fallback_stop_loss_applied",
        }

    return stop_loss, take_profit, {
        "mode": "standard",
        "initial_broker_stop_loss": stop_loss,
        "planned_stop_loss": stop_loss,
        "take_profit": take_profit,
        "reason": "plan_stop_loss_applied",
    }


def _prop_firm_guard_preflight(
    adapter: MT5Adapter,
    trading_cfg: Dict[str, Any],
    symbol: str,
    side: str,
    volume: float,
    entry: float,
    stop_loss: float,
) -> Dict[str, Any]:
    """Fail closed before an order can consume a prop-firm risk allowance."""
    cfg = dict((trading_cfg.get("prop_firm_guard") or {}))
    sizing_cfg = dict(((trading_cfg.get("risk") or {}).get("account_sizing") or {}))
    sizing_enabled = bool(sizing_cfg.get("enabled", False))
    effective_volume = float(volume or 0.0)
    sizing: Dict[str, Any] = {"enabled": sizing_enabled, "requested_volume": effective_volume}
    if sizing_enabled:
        if float(stop_loss or 0.0) <= 0.0:
            return {"ok": False, "enabled": True, "reason": "account_sizing_requires_hard_stop", "sizing": sizing}
        account = adapter.get_account_snapshot()
        symbol_spec = adapter.get_symbol_trade_spec(symbol)
        sizing["account"] = account
        sizing["symbol_spec"] = symbol_spec
        if not account.get("ok") or not symbol_spec.get("ok"):
            return {"ok": False, "enabled": True, "reason": "account_sizing_broker_metadata_failed", "sizing": sizing}
        equity = max(float(account.get("equity", 0.0) or 0.0), 0.0)
        risk_pct = max(float((trading_cfg.get("risk") or {}).get("risk_per_trade_pct", 0.5) or 0.5), 0.0)
        allowance = equity * risk_pct / 100.0
        one_lot = adapter.estimate_trade_loss(
            symbol=symbol, side=side, volume=1.0, entry=entry, stop_loss=stop_loss
        )
        sizing["one_lot_loss"] = one_lot
        if not one_lot.get("ok") or float(one_lot.get("estimated_loss", 0.0) or 0.0) <= 0.0:
            return {"ok": False, "enabled": True, "reason": "account_sizing_loss_estimate_failed", "sizing": sizing}
        risk_sized_volume = allowance / float(one_lot["estimated_loss"])
        volume_min = max(float(symbol_spec.get("volume_min", 0.0) or 0.0), 0.0)
        volume_step = max(float(symbol_spec.get("volume_step", volume_min) or volume_min), 1e-12)
        requested_broker_volume = effective_volume
        minimum_volume_floor_applied = (
            0.0 < requested_broker_volume < volume_min
            and risk_sized_volume >= volume_min
        )
        if minimum_volume_floor_applied:
            requested_broker_volume = volume_min
        raw_volume = (
            min(risk_sized_volume, requested_broker_volume)
            if requested_broker_volume > 0.0
            else risk_sized_volume
        )
        volume_max = max(float(symbol_spec.get("volume_max", raw_volume) or raw_volume), 0.0)
        stepped_volume = int((min(raw_volume, volume_max) + 1e-12) / volume_step) * volume_step
        precision = max(0, min(8, len(f"{volume_step:.8f}".rstrip("0").split(".")[-1])))
        effective_volume = round(stepped_volume, precision)
        sizing.update({"equity": equity, "risk_pct": risk_pct, "risk_allowance": allowance,
                       "risk_sized_volume": risk_sized_volume, "raw_volume": raw_volume,
                       "effective_volume": effective_volume,
                       "minimum_volume_floor_applied": minimum_volume_floor_applied})
        if effective_volume < volume_min or effective_volume <= 0.0:
            return {"ok": False, "enabled": True, "reason": "minimum_broker_volume_exceeds_risk_budget", "sizing": sizing}

    if not bool(cfg.get("enabled", False)):
        return {"ok": True, "enabled": False, "reason": "guard_disabled", "sized_volume": effective_volume, "sizing": sizing}

    result: Dict[str, Any] = {"ok": False, "enabled": True, "checks": {}}

    require_hard_stop = bool(cfg.get("require_hard_stop", True))
    if require_hard_stop and float(stop_loss or 0.0) <= 0.0:
        result["reason"] = "hard_stop_required"
        return result

    positions_res = adapter.list_open_positions()
    orders_res = adapter.list_pending_orders()
    if not positions_res.get("ok") or not orders_res.get("ok"):
        result["reason"] = "broker_exposure_check_failed"
        result["positions_check"] = positions_res
        result["orders_check"] = orders_res
        return result

    positions = list(positions_res.get("positions") or [])
    orders = list(orders_res.get("orders") or [])
    result["checks"]["open_positions"] = len(positions)
    result["checks"]["pending_orders"] = len(orders)
    if bool(cfg.get("block_on_any_account_exposure", True)) and (positions or orders):
        result["reason"] = "existing_account_exposure"
        result["exposure"] = {
            "positions": positions,
            "orders": orders,
        }
        return result

    account = adapter.get_account_snapshot()
    if not account.get("ok"):
        result["reason"] = "account_snapshot_failed"
        result["account"] = account
        return result
    result["account"] = account

    expected_login_env = str(cfg.get("expected_login_env") or "").strip()
    if expected_login_env:
        expected_login_raw = str(os.environ.get(expected_login_env, "") or "").strip()
        if not expected_login_raw:
            result["reason"] = "expected_login_environment_missing"
            return result
        if str(int(account.get("login", 0) or 0)) != expected_login_raw:
            result["reason"] = "unexpected_broker_account"
            return result

    expected_server = str(cfg.get("expected_server") or "").strip()
    if expected_server and str(account.get("server") or "").strip().lower() != expected_server.lower():
        result["reason"] = "unexpected_broker_server"
        return result

    expected_currency = str(cfg.get("expected_currency") or "").strip()
    if expected_currency and str(account.get("currency") or "").strip().upper() != expected_currency.upper():
        result["reason"] = "unexpected_account_currency"
        return result

    if not bool(account.get("trade_allowed", False)) or not bool(account.get("trade_expert", False)):
        result["reason"] = "broker_does_not_allow_expert_trading"
        return result

    loss_estimate = adapter.estimate_trade_loss(
        symbol=str(symbol),
        side=str(side),
        volume=float(effective_volume),
        entry=float(entry),
        stop_loss=float(stop_loss),
    )
    result["loss_estimate"] = loss_estimate
    if not loss_estimate.get("ok"):
        result["reason"] = "planned_loss_estimate_failed"
        return result

    estimated_loss = float(loss_estimate.get("estimated_loss", 0.0) or 0.0)
    hard_single_trade_loss = max(float(cfg.get("hard_single_trade_loss_usd", 0.0) or 0.0), 0.0)
    internal_trade_loss = max(float(cfg.get("internal_trade_loss_cap_usd", hard_single_trade_loss) or hard_single_trade_loss), 0.0)
    if hard_single_trade_loss > 0.0 and estimated_loss >= hard_single_trade_loss:
        result["reason"] = "hard_single_trade_loss_limit"
        return result
    if internal_trade_loss > 0.0 and estimated_loss > internal_trade_loss:
        result["reason"] = "internal_trade_loss_cap"
        return result

    day_pnl = adapter.get_utc_day_realized_pnl()
    result["utc_day"] = day_pnl
    if not day_pnl.get("ok"):
        result["reason"] = "utc_day_pnl_check_failed"
        return result

    realized_pnl = float(day_pnl.get("realized_pnl", 0.0) or 0.0)
    projected_daily_loss = max(-realized_pnl, 0.0) + estimated_loss
    result["checks"]["projected_daily_loss"] = projected_daily_loss
    hard_daily_loss = max(float(cfg.get("hard_daily_loss_usd", 0.0) or 0.0), 0.0)
    internal_daily_loss = max(float(cfg.get("internal_daily_loss_cap_usd", hard_daily_loss) or hard_daily_loss), 0.0)
    if hard_daily_loss > 0.0 and projected_daily_loss >= hard_daily_loss:
        result["reason"] = "hard_daily_loss_limit"
        return result
    if internal_daily_loss > 0.0 and projected_daily_loss > internal_daily_loss:
        result["reason"] = "internal_daily_loss_cap"
        return result

    initial_balance = max(float(cfg.get("initial_balance_usd", 0.0) or 0.0), 0.0)
    hard_shield_pct = min(max(float(cfg.get("hard_dynamic_shield_pct", 0.0) or 0.0), 0.0), 100.0)
    balance = float(account.get("balance", 0.0) or 0.0)
    equity = float(account.get("equity", 0.0) or 0.0)
    projected_equity = equity - estimated_loss
    result["checks"]["projected_equity"] = projected_equity

    if initial_balance > 0.0 and hard_shield_pct > 0.0:
        observed_peak = max(initial_balance, balance, equity)
        hard_floor = min(observed_peak * (1.0 - hard_shield_pct / 100.0), initial_balance)
        result["checks"]["observed_dynamic_shield_floor"] = hard_floor
        if projected_equity <= hard_floor:
            result["reason"] = "dynamic_risk_shield_buffer"
            return result

    internal_equity_buffer = max(float(cfg.get("internal_equity_buffer_usd", 0.0) or 0.0), 0.0)
    if initial_balance > 0.0 and internal_equity_buffer > 0.0:
        internal_floor = max(initial_balance - internal_equity_buffer, balance - internal_equity_buffer)
        result["checks"]["internal_equity_floor"] = internal_floor
        if projected_equity <= internal_floor:
            result["reason"] = "internal_equity_buffer"
            return result

    result["ok"] = True
    result["reason"] = "all_prop_firm_checks_passed"
    result["sized_volume"] = effective_volume
    result["sizing"] = sizing
    return result


def _place_programmed_order(
    adapter: MT5Adapter,
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    plan: Dict[str, Any],
    expiration_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    exec_cfg = (trading_cfg.get("execution") or {})
    symbol = str(exec_cfg.get("symbol") or app_config.get("symbol") or "XAUUSD")
    stop_loss, take_profit, _risk_meta = _order_risk_levels_for_plan(plan, trading_cfg)
    return adapter.place_programmed_order(
        symbol=symbol,
        side=str(plan.get("decision", "hold")).lower(),
        volume=_order_volume_for_plan(plan, trading_cfg),
        entry=float(plan.get("entry", 0.0)),
        stop_loss=stop_loss,
        take_profit=take_profit,
        expiration_utc=expiration_utc,
    )


def _is_tsmm_trade_record(record: Dict[str, Any]) -> bool:
    payload = dict(record or {})
    comment = str(payload.get("comment") or "")
    magic = int(payload.get("magic", 0) or 0)
    return ("TSMM" in comment) or (magic in {7070001, 7070002})


def _trade_record_side(record: Dict[str, Any]) -> str:
    payload = dict(record or {})
    explicit_side = str(payload.get("side") or "").strip().lower()
    if explicit_side in {"buy", "sell"}:
        return explicit_side

    record_type = int(payload.get("type", -1) or -1)
    if record_type in {0, 2, 4, 6}:
        return "buy"
    if record_type in {1, 3, 5, 7, -1}:
        return "sell"
    return "unknown"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(float(value))
    except Exception:
        return int(default)


def _intentional_same_side_stack_policy(plan: Dict[str, Any], trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    tj = (trading_cfg.get("trading_job") or {})
    stack_cfg = dict((tj.get("intentional_same_side_stacking") or {}))

    enabled = bool(stack_cfg.get("enabled", False))
    max_same_side_orders = max(_safe_int(stack_cfg.get("max_same_side_orders", 1), 1), 1)
    auto_enabled = bool(stack_cfg.get("auto_enable_on_high_confidence", False))
    auto_stack_confidence_threshold = _safe_float(stack_cfg.get("auto_stack_confidence_threshold", 0.0), 0.0)
    auto_stack_target_orders = max(_safe_int(stack_cfg.get("auto_stack_target_orders", 1), 1), 1)

    explicit_keys = (
        "intentional_same_side_orders",
        "intended_same_side_orders",
        "same_side_order_count",
        "stack_order_count",
        "intended_order_count",
        "target_same_side_orders",
    )
    explicit_target_orders = 0
    for key in explicit_keys:
        explicit_target_orders = max(explicit_target_orders, _safe_int((plan or {}).get(key), 0))

    execution_intent = (plan or {}).get("execution_intent")
    if isinstance(execution_intent, dict):
        for key in explicit_keys:
            explicit_target_orders = max(explicit_target_orders, _safe_int(execution_intent.get(key), 0))

    confidence = _safe_float((plan or {}).get("confidence"), 0.0)
    target_orders = 1
    reason = "duplicate_guard_default"

    if enabled:
        if explicit_target_orders > 1:
            target_orders = min(max(explicit_target_orders, 1), max_same_side_orders)
            reason = "explicit_stack_intent"
        elif auto_enabled and auto_stack_target_orders > 1 and confidence >= auto_stack_confidence_threshold:
            target_orders = min(auto_stack_target_orders, max_same_side_orders)
            reason = "high_confidence_auto_stack"

    allowed_existing_similar_orders = max(target_orders - 1, 0)
    return {
        "enabled": enabled,
        "target_orders": target_orders,
        "allowed_existing_similar_orders": allowed_existing_similar_orders,
        "max_same_side_orders": max_same_side_orders,
        "explicit_target_orders": explicit_target_orders,
        "auto_enabled": auto_enabled,
        "auto_stack_confidence_threshold": auto_stack_confidence_threshold,
        "auto_stack_target_orders": auto_stack_target_orders,
        "confidence": confidence,
        "reason": reason,
    }


def _forced_agent_a_plan_from_env() -> Dict[str, Any]:
    raw = str(os.environ.get("TSMM_FORCE_AGENT_A_PLAN_JSON", "") or "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _apply_forced_agent_a_plan_override(
    base_plan: Dict[str, Any],
    forced_plan: Dict[str, Any],
    trading_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    plan = dict(base_plan or {})
    override = dict(forced_plan or {})

    decision = str(override.get("decision") or "").strip().lower()
    if decision in {"buy", "sell"}:
        plan["decision"] = decision

    model_name = str(override.get("model") or "").strip()
    if model_name:
        plan["model"] = model_name

    numeric_keys = (
        "entry",
        "stop_loss",
        "take_profit",
        "volume",
        "confidence",
        "cm_accuracy",
        "signal_score",
        "success_probability",
        "input_fooling_risk",
    )
    for key in numeric_keys:
        if key not in override:
            continue
        value = _safe_float(override.get(key), _safe_float(plan.get(key), 0.0))
        if key in {"entry", "volume"} and value <= 0.0:
            continue
        if key in {"stop_loss", "take_profit"} and value <= 0.0:
            continue
        plan[key] = value

    for key in ("analysis_grounding_timeframe", "analysis_grounding_timeframe_minutes", "order_submission_mode"):
        if key in override and str(override.get(key) or "").strip():
            plan[key] = override.get(key)

    plan["forced_plan_override"] = True
    plan["forced_plan_override_source"] = str(override.get("source") or "TSMM_FORCE_AGENT_A_PLAN_JSON")
    notes = list(plan.get("risk_notes") or [])
    note = "Agent A plan override applied from TSMM_FORCE_AGENT_A_PLAN_JSON."
    if note not in notes:
        notes.append(note)
    plan["risk_notes"] = notes

    if "volume" not in plan or _safe_float(plan.get("volume"), 0.0) <= 0.0:
        exec_cfg = (trading_cfg.get("execution") or {})
        plan["volume"] = _safe_float(exec_cfg.get("default_volume", 0.01), 0.01)

    return plan


def _opposing_countertrade_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict((trading_cfg.get("opposing_countertrade") or {}))
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "allow_source_buy": bool(cfg.get("allow_source_buy", False)),
        "min_source_confidence": _safe_float(cfg.get("min_source_confidence", 0.55), 0.55),
        "target_price_tolerance_abs": max(_safe_float(cfg.get("target_price_tolerance_abs", 2.0), 2.0), 0.0),
        "stop_distance_multiplier": max(_safe_float(cfg.get("stop_distance_multiplier", 1.0), 1.0), 0.1),
        "launch_when_source_submission_mode": str(cfg.get("launch_when_source_submission_mode", "programmed") or "programmed").strip().lower() or "programmed",
        "enforce_mirror_parity": bool(cfg.get("enforce_mirror_parity", True)),
        "mirror_parity_wait_seconds": max(int(cfg.get("mirror_parity_wait_seconds", 45) or 45), 1),
        "mirror_parity_poll_seconds": max(int(cfg.get("mirror_parity_poll_seconds", 3) or 3), 1),
        "mirror_parity_entry_tolerance_abs": max(_safe_float(cfg.get("mirror_parity_entry_tolerance_abs", 0.15), 0.15), 0.0),
        "mirror_parity_volume_tolerance": max(_safe_float(cfg.get("mirror_parity_volume_tolerance", 1e-9), 1e-9), 0.0),
        "mirror_parity_tsmm_only": bool(cfg.get("mirror_parity_tsmm_only", True)),
    }


def _build_opposing_countertrade_plan(
    source_plan: Dict[str, Any],
    trading_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = _opposing_countertrade_cfg(trading_cfg)
    source_decision = str((source_plan or {}).get("decision") or "").strip().lower()
    if source_decision not in {"buy", "sell"}:
        return {"ok": False, "reason": "source_plan_not_directional"}
    if source_decision == "buy" and not bool(cfg.get("allow_source_buy", False)):
        return {"ok": False, "reason": "source_buy_not_enabled"}

    source_confidence = _safe_float((source_plan or {}).get("confidence"), 0.0)
    min_confidence = _safe_float(cfg.get("min_source_confidence", 0.55), 0.55)
    if source_confidence < min_confidence:
        return {
            "ok": False,
            "reason": f"source_confidence_below_threshold:{source_confidence:.4f}<{min_confidence:.4f}",
        }

    source_entry = _safe_float((source_plan or {}).get("entry"), 0.0)
    source_take_profit = _safe_float((source_plan or {}).get("take_profit"), 0.0)
    source_stop_loss = _safe_float((source_plan or {}).get("stop_loss"), 0.0)
    if source_entry <= 0.0 or source_take_profit <= 0.0:
        return {"ok": False, "reason": "source_entry_or_take_profit_missing"}

    risk_cfg = (trading_cfg.get("risk") or {})
    stop_loss_pct = max(_safe_float(risk_cfg.get("stop_loss_pct", 0.8), 0.8), 0.01) / 100.0
    stop_multiplier = _safe_float(cfg.get("stop_distance_multiplier", 1.0), 1.0)

    if source_decision == "sell":
        target_entry = source_take_profit
        if target_entry >= source_entry:
            return {"ok": False, "reason": "sell_plan_target_not_below_entry"}
        opposite_decision = "buy"
        opposite_take_profit = source_entry
        if source_stop_loss > source_entry:
            source_stop_distance = abs(source_stop_loss - source_entry)
        else:
            source_stop_distance = max(source_entry * stop_loss_pct, abs(source_entry - target_entry) * 0.5)
        opposite_stop_loss = target_entry - (source_stop_distance * stop_multiplier)
    else:
        target_entry = source_take_profit
        if target_entry <= source_entry:
            return {"ok": False, "reason": "buy_plan_target_not_above_entry"}
        opposite_decision = "sell"
        opposite_take_profit = source_entry
        if 0.0 < source_stop_loss < source_entry:
            source_stop_distance = abs(source_entry - source_stop_loss)
        else:
            source_stop_distance = max(source_entry * stop_loss_pct, abs(target_entry - source_entry) * 0.5)
        opposite_stop_loss = target_entry + (source_stop_distance * stop_multiplier)

    if opposite_stop_loss <= 0.0:
        return {"ok": False, "reason": "opposite_stop_loss_invalid"}

    exec_cfg = (trading_cfg.get("execution") or {})
    volume = _safe_float((source_plan or {}).get("volume"), _safe_float(exec_cfg.get("default_volume", 0.01), 0.01))
    if volume <= 0.0:
        volume = _safe_float(exec_cfg.get("default_volume", 0.01), 0.01)

    source_model = str((source_plan or {}).get("model") or "agent_a")
    counter_plan = {
        "decision": opposite_decision,
        "entry": round(float(target_entry), 6),
        "stop_loss": round(float(opposite_stop_loss), 6),
        "take_profit": round(float(opposite_take_profit), 6),
        "volume": round(float(volume), 6),
        "confidence": float(source_confidence),
        "cm_accuracy": _safe_float((source_plan or {}).get("cm_accuracy"), 0.0),
        "signal_score": _safe_float((source_plan or {}).get("signal_score"), 0.0),
        "model": f"opposing_countertrade_from_{source_model}",
        "forced_plan_source": "opposing_countertrade",
        "source_decision": source_decision,
        "source_entry": float(source_entry),
        "source_take_profit": float(source_take_profit),
        "source_confidence": float(source_confidence),
    }

    return {
        "ok": True,
        "source_decision": source_decision,
        "source_confidence": float(source_confidence),
        "min_source_confidence": float(min_confidence),
        "target_entry": float(target_entry),
        "plan": counter_plan,
    }


def _is_countertrade_target_reached(
    source_decision: str,
    current_price: Optional[float],
    target_entry: float,
    tolerance_abs: float,
) -> bool:
    if current_price is None:
        return False
    current_value = _safe_float(current_price, 0.0)
    target_value = _safe_float(target_entry, 0.0)
    if current_value <= 0.0 or target_value <= 0.0:
        return False

    tol = max(_safe_float(tolerance_abs, 0.0), 0.0)
    side = str(source_decision or "").strip().lower()
    if side == "sell":
        return current_value <= (target_value + tol)
    if side == "buy":
        return current_value >= (target_value - tol)
    return False


def _symbol_mid_price(adapter: MT5Adapter, symbol: str) -> Optional[float]:
    mt5 = getattr(adapter, "_mt5", None)
    if mt5 is None:
        return None
    try:
        tick = mt5.symbol_info_tick(str(symbol or ""))
    except Exception:
        return None
    if tick is None:
        return None

    bid = _safe_float(getattr(tick, "bid", 0.0), 0.0)
    ask = _safe_float(getattr(tick, "ask", 0.0), 0.0)
    if bid > 0.0 and ask > 0.0:
        return (bid + ask) / 2.0
    if bid > 0.0:
        return bid
    if ask > 0.0:
        return ask
    return None


def _launch_opposing_countertrade_start(
    trading_cfg: Dict[str, Any],
    logger,
    counter_plan: Dict[str, Any],
    submission_mode: str,
) -> Dict[str, Any]:
    new_job_id = _new_job_id(trading_cfg)
    env = os.environ.copy()
    env.setdefault("CONFIG_PATH", "config/config.yaml")
    env["TRADING_CONFIG_PATH"] = _current_trading_config_path()
    env["TSMM_AGENT_A_AUTO_CREATED"] = "1"
    env["TSMM_FORCE_AGENT_A_PLAN_JSON"] = json.dumps(counter_plan, ensure_ascii=True, separators=(",", ":"), default=str)
    env["TSMM_SUPPRESS_OPPOSING_COUNTERTRADE"] = "1"
    env.pop("TSMM_ACCOUNT_MIRROR_SUPPRESS", None)
    env.pop("TSMM_ACCOUNT_MIRROR_SOURCE_JOB_ID", None)
    env.pop("TSMM_ACCOUNT_MIRROR_SOURCE_CONFIG_PATH", None)
    env.pop("TSMM_ACCOUNT_MIRROR_SOURCE_PROFILE", None)

    command = [
        sys.executable,
        "app.py",
        "trading-job",
        "start",
        "--job-id",
        new_job_id,
        "--submission-mode",
        str(submission_mode or "programmed").strip().lower(),
        "--autonomous-trigger",
        "opposing_countertrade",
    ]

    cmds = list(command)
    if os.name == "nt" and len(cmds) > 0 and "python" in str(cmds[0]).lower():
        pass  # CREATE_NO_WINDOW handled by sitecustomize
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    try:
        proc = subprocess.Popen(
            cmds,
            cwd=str(_project_root()),
            env=env,
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        if logger is not None:
            logger.exception("Failed to launch opposing countertrade start")
        return {"ok": False, "error": str(exc), "job_id": new_job_id, "command": command}

    return {"ok": True, "job_id": new_job_id, "pid": int(proc.pid), "command": command}


def _maybe_launch_opposing_countertrade(
    adapter: MT5Adapter,
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    logger,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    prior = dict((state.get("opposing_countertrade") or {}) if isinstance(state, dict) else {})
    if bool(prior.get("attempted", False)):
        return prior

    result: Dict[str, Any] = {
        "attempted": True,
        "timestamp_utc": _iso(_now_utc()),
        "ok": False,
        "skipped": True,
    }

    cfg = _opposing_countertrade_cfg(trading_cfg)
    if not bool(cfg.get("enabled", False)):
        result["reason"] = "disabled"
        state["opposing_countertrade"] = result
        return result

    if _truthy_env("TSMM_SUPPRESS_OPPOSING_COUNTERTRADE"):
        result["reason"] = "suppressed_by_env"
        state["opposing_countertrade"] = result
        return result

    request_ctx = dict((state.get("request_context") or {}) if isinstance(state, dict) else {})
    autonomous_trigger = str((request_ctx.get("autonomous_trigger") or "")).strip().lower()
    if autonomous_trigger == "opposing_countertrade":
        result["reason"] = "autonomous_countertrade_job"
        state["opposing_countertrade"] = result
        return result

    mirror_state = dict((state.get("mirror") or {}) if isinstance(state, dict) else {})
    if str(mirror_state.get("role") or "").strip().lower() == "mirror":
        result["reason"] = "mirror_role_skips_countertrade"
        state["opposing_countertrade"] = result
        return result

    source_submission_mode = str(state.get("order_submission_mode") or "").strip().lower()
    required_mode = str(cfg.get("launch_when_source_submission_mode") or "").strip().lower()
    if required_mode in {"market", "programmed"} and source_submission_mode != required_mode:
        result["reason"] = f"source_submission_mode_not_{required_mode}"
        state["opposing_countertrade"] = result
        return result

    source_order = dict((state.get("order") or {}) if isinstance(state, dict) else {})
    if int(source_order.get("deal_ticket", 0) or 0) > 0:
        result["reason"] = "source_order_already_filled"
        state["opposing_countertrade"] = result
        return result

    source_plan = dict((state.get("plan") or {}) if isinstance(state, dict) else {})
    plan_build = _build_opposing_countertrade_plan(source_plan, trading_cfg)
    if not bool(plan_build.get("ok", False)):
        result["reason"] = str(plan_build.get("reason") or "countertrade_plan_unavailable")
        state["opposing_countertrade"] = result
        return result

    counter_plan = dict(plan_build.get("plan") or {})
    source_decision = str(plan_build.get("source_decision") or "").strip().lower()
    source_confidence = _safe_float(plan_build.get("source_confidence"), 0.0)
    min_confidence = _safe_float(plan_build.get("min_source_confidence"), 0.0)
    target_entry = _safe_float(plan_build.get("target_entry"), 0.0)
    target_side = str(counter_plan.get("decision") or "").strip().lower()

    exec_cfg = (trading_cfg.get("execution") or {})
    symbol = str(exec_cfg.get("symbol") or app_config.get("symbol") or "XAUUSD")
    target_volume = _safe_float(counter_plan.get("volume"), _safe_float(exec_cfg.get("default_volume", 0.01), 0.01))
    tj = (trading_cfg.get("trading_job") or {})

    dedup = _find_similar_mt5_exposure(
        adapter=adapter,
        symbol=symbol,
        side=target_side,
        entry=target_entry,
        volume=target_volume,
        entry_tolerance=_safe_float(tj.get("duplicate_entry_tolerance", 0.15), 0.15),
        volume_tolerance=_safe_float(tj.get("duplicate_volume_tolerance", 1e-9), 1e-9),
        tsmm_only=bool(tj.get("duplicate_tsmm_only", True)),
    )
    if dedup.get("pending_orders") or dedup.get("open_positions"):
        result["reason"] = "existing_opposite_exposure"
        result["dedup"] = dedup
        state["opposing_countertrade"] = result
        return result

    current_price = _symbol_mid_price(adapter, symbol)
    near_target = _is_countertrade_target_reached(
        source_decision=source_decision,
        current_price=current_price,
        target_entry=target_entry,
        tolerance_abs=_safe_float(cfg.get("target_price_tolerance_abs", 2.0), 2.0),
    )
    counter_submission_mode = "market" if near_target else "programmed"

    launch = _launch_opposing_countertrade_start(
        trading_cfg=trading_cfg,
        logger=logger,
        counter_plan=counter_plan,
        submission_mode=counter_submission_mode,
    )

    result.update(
        {
            "source_decision": source_decision,
            "source_confidence": source_confidence,
            "min_source_confidence": min_confidence,
            "target_entry": target_entry,
            "current_price": current_price,
            "near_target": bool(near_target),
            "counter_plan": counter_plan,
            "counter_submission_mode": counter_submission_mode,
            "launch": launch,
        }
    )

    signal_json_path = str(state.get("signal_json_path") or "").strip()
    resolved_job_id = str(state.get("job_id") or "").strip()
    if bool(launch.get("ok", False)):
        result["ok"] = True
        result["skipped"] = False
        result["reason"] = "launched"
        state.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_a",
                kind="opposing_countertrade_started",
                message=(
                    "Opposing countertrade automation launched from a high-confidence programmed setup. "
                    f"source_decision={source_decision}, counter_decision={target_side}, "
                    f"counter_mode={counter_submission_mode}, source_entry={source_plan.get('entry')}, "
                    f"counter_entry={target_entry}, current_price={round(float(current_price), 6) if current_price is not None else 'n/a'}, "
                    f"new_job_id={launch.get('job_id')}."
                ),
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={
                    "signal_json_path": signal_json_path,
                    "opposing_countertrade": result,
                },
                force_telegram=True,
                job_id=resolved_job_id,
            )
        )
    else:
        result["ok"] = False
        result["skipped"] = bool(launch.get("skipped", False))
        result["reason"] = str(launch.get("error") or launch.get("reason") or "launch_failed")
        state.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_a",
                kind="opposing_countertrade_failed",
                message=(
                    "Opposing countertrade automation could not start a new trading job. "
                    f"reason={result.get('reason')}"
                ),
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={
                    "signal_json_path": signal_json_path,
                    "opposing_countertrade": result,
                },
                force_telegram=True,
                job_id=resolved_job_id,
            )
        )

    state["opposing_countertrade"] = result
    return result


def _countertrade_mirror_parity_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _opposing_countertrade_cfg(trading_cfg)
    tj = (trading_cfg.get("trading_job") or {})
    default_entry_tolerance = _safe_float(tj.get("duplicate_entry_tolerance", 0.15), 0.15)
    default_volume_tolerance = _safe_float(tj.get("duplicate_volume_tolerance", 1e-9), 1e-9)

    return {
        "enabled": bool(cfg.get("enforce_mirror_parity", True)),
        "wait_seconds": max(int(cfg.get("mirror_parity_wait_seconds", 45) or 45), 1),
        "poll_seconds": max(int(cfg.get("mirror_parity_poll_seconds", 3) or 3), 1),
        "entry_tolerance": max(
            _safe_float(cfg.get("mirror_parity_entry_tolerance_abs", default_entry_tolerance), default_entry_tolerance),
            0.0,
        ),
        "volume_tolerance": max(
            _safe_float(cfg.get("mirror_parity_volume_tolerance", default_volume_tolerance), default_volume_tolerance),
            0.0,
        ),
        "tsmm_only": bool(cfg.get("mirror_parity_tsmm_only", tj.get("duplicate_tsmm_only", True))),
    }


def _revert_local_countertrade_exposure(
    adapter: MT5Adapter,
    state: Dict[str, Any],
    *,
    order_ticket: int,
) -> Dict[str, Any]:
    ticket = int(order_ticket or 0)
    result: Dict[str, Any] = {
        "attempted": True,
        "order_ticket": ticket,
        "ok": False,
        "method": "",
    }

    cancel_res = (
        adapter.cancel_pending_order(ticket)
        if ticket > 0
        else {"ok": False, "skipped": True, "reason": "missing_order_ticket"}
    )
    result["cancel_result"] = cancel_res
    if bool(cancel_res.get("ok", False)) and not bool(cancel_res.get("skipped", False)):
        result["ok"] = True
        result["method"] = "cancel_pending_order"
        return result

    position: Dict[str, Any] = {}
    if ticket > 0:
        lookup_res = adapter.find_position_by_order(ticket)
        result["position_lookup"] = lookup_res
        if bool(lookup_res.get("ok", False)) and isinstance(lookup_res.get("position"), dict):
            position = dict(lookup_res.get("position") or {})

    if not position:
        state_position = dict((state.get("position") or {}) if isinstance(state, dict) else {})
        if int(state_position.get("ticket", 0) or 0) > 0:
            position = state_position
            result["position_lookup_fallback"] = {
                "ok": True,
                "source": "state.position",
                "position": state_position,
            }

    if not position:
        order_block = dict((state.get("order") or {}) if isinstance(state, dict) else {})
        order_position = dict((order_block.get("position") or {}) if isinstance(order_block.get("position"), dict) else {})
        if int(order_position.get("ticket", 0) or 0) > 0:
            position = order_position
            result["position_lookup_fallback"] = {
                "ok": True,
                "source": "state.order.position",
                "position": order_position,
            }

    if position:
        position_ticket = int(position.get("ticket", 0) or 0)
        if position_ticket > 0:
            close_res = adapter.close_position_by_ticket(position_ticket)
            result["close_result"] = close_res
            result["method"] = "close_position_by_ticket"
            result["ok"] = bool(close_res.get("ok", False))
            return result

    if bool(cancel_res.get("ok", False)) and bool(cancel_res.get("skipped", False)):
        result["ok"] = True
        result["method"] = str(cancel_res.get("reason") or "order_not_pending")
        return result

    result["method"] = "revert_failed"
    return result


def _enforce_opposing_countertrade_mirror_parity(
    adapter: MT5Adapter,
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "attempted": False,
        "timestamp_utc": _iso(_now_utc()),
        "ok": False,
        "skipped": True,
    }

    request_ctx = dict((state.get("request_context") or {}) if isinstance(state, dict) else {})
    autonomous_trigger = str((request_ctx.get("autonomous_trigger") or "")).strip().lower()
    if autonomous_trigger != "opposing_countertrade":
        result["reason"] = "not_opposing_countertrade_trigger"
        return result

    result["attempted"] = True
    parity_cfg = _countertrade_mirror_parity_cfg(trading_cfg)
    if not bool(parity_cfg.get("enabled", False)):
        result["reason"] = "parity_guard_disabled"
        return result

    order = dict((state.get("order") or {}) if isinstance(state, dict) else {})
    order_ticket = int(order.get("order_ticket", 0) or 0)
    if order_ticket <= 0:
        result["reason"] = "missing_order_ticket"
        return result

    plan = dict((state.get("plan") or {}) if isinstance(state, dict) else {})
    exec_cfg = (trading_cfg.get("execution") or {})
    symbol = str(plan.get("symbol") or order.get("symbol") or exec_cfg.get("symbol") or app_config.get("symbol") or "XAUUSD").strip() or "XAUUSD"
    side = str(plan.get("decision") or order.get("side") or "").strip().lower()
    entry = _safe_float(plan.get("entry"), _safe_float(order.get("price_open"), 0.0))
    volume = _safe_float(plan.get("volume"), _safe_float(order.get("volume"), _safe_float(exec_cfg.get("default_volume", 0.01), 0.01)))
    if side not in {"buy", "sell"}:
        result["reason"] = "invalid_plan_side"
        return result
    if entry <= 0.0:
        result["reason"] = "invalid_plan_entry"
        return result

    mirror_state = dict((state.get("mirror") or {}) if isinstance(state, dict) else {})
    peer_cfg_path = str(mirror_state.get("peer_trading_config_path") or "").strip()
    peer_job_id = str(mirror_state.get("peer_job_id") or "").strip()
    peer_profile = str(mirror_state.get("peer_profile") or "").strip()
    if not peer_cfg_path:
        mirror_cfg = _account_mirror_cfg(trading_cfg)
        if bool(mirror_cfg.get("enabled", False)):
            peer_cfg_path = str(mirror_cfg.get("peer_trading_config_path") or "").strip()
            if not peer_profile:
                peer_profile = str(mirror_cfg.get("peer_profile") or "").strip()

    if not peer_cfg_path:
        result["reason"] = "missing_peer_trading_config_path"
        return result
    if os.path.normcase(peer_cfg_path) == os.path.normcase(_current_trading_config_path()):
        result["reason"] = "peer_trading_config_matches_current"
        return result

    result.update(
        {
            "symbol": symbol,
            "side": side,
            "entry": entry,
            "volume": volume,
            "peer_job_id": peer_job_id,
            "peer_trading_config_path": peer_cfg_path,
            "peer_profile": peer_profile or "PEER",
        }
    )

    def _fail_closed(reason: str, *, peer_error: str = "", checks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        failed = dict(result)
        failed["ok"] = False
        failed["skipped"] = False
        failed["reason"] = reason
        if peer_error:
            failed["peer_error"] = str(peer_error)
        if checks is not None:
            failed["parity_checks"] = checks
        failed["reverted"] = True
        failed["local_revert"] = _revert_local_countertrade_exposure(adapter, state, order_ticket=order_ticket)
        if peer_job_id:
            try:
                failed["peer_kill"] = _propagate_mirror_job_action("kill", output_dir, state)
            except Exception as exc:
                failed["peer_kill"] = {"ok": False, "error": str(exc)}
        else:
            failed["peer_kill"] = {"ok": False, "skipped": True, "reason": "missing_peer_job_id"}
        return failed

    try:
        peer_trading_cfg = load_trading_config(peer_cfg_path)
    except Exception as exc:
        return _fail_closed("peer_config_load_failed", peer_error=str(exc))

    if not peer_profile:
        peer_profile = _account_profile_label(peer_trading_cfg)
        result["peer_profile"] = peer_profile

    peer_mt5_cfg = (((peer_trading_cfg.get("broker") or {}).get("mt5") or {}))
    peer_adapter = MT5Adapter(peer_mt5_cfg)
    ok_peer_conn, msg_peer_conn = peer_adapter.connect()
    if not ok_peer_conn:
        return _fail_closed("peer_mt5_connect_failed", peer_error=str(msg_peer_conn))

    checks: List[Dict[str, Any]] = []
    try:
        deadline = time.time() + int(parity_cfg.get("wait_seconds", 45) or 45)
        poll_seconds = max(int(parity_cfg.get("poll_seconds", 3) or 3), 1)
        while True:
            peer_exposure = _find_similar_mt5_exposure(
                adapter=peer_adapter,
                symbol=symbol,
                side=side,
                entry=entry,
                volume=volume,
                entry_tolerance=_safe_float(parity_cfg.get("entry_tolerance", 0.15), 0.15),
                volume_tolerance=_safe_float(parity_cfg.get("volume_tolerance", 1e-9), 1e-9),
                tsmm_only=bool(parity_cfg.get("tsmm_only", True)),
            )
            pending_count = len(peer_exposure.get("pending_orders") or [])
            open_count = len(peer_exposure.get("open_positions") or [])
            checks.append(
                {
                    "checked_at_utc": _iso(_now_utc()),
                    "pending_count": pending_count,
                    "open_count": open_count,
                    "ok": bool(peer_exposure.get("ok", False)),
                }
            )

            if not bool(peer_exposure.get("ok", False)):
                peer_error = str(peer_exposure.get("message") or peer_exposure.get("error") or "peer_exposure_check_failed")
                return _fail_closed("peer_exposure_check_failed", peer_error=peer_error, checks=checks)

            if pending_count > 0 or open_count > 0:
                result.update(
                    {
                        "ok": True,
                        "skipped": False,
                        "reason": "peer_exposure_detected",
                        "peer_exposure": peer_exposure,
                        "parity_checks": checks,
                        "reverted": False,
                    }
                )
                return result

            if time.time() >= deadline:
                break
            time.sleep(poll_seconds)
    finally:
        peer_adapter.shutdown()

    return _fail_closed("peer_exposure_not_detected_before_timeout", checks=checks)


def _finalize_countertrade_parity_reverted_state(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    state: Dict[str, Any],
    state_file: str,
    signal_json_path: str,
    resolved_job_id: str,
    parity_result: Dict[str, Any],
) -> Dict[str, Any]:
    local_revert = dict((parity_result.get("local_revert") or {}) if isinstance(parity_result, dict) else {})
    local_ok = bool(local_revert.get("ok", False))
    reason = str(parity_result.get("reason") or "parity_check_failed")
    peer_profile = str(parity_result.get("peer_profile") or "PEER")
    local_action = str(local_revert.get("method") or "none")

    state.setdefault("notifications", []).append(
        _notify(
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            channel="agent_a",
            kind="opposing_countertrade_parity_reverted",
            message=(
                "Opposing countertrade parity guard reverted the local MT5 exposure because linked-account parity "
                f"was not observed in time. peer_profile={peer_profile}, reason={reason}, local_action={local_action}."
            ),
            requires_approval=False,
            emergency=True,
            approval_deadline_utc=None,
            metadata={
                "signal_json_path": signal_json_path,
                "order": state.get("order"),
                "opposing_countertrade_parity": parity_result,
            },
            force_telegram=True,
            job_id=resolved_job_id,
        )
    )

    state["status"] = "completed" if local_ok else "failed"
    state["closed_reason"] = "opposing_countertrade_parity_reverted" if local_ok else "opposing_countertrade_parity_revert_failed"
    state["ended_at"] = _iso(_now_utc())
    _save_job_state(output_dir, trading_cfg, state_file, state)
    return state


def _retcode_failure_hint(retcode: Any) -> str:
    code = str(retcode or "").strip()
    hints = {
        "10027": "client autotrading disabled; verify MT5 AutoTrading and account permissions",
        "10018": "market closed",
        "10016": "invalid stops",
        "10030": "invalid order filling mode",
        "10013": "invalid request/filling for broker",
    }
    return str(hints.get(code, "") or "").strip()


def _append_retcode_hint(message: str, retcode: Any) -> str:
    text = str(message or "").strip()
    if not text:
        return text
    hint = _retcode_failure_hint(retcode)
    if not hint:
        return text
    if hint.lower() in text.lower():
        return text
    return f"{text} ({hint})"


def _order_failure_message(order_res: Dict[str, Any]) -> str:
    payload = dict(order_res or {})
    for key in ("message", "error", "reason", "details"):
        value = str(payload.get(key) or "").strip()
        if value:
            retcode_match = re.search(r"retcode\s*=\s*(\d+)", value, flags=re.IGNORECASE)
            if retcode_match:
                return _append_retcode_hint(value, retcode_match.group(1))
            return value

    retcode = payload.get("retcode")
    retcode_text = str(retcode).strip() if retcode is not None else ""
    comment = str(payload.get("comment") or payload.get("mt5_comment") or "").strip()
    if retcode_text:
        base = f"retcode={retcode_text} ({comment})" if comment else f"retcode={retcode_text}"
        return _append_retcode_hint(base, retcode_text)

    return "unknown"


def _find_similar_mt5_exposure(
    adapter: MT5Adapter,
    symbol: str,
    side: str,
    entry: float,
    volume: float,
    entry_tolerance: float,
    volume_tolerance: float,
    *,
    tsmm_only: bool = True,
) -> Dict[str, Any]:
    target_symbol = str(symbol or "").strip()
    target_side = str(side or "").strip().lower()
    target_entry = _safe_float(entry, 0.0)
    target_volume = _safe_float(volume, 0.0)
    entry_tol = max(_safe_float(entry_tolerance, 0.15), 0.0)
    volume_tol = max(_safe_float(volume_tolerance, 1e-9), 0.0)

    pending_matches: List[Dict[str, Any]] = []
    open_position_matches: List[Dict[str, Any]] = []

    pending_res = adapter.list_pending_orders()
    if bool(pending_res.get("ok", False)):
        for order in (pending_res.get("orders") or []):
            if not isinstance(order, dict):
                continue
            if tsmm_only and not _is_tsmm_trade_record(order):
                continue
            if target_symbol and str(order.get("symbol") or "").strip() != target_symbol:
                continue
            if target_side and _trade_record_side(order) != target_side:
                continue

            order_entry = _safe_float(order.get("price_open"), 0.0)
            if order_entry <= 0.0:
                continue
            if abs(order_entry - target_entry) > entry_tol:
                continue

            order_volume = _safe_float(order.get("volume"), 0.0)
            if target_volume > 0.0 and abs(order_volume - target_volume) > volume_tol:
                continue

            match = dict(order)
            match["entry_gap"] = round(abs(order_entry - target_entry), 6)
            pending_matches.append(match)

    open_positions_res = adapter.list_open_positions()
    if bool(open_positions_res.get("ok", False)):
        for position in (open_positions_res.get("positions") or []):
            if not isinstance(position, dict):
                continue
            if tsmm_only and not _is_tsmm_trade_record(position):
                continue
            if target_symbol and str(position.get("symbol") or "").strip() != target_symbol:
                continue
            if target_side and _trade_record_side(position) != target_side:
                continue

            position_entry = _safe_float(position.get("price_open"), 0.0)
            if position_entry <= 0.0:
                continue
            if abs(position_entry - target_entry) > entry_tol:
                continue

            position_volume = _safe_float(position.get("volume"), 0.0)
            if target_volume > 0.0 and abs(position_volume - target_volume) > volume_tol:
                continue

            match = dict(position)
            match["entry_gap"] = round(abs(position_entry - target_entry), 6)
            open_position_matches.append(match)

    pending_matches.sort(key=lambda item: (_safe_float(item.get("entry_gap"), 0.0), int(item.get("order_ticket", 0) or 0)))
    open_position_matches.sort(key=lambda item: (_safe_float(item.get("entry_gap"), 0.0), int(item.get("ticket", 0) or 0)))

    return {
        "ok": True,
        "pending_orders": pending_matches,
        "open_positions": open_position_matches,
        "checked_at_utc": _iso(_now_utc()),
    }


def _find_symbol_tsmm_exposure(
    adapter: MT5Adapter,
    symbol: str,
    *,
    tsmm_only: bool = True,
) -> Dict[str, Any]:
    target_symbol = str(symbol or "").strip()
    pending_matches: List[Dict[str, Any]] = []
    open_position_matches: List[Dict[str, Any]] = []

    pending_res = adapter.list_pending_orders()
    if bool(pending_res.get("ok", False)):
        for order in (pending_res.get("orders") or []):
            if not isinstance(order, dict):
                continue
            if tsmm_only and not _is_tsmm_trade_record(order):
                continue
            if target_symbol and str(order.get("symbol") or "").strip() != target_symbol:
                continue
            pending_matches.append(dict(order))

    open_positions_res = adapter.list_open_positions()
    if bool(open_positions_res.get("ok", False)):
        for position in (open_positions_res.get("positions") or []):
            if not isinstance(position, dict):
                continue
            if tsmm_only and not _is_tsmm_trade_record(position):
                continue
            if target_symbol and str(position.get("symbol") or "").strip() != target_symbol:
                continue
            open_position_matches.append(dict(position))

    pending_matches.sort(key=lambda item: int(item.get("order_ticket", 0) or 0))
    open_position_matches.sort(key=lambda item: int(item.get("ticket", 0) or 0))

    return {
        "ok": True,
        "pending_orders": pending_matches,
        "open_positions": open_position_matches,
        "checked_at_utc": _iso(_now_utc()),
    }


def _programmed_order_expiration_minutes(
    trading_cfg: Dict[str, Any],
    autonomous_trigger: str = "",
) -> int:
    tj = (trading_cfg.get("trading_job") or {})
    trigger = str(autonomous_trigger or "").strip().lower()
    if trigger == "autonomous_followup":
        autonomy = dict(trading_cfg.get("autonomous_trading") or {})
        raw_minutes = autonomy.get("opportunity_order_expiration_minutes")
        if raw_minutes is not None:
            return max(int(raw_minutes or 1), 1)
    raw_minutes = tj.get("programmed_order_expiration_minutes", tj.get("max_wait_fill_minutes", 420))
    return max(int(raw_minutes or 1), 1)


def _programmed_order_expiration_utc(
    trading_cfg: Dict[str, Any],
    now_utc: Optional[datetime] = None,
    autonomous_trigger: str = "",
) -> datetime:
    base_now = now_utc or _now_utc()
    return base_now + timedelta(
        minutes=_programmed_order_expiration_minutes(trading_cfg, autonomous_trigger)
    )


def _timeframe_to_minutes(raw_timeframe: Any) -> Optional[int]:
    token = str(raw_timeframe or "").strip().lower()
    mapping = {
        "1m": 1,
        "5m": 5,
        "10m": 10,
        "15m": 15,
        "30m": 30,
        "45m": 45,
        "1h": 60,
        "2h": 120,
        "3h": 180,
        "4h": 240,
        "6h": 360,
        "7h": 420,
        "8h": 480,
        "12h": 720,
        "24h": 1440,
        "1d": 1440,
        "1w": 10080,
    }
    return mapping.get(token)


def _parse_state_datetime(raw_value: Any) -> Optional[datetime]:
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def resolve_trading_start_request(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    requested_submission_mode: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    current_now = now_utc or _now_utc()
    tj = (trading_cfg.get("trading_job") or {})
    session_hours = max(int(tj.get("session_hours", 7) or 7), 1)
    session_start = current_now - timedelta(hours=session_hours)
    first_grounding = str(tj.get("first_session_grounding_timeframe") or "7h").strip().lower() or "7h"
    repeat_grounding = str(tj.get("repeat_session_grounding_timeframe") or "3h").strip().lower() or "3h"
    live_grounding_candidates = [
        str(x).strip().lower()
        for x in (tj.get("live_request_grounding_timeframes") or ["1h", "30m", "10m"])
        if str(x).strip()
    ] or ["1h", "30m", "10m"]

    registry = _load_state(str(_job_registry_path(output_dir, trading_cfg)))
    prior_jobs = 0
    for meta in (registry.get("jobs") or {}).values():
        item = dict(meta or {})
        started_at = _parse_state_datetime(item.get("started_at"))
        if started_at is None or started_at < session_start or started_at > current_now:
            continue
        status = str(item.get("status") or "").strip().lower()
        if status in {"killed", "stopped"}:
            continue
        prior_jobs += 1

    requested_mode = str(requested_submission_mode or "").strip().lower()
    if requested_mode not in {"programmed", "market"}:
        requested_mode = str((tj.get("order_submission_mode") or "programmed")).strip().lower()
    if requested_mode not in {"programmed", "market"}:
        requested_mode = "programmed"

    session_operation_index = prior_jobs + 1
    is_first_operation = session_operation_index == 1
    effective_submission_mode = "programmed" if is_first_operation else requested_mode

    if is_first_operation:
        grounding_timeframe = first_grounding
        grounding_reason = "first_session_operation"
    elif effective_submission_mode == "market":
        grounding_timeframe = live_grounding_candidates[0]
        grounding_reason = "explicit_live_request"
    else:
        grounding_timeframe = repeat_grounding
        grounding_reason = "followup_session_operation"

    grounding_minutes = _timeframe_to_minutes(grounding_timeframe) or _timeframe_to_minutes(first_grounding) or 420
    return {
        "requested_submission_mode": requested_mode,
        "effective_submission_mode": effective_submission_mode,
        "session_operation_index": session_operation_index,
        "is_first_operation_in_session": bool(is_first_operation),
        "grounding_timeframe": grounding_timeframe,
        "grounding_timeframe_minutes": grounding_minutes,
        "grounding_reason": grounding_reason,
        "session_hours": session_hours,
    }


def _autonomous_followup_meets_entry_thresholds(plan: Dict[str, Any], trading_cfg: Dict[str, Any]) -> Tuple[bool, str]:
    cfg = dict(trading_cfg.get("autonomous_trading") or {})

    def _num(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    reasons: List[str] = []
    decision = str((plan or {}).get("decision") or "").strip().lower()
    if decision not in {"buy", "sell"}:
        reasons.append("decision_not_directional")

    success_probability = _num((plan or {}).get("success_probability"), 0.0)
    min_success_probability = _num(cfg.get("min_success_probability_for_followup", 0.58), 0.58)
    if success_probability < min_success_probability:
        reasons.append(f"success_probability<{min_success_probability:.2f}")

    confidence = _num((plan or {}).get("confidence"), 0.0)
    min_confidence = _num(cfg.get("min_confidence_for_followup", 0.55), 0.55)
    if confidence < min_confidence:
        reasons.append(f"confidence<{min_confidence:.2f}")

    cm_accuracy = _num((plan or {}).get("cm_accuracy"), 0.0)
    min_cm_accuracy = _num(cfg.get("min_cm_accuracy_for_followup", 0.52), 0.52)
    if cm_accuracy < min_cm_accuracy:
        reasons.append(f"cm_accuracy<{min_cm_accuracy:.2f}")

    input_fooling_risk = _num((plan or {}).get("input_fooling_risk"), 0.0)
    max_input_fooling_risk = _num(cfg.get("max_input_fooling_risk_for_followup", 0.45), 0.45)
    if input_fooling_risk > max_input_fooling_risk:
        reasons.append(f"input_fooling_risk>{max_input_fooling_risk:.2f}")

    if bool(cfg.get("require_consensus_alignment_for_followup", True)):
        enrichment = dict((plan or {}).get("enrichment") or {})
        alignment = str(enrichment.get("alignment") or "unknown").strip().lower()
        if alignment == "opposed":
            reasons.append("consensus_alignment_opposed")

    if reasons:
        return False, "; ".join(reasons)
    return True, "thresholds_passed"


def _agent_a_order_submission_mode(plan: Dict[str, Any], trading_cfg: Dict[str, Any]) -> str:
    raw_mode = str((plan or {}).get("order_submission_mode") or ((trading_cfg.get("trading_job") or {}).get("order_submission_mode", "programmed")) or "programmed").strip().lower()
    if raw_mode in {"market", "immediate", "direct", "instant"}:
        return "market"
    return "programmed"


def _wait_fill_and_get_position(
    adapter: MT5Adapter,
    order_ticket: int,
    side: str,
    max_wait_sec: int,
    poll_sec: int,
) -> Dict[str, Any]:
    deadline = time.time() + max(max_wait_sec, 1)
    while time.time() <= deadline:
        open_pos = adapter.find_position_by_order(order_ticket)
        if open_pos.get("ok") and open_pos.get("position"):
            return {"filled": True, "position": open_pos.get("position")}
        time.sleep(max(poll_sec, 1))

    return {"filled": False, "position": None, "reason": f"Order not filled within {max_wait_sec}s"}


def _try_programmed_market_fallback(
    *,
    adapter: MT5Adapter,
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    state: Dict[str, Any],
    order_ticket: int,
) -> Dict[str, Any]:
    """Convert an expiring programmed order only while the joint signal is valid."""
    trading_job_cfg = dict(trading_cfg.get("trading_job") or {})
    cfg = dict(trading_job_cfg.get("market_fallback") or {})
    if not bool(cfg.get("enabled", False)):
        return {"ok": False, "skipped": True, "reason": "market_fallback_disabled"}

    request_context = dict(state.get("request_context") or {})
    trigger = str(request_context.get("autonomous_trigger") or "").strip().lower()
    allowed = {
        str(value).strip().lower()
        for value in (cfg.get("allowed_triggers") or ["mandatory_session"])
        if str(value).strip()
    }
    if trigger not in allowed:
        return {"ok": False, "skipped": True, "reason": f"trigger_not_allowed:{trigger or 'manual'}"}

    plan = dict(state.get("plan") or {})
    side = str(plan.get("decision") or "").strip().lower()
    if side not in {"buy", "sell"}:
        return {"ok": False, "skipped": True, "reason": "plan_side_not_directional"}

    timeout = max(float(cfg.get("assessment_timeout_seconds", 5.0) or 5.0), 0.5)
    assessment = _collect_all_model_assessment_signals(trading_cfg, timeout_sec=timeout)
    current_price = _symbol_mid_price(
        adapter,
        str(((trading_cfg.get("execution") or {}).get("symbol") or app_config.get("symbol") or "XAUUSD")),
    )
    policy = evaluate_joint_ohlc_policy(assessment, trading_cfg, market_price=current_price)
    minimum_score = abs(_safe_float(cfg.get("min_direction_score", 0.12), 0.12))
    if str(policy.get("decision") or "hold") != side:
        return {
            "ok": False,
            "skipped": True,
            "reason": "market_fallback_signal_no_longer_matches",
            "assessment": assessment,
            "signal_policy": policy,
        }
    if abs(_safe_float(policy.get("score"), 0.0)) < minimum_score:
        return {
            "ok": False,
            "skipped": True,
            "reason": "market_fallback_signal_below_threshold",
            "assessment": assessment,
            "signal_policy": policy,
        }
    if current_price is None or current_price <= 0.0:
        return {"ok": False, "skipped": True, "reason": "market_fallback_live_price_unavailable"}

    cancel_result = adapter.cancel_pending_order(order_ticket)
    if not bool(cancel_result.get("ok", False)) or bool(cancel_result.get("skipped", False)):
        return {
            "ok": False,
            "skipped": True,
            "reason": "pending_order_not_confirmed_cancelled",
            "cancel_result": cancel_result,
        }

    old_entry = _safe_float(plan.get("entry"), current_price)
    old_stop = plan.get("stop_loss")
    old_take = plan.get("take_profit")
    stop_distance = abs(old_entry - _safe_float(old_stop, old_entry))
    target_distance = abs(_safe_float(old_take, old_entry) - old_entry)
    fallback_plan = dict(plan)
    fallback_plan["entry"] = float(current_price)
    fallback_plan["order_submission_mode"] = "market"
    fallback_plan["market_fallback"] = {
        "enabled": True,
        "source_order_ticket": int(order_ticket),
        "signal_policy": policy,
        "assessment_scope": assessment.get("assessment_scope"),
    }
    if side == "buy":
        fallback_plan["stop_loss"] = current_price - stop_distance if old_stop is not None else None
        fallback_plan["take_profit"] = current_price + target_distance
    else:
        fallback_plan["stop_loss"] = current_price + stop_distance if old_stop is not None else None
        fallback_plan["take_profit"] = current_price - target_distance

    symbol = str(((trading_cfg.get("execution") or {}).get("symbol") or app_config.get("symbol") or "XAUUSD"))
    stop_loss, take_profit, risk_meta = _order_risk_levels_for_plan(fallback_plan, trading_cfg)
    volume = _order_volume_for_plan(fallback_plan, trading_cfg)
    guard = _prop_firm_guard_preflight(
        adapter=adapter,
        trading_cfg=trading_cfg,
        symbol=symbol,
        side=side,
        volume=volume,
        entry=float(current_price),
        stop_loss=stop_loss,
    )
    if not bool(guard.get("ok", False)):
        return {
            "ok": False,
            "skipped": False,
            "reason": f"market_fallback_risk_guard_blocked:{guard.get('reason', 'unknown')}",
            "cancel_result": cancel_result,
            "guard": guard,
            "signal_policy": policy,
        }

    effective_volume = _safe_float(guard.get("sized_volume"), volume)
    fallback_plan["volume"] = effective_volume
    order_result = adapter.place_market_order(
        symbol=symbol,
        side=side,
        volume=effective_volume,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )
    return {
        "ok": bool(order_result.get("ok", False)),
        "skipped": False,
        "reason": "market_fallback_submitted" if order_result.get("ok") else "market_fallback_order_failed",
        "cancel_result": cancel_result,
        "guard": guard,
        "risk": risk_meta,
        "signal_policy": policy,
        "assessment": assessment,
        "plan": fallback_plan,
        "order": order_result,
    }


def _finalize_agent_a_order_submission(
    adapter: MT5Adapter,
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    logger,
    state: Dict[str, Any],
    state_file: str,
    order_res: Dict[str, Any],
    wait_fill_sec: int,
    max_wait_fill_minutes: int,
) -> Dict[str, Any]:
    plan = dict(state.get("plan") or {})
    decision = str(plan.get("decision") or "hold").strip().lower()
    signal_json_path = str(state.get("signal_json_path") or "").strip()
    resolved_job_id = str(state.get("job_id") or "").strip()
    programmed_expiration = str(state.get("programmed_order_expiration_utc") or "").strip()
    normalized_order_res = dict(order_res or {})
    normalized_order_ticket = int(normalized_order_res.get("order_ticket", 0) or 0)
    # Recovered pending orders can be serialized from MT5 state without an explicit "ok" flag.
    if normalized_order_ticket > 0 and not bool(normalized_order_res.get("ok", False)):
        normalized_order_res["ok"] = True

    state.pop("market_reopen_at_utc", None)
    state["order"] = normalized_order_res
    if programmed_expiration:
        state["programmed_order_expiration_utc"] = programmed_expiration
        if isinstance(state.get("order"), dict) and not str((state["order"].get("expiration_utc") or "")).strip():
            state["order"]["expiration_utc"] = programmed_expiration
    if not normalized_order_res.get("ok"):
        failure_detail = _order_failure_message(normalized_order_res)
        state.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_a",
                kind="order_failed",
                message=f"Agent A MT5 order placement failed: {failure_detail}",
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={"signal_json_path": signal_json_path, "order": normalized_order_res},
                force_telegram=True,
                job_id=resolved_job_id,
            )
        )
        state["status"] = "failed"
        state["closed_reason"] = f"order_place_failed: {failure_detail}"
        state["ended_at"] = _iso(_now_utc())
        _save_job_state(output_dir, trading_cfg, state_file, state)
        return state

    if programmed_expiration:
        state.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_a",
                kind="order_placed_pending",
                message=(
                    "Agent A programmed MT5 order was placed successfully. "
                    f"It will expire at {programmed_expiration} UTC if it is not filled before then."
                ),
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={"signal_json_path": signal_json_path, "order": state.get("order")},
                force_telegram=True,
                job_id=resolved_job_id,
            )
        )

    countertrade_res = _maybe_launch_opposing_countertrade(
        adapter=adapter,
        app_config=app_config,
        trading_cfg=trading_cfg,
        output_dir=output_dir,
        logger=logger,
        state=state,
    )
    if countertrade_res:
        state["opposing_countertrade"] = countertrade_res

    parity_res = _enforce_opposing_countertrade_mirror_parity(
        adapter=adapter,
        app_config=app_config,
        trading_cfg=trading_cfg,
        output_dir=output_dir,
        state=state,
    )
    if bool(parity_res.get("attempted", False)):
        state["opposing_countertrade_parity"] = parity_res
        if bool(parity_res.get("reverted", False)):
            return _finalize_countertrade_parity_reverted_state(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                state=state,
                state_file=state_file,
                signal_json_path=signal_json_path,
                resolved_job_id=resolved_job_id,
                parity_result=parity_res,
            )

    if programmed_expiration or bool(countertrade_res.get("attempted", False)) or bool(parity_res.get("attempted", False)):
        _save_job_state(output_dir, trading_cfg, state_file, state)

    order_ticket = int(normalized_order_res.get("order_ticket", 0) or 0)
    fallback_wait_cfg = dict(((trading_cfg.get("trading_job") or {}).get("market_fallback") or {}))
    effective_wait_minutes = max_wait_fill_minutes
    if bool(fallback_wait_cfg.get("enabled", False)):
        effective_wait_minutes = max(
            max_wait_fill_minutes - max(int(fallback_wait_cfg.get("minutes_before_expiry", 15) or 15), 0),
            1,
        )
    filled = _wait_fill_and_get_position(
        adapter=adapter,
        order_ticket=order_ticket,
        side=decision,
        max_wait_sec=effective_wait_minutes * 60,
        poll_sec=wait_fill_sec,
    )

    if not filled.get("filled"):
        fallback = _try_programmed_market_fallback(
            adapter=adapter,
            app_config=app_config,
            trading_cfg=trading_cfg,
            state=state,
            order_ticket=order_ticket,
        ) if order_ticket else {"ok": False, "skipped": True, "reason": "missing_order_ticket"}
        state["market_fallback"] = fallback
        if bool(fallback.get("ok", False)):
            state["plan"] = dict(fallback.get("plan") or plan)
            state["order_submission_mode"] = "market_fallback"
            state.pop("programmed_order_expiration_utc", None)
            state.setdefault("notifications", []).append(
                _notify(
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    channel="agent_a",
                    kind="market_fallback_submitted",
                    message=(
                        "The programmed entry was still supported near expiry, so TSMM cancelled it and "
                        "submitted a guarded market entry after rechecking the joint OHLC signal."
                    ),
                    requires_approval=False,
                    emergency=False,
                    approval_deadline_utc=None,
                    metadata={"signal_json_path": signal_json_path, "market_fallback": fallback},
                    force_telegram=True,
                    job_id=resolved_job_id,
                )
            )
            _save_job_state(output_dir, trading_cfg, state_file, state)
            return _finalize_agent_a_market_order_submission(
                adapter=adapter,
                app_config=app_config,
                trading_cfg=trading_cfg,
                output_dir=output_dir,
                logger=logger,
                state=state,
                state_file=state_file,
                order_res=dict(fallback.get("order") or {}),
            )

        cancel_res = dict(fallback.get("cancel_result") or {})
        if not cancel_res:
            cancel_res = adapter.cancel_pending_order(order_ticket) if order_ticket else {"ok": False, "skipped": True, "reason": "missing_order_ticket"}
        filled["cancel_result"] = cancel_res
        filled["market_fallback"] = fallback
        state.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_a",
                kind="order_pending_timeout",
                message="Agent A order was sent to MT5 but was not filled within the configured session window. The pending MT5 order was canceled to avoid unmanaged later fills.",
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={"signal_json_path": signal_json_path, "order": normalized_order_res, "fill_status": filled},
                force_telegram=True,
                job_id=resolved_job_id,
            )
        )
        state["status"] = "completed"
        state["closed_reason"] = "order_not_filled_in_session"
        state["fill_status"] = filled
        state["ended_at"] = _iso(_now_utc())
        _save_job_state(output_dir, trading_cfg, state_file, state)
        return state

    state["stage"] = "agent_b"
    state["mode"] = "mode_b"
    state["status"] = "agent_b_running"
    state["position"] = filled.get("position")
    state["agent_b_started_at"] = _iso(_now_utc())
    state.setdefault("notifications", []).append(
        _notify(
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            channel="agent_a",
            kind="order_filled",
            message="Agent A MT5 order filled successfully. Agent B supervision has started.",
            requires_approval=False,
            emergency=False,
            approval_deadline_utc=None,
            metadata={"signal_json_path": signal_json_path, "order": normalized_order_res, "position": filled.get('position')},
            force_telegram=True,
            job_id=resolved_job_id,
        )
    )
    _save_job_state(output_dir, trading_cfg, state_file, state)

    mirror_entry_res = _mirror_agent_a_entry_on_peer_preflight_failure(
        app_config=app_config,
        source_trading_cfg=trading_cfg,
        output_dir=output_dir,
        source_state=state,
    )
    if not bool(mirror_entry_res.get("skipped", False)):
        state["mirror_entry"] = mirror_entry_res
        if bool(mirror_entry_res.get("ok", False)):
            mirror_profile = str(mirror_entry_res.get("peer_profile") or "PEER")
            fallback_reason = str(mirror_entry_res.get("fallback_reason") or "preflight failure")
            state.setdefault("notifications", []).append(
                _notify(
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    channel="agent_a",
                    kind="mirror_entry_fallback",
                    message=(
                        f"Linked-account market mirror fallback succeeded for {mirror_profile}. "
                        f"The peer order was placed directly after {fallback_reason}."
                    ),
                    requires_approval=False,
                    emergency=False,
                    approval_deadline_utc=None,
                    metadata={"signal_json_path": signal_json_path, "mirror_entry": mirror_entry_res},
                    force_telegram=True,
                    job_id=resolved_job_id,
                )
            )
        else:
            state.setdefault("notifications", []).append(
                _notify(
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    channel="agent_a",
                    kind="mirror_entry_failed",
                    message=(
                        "Linked-account market mirror fallback failed after peer preflight data-sync failure. "
                        f"error={mirror_entry_res.get('error', mirror_entry_res.get('reason', 'unknown'))}."
                    ),
                    requires_approval=False,
                    emergency=False,
                    approval_deadline_utc=None,
                    metadata={"signal_json_path": signal_json_path, "mirror_entry": mirror_entry_res},
                    force_telegram=True,
                    job_id=resolved_job_id,
                )
            )
        _save_job_state(output_dir, trading_cfg, state_file, state)

    out = _run_agent_b_loop(state, trading_cfg, output_dir, state_file, logger)
    out["followup_agent_a"] = _launch_followup_agent_a_start(
        app_config=app_config,
        trading_cfg=trading_cfg,
        output_dir=output_dir,
        logger=logger,
        completed_state=out,
    )
    _save_job_state(output_dir, trading_cfg, state_file, out)
    return out


def _finalize_agent_a_market_order_submission(
    adapter: MT5Adapter,
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    logger,
    state: Dict[str, Any],
    state_file: str,
    order_res: Dict[str, Any],
) -> Dict[str, Any]:
    signal_json_path = str(state.get("signal_json_path") or "").strip()
    resolved_job_id = str(state.get("job_id") or "").strip()
    normalized_order_res = dict(order_res or {})

    state.pop("market_reopen_at_utc", None)
    state.pop("programmed_order_expiration_utc", None)
    state["order"] = normalized_order_res
    if not normalized_order_res.get("ok"):
        failure_detail = _order_failure_message(normalized_order_res)
        state.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_a",
                kind="order_failed",
                message=f"Agent A MT5 market order placement failed: {failure_detail}",
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={"signal_json_path": signal_json_path, "order": normalized_order_res},
                force_telegram=True,
                job_id=resolved_job_id,
            )
        )
        state["status"] = "failed"
        state["closed_reason"] = f"market_order_place_failed: {failure_detail}"
        state["ended_at"] = _iso(_now_utc())
        _save_job_state(output_dir, trading_cfg, state_file, state)
        return state

    position = normalized_order_res.get("position") if isinstance(normalized_order_res.get("position"), dict) else None
    if not position:
        state["status"] = "failed"
        state["closed_reason"] = "market_order_position_not_found"
        state["ended_at"] = _iso(_now_utc())
        _save_job_state(output_dir, trading_cfg, state_file, state)
        return state

    state["position"] = position
    parity_res = _enforce_opposing_countertrade_mirror_parity(
        adapter=adapter,
        app_config=app_config,
        trading_cfg=trading_cfg,
        output_dir=output_dir,
        state=state,
    )
    if bool(parity_res.get("attempted", False)):
        state["opposing_countertrade_parity"] = parity_res
        if bool(parity_res.get("reverted", False)):
            return _finalize_countertrade_parity_reverted_state(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                state=state,
                state_file=state_file,
                signal_json_path=signal_json_path,
                resolved_job_id=resolved_job_id,
                parity_result=parity_res,
            )

    state["stage"] = "agent_b"
    state["mode"] = "mode_b"
    state["status"] = "agent_b_running"
    state["agent_b_started_at"] = _iso(_now_utc())
    state.setdefault("notifications", []).append(
        _notify(
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            channel="agent_a",
            kind="order_filled",
            message="Agent A market order filled immediately. Agent B supervision has started.",
            requires_approval=False,
            emergency=False,
            approval_deadline_utc=None,
            metadata={"signal_json_path": signal_json_path, "order": normalized_order_res, "position": position},
            force_telegram=True,
            job_id=resolved_job_id,
        )
    )
    _save_job_state(output_dir, trading_cfg, state_file, state)

    out = _run_agent_b_loop(state, trading_cfg, output_dir, state_file, logger)
    out["followup_agent_a"] = _launch_followup_agent_a_start(
        app_config=app_config,
        trading_cfg=trading_cfg,
        output_dir=output_dir,
        logger=logger,
        completed_state=out,
    )
    _save_job_state(output_dir, trading_cfg, state_file, out)
    return out


def _recover_agent_b_state_from_live_position(
    state: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    if str((state or {}).get("ended_at") or "").strip() or str((state or {}).get("closed_reason") or "").strip():
        return {"ok": False, "reason": "job_already_closed"}

    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    order_block = (state.get("order") or {}) if isinstance(state, dict) else {}
    order_ticket = int(order_block.get("order_ticket", 0) or 0)
    position_block = (state.get("position") or {}) if isinstance(state, dict) else {}
    position_ticket = int(position_block.get("ticket", 0) or 0)
    exec_cfg = (trading_cfg.get("execution") or {})
    symbol = str(exec_cfg.get("symbol") or plan.get("symbol") or "XAUUSD")
    decision = str(plan.get("decision") or "").strip().lower()
    volume = float(plan.get("volume", exec_cfg.get("default_volume", 0.01)) or 0.01)
    entry = float(plan.get("entry", 0.0) or 0.0)
    stop_loss = float(plan.get("stop_loss", 0.0) or 0.0)
    take_profit = float(plan.get("take_profit", 0.0) or 0.0)

    if decision not in {"buy", "sell"}:
        return {"ok": False, "reason": "unsupported_plan_decision"}

    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, msg_conn = adapter.connect()
    if not ok_conn:
        return {"ok": False, "reason": f"mt5_connect_failed:{msg_conn}"}

    try:
        found = adapter.get_position_by_ticket(position_ticket) if position_ticket > 0 else {"ok": True, "position": None}
        recovery_method = "position_ticket"
        if found.get("ok") and not found.get("position"):
            found = adapter.find_position_by_order(order_ticket) if order_ticket > 0 else {"ok": True, "position": None}
            recovery_method = "order_ticket"
        if found.get("ok") and not found.get("position"):
            found = adapter.find_live_position_by_plan(
                symbol=symbol,
                volume=volume,
                entry=entry,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            recovery_method = "plan_match"
    finally:
        adapter.shutdown()

    if not found.get("ok"):
        return {"ok": False, "reason": found.get("message", "find_position_failed")}

    position = found.get("position")
    if not position:
        return {"ok": False, "reason": "position_not_found"}

    recovered_state = dict(state)
    recovered_state["stage"] = "agent_b"
    recovered_state["mode"] = "mode_b"
    recovered_state["status"] = "agent_b_running"
    recovered_state["position"] = position
    recovered_state["recovery_method"] = recovery_method
    pos_time = int((position or {}).get("time", 0) or 0)
    recovered_state["agent_b_started_at"] = _iso(datetime.utcfromtimestamp(pos_time)) if pos_time > 0 else _iso(_now_utc())
    recovered_state.pop("ended_at", None)
    recovered_state.pop("closed_reason", None)
    recovered_state.setdefault("notifications", []).append(
        _notify(
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            channel="agent_b",
            kind="position_recovered",
            message="Recovered a live MT5 position from the approved Agent A order and attached it to Agent B supervision.",
            requires_approval=False,
            emergency=False,
            approval_deadline_utc=None,
            metadata={"order_ticket": order_ticket, "position": position, "recovery_method": recovery_method},
            force_telegram=True,
            job_id=str(recovered_state.get("job_id") or "").strip(),
        )
    )
    return {"ok": True, "state": recovered_state, "order_ticket": order_ticket, "position": position}


def _recover_agent_a_state_from_pending_order(
    state: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
) -> Dict[str, Any]:
    if str((state or {}).get("stage") or "").strip().lower() != "agent_a":
        return {"ok": False, "reason": "not_agent_a"}
    if not bool((state or {}).get("agent_a_approved")):
        return {"ok": False, "reason": "agent_a_not_approved"}
    if str((state or {}).get("ended_at") or "").strip() or str((state or {}).get("closed_reason") or "").strip():
        return {"ok": False, "reason": "job_already_closed"}

    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    exec_cfg = (trading_cfg.get("execution") or {})
    symbol = str(exec_cfg.get("symbol") or plan.get("symbol") or "XAUUSD")
    volume = float(plan.get("volume", exec_cfg.get("default_volume", 0.01)) or 0.01)
    entry = float(plan.get("entry", 0.0) or 0.0)
    stop_loss = float(plan.get("stop_loss", 0.0) or 0.0)
    take_profit = float(plan.get("take_profit", 0.0) or 0.0)

    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, msg_conn = adapter.connect()
    if not ok_conn:
        return {"ok": False, "reason": f"mt5_connect_failed:{msg_conn}"}

    try:
        found = adapter.find_pending_order_by_plan(
            symbol=symbol,
            volume=volume,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
    finally:
        adapter.shutdown()

    if not found.get("ok"):
        return {"ok": False, "reason": found.get("message", "find_pending_order_failed")}

    order = found.get("order")
    if not order:
        return {"ok": False, "reason": "pending_order_not_found"}

    recovered_state = dict(state)
    recovered_state["status"] = "agent_a_completed"
    recovered_state["order"] = order
    recovered_state["recovery_method"] = "pending_order_plan_match"
    recovered_state.pop("ended_at", None)
    recovered_state.pop("closed_reason", None)
    recovered_state.setdefault("notifications", []).append(
        _notify(
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            channel="agent_a",
            kind="order_recovered",
            message="Recovered a pending MT5 order from an approved Agent A job and resumed fill monitoring.",
            requires_approval=False,
            emergency=False,
            approval_deadline_utc=None,
            metadata={"order": order, "recovery_method": "pending_order_plan_match"},
            force_telegram=True,
            job_id=str(recovered_state.get("job_id") or "").strip(),
        )
    )
    return {"ok": True, "state": recovered_state, "order": order}


def _agent_b_close_policy(
    side: str,
    consensus: str,
    score: float,
    close_threshold: float,
) -> bool:
    if side == "buy" and consensus == "sell" and score <= -abs(close_threshold):
        return True
    if side == "sell" and consensus == "buy" and score >= abs(close_threshold):
        return True
    return False


def _build_agent_b_management_plan(
    state: Dict[str, Any],
    position: Dict[str, Any],
    consensus: str,
    score: float,
    close_threshold: float,
    now: datetime,
    poll_seconds: int,
    base_deadline: datetime,
    extension_approved: bool,
) -> Dict[str, Any]:
    side = str(((state.get("plan") or {}).get("decision") or "hold")).lower()
    should_close = _agent_b_close_policy(side=side, consensus=consensus, score=score, close_threshold=close_threshold)

    if should_close:
        recommendation = "close_position"
        rationale = (
            f"Mode B consensus flipped against the active {side} position with score {score:.3f}, "
            f"crossing the close threshold {close_threshold:.3f}."
        )
    elif now >= base_deadline and not extension_approved:
        recommendation = "request_extension"
        rationale = "The base session horizon has been reached; extension approval is required to keep managing the position."
    elif consensus == side:
        recommendation = "maintain_position"
        rationale = f"Mode B consensus remains aligned with the active {side} position at score {score:.3f}."
    elif consensus == "hold":
        recommendation = "monitor_closely"
        rationale = "Mode B consensus is neutral, so the position should be watched closely without changing it yet."
    else:
        recommendation = "prepare_defensive_exit"
        rationale = (
            f"Mode B consensus is leaning {consensus} against the active {side} position, "
            f"but score {score:.3f} has not crossed the forced-close threshold {close_threshold:.3f}."
        )

    next_review_at = now + timedelta(seconds=max(int(poll_seconds), 1))
    return {
        "timestamp_utc": _iso(now),
        "position_ticket": int((position or {}).get("ticket", 0) or 0),
        "position_side": side,
        "position_symbol": str((position or {}).get("symbol") or ""),
        "position_volume": float((position or {}).get("volume", 0.0) or 0.0),
        "position_entry": float((position or {}).get("price_open", 0.0) or 0.0),
        "consensus": consensus,
        "consensus_score": float(score),
        "close_threshold": float(close_threshold),
        "recommendation": recommendation,
        "should_close": bool(should_close),
        "close_reason": f"mode_b_consensus_close({consensus},{score:.3f})" if should_close else "",
        "rationale": rationale,
        "base_deadline_utc": _iso(base_deadline),
        "extension_approved": bool(extension_approved),
        "next_review_utc": _iso(next_review_at),
    }


def _opposing_programmed_tp_target(
    state: Dict[str, Any],
    position: Dict[str, Any],
    trading_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    state_path_raw = str((state or {}).get("state_path") or "").strip()
    if not state_path_raw:
        return None

    try:
        state_path = Path(state_path_raw)
        jobs_root = state_path.parent.parent
    except Exception:
        return None

    if not jobs_root.exists() or not jobs_root.is_dir():
        return None

    side = str((((state.get("plan") or {}) if isinstance(state, dict) else {}).get("decision") or "")).strip().lower()
    if side not in {"buy", "sell"}:
        return None

    try:
        entry = float((position or {}).get("price_open") or ((state.get("plan") or {}).get("entry")) or 0.0)
        current_price = float((position or {}).get("price_current") or entry or 0.0)
    except Exception:
        return None

    if entry <= 0.0 or current_price <= 0.0:
        return None

    target_symbol = str((position or {}).get("symbol") or ((state.get("plan") or {}).get("symbol")) or ((trading_cfg.get("execution") or {}).get("symbol")) or "").strip()
    active_job_id = str((state or {}).get("job_id") or "").strip()
    min_delta = float(((trading_cfg.get("mode_b") or {}).get("min_sltp_adjust_abs", 0.05) or 0.05))

    candidate: Optional[Dict[str, Any]] = None
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        job_id = str(job_dir.name or "").strip()
        if not job_id or job_id == active_job_id:
            continue

        peer_state = _load_state(str(job_dir / "trading_job_state.json"))
        if not peer_state:
            continue
        if str((peer_state.get("status") or "")).strip().lower() != "agent_a_completed":
            continue
        if str((peer_state.get("stage") or "")).strip().lower() != "agent_a":
            continue
        if str((peer_state.get("order_submission_mode") or "")).strip().lower() != "programmed":
            continue
        if str((peer_state.get("ended_at") or "")).strip() or str((peer_state.get("closed_reason") or "")).strip():
            continue

        peer_plan = (peer_state.get("plan") or {}) if isinstance(peer_state.get("plan"), dict) else {}
        peer_order = (peer_state.get("order") or {}) if isinstance(peer_state.get("order"), dict) else {}
        if not peer_order:
            continue

        peer_side = str(peer_plan.get("decision") or "").strip().lower()
        if peer_side not in {"buy", "sell"} or peer_side == side:
            continue

        peer_symbol = str(peer_order.get("symbol") or peer_plan.get("symbol") or "").strip()
        if target_symbol and peer_symbol and peer_symbol != target_symbol:
            continue

        try:
            peer_entry = float(peer_order.get("price_open") or peer_plan.get("entry") or 0.0)
        except Exception:
            continue
        if peer_entry <= 0.0:
            continue

        if side == "buy":
            if peer_entry <= entry + min_delta:
                continue
            if peer_entry <= current_price + min_delta:
                continue
            if candidate is None or peer_entry < float(candidate["entry"]):
                candidate = {"entry": peer_entry, "job_id": job_id, "side": peer_side}
        else:
            if peer_entry >= entry - min_delta:
                continue
            if peer_entry >= current_price - min_delta:
                continue
            if candidate is None or peer_entry > float(candidate["entry"]):
                candidate = {"entry": peer_entry, "job_id": job_id, "side": peer_side}

    return candidate


def _agent_b_risk_adjustment(
    state: Dict[str, Any],
    position: Dict[str, Any],
    current_plan: Dict[str, Any],
    trading_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    mb_cfg = (trading_cfg.get("mode_b") or {})
    if not bool(mb_cfg.get("manage_existing_positions", True)):
        return None

    side = str(current_plan.get("position_side") or "").lower()
    if side not in {"buy", "sell"}:
        return None

    risk_cfg = (trading_cfg.get("risk") or {})
    trailing_cfg = (risk_cfg.get("trailing") or {})
    base_plan = (state.get("plan") or {}) if isinstance(state.get("plan"), dict) else {}

    try:
        entry = float((position or {}).get("price_open") or base_plan.get("entry") or 0.0)
        current_price = float((position or {}).get("price_current") or entry or 0.0)
    except Exception:
        return None

    if entry <= 0.0 or current_price <= 0.0:
        return None

    def _float_or_none(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except Exception:
            return None
        return out if abs(out) > 1e-9 else None

    current_sl = _float_or_none((position or {}).get("sl"))
    current_tp = _float_or_none((position or {}).get("tp"))
    base_sl = _float_or_none(base_plan.get("stop_loss"))
    base_tp = _float_or_none(base_plan.get("take_profit"))

    sl_pct = float(risk_cfg.get("stop_loss_pct", 0.8) or 0.8) / 100.0
    tp_pct = float(risk_cfg.get("take_profit_pct", 1.6) or 1.6) / 100.0
    min_delta = float(mb_cfg.get("min_sltp_adjust_abs", 0.05) or 0.05)
    trailing_enabled = bool(trailing_cfg.get("enabled", True))
    trail_gap_candidates = [
        entry * (float(trailing_cfg.get("trail_pct_base", 0.5) or 0.5) / 100.0),
        min_delta,
    ]
    volatility_meta = dict(base_plan.get("volatility_protection") or {})
    atr = _safe_float(volatility_meta.get("atr"), 0.0)
    if atr > 0.0:
        trail_gap_candidates.append(
            atr * max(_safe_float(trailing_cfg.get("trail_atr_multiplier", 1.0), 1.0), 0.1)
        )
    trail_gap = max(trail_gap_candidates)
    base_sl_distance = abs(entry - base_sl) if base_sl is not None else entry * sl_pct
    base_tp_distance = abs(base_tp - entry) if base_tp is not None else entry * tp_pct
    breakeven_activation_ratio = float(trailing_cfg.get("breakeven_activation_ratio", 0.75) or 0.75)
    breakeven_activation_move = max(base_sl_distance * breakeven_activation_ratio, trail_gap)
    favorable_move = current_price - entry if side == "buy" else entry - current_price
    execution_cfg = dict(trading_cfg.get("execution") or {})
    round_trip_bps = 2.0 * (
        _safe_float(execution_cfg.get("spread_bps"), 0.0)
        + _safe_float(execution_cfg.get("slippage_bps"), 0.0)
    )
    cost_buffer = (
        entry * round_trip_bps / 10000.0
        * max(_safe_float(trailing_cfg.get("breakeven_cost_buffer_multiple", 1.0), 1.0), 0.0)
    )

    recommendation = str(current_plan.get("recommendation") or "").lower()
    risk_recommendation = recommendation
    if recommendation == "request_extension":
        consensus = str(current_plan.get("consensus") or "").lower()
        if consensus == side:
            risk_recommendation = "maintain_position"
        elif consensus == "hold":
            risk_recommendation = "monitor_closely"
        else:
            risk_recommendation = "prepare_defensive_exit"
    consensus_score = abs(float(current_plan.get("consensus_score", 0.0) or 0.0))
    close_threshold = abs(float(current_plan.get("close_threshold", 0.25) or 0.25))

    desired_sl = current_sl if current_sl is not None else base_sl
    desired_tp = current_tp if current_tp is not None else base_tp
    actions: list[str] = []
    reasons: list[str] = []

    delayed_cfg = _delayed_stop_loss_cfg(trading_cfg)
    protection_meta = (base_plan.get("stop_loss_protection") or state.get("stop_loss_protection") or {}) if isinstance(state, dict) else {}
    delayed_active = (
        bool(delayed_cfg.get("enabled", False))
        and current_sl is None
        and (
            str(protection_meta.get("mode") or "").strip().lower() == "delayed_protection"
            or _plan_requests_delayed_stop_loss(base_plan, trading_cfg)
        )
    )
    if delayed_active:
        age_seconds = 0.0
        started_raw = str(state.get("agent_b_started_at") or state.get("started_at") or "").strip()
        try:
            if started_raw:
                age_seconds = max((_now_utc() - datetime.strptime(started_raw, "%Y-%m-%d %H:%M:%S")).total_seconds(), 0.0)
        except Exception:
            age_seconds = 0.0
        max_unprotected_seconds = max(_safe_float(delayed_cfg.get("max_unprotected_seconds", 900), 900), 1.0)
        max_adverse_pct = max(_safe_float(delayed_cfg.get("max_unprotected_adverse_pct", 0.25), 0.25), 0.01)
        adverse_move = max((entry - current_price) if side == "buy" else (current_price - entry), 0.0)
        adverse_pct = (adverse_move / entry) * 100.0 if entry > 0.0 else 0.0
        planned_stop = _safe_float(protection_meta.get("planned_stop_loss"), 0.0)
        if planned_stop <= 0.0:
            planned_stop = _fallback_stop_loss_for_plan(base_plan, trading_cfg)
        attach_due_to_time = age_seconds >= max_unprotected_seconds
        attach_due_to_adverse = adverse_pct >= max_adverse_pct
        attach_due_to_defense = risk_recommendation in {"monitor_closely", "prepare_defensive_exit"}
        if planned_stop > 0.0 and (attach_due_to_time or attach_due_to_adverse or attach_due_to_defense):
            out: Dict[str, Any] = {
                "action": "attach_delayed_stop_loss",
                "rationale": (
                    "attached delayed stop loss after "
                    f"unprotected_seconds={int(age_seconds)}, adverse_pct={adverse_pct:.3f}, "
                    f"risk_recommendation={risk_recommendation}"
                ),
                "favorable_move": round(float(favorable_move), 6),
                "breakeven_activation_move": round(float(breakeven_activation_move), 6),
                "previous_stop_loss": current_sl,
                "previous_take_profit": current_tp,
                "stop_loss": round(float(planned_stop), 6),
            }
            if desired_tp is not None:
                out["take_profit"] = round(float(desired_tp), 6)
            return out

    if favorable_move <= min_delta:
        return None

    if risk_recommendation in {"monitor_closely", "prepare_defensive_exit"}:
        if favorable_move >= breakeven_activation_move:
            breakeven_sl = entry + cost_buffer if side == "buy" else entry - cost_buffer
            if side == "buy":
                candidate_sl = max(desired_sl if desired_sl is not None else breakeven_sl, breakeven_sl)
                if current_sl is None or candidate_sl - current_sl >= min_delta:
                    desired_sl = candidate_sl
                    actions.append("tighten_stop_loss")
                    reasons.append("moved stop loss to break-even after a meaningful favorable move while supervision turned defensive")
            else:
                candidate_sl = min(desired_sl if desired_sl is not None else breakeven_sl, breakeven_sl)
                if current_sl is None or current_sl - candidate_sl >= min_delta:
                    desired_sl = candidate_sl
                    actions.append("tighten_stop_loss")
                    reasons.append("moved stop loss to break-even after a meaningful favorable move while supervision turned defensive")
    elif risk_recommendation == "maintain_position" and trailing_enabled:
        if side == "buy":
            candidate_floor = entry if favorable_move >= breakeven_activation_move else (desired_sl if desired_sl is not None else base_sl if base_sl is not None else entry - base_sl_distance)
            candidate_sl = max(desired_sl if desired_sl is not None else candidate_floor, candidate_floor, current_price - trail_gap)
            if current_sl is None or candidate_sl - current_sl >= min_delta:
                desired_sl = candidate_sl
                actions.append("trail_stop_loss")
                reasons.append("locked in profit under aligned bullish supervision")
        else:
            candidate_ceiling = entry if favorable_move >= breakeven_activation_move else (desired_sl if desired_sl is not None else base_sl if base_sl is not None else entry + base_sl_distance)
            candidate_sl = min(desired_sl if desired_sl is not None else candidate_ceiling, candidate_ceiling, current_price + trail_gap)
            if current_sl is None or current_sl - candidate_sl >= min_delta:
                desired_sl = candidate_sl
                actions.append("trail_stop_loss")
                reasons.append("locked in profit under aligned bearish supervision")

        current_confidence = _safe_float(
            current_plan.get(
                "success_probability",
                current_plan.get("average_confidence", consensus_score),
            ),
            0.0,
        )
        minimum_extension_confidence = _safe_float(
            trailing_cfg.get("min_confidence_to_extend", 0.0), 0.0
        )
        if (
            consensus_score >= max(close_threshold, 0.4)
            and current_confidence >= minimum_extension_confidence
        ):
            extension_distance = max(base_tp_distance * 0.5, trail_gap)
            if side == "buy":
                candidate_tp = max(desired_tp if desired_tp is not None else current_price, current_price + extension_distance)
                if current_tp is None or candidate_tp - current_tp >= min_delta:
                    desired_tp = candidate_tp
                    actions.append("extend_take_profit")
                    reasons.append("extended take profit while consensus remained strongly aligned")
            else:
                candidate_tp = min(desired_tp if desired_tp is not None else current_price, current_price - extension_distance)
                if current_tp is None or current_tp - candidate_tp >= min_delta:
                    desired_tp = candidate_tp
                    actions.append("extend_take_profit")
                    reasons.append("extended take profit while consensus remained strongly aligned")

    opposing_target = _opposing_programmed_tp_target(state, position, trading_cfg)
    if opposing_target is not None:
        opposing_entry = float(opposing_target.get("entry", 0.0) or 0.0)
        if side == "buy":
            if (current_tp is None or current_tp - opposing_entry >= min_delta) and (desired_tp is None or desired_tp - opposing_entry >= min_delta):
                desired_tp = opposing_entry
                actions = [action for action in actions if action != "extend_take_profit"]
                actions.append("tighten_take_profit")
                reasons.append(
                    f"tightened take profit to opposing programmed sell entry {opposing_entry:.5f} from {opposing_target.get('job_id', 'unknown')}"
                )
        else:
            if (current_tp is None or opposing_entry - current_tp >= min_delta) and (desired_tp is None or opposing_entry - desired_tp >= min_delta):
                desired_tp = opposing_entry
                actions = [action for action in actions if action != "extend_take_profit"]
                actions.append("tighten_take_profit")
                reasons.append(
                    f"tightened take profit to opposing programmed buy entry {opposing_entry:.5f} from {opposing_target.get('job_id', 'unknown')}"
                )

    if not actions:
        return None

    if desired_sl is None and desired_tp is None:
        return None

    out: Dict[str, Any] = {
        "action": "+".join(actions),
        "rationale": "; ".join(reasons),
        "favorable_move": round(float(favorable_move), 6),
        "breakeven_activation_move": round(float(breakeven_activation_move), 6),
        "estimated_round_trip_cost_buffer": round(float(cost_buffer), 6),
        "trail_gap": round(float(trail_gap), 6),
        "previous_stop_loss": current_sl,
        "previous_take_profit": current_tp,
    }
    if desired_sl is not None:
        out["stop_loss"] = round(float(desired_sl), 6)
    if desired_tp is not None:
        out["take_profit"] = round(float(desired_tp), 6)
    return out


def _agent_b_plan_changed(previous: Optional[Dict[str, Any]], current: Dict[str, Any]) -> bool:
    if not isinstance(previous, dict) or not previous:
        return True
    if str(previous.get("recommendation") or "") != str(current.get("recommendation") or ""):
        return True
    if str(previous.get("consensus") or "") != str(current.get("consensus") or ""):
        return True
    prev_score = float(previous.get("consensus_score", 0.0) or 0.0)
    curr_score = float(current.get("consensus_score", 0.0) or 0.0)
    if abs(prev_score - curr_score) >= 0.05:
        return True

    prev_adj = previous.get("risk_adjustment") if isinstance(previous.get("risk_adjustment"), dict) else {}
    curr_adj = current.get("risk_adjustment") if isinstance(current.get("risk_adjustment"), dict) else {}
    if str(prev_adj.get("action") or "") != str(curr_adj.get("action") or ""):
        return True
    prev_sl = float(prev_adj.get("stop_loss", 0.0) or 0.0)
    curr_sl = float(curr_adj.get("stop_loss", 0.0) or 0.0)
    if abs(prev_sl - curr_sl) >= 0.05:
        return True
    prev_tp = float(prev_adj.get("take_profit", 0.0) or 0.0)
    curr_tp = float(curr_adj.get("take_profit", 0.0) or 0.0)
    return abs(prev_tp - curr_tp) >= 0.05


def _agent_b_risk_adjustment_signature(risk_adjustment: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "action": risk_adjustment.get("action"),
            "stop_loss": risk_adjustment.get("stop_loss"),
            "take_profit": risk_adjustment.get("take_profit"),
        },
        sort_keys=True,
        default=str,
    )


def _apply_agent_b_risk_adjustment(
    adapter: MT5Adapter,
    state: Dict[str, Any],
    pos_ticket: int,
    risk_adjustment: Dict[str, Any],
) -> Dict[str, Any]:
    signature = _agent_b_risk_adjustment_signature(risk_adjustment)
    previous = state.get("last_risk_adjustment") if isinstance(state.get("last_risk_adjustment"), dict) else {}
    previous_result = previous.get("result") if isinstance(previous.get("result"), dict) else {}
    already_applied = bool(previous_result.get("ok", False)) and signature == str(previous.get("signature") or "")
    if already_applied:
        return {"attempted": False, "signature": signature, "result": previous_result}

    modify_res = adapter.modify_position_risk(
        pos_ticket,
        stop_loss=risk_adjustment.get("stop_loss"),
        take_profit=risk_adjustment.get("take_profit"),
    )
    state["last_risk_adjustment"] = {
        "timestamp_utc": _iso(_now_utc()),
        "signature": signature,
        "action": risk_adjustment.get("action"),
        "stop_loss": risk_adjustment.get("stop_loss"),
        "take_profit": risk_adjustment.get("take_profit"),
        "previous_stop_loss": risk_adjustment.get("previous_stop_loss"),
        "previous_take_profit": risk_adjustment.get("previous_take_profit"),
        "result": modify_res,
    }
    return {"attempted": True, "signature": signature, "result": modify_res}


def _maybe_llm_assist(trading_cfg: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    llm_cfg = (trading_cfg.get("llm") or {})
    if not bool(llm_cfg.get("enabled", False)):
        return {"ok": False, "skipped": True, "reason": "llm disabled"}

    providers_path = str(llm_cfg.get("providers_config_path", "config/llm_providers.yaml"))
    providers_cfg = load_llm_providers_config(providers_path)
    provider_name = str(llm_cfg.get("provider") or providers_cfg.get("default_provider", "")).strip()
    if not provider_name:
        return {"ok": False, "error": "no provider configured"}

    timeout_sec = int(llm_cfg.get("timeout_seconds", 30) or 30)
    return call_llm(provider_name=provider_name, prompt=prompt, providers_cfg=providers_cfg, timeout_sec=timeout_sec)


def _trim_json_for_prompt(payload: Dict[str, Any], max_chars: int) -> str:
    raw = json.dumps(payload, indent=2, default=str)
    cap = max(int(max_chars or 8000), 1200)
    if len(raw) <= cap:
        return raw
    return raw[:cap] + "\n...<trimmed for context window>"


def _get_memory_store(trading_cfg: Dict[str, Any], output_dir: str) -> Optional[AgentMemoryStore]:
    mem_cfg = (trading_cfg.get("memory") or {})
    if not bool(mem_cfg.get("enabled", False)):
        return None
    db_path = str(mem_cfg.get("db_path") or os.path.join(output_dir, "runtime", "agent_memory.sqlite"))
    emb_dim = int(mem_cfg.get("embedding_dim", 256) or 256)
    try:
        return AgentMemoryStore(db_path=db_path, embedding_dim=emb_dim)
    except Exception:
        return None


def _approval_policy(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    ap = (trading_cfg.get("approval_policy") or {})
    risk_cfg = (trading_cfg.get("risk") or {})
    return {
        "normal_timeout_seconds": int(ap.get("normal_timeout_seconds", 7200) or 7200),
        "fast_timeout_seconds": int(ap.get("fast_timeout_seconds", 120) or 120),
        "emergency_lead_minutes": int(ap.get("emergency_lead_minutes", 5) or 5),
        "channel_user_triggered_only": bool(ap.get("channel_user_triggered_only", True)),
        "auto_approve_mandatory_session_programmed": bool(ap.get("auto_approve_mandatory_session_programmed", True)),
        "auto_approve_opposing_countertrade": bool(ap.get("auto_approve_opposing_countertrade", True)),
        "auto_approve_below_agent_b_count": max(int(ap.get("auto_approve_below_agent_b_count", risk_cfg.get("max_open_positions", 3)) or risk_cfg.get("max_open_positions", 3) or 3), 0),
    }


def _active_agent_b_position_count(output_dir: str, trading_cfg: Dict[str, Any]) -> int:
    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, _msg_conn = adapter.connect()
    if ok_conn:
        try:
            listed = adapter.list_open_positions()
            if bool(listed.get("ok", False)):
                target_symbol = str(((trading_cfg.get("execution") or {}).get("symbol") or "")).strip()
                tickets: set[int] = set()
                count = 0
                for position in (listed.get("positions") or []):
                    if not isinstance(position, dict):
                        continue
                    if target_symbol and str(position.get("symbol") or "").strip() != target_symbol:
                        continue
                    comment = str(position.get("comment") or "")
                    magic = int(position.get("magic", 0) or 0)
                    if "TSMM" not in comment and magic not in {7070001, 7070002}:
                        continue
                    ticket = int(position.get("ticket", 0) or 0)
                    if ticket > 0 and ticket in tickets:
                        continue
                    if ticket > 0:
                        tickets.add(ticket)
                    count += 1
                return count
        finally:
            adapter.shutdown()

    registry = _load_state(str(_job_registry_path(output_dir, trading_cfg)))
    candidate_ids: list[str] = []
    for raw_id in (registry.get("active_job_ids") or []):
        clean_id = str(raw_id).strip()
        if clean_id and clean_id not in candidate_ids:
            candidate_ids.append(clean_id)
    for raw_id in (registry.get("jobs") or {}).keys():
        clean_id = str(raw_id).strip()
        if clean_id and clean_id not in candidate_ids:
            candidate_ids.append(clean_id)

    tickets: set[int] = set()
    count = 0
    for job_id in candidate_ids:
        state = _load_state(_state_path(output_dir, trading_cfg, job_id))
        if not isinstance(state, dict) or not state:
            continue
        if str((state.get("status") or "")).strip().lower() != "agent_b_running":
            continue
        if str((state.get("stage") or "")).strip().lower() != "agent_b":
            continue
        if str((state.get("ended_at") or "")).strip() or str((state.get("closed_reason") or "")).strip():
            continue
        ticket = int(((state.get("position") or {}).get("ticket", 0)) or 0)
        if ticket > 0 and ticket in tickets:
            continue
        if ticket > 0:
            tickets.add(ticket)
        count += 1
    return count


def _agent_a_approval_decision(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    *,
    auto_created: bool,
    autonomous_trigger: str,
    submission_mode: str,
) -> tuple[bool, str, int, int]:
    agent_cfg = (trading_cfg.get("agent") or {})
    policy = _approval_policy(trading_cfg)
    active_agent_b_count = _active_agent_b_position_count(output_dir, trading_cfg)
    threshold = int(policy.get("auto_approve_below_agent_b_count", 0) or 0)

    if bool(policy.get("auto_approve_mandatory_session_programmed", True)) and autonomous_trigger == "mandatory_session" and submission_mode == "programmed":
        return False, "mandatory_session_programmed", active_agent_b_count, threshold
    if bool(policy.get("auto_approve_opposing_countertrade", True)) and autonomous_trigger == "opposing_countertrade":
        return False, "opposing_countertrade", active_agent_b_count, threshold
    if auto_created and bool(agent_cfg.get("followup_agent_a_requires_approval", False)):
        return True, "followup_manual_approval_required", active_agent_b_count, threshold
    if threshold > 0 and active_agent_b_count < threshold:
        return False, "below_agent_b_threshold", active_agent_b_count, threshold
    if auto_created and not bool(agent_cfg.get("followup_agent_a_requires_approval", False)):
        return False, "auto_created", active_agent_b_count, threshold
    return True, "manual_approval_required", active_agent_b_count, threshold


def _agent_a_fallback_candidates(trading_cfg: Dict[str, Any]) -> list[Dict[str, str]]:
    cfg = (trading_cfg.get("agent_a_fallback") or {})
    if not bool(cfg.get("enabled", False)):
        return []

    out: list[Dict[str, str]] = []
    for item in (cfg.get("attempts") or []):
        if isinstance(item, dict):
            cfg_path = str(item.get("config_path") or "").strip()
            model = str(item.get("model") or "").strip()
            timeframe = str(item.get("timeframe") or "").strip()
        else:
            cfg_path = ""
            model = ""
            timeframe = ""

        if not timeframe and cfg_path:
            m = re.search(r"high([^\\/]+)Results", cfg_path, flags=re.IGNORECASE)
            if m:
                timeframe = m.group(1)
        if cfg_path:
            out.append({"config_path": cfg_path, "model": model, "timeframe": timeframe})
    return out


def _realign_plan_risk_levels(plan: Dict[str, Any], side: str) -> Dict[str, Any]:
    payload = dict(plan or {})
    desired_side = str(side or "").strip().lower()
    if desired_side not in {"buy", "sell"}:
        return payload

    try:
        entry = float(payload.get("entry"))
        stop_loss = float(payload.get("stop_loss"))
        take_profit = float(payload.get("take_profit"))
    except Exception:
        return payload

    sl_distance = abs(entry - stop_loss)
    tp_distance = abs(take_profit - entry)
    if desired_side == "buy":
        payload["stop_loss"] = round(entry - sl_distance, 6)
        payload["take_profit"] = round(entry + tp_distance, 6)
    else:
        payload["stop_loss"] = round(entry + sl_distance, 6)
        payload["take_profit"] = round(entry - tp_distance, 6)
    return payload


def _parse_r2_from_name(path: Path) -> float:
    stem = path.stem.lower()
    m = re.search(r"_(\d{4,6})$", stem)
    if m:
        digits = m.group(1)
        return float(int(digits) / (10 ** (len(digits) - 1)))
    m2 = re.search(r"(\d+\.\d+)", stem)
    if m2:
        return float(m2.group(1))
    return 0.0


def _discover_dynamic_fallback_candidates(trading_cfg: Dict[str, Any]) -> list[Dict[str, str]]:
    cfg = (trading_cfg.get("agent_a_fallback") or {})
    if not bool(cfg.get("enabled", False)) or not bool(cfg.get("auto_discover", True)):
        return []

    target_families = [
        str(x).strip().lower()
        for x in (cfg.get("target_families") or ["high", "low", "close", "open"])
        if str(x).strip()
    ]
    timeframe_priority = [
        str(x).strip()
        for x in (cfg.get("prefer_timeframes") or ["3h", "1h", "30m", "10m", "12h", "24h", "1w"])
        if str(x).strip()
    ]
    timeframe_rank = {tf: i for i, tf in enumerate(timeframe_priority)}
    max_attempts = max(int(cfg.get("max_attempts", 24) or 24), 1)

    endpoint_map = dict(trading_cfg.get("model_endpoints") or {})
    project_root = Path(__file__).resolve().parents[1]
    config_root = project_root / "config"
    if not config_root.exists():
        return []

    discovered: list[Dict[str, Any]] = []
    for family in target_families:
        for tf in endpoint_map.keys():
            tf_label = str(tf).strip()
            if not tf_label or tf_label == "7h":
                continue

            tf_dir = config_root / f"{family}{tf_label}Results"
            if not tf_dir.exists() or not tf_dir.is_dir():
                continue

            best_path: Path | None = None
            best_model = ""
            best_r2 = -1.0
            for model_dir in [d for d in tf_dir.iterdir() if d.is_dir()]:
                for cfg_file in list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml")):
                    r2 = _parse_r2_from_name(cfg_file)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_path = cfg_file
                        best_model = model_dir.name

            if best_path is None:
                continue

            discovered.append(
                {
                    "config_path": str(best_path),
                    "model": best_model,
                    "timeframe": tf_label,
                    "family": family,
                    "r2": best_r2,
                    "timeframe_rank": timeframe_rank.get(tf_label, 999),
                }
            )

    discovered = sorted(
        discovered,
        key=lambda x: (int(x.get("timeframe_rank", 999)), -float(x.get("r2", 0.0))),
    )
    return [
        {
            "config_path": str(x.get("config_path") or ""),
            "model": str(x.get("model") or ""),
            "timeframe": str(x.get("timeframe") or ""),
        }
        for x in discovered[:max_attempts]
    ]


def _notify(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    channel: str,
    kind: str,
    message: str,
    requires_approval: bool,
    emergency: bool,
    approval_deadline_utc: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
    force_telegram: bool = False,
    job_id: Optional[str] = None,
    state_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_job_id = str(job_id or ((metadata or {}).get("job_id") if isinstance(metadata, dict) else "") or "").strip()
    payload_metadata = dict(metadata or {})
    if resolved_job_id:
        payload_metadata["job_id"] = resolved_job_id

    def _telegram_category(current_channel: str, current_kind: str, needs_approval: bool) -> str:
        if current_kind == "approval_request" and current_channel == "agent_a":
            return "NEW OPERATION PROPOSAL"
        if current_kind in {"plan_ready", "followup_start_requested"} and current_channel == "agent_a":
            return "ENTRY ANALYSIS"
        if current_kind in {"approval_bypassed", "order_placed_pending", "order_filled", "order_pending_timeout"} and current_channel == "agent_a":
            return "ENTRY EXECUTION"
        if current_channel == "agent_b" and current_kind in {"management_plan", "risk_update", "job_finished"}:
            return "LIVE POSITION SUPERVISION"
        if needs_approval:
            return "APPROVAL ACTION"
        return "SYSTEM UPDATE"

    ch = publish_channel_message(
        output_dir=output_dir,
        trading_cfg=trading_cfg,
        channel=channel,
        kind=kind,
        message=message,
        requires_approval=requires_approval,
        emergency=emergency,
        approval_deadline_utc=approval_deadline_utc,
        metadata=payload_metadata,
    )
    email_cfg = (trading_cfg.get("email_notifications") or {})
    telegram_cfg = (trading_cfg.get("telegram_notifications") or {})
    email_out = {"ok": False, "skipped": True, "reason": "not_required"}
    telegram_out = {"ok": False, "skipped": True, "reason": "not_required"}

    should_push = bool(requires_approval) or bool(emergency)
    if bool(telegram_cfg.get("send_on_all_messages", False)):
        should_push = True
    if bool(force_telegram):
        should_push = True

    account_label = _account_profile_label(trading_cfg)

    if bool(email_cfg.get("enabled", False)) and should_push:
        scoped_message = f"job_id: {resolved_job_id}\n\n{message}" if resolved_job_id else message
        email_subject = f"TSMM Notification: {channel}/{kind}"
        email_body = (
            f"time_utc: {_iso(_now_utc())}\n"
            f"account: {account_label}\n"
            f"channel: {channel}\n"
            f"type: {kind}\n"
            f"job_id: {resolved_job_id or 'n/a'}\n"
            f"emergency: {bool(emergency)}\n"
            f"requires_approval: {bool(requires_approval)}\n"
            f"approval_deadline_utc: {approval_deadline_utc}\n\n"
            f"{scoped_message}"
        )
        email_out = send_email_notification(email_cfg=email_cfg, subject=email_subject, body=email_body)

    if bool(telegram_cfg.get("enabled", False)) and should_push:
        scoped_message = f"job_id: {resolved_job_id}\n\n{message}" if resolved_job_id else message
        category = _telegram_category(channel, kind, requires_approval)
        tg_message = (
            f"*TSMM Notification*\n"
            f"account: {account_label}\n"
            f"category: {category}\n"
            f"time_utc: {_iso(_now_utc())}\n"
            f"channel: {channel}\n"
            f"type: {kind}\n"
            f"job_id: {resolved_job_id or 'n/a'}\n"
            f"emergency: {bool(emergency)}\n"
            f"requires_approval: {bool(requires_approval)}\n"
            f"approval_deadline_utc: {approval_deadline_utc}\n\n"
            f"{scoped_message}"
        )
        telegram_out = send_telegram_broadcast(
            telegram_cfg=telegram_cfg,
            message=tg_message,
            subscribers_path=_telegram_subscribers_path(output_dir, trading_cfg),
        )
        if not bool(telegram_out.get("ok", False)):
            payload_metadata["telegram_delivery_error"] = str(
                telegram_out.get("error")
                or telegram_out.get("reason")
                or "telegram_send_failed"
            ).strip()
        payload_metadata["notification_delivery"] = {
            "email": email_out,
            "telegram": telegram_out,
        }

    try:
        feedback_state = dict(state_context or {})
        if not feedback_state and resolved_job_id:
            feedback_state = _load_state(_state_path(output_dir, trading_cfg, resolved_job_id))
        log_notification_feedback(
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            channel=channel,
            kind=kind,
            message=message,
            metadata=payload_metadata,
            state=feedback_state,
            job_id=resolved_job_id,
        )
    except Exception:
        pass

    return {"channel": ch, "email": email_out, "telegram": telegram_out}


def _job_result_summary_message(state: Dict[str, Any]) -> str:
    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    position = (state.get("position") or {}) if isinstance(state, dict) else {}
    close_result = (state.get("close_result") or {}) if isinstance(state, dict) else {}
    parts = ["Trading job finished."]
    parts.append(f"status={state.get('status', 'n/a')}")
    parts.append(f"closed_reason={state.get('closed_reason', 'n/a')}")
    parts.append(f"decision={plan.get('decision', 'n/a')}")
    parts.append(f"entry={plan.get('entry', 'n/a')}")

    ticket = position.get("ticket") or close_result.get("ticket") or close_result.get("position_ticket") or "n/a"
    parts.append(f"mt5_ticket={ticket}")
    if position.get("symbol"):
        parts.append(f"symbol={position.get('symbol')}")
    if position.get("volume") is not None:
        parts.append(f"volume={position.get('volume')}")
    if position.get("price_open") is not None:
        parts.append(f"price_open={position.get('price_open')}")
    if position.get("price_current") is not None:
        parts.append(f"last_price={position.get('price_current')}")
    if position.get("profit") is not None:
        parts.append(f"profit={position.get('profit')}")
    if state.get("started_at"):
        parts.append(f"started_at={state.get('started_at')}")
    if state.get("ended_at"):
        parts.append(f"ended_at={state.get('ended_at')}")
    return "; ".join(parts)


def _append_job_finished_notification(
    state: Dict[str, Any],
    output_dir: str,
    trading_cfg: Dict[str, Any],
    channel: str,
    job_id: str,
) -> None:
    notifications = state.setdefault("notifications", [])
    for existing in notifications:
        payload = ((existing or {}).get("channel") or {}) if isinstance(existing, dict) else {}
        if str(payload.get("kind") or "").strip() == "job_finished":
            return
    notifications.append(
        _notify(
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            channel=channel,
            kind="job_finished",
            message=_job_result_summary_message(state),
            requires_approval=False,
            emergency=False,
            approval_deadline_utc=None,
            metadata={"close_result": state.get("close_result"), "position": state.get("position")},
            force_telegram=True,
            job_id=job_id,
        )
    )


def _attempt_agent_b_close(
    *,
    adapter: MT5Adapter,
    state: Dict[str, Any],
    pos_ticket: int,
    output_dir: str,
    trading_cfg: Dict[str, Any],
    job_id: str,
    closed_reason: str,
    final_status: str,
    failure_message: str,
    propagate_mirror_action: bool = True,
) -> bool:
    close_res = adapter.close_position_by_ticket(pos_ticket)
    state["close_result"] = close_res

    if close_res.get("ok"):
        close_outcome = adapter.get_position_close_outcome(pos_ticket)
        if isinstance(close_outcome, dict) and close_outcome.get("ok"):
            state["close_outcome"] = close_outcome

        if propagate_mirror_action:
            mirror_res = _mirror_agent_b_position_action(
                action="close",
                output_dir=output_dir,
                source_trading_cfg=trading_cfg,
                source_state=state,
                source_job_id=job_id,
                closed_reason=closed_reason,
                final_status=final_status,
            )
            state["close_result"] = dict(close_res)
            state["close_result"]["mirror_result"] = mirror_res
            if not mirror_res.get("ok") and not mirror_res.get("skipped"):
                state.setdefault("notifications", []).append(
                    _notify(
                        output_dir=output_dir,
                        trading_cfg=trading_cfg,
                        channel="agent_b",
                        kind="system_update",
                        message=(
                            "Agent B closed the local MT5 position, but the linked-account close mirror failed. "
                            f"peer_profile={mirror_res.get('peer_profile', 'n/a')}, details={mirror_res.get('error') or mirror_res.get('result')}"
                        ),
                        requires_approval=False,
                        emergency=False,
                        approval_deadline_utc=None,
                        metadata={"mirror_result": mirror_res, "close_result": close_res},
                        force_telegram=True,
                        job_id=job_id,
                    )
                )
        state["status"] = final_status
        state["closed_reason"] = closed_reason
        state["ended_at"] = _iso(_now_utc())
        state.pop("pending_close_reason", None)
        state.pop("pending_close_status", None)
        state.pop("last_close_failure", None)
        _append_job_finished_notification(state, output_dir, trading_cfg, "agent_b", job_id)
        try:
            from utils.trade_memory import TradeMemory as _TMem
            _TMem().record(_TMem.from_registry_job(state))
        except Exception:
            pass
        return True

    pos = adapter.get_position_by_ticket(pos_ticket)
    live_position = (pos.get("position") or {}) if isinstance(pos, dict) else {}
    if not pos.get("ok") or not live_position:
        state["status"] = final_status
        state["closed_reason"] = closed_reason
        state["ended_at"] = _iso(_now_utc())
        state.pop("pending_close_reason", None)
        state.pop("pending_close_status", None)
        state.pop("last_close_failure", None)
        _append_job_finished_notification(state, output_dir, trading_cfg, "agent_b", job_id)
        return True

    state["status"] = "agent_b_running"
    state["position"] = live_position
    state["pending_close_reason"] = closed_reason
    state["pending_close_status"] = final_status
    state.pop("ended_at", None)

    failure_signature = json.dumps(
        {
            "closed_reason": closed_reason,
            "message": close_res.get("message"),
            "retcode": close_res.get("retcode"),
            "ticket": pos_ticket,
        },
        sort_keys=True,
        default=str,
    )
    last_failure_signature = str(((state.get("last_close_failure") or {}).get("signature") or ""))
    if failure_signature != last_failure_signature:
        state.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_b",
                kind="system_update",
                message=failure_message,
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={"close_reason": closed_reason, "close_result": close_res, "position": live_position},
                force_telegram=True,
                job_id=job_id,
            )
        )
    state["last_close_failure"] = {
        "timestamp_utc": _iso(_now_utc()),
        "signature": failure_signature,
        "close_reason": closed_reason,
        "result": close_res,
        "position": live_position,
    }
    return False


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _close_outcome_reason_label(state: Dict[str, Any]) -> str:
    outcome = (state.get("close_outcome") or {}) if isinstance(state, dict) else {}
    return str(outcome.get("reason_label") or "").strip().lower()


def _is_manual_close_outcome(state: Dict[str, Any]) -> bool:
    outcome = (state.get("close_outcome") or {}) if isinstance(state, dict) else {}
    if not isinstance(outcome, dict):
        return False

    reason_label = str(outcome.get("reason_label") or "").strip().lower()
    if reason_label in {"client", "mobile", "web", "manual"}:
        return True

    # Some brokers may only include hints in the deal comment for manual closures.
    comment = str(outcome.get("comment") or "").strip().lower()
    if "manual" in comment and reason_label not in {"expert", "sl", "tp", "so"}:
        return True
    return False


def _is_manual_or_external_close(state: Dict[str, Any]) -> bool:
    closed_reason = str((state or {}).get("closed_reason") or "").strip().lower()
    if closed_reason in {"manual_stop", "manual_close_via_telegram", "position_not_found_assumed_closed"}:
        return True
    if bool((state or {}).get("manual_or_external_close_detected", False)):
        return True
    return _is_manual_close_outcome(state)


def _should_auto_request_followup_agent_a(state: Dict[str, Any], trading_cfg: Optional[Dict[str, Any]] = None) -> bool:
    closed_reason = str((state or {}).get("closed_reason") or "").strip().lower()
    if not closed_reason:
        return False
    if _is_manual_or_external_close(state):
        return False
    if closed_reason.startswith("mode_b_consensus_close("):
        return True
    if closed_reason in {"hard_deadline_reached", "extension_not_approved_in_window", "extension_rejected_or_timeout"}:
        return True
    return False


def _followup_agent_a_submission_mode(trading_cfg: Dict[str, Any]) -> str:
    agent_cfg = (trading_cfg.get("agent") or {}) if isinstance(trading_cfg, dict) else {}
    mode = str(agent_cfg.get("followup_agent_a_submission_mode") or "programmed").strip().lower()
    if mode not in {"programmed", "market"}:
        mode = "programmed"
    return mode


def _launch_followup_agent_a_start(
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    logger,
    completed_state: Dict[str, Any],
) -> Dict[str, Any]:
    if not _should_auto_request_followup_agent_a(completed_state, trading_cfg):
        skip_reason = "manual_or_external_close" if _is_manual_or_external_close(completed_state) else "close_reason_not_eligible"
        return {"ok": False, "skipped": True, "reason": skip_reason}

    new_job_id = _new_job_id(trading_cfg)
    submission_mode = _followup_agent_a_submission_mode(trading_cfg)
    env = os.environ.copy()
    env.setdefault("CONFIG_PATH", "config/config.yaml")
    env["TRADING_CONFIG_PATH"] = _current_trading_config_path()
    env["TSMM_AGENT_A_AUTO_CREATED"] = "1"
    env.pop("TSMM_ACCOUNT_MIRROR_SUPPRESS", None)
    env.pop("TSMM_ACCOUNT_MIRROR_SOURCE_JOB_ID", None)
    env.pop("TSMM_ACCOUNT_MIRROR_SOURCE_CONFIG_PATH", None)
    env.pop("TSMM_ACCOUNT_MIRROR_SOURCE_PROFILE", None)
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    try:
        proc = subprocess.Popen(
            [
                sys.executable,
                "app.py",
                "trading-job",
                "start",
                "--job-id",
                new_job_id,
                "--submission-mode",
                submission_mode,
            ],
            cwd=str(_project_root()),
            env=env,
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.exception("Failed to launch follow-up Agent A start after job close")
        return {"ok": False, "error": str(exc), "job_id": new_job_id}

    completed_state.setdefault("notifications", []).append(
        _notify(
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            channel="agent_a",
            kind="followup_start_requested",
            message=(
                "Previous trading job finished with an auto-reentry eligible close outcome. "
                f"Starting a new Agent A entry search now using {submission_mode} order submission mode. "
                f"new_job_id={new_job_id}; pid={int(proc.pid)}"
            ),
            requires_approval=False,
            emergency=False,
            approval_deadline_utc=None,
            metadata={
                "previous_job_id": completed_state.get("job_id"),
                "new_job_id": new_job_id,
                "followup_submission_mode": submission_mode,
                "trigger_closed_reason": completed_state.get("closed_reason"),
                "close_outcome": completed_state.get("close_outcome"),
            },
            force_telegram=True,
            job_id=str(completed_state.get("job_id") or "").strip(),
        )
    )
    return {"ok": True, "job_id": new_job_id, "pid": int(proc.pid)}


def _write_agent_a_signal_json(
    output_dir: str,
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    results: Dict[str, Any],
    mode_a_result: Dict[str, Any],
    llm_analysis: Dict[str, Any],
    sentiment: Dict[str, Any],
    memory_context: str,
) -> str:
    report_cfg = (trading_cfg.get("reporting") or {})
    rep_dir = str(report_cfg.get("output_dir") or os.path.join(output_dir, "trading_plans"))
    os.makedirs(rep_dir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(rep_dir, f"agent_a_signal_{ts}.json")

    plan = (mode_a_result.get("plan") or {})
    backtest = (mode_a_result.get("backtest") or {})
    eval_block = (results.get("evaluation") or {}).get(str(plan.get("model", "")), {})
    signal_payload = {
        "created_at_utc": _iso(_now_utc()),
        "mode": "mode_a",
        "symbol": ((trading_cfg.get("execution") or {}).get("symbol") or app_config.get("target_col") or "UNKNOWN"),
        "session_hours": float(((trading_cfg.get("trading_job") or {}).get("session_hours", 7) or 7)),
        "technical_analysis": {
            "selected_model": plan.get("model"),
            "decision": plan.get("decision"),
            "entry": plan.get("entry"),
            "stop_loss": plan.get("stop_loss"),
            "take_profit": plan.get("take_profit"),
            "volume": plan.get("volume"),
            "confidence": plan.get("confidence"),
            "cm_accuracy": plan.get("cm_accuracy"),
            "signal_score": plan.get("signal_score"),
            "success_probability": plan.get("success_probability"),
            "input_fooling_risk": plan.get("input_fooling_risk"),
            "feature_forecasts_step1": plan.get("feature_forecasts_step1"),
            "rationale": plan.get("rationale"),
            "risk_notes": plan.get("risk_notes"),
        },
        "evaluation_metrics": {
            "metrics": (eval_block.get("metrics") or {}),
            "confusion_matrix": (eval_block.get("confusion_matrix") or {}),
            "confidence_levels": (eval_block.get("confidence_levels") or []),
            "input_fooling_risk": (eval_block.get("input_fooling_risk") or {}),
            "explosion_detection": (eval_block.get("explosion_detection") or {}),
        },
        "backtest": {
            "n_trades": backtest.get("n_trades"),
            "win_rate": backtest.get("win_rate"),
            "total_return_pct": backtest.get("total_return_pct"),
            "max_drawdown_pct": backtest.get("max_drawdown_pct"),
        },
        "market_sentiment": sentiment,
        "llm_analysis": llm_analysis,
        "memory_context": memory_context,
        "report_path": mode_a_result.get("report_path"),
        "state_path": mode_a_result.get("state_path"),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(signal_payload, f, indent=2, default=str)
    return path


def _run_agent_b_loop(
    state: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    state_file: str,
    logger,
) -> Dict[str, Any]:
    mb_cfg = (trading_cfg.get("mode_b") or {})
    tj = (trading_cfg.get("trading_job") or {})
    agent_cfg = (trading_cfg.get("agent") or {})

    poll_seconds = int(mb_cfg.get("poll_seconds", tj.get("poll_seconds", 60)) or 60)
    session_hours = float(tj.get("session_hours", 24.0) or 24.0)
    approval_after_hours = max(float(tj.get("approval_after_hours", session_hours) or session_hours), 24.0)
    extension_max_hours = float(tj.get("extension_max_hours", 168.0) or 168.0)
    extension_window_minutes = int(tj.get("extension_request_window_minutes", 15) or 15)
    approval_timeout_seconds = int(tj.get("approval_timeout_seconds", 900) or 900)
    channels = agent_cfg.get("approval_channels", ["popup", "terminal"])

    close_threshold = float((mb_cfg.get("close_consensus_threshold", 0.25) or 0.25))
    llm_cfg = (trading_cfg.get("llm") or {})
    llm_assist_every_seconds = int(llm_cfg.get("assist_every_seconds", 300) or 300)
    mem_store = _get_memory_store(trading_cfg, output_dir)
    mem_cfg = (trading_cfg.get("memory") or {})
    policy = _approval_policy(trading_cfg)
    job_id = str(state.get("job_id") or "").strip()

    started_at_raw = str(state.get("agent_b_started_at") or state["started_at"])
    started_at = datetime.strptime(started_at_raw, "%Y-%m-%d %H:%M:%S")
    base_deadline = started_at + timedelta(hours=approval_after_hours)
    hard_deadline = started_at + timedelta(hours=extension_max_hours)

    side = str(((state.get("plan") or {}).get("decision") or "hold")).lower()
    pos_ticket = int((state.get("position") or {}).get("ticket", 0) or 0)

    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, msg_conn = adapter.connect()
    if not ok_conn:
        state["status"] = "failed"
        state["closed_reason"] = f"mode_b_mt5_connect_failed: {msg_conn}"
        state["ended_at"] = _iso(_now_utc())
        return state

    try:
        extension_requested = bool(state.get("extension_requested", False))
        extension_approved = bool(state.get("extension_approved", False))
        last_llm_assist_ts = 0.0
        prev_consensus = None
        prev_score = None

        while True:
            if _should_stop(output_dir, trading_cfg, job_id):
                logger.warning("Manual stop flag detected, closing active position.")
                if _attempt_agent_b_close(
                    adapter=adapter,
                    state=state,
                    pos_ticket=pos_ticket,
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    job_id=job_id,
                    closed_reason="manual_stop",
                    final_status="stopped",
                    failure_message="Agent B attempted to stop the live MT5 position, but MT5 rejected the close request. Supervision will continue until the position is confirmed closed.",
                ):
                    return state
                _save_job_state(output_dir, trading_cfg, state_file, state)
                time.sleep(max(poll_seconds, 1))
                continue

            now = _now_utc()
            if now >= hard_deadline:
                if _attempt_agent_b_close(
                    adapter=adapter,
                    state=state,
                    pos_ticket=pos_ticket,
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    job_id=job_id,
                    closed_reason="hard_deadline_reached",
                    final_status="closed",
                    failure_message="Agent B reached the hard deadline and attempted to close the MT5 position, but MT5 rejected the request. The job will remain active and keep retrying until closure is confirmed.",
                ):
                    return state
                _save_job_state(output_dir, trading_cfg, state_file, state)
                time.sleep(max(poll_seconds, 1))
                continue

            pos = adapter.get_position_by_ticket(pos_ticket)
            if not pos.get("ok") or not pos.get("position"):
                close_outcome = adapter.get_position_close_outcome(pos_ticket)
                if close_outcome.get("ok"):
                    state["close_outcome"] = close_outcome
                state["status"] = "closed"
                state["closed_reason"] = "position_not_found_assumed_closed"
                state["manual_or_external_close_detected"] = True
                state["ended_at"] = _iso(_now_utc())
                _append_job_finished_notification(state, output_dir, trading_cfg, "agent_b", job_id)
                return state

            live_position = pos.get("position") or {}
            state["position"] = live_position

            pending_close_reason = str(state.get("pending_close_reason") or "").strip()
            pending_close_status = str(state.get("pending_close_status") or "closed").strip() or "closed"
            if pending_close_reason:
                logger.warning("Retrying pending Agent B close for job %s: %s", job_id, pending_close_reason)
                if _attempt_agent_b_close(
                    adapter=adapter,
                    state=state,
                    pos_ticket=pos_ticket,
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    job_id=job_id,
                    closed_reason=pending_close_reason,
                    final_status=pending_close_status,
                    failure_message="Agent B retried a previously failed MT5 close request. The position remains under supervision until MT5 confirms the close.",
                ):
                    return state
                state["last_pending_close_retry_at"] = _iso(_now_utc())

            fail_safe_plan = {
                "position_side": side,
                "recommendation": "maintain_position",
                "consensus_score": 0.0,
                "close_threshold": close_threshold,
            }
            fail_safe_adjustment = _agent_b_risk_adjustment(
                state=state,
                position=live_position,
                current_plan=fail_safe_plan,
                trading_cfg=trading_cfg,
            )
            if fail_safe_adjustment and fail_safe_adjustment.get("action") == "attach_delayed_stop_loss":
                fail_safe_apply = _apply_agent_b_risk_adjustment(
                    adapter=adapter,
                    state=state,
                    pos_ticket=pos_ticket,
                    risk_adjustment=fail_safe_adjustment,
                )
                fail_safe_result = fail_safe_apply.get("result") or {}
                if fail_safe_result.get("ok") and isinstance(fail_safe_result.get("position"), dict):
                    state["position"] = fail_safe_result.get("position")
                    live_position = fail_safe_result.get("position")
                if fail_safe_apply.get("attempted"):
                    if fail_safe_result.get("ok") and not fail_safe_result.get("skipped"):
                        mirror_res = _mirror_agent_b_position_action(
                            action="risk_update",
                            output_dir=output_dir,
                            source_trading_cfg=trading_cfg,
                            source_state=state,
                            source_job_id=job_id,
                            risk_adjustment=fail_safe_adjustment,
                        )
                        state["last_risk_adjustment"]["mirror_result"] = mirror_res
                    logger.warning(
                        "Agent B delayed-stop fail-safe attempted action=%s ok=%s",
                        fail_safe_adjustment.get("action"),
                        bool(fail_safe_result.get("ok", False)),
                    )
                    _save_job_state(output_dir, trading_cfg, state_file, state)

            data_sync = _ensure_assessment_data_fresh(trading_cfg=trading_cfg, logger=logger)
            state["last_data_sync"] = data_sync
            state["last_data_sync_at"] = _iso(now)
            if not bool(data_sync.get("ok", False)):
                logger.warning("Skipping Agent B assessment because master sync is not aligned: %s", data_sync)
                state["last_mode_b_skip_reason"] = f"data_sync_failed:{data_sync.get('error', 'unknown')}"
                _save_job_state(output_dir, trading_cfg, state_file, state)
                time.sleep(max(poll_seconds, 1))
                continue

            sig = _collect_all_model_assessment_signals(trading_cfg=trading_cfg, timeout_sec=3.0)
            consensus = str(sig.get("consensus", "hold"))
            score = float(sig.get("consensus_score", 0.0) or 0.0)
            state["mode_b"] = sig
            state["last_mode_b_tick"] = _iso(now)
            current_plan = _build_agent_b_management_plan(
                state=state,
                position=live_position,
                consensus=consensus,
                score=score,
                close_threshold=close_threshold,
                now=now,
                poll_seconds=poll_seconds,
                base_deadline=base_deadline,
                extension_approved=extension_approved,
            )
            previous_plan = state.get("agent_b_plan") if isinstance(state.get("agent_b_plan"), dict) else None
            risk_adjustment = _agent_b_risk_adjustment(
                state=state,
                position=live_position,
                current_plan=current_plan,
                trading_cfg=trading_cfg,
            )
            if risk_adjustment:
                current_plan["risk_adjustment"] = risk_adjustment
            state["agent_b_plan"] = current_plan
            try:
                log_agent_b_sample_feedback(
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    state=state,
                    signals=sig,
                    current_plan=current_plan,
                )
            except Exception:
                pass

            if _agent_b_plan_changed(previous_plan, current_plan):
                risk_action = ((current_plan.get("risk_adjustment") or {}).get("action") if isinstance(current_plan.get("risk_adjustment"), dict) else None)
                plan_message = (
                    "Agent B supervision plan updated. "
                    f"recommendation={current_plan.get('recommendation')}, consensus={consensus}, "
                    f"score={score:.3f}, next_review={current_plan.get('next_review_utc')}"
                    f"{'; risk_action=' + str(risk_action) if risk_action else ''}."
                )
                state.setdefault("notifications", []).append(
                    _notify(
                        output_dir=output_dir,
                        trading_cfg=trading_cfg,
                        channel="agent_b",
                        kind="management_plan",
                        message=plan_message,
                        requires_approval=False,
                        emergency=False,
                        approval_deadline_utc=None,
                        metadata={"agent_b_plan": current_plan},
                        force_telegram=True,
                        job_id=job_id,
                    )
                )

            if risk_adjustment:
                adjustment_apply = _apply_agent_b_risk_adjustment(
                    adapter=adapter,
                    state=state,
                    pos_ticket=pos_ticket,
                    risk_adjustment=risk_adjustment,
                )
                if adjustment_apply.get("attempted"):
                    modify_res = adjustment_apply.get("result") or {}
                    if modify_res.get("ok") and not modify_res.get("skipped"):
                        mirror_res = _mirror_agent_b_position_action(
                            action="risk_update",
                            output_dir=output_dir,
                            source_trading_cfg=trading_cfg,
                            source_state=state,
                            source_job_id=job_id,
                            risk_adjustment=risk_adjustment,
                        )
                        state["last_risk_adjustment"]["mirror_result"] = mirror_res
                        if isinstance(modify_res.get("position"), dict):
                            state["position"] = modify_res.get("position")
                        state.setdefault("notifications", []).append(
                            _notify(
                                output_dir=output_dir,
                                trading_cfg=trading_cfg,
                                channel="agent_b",
                                kind="risk_update",
                                message=(
                                    "Agent B updated MT5 risk levels for the live position. "
                                    f"action={risk_adjustment.get('action')}, "
                                    f"sl={risk_adjustment.get('stop_loss', 'unchanged')}, "
                                    f"tp={risk_adjustment.get('take_profit', 'unchanged')}."
                                ),
                                requires_approval=False,
                                emergency=False,
                                approval_deadline_utc=None,
                                metadata={"risk_adjustment": risk_adjustment, "result": modify_res},
                                force_telegram=True,
                                job_id=job_id,
                            )
                        )
                        if not mirror_res.get("ok") and not mirror_res.get("skipped"):
                            state.setdefault("notifications", []).append(
                                _notify(
                                    output_dir=output_dir,
                                    trading_cfg=trading_cfg,
                                    channel="agent_b",
                                    kind="system_update",
                                    message=(
                                        "Agent B updated local MT5 risk levels, but the linked-account mirror failed. "
                                        f"peer_profile={mirror_res.get('peer_profile', 'n/a')}, details={mirror_res.get('error') or mirror_res.get('result')}"
                                    ),
                                    requires_approval=False,
                                    emergency=False,
                                    approval_deadline_utc=None,
                                    metadata={"mirror_result": mirror_res, "risk_adjustment": risk_adjustment},
                                    force_telegram=True,
                                    job_id=job_id,
                                )
                            )

            if bool(llm_cfg.get("enabled", False)) and (time.time() - last_llm_assist_ts >= max(llm_assist_every_seconds, 1)):
                mem_ctx = ""
                if mem_store is not None:
                    mem_ctx = mem_store.build_context_block(
                        query=f"agent_b side={side} consensus={consensus} score={score:.4f}",
                        limit=int(mem_cfg.get("retrieval_top_k", 5) or 5),
                        symbol=str(((trading_cfg.get("execution") or {}).get("symbol") or "UNKNOWN")),
                        kinds=(mem_cfg.get("retrieval_kinds") or None),
                    )
                assist_prompt = (
                    "Agent B assistance request. Provide concise risk-aware advice in 5 bullets max. "
                    f"Current side={side}, consensus={consensus}, score={score:.4f}, "
                    f"position_ticket={pos_ticket}.\nMEMORY_CONTEXT:\n{mem_ctx or '<none>'}"
                )
                llm_out = _maybe_llm_assist(trading_cfg, assist_prompt)
                state.setdefault("agent_b_assistance", []).append(
                    {
                        "timestamp": _iso(_now_utc()),
                        "provider": llm_out.get("provider"),
                        "ok": bool(llm_out.get("ok", False)),
                        "text": llm_out.get("text", "") if bool(llm_out.get("ok", False)) else "",
                        "error": llm_out.get("error") if not bool(llm_out.get("ok", False)) else None,
                    }
                )
                state["agent_b_assistance"] = state.get("agent_b_assistance", [])[-20:]
                if mem_store is not None:
                    mem_store.add_memory(
                        kind="assistant_note",
                        timeframe="mode_b",
                        symbol=str(((trading_cfg.get("execution") or {}).get("symbol") or "UNKNOWN")),
                        title=f"AgentB-{consensus}",
                        text_payload=str(llm_out.get("text") or llm_out.get("error") or ""),
                        metadata={
                            "ok": bool(llm_out.get("ok", False)),
                            "provider": llm_out.get("provider"),
                            "consensus": consensus,
                            "score": score,
                        },
                    )
                last_llm_assist_ts = time.time()

            if _agent_b_close_policy(side=side, consensus=consensus, score=score, close_threshold=close_threshold):
                if _attempt_agent_b_close(
                    adapter=adapter,
                    state=state,
                    pos_ticket=pos_ticket,
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    job_id=job_id,
                    closed_reason=f"mode_b_consensus_close({consensus},{score:.3f})",
                    final_status="closed",
                    failure_message="Agent B attempted to close the MT5 position after a consensus reversal, but MT5 rejected the request. The position is still being monitored and has not been treated as closed.",
                ):
                    return state
                _save_job_state(output_dir, trading_cfg, state_file, state)
                time.sleep(max(poll_seconds, 1))
                continue

            if now >= base_deadline and not extension_approved:
                if not extension_requested:
                    extension_requested = True
                    state["extension_requested"] = True
                    state["extension_requested_at"] = _iso(now)

                extension_window_deadline = base_deadline + timedelta(minutes=extension_window_minutes)
                if now > extension_window_deadline:
                    if _attempt_agent_b_close(
                        adapter=adapter,
                        state=state,
                        pos_ticket=pos_ticket,
                        output_dir=output_dir,
                        trading_cfg=trading_cfg,
                        job_id=job_id,
                        closed_reason="extension_not_approved_in_window",
                        final_status="closed",
                        failure_message="Agent B attempted to close the MT5 position because the extension approval window expired, but MT5 rejected the request. The position remains tracked until closure is confirmed.",
                    ):
                        return state
                    _save_job_state(output_dir, trading_cfg, state_file, state)
                    time.sleep(max(poll_seconds, 1))
                    continue

                approved = request_approval(
                    title="TSMM Agent B Extension Request",
                    message=(
                        "Agent B requests extending this trading job beyond the 7h session. "
                        "Approve extension (up to 1 week max)?"
                    ),
                    timeout_sec=int(policy.get("normal_timeout_seconds", approval_timeout_seconds)),
                    channels=channels,
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    approval_metadata={"channel": "agent_b", "kind": "approval_request", "consensus": consensus, "score": score, "side": side},
                    job_id=job_id,
                )
                deadline_norm = _now_utc() + timedelta(seconds=int(policy.get("normal_timeout_seconds", approval_timeout_seconds)))
                extension_story = (
                    "Non-emergency approval requested: Agent B seeks extension to continue managing open position. "
                    "Reasoning: session horizon reached while signal/risk state still supports managed continuation. "
                    f"Consensus={consensus}, score={score:.4f}, side={side}."
                )
                notify_res = _notify(
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    channel="agent_b",
                    kind="approval_request",
                    message=extension_story,
                    requires_approval=True,
                    emergency=False,
                    approval_deadline_utc=_iso(deadline_norm),
                    metadata={"consensus": consensus, "score": score, "side": side},
                    job_id=job_id,
                )
                state.setdefault("notifications", []).append(notify_res)
                if approved:
                    extension_approved = True
                    state["extension_approved"] = True
                    state["extension_approved_at"] = _iso(_now_utc())
                else:
                    if _attempt_agent_b_close(
                        adapter=adapter,
                        state=state,
                        pos_ticket=pos_ticket,
                        output_dir=output_dir,
                        trading_cfg=trading_cfg,
                        job_id=job_id,
                        closed_reason="extension_rejected_or_timeout",
                        final_status="closed",
                        failure_message="Agent B attempted to close the MT5 position after the extension request was rejected or timed out, but MT5 rejected the close request. The position remains under supervision until closure is confirmed.",
                    ):
                        return state
                    _save_job_state(output_dir, trading_cfg, state_file, state)
                    time.sleep(max(poll_seconds, 1))
                    continue

            _save_job_state(output_dir, trading_cfg, state_file, state)
            prev_consensus = consensus
            prev_score = score
            time.sleep(max(poll_seconds, 1))
    finally:
        adapter.shutdown()


def start_trading_job(
    app_config: Dict[str, Any],
    results: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    logger,
    selected_model: Optional[str] = None,
    job_id: Optional[str] = None,
    submission_mode_override: Optional[str] = None,
    request_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_job_id = str(job_id or "").strip() or _new_job_id(trading_cfg)
    _clear_stop_flag(output_dir, trading_cfg, resolved_job_id)
    state_file = _state_path(output_dir, trading_cfg, resolved_job_id)

    agent_cfg = (trading_cfg.get("agent") or {})
    channels = agent_cfg.get("approval_channels", ["popup", "terminal"])
    policy = _approval_policy(trading_cfg)

    request_ctx = dict(request_context or {})
    effective_submission_mode = str(request_ctx.get("effective_submission_mode") or submission_mode_override or "").strip().lower()
    if effective_submission_mode:
        request_ctx["effective_submission_mode"] = effective_submission_mode

    forced_plan_override = _forced_agent_a_plan_from_env()
    if forced_plan_override:
        request_ctx["forced_plan_override"] = forced_plan_override
        request_ctx["forced_plan_override_source"] = "TSMM_FORCE_AGENT_A_PLAN_JSON"

    incoming_mirror = _incoming_account_mirror_context()
    if incoming_mirror:
        request_ctx["mirror"] = incoming_mirror

    data_sync = _ensure_assessment_data_fresh(trading_cfg=trading_cfg, logger=logger, app_config=app_config)
    if not bool(data_sync.get("ok", False)):
        failed_state = {
            "job_id": resolved_job_id,
            "job_type": "trading_job",
            "status": "failed",
            "stage": "preflight",
            "mode": "mode_a",
            "runner_pid": _runner_pid(),
            "started_at": _iso(_now_utc()),
            "ended_at": _iso(_now_utc()),
            "closed_reason": f"data_sync_failed:{data_sync.get('error', 'unknown')}",
            "last_data_sync": data_sync,
            "state_path": state_file,
        }
        if isinstance(request_ctx.get("mirror"), dict):
            failed_state["mirror"] = dict(request_ctx.get("mirror") or {})
        _save_job_state(output_dir, trading_cfg, state_file, failed_state)
        return {"ok": False, "error": failed_state["closed_reason"], "state": failed_state}

    mirror_state = dict(request_ctx.get("mirror") or {}) if isinstance(request_ctx.get("mirror"), dict) else {}
    if mirror_state:
        request_ctx["mirror"] = mirror_state

    # Agent A: build plan/report.
    mode_a_result = run_investing_agent(
        app_config=app_config,
        results=results,
        trading_cfg=trading_cfg,
        output_dir=output_dir,
        selected_model=selected_model,
        mode_override="mode_a",
    )
    plan = (mode_a_result.get("plan") or {})
    forced_plan = dict(request_ctx.get("forced_plan_override") or {}) if isinstance(request_ctx.get("forced_plan_override"), dict) else {}
    if forced_plan:
        if str(forced_plan.get("source") or "").strip() == "":
            forced_plan["source"] = str(request_ctx.get("forced_plan_override_source") or "TSMM_FORCE_AGENT_A_PLAN_JSON")
        plan = _apply_forced_agent_a_plan_override(plan, forced_plan, trading_cfg)
    if effective_submission_mode:
        plan["order_submission_mode"] = effective_submission_mode
    if request_ctx:
        plan["analysis_grounding_timeframe"] = request_ctx.get("grounding_timeframe")
        plan["analysis_grounding_timeframe_minutes"] = request_ctx.get("grounding_timeframe_minutes")
        plan["session_operation_index"] = request_ctx.get("session_operation_index")

    fallback_attempts = _agent_a_fallback_candidates(trading_cfg)
    dynamic_candidates = _discover_dynamic_fallback_candidates(trading_cfg)
    if dynamic_candidates:
        seen = {str(c.get("config_path") or "") for c in fallback_attempts}
        for item in dynamic_candidates:
            cfg_path = str(item.get("config_path") or "")
            if cfg_path and cfg_path not in seen:
                fallback_attempts.append(item)
                seen.add(cfg_path)
    fallback_cfg = (trading_cfg.get("agent_a_fallback") or {})
    fallback_max_attempts = max(int(fallback_cfg.get("max_attempts", 24) or 24), 1)
    fallback_log = []
    signal_policy_enabled = bool((trading_cfg.get("signal_policy") or {}).get("enabled", False))
    if signal_policy_enabled and not bool(fallback_cfg.get("allow_signal_policy_bypass", False)):
        # A single endpoint must not resurrect a trade rejected by the joint
        # quality, range, probability, and cost gates.
        fallback_attempts = []
        fallback_log.append(
            {
                "skipped": True,
                "reason": "joint_signal_policy_rejection_is_authoritative",
            }
        )
    initial_decision = str(plan.get("decision", "hold")).lower()
    if initial_decision not in {"buy", "sell"} and fallback_attempts:
        endpoints_cfg = dict(trading_cfg.get("model_endpoints") or {})
        for idx, attempt in enumerate(fallback_attempts[:fallback_max_attempts], start=1):
            cfg_path = attempt.get("config_path") or ""
            model_name = attempt.get("model") or selected_model
            timeframe = str(attempt.get("timeframe") or "").strip()
            if not cfg_path:
                continue

            endpoint_item = endpoints_cfg.get(timeframe)
            if not timeframe or endpoint_item is None:
                fallback_log.append(
                    {
                        "attempt": idx,
                        "config_path": cfg_path,
                        "timeframe": timeframe,
                        "model": model_name,
                        "error": "missing_timeframe_endpoint",
                    }
                )
                continue

            try:
                mtf = _collect_mode_b_signals(
                    model_endpoints={timeframe: endpoint_item},
                    trading_cfg=trading_cfg,
                    timeout_sec=3.0,
                    config_overrides={timeframe: cfg_path},
                )
                tf_sig = ((mtf.get("timeframes") or {}).get(timeframe) or {})
                s = int(tf_sig.get("signal", 0) or 0)
                run_decision = "buy" if s > 0 else ("sell" if s < 0 else "hold")
                run_conf = float(tf_sig.get("confidence", 0.5) or 0.5)

                fallback_log.append(
                    {
                        "attempt": idx,
                        "config_path": cfg_path,
                        "timeframe": timeframe,
                        "model": model_name,
                        "decision": run_decision,
                        "confidence": run_conf,
                    }
                )
                if run_decision in {"buy", "sell"}:
                    prior_decision = str(plan.get("decision") or "hold").lower()
                    plan["decision"] = run_decision
                    plan["confidence"] = run_conf
                    if timeframe:
                        plan["fallback_timeframe"] = timeframe
                    if model_name:
                        plan["model"] = model_name
                    plan["fallback_config_path"] = cfg_path
                    if prior_decision != run_decision:
                        plan = _realign_plan_risk_levels(plan, run_decision)
                        plan.setdefault("risk_notes", []).append(
                            f"Fallback selected {run_decision} on timeframe {timeframe or 'n/a'}; risk levels realigned to the fallback side."
                        )
                    break
            except Exception as e:
                fallback_log.append(
                    {
                        "attempt": idx,
                        "config_path": cfg_path,
                        "timeframe": timeframe,
                        "model": model_name,
                        "error": str(e),
                    }
                )
    symbol = str(((trading_cfg.get("execution") or {}).get("symbol") or app_config.get("target_col") or "UNKNOWN"))

    # Optional persistent memory/knowledge layer.
    mem_cfg = (trading_cfg.get("memory") or {})
    mem_store = _get_memory_store(trading_cfg, output_dir)
    if mem_store is not None and bool(mem_cfg.get("ingest_kb_on_start", False)):
        kb_paths = [str(p) for p in (mem_cfg.get("knowledge_base_paths") or [])]
        mem_store.ingest_documents(kb_paths, kind="kb_document", symbol=symbol)

    # Optional intelligence layer: pull external market sentiment for context.
    sentiment_cfg = (trading_cfg.get("sentiment") or {})
    if bool(sentiment_cfg.get("enabled", True)):
        sentiment = aggregate_market_sentiment(sentiment_cfg)
    else:
        sentiment = {"enabled": False, "reason": "disabled_in_config", "sources": [], "aggregate": {}}

    llm_mode_a_cfg = (trading_cfg.get("llm") or {})
    llm_analysis: Dict[str, Any] = {"enabled": bool(llm_mode_a_cfg.get("enabled", False))}
    memory_context = ""
    if mem_store is not None:
        memory_context = mem_store.build_context_block(
            query=(
                f"symbol={symbol} model={plan.get('model')} decision={plan.get('decision')} "
                f"signal_score={plan.get('signal_score')} confidence={plan.get('confidence')}"
            ),
            limit=int(mem_cfg.get("retrieval_top_k", 6) or 6),
            symbol=symbol,
            kinds=(mem_cfg.get("retrieval_kinds") or None),
        )

    if bool(llm_mode_a_cfg.get("enabled", False)) and bool(((trading_cfg.get("mode_a") or {}).get("use_llm_explanation", False))):
        sa_cfg = (llm_mode_a_cfg.get("signal_analysis") or {})
        context_cap = int(sa_cfg.get("max_context_chars", 14000) or 14000)
        sentiment_json = _trim_json_for_prompt(sentiment, max_chars=max(int(context_cap * 0.35), 2000))
        tech_json = _trim_json_for_prompt(
            {
                "plan": plan,
                "backtest": mode_a_result.get("backtest", {}),
                "evaluation": (results.get("evaluation") or {}).get(str(plan.get("model", "")), {}),
            },
            max_chars=max(int(context_cap * 0.65), 4000),
        )
        prompt = (
            "You are Agent A signal analyst. Produce detailed, risk-aware market operation reasoning. "
            "Do not change numerical plan fields; explain and assess confidence, failure modes, and scenario handling.\n\n"
            "TECHNICAL_CONTEXT_JSON:\n"
            f"{tech_json}\n\n"
            "SENTIMENT_CONTEXT_JSON:\n"
            f"{sentiment_json}\n\n"
            "MEMORY_CONTEXT:\n"
            f"{memory_context or '<none>'}\n\n"
            "Return concise structured explanation with sections: thesis, confirmation, contradictions, risk_controls, invalidation."
        )
        llm_out = _maybe_llm_assist(trading_cfg, prompt)
        if bool(llm_out.get("ok", False)) and llm_out.get("text"):
            plan["llm_reasoning"] = llm_out.get("text")
            llm_analysis = {
                "enabled": True,
                "ok": True,
                "provider": llm_out.get("provider"),
                "provider_type": llm_out.get("provider_type"),
                "text": llm_out.get("text"),
            }
        else:
            plan["llm_reasoning_error"] = llm_out.get("error", "llm call failed")
            llm_analysis = {
                "enabled": True,
                "ok": False,
                "provider": llm_out.get("provider"),
                "provider_type": llm_out.get("provider_type"),
                "error": llm_out.get("error", "llm call failed"),
            }

    signal_json_path = _write_agent_a_signal_json(
        output_dir=output_dir,
        app_config=app_config,
        trading_cfg=trading_cfg,
        results=results,
        mode_a_result=mode_a_result,
        llm_analysis=llm_analysis,
        sentiment=sentiment,
        memory_context=memory_context,
    )

    autonomous_trigger = str((request_ctx or {}).get("autonomous_trigger") or "").strip().lower()
    is_mirror_exact_copy = bool(plan.get("forced_plan_override")) and str(plan.get("forced_plan_override_source") or "").strip() == "account_mirror_exact_copy"
    if autonomous_trigger == "autonomous_followup" and not is_mirror_exact_copy:
        should_trade, autonomous_reason = _autonomous_followup_meets_entry_thresholds(plan, trading_cfg)
        if not should_trade:
            state = {
                "job_id": resolved_job_id,
                "job_type": "trading_job",
                "status": "completed",
                "runner_pid": _runner_pid(),
                "started_at": _iso(_now_utc()),
                "ended_at": _iso(_now_utc()),
                "stage": "agent_a",
                "mode": "mode_a",
                "plan": plan,
                "report_path": mode_a_result.get("report_path"),
                "signal_json_path": signal_json_path,
                "state_path": state_file,
                "mode_a": mode_a_result,
                "sentiment": sentiment,
                "llm_analysis": llm_analysis,
                "memory_context": memory_context,
                "last_data_sync": data_sync,
                "request_context": request_ctx,
                "notifications": [],
                "closed_reason": "autonomous_followup_filtered",
                "autonomous_filter_reason": autonomous_reason,
                "order_submission_mode": _agent_a_order_submission_mode(plan, trading_cfg),
            }
            if mirror_state:
                state["mirror"] = mirror_state
            if fallback_log:
                state["agent_a_fallback_attempts"] = fallback_log
            _save_job_state(output_dir, trading_cfg, state_file, state)
            try:
                log_agent_a_plan_feedback(output_dir=output_dir, trading_cfg=trading_cfg, state=state)
            except Exception:
                pass
            return state

    # Launch account mirror only AFTER the autonomous-followup filter has passed,
    # so FTMO does not receive mirror jobs for trades that Pepperstone filters out.
    mirror_launch = _launch_account_mirror_start(
        app_config=app_config,
        trading_cfg=trading_cfg,
        output_dir=output_dir,
        logger=logger,
        selected_model=selected_model,
        source_job_id=resolved_job_id,
        request_context=request_ctx,
        source_plan=plan,
    )
    if not mirror_state and not bool(mirror_launch.get("skipped", False)):
        mirror_state = dict(mirror_launch)
    if mirror_state:
        request_ctx["mirror"] = mirror_state

    story = (
        "Agent A completed signal analysis for a single-session operation. "
        f"Decision={plan.get('decision')}, model={plan.get('model')}, confidence={plan.get('confidence')}, "
        f"cm_accuracy={plan.get('cm_accuracy')}, signal_score={plan.get('signal_score')}, "
        f"entry={plan.get('entry')}, sl={plan.get('stop_loss')}, tp={plan.get('take_profit')}, "
        f"grounding_timeframe={plan.get('analysis_grounding_timeframe', 'n/a')}, submission_mode={plan.get('order_submission_mode', 'n/a')}."
    )
    notify_res = _notify(
        output_dir=output_dir,
        trading_cfg=trading_cfg,
        channel="agent_a",
        kind="plan_ready",
        message=story,
        requires_approval=False,
        emergency=False,
        approval_deadline_utc=None,
        metadata={"signal_json_path": signal_json_path},
        force_telegram=True,
        job_id=resolved_job_id,
    )
    if mem_store is not None:
        mem_store.add_memory(
            kind="signal",
            timeframe="7h",
            symbol=symbol,
            title=f"AgentA-{plan.get('model', 'unknown')}",
            text_payload=json.dumps(
                {
                    "decision": plan.get("decision"),
                    "entry": plan.get("entry"),
                    "stop_loss": plan.get("stop_loss"),
                    "take_profit": plan.get("take_profit"),
                    "confidence": plan.get("confidence"),
                    "cm_accuracy": plan.get("cm_accuracy"),
                    "signal_score": plan.get("signal_score"),
                    "input_fooling_risk": plan.get("input_fooling_risk"),
                    "llm_analysis": llm_analysis,
                    "sentiment": sentiment.get("aggregate", {}),
                },
                default=str,
            ),
            metadata={"signal_json_path": signal_json_path},
        )
    state = {
        "job_id": resolved_job_id,
        "job_type": "trading_job",
        "status": "agent_a_completed",
        "runner_pid": _runner_pid(),
        "started_at": _iso(_now_utc()),
        "stage": "agent_a",
        "mode": "mode_a",
        "plan": plan,
        "report_path": mode_a_result.get("report_path"),
        "signal_json_path": signal_json_path,
        "state_path": state_file,
        "mode_a": mode_a_result,
        "sentiment": sentiment,
        "llm_analysis": llm_analysis,
        "memory_context": memory_context,
        "last_data_sync": data_sync,
        "request_context": request_ctx,
        "notifications": [notify_res],
    }
    if mirror_state:
        state["mirror"] = mirror_state
    auto_created = str(os.environ.get("TSMM_AGENT_A_AUTO_CREATED", "")).strip().lower() in {"1", "true", "yes", "y"}
    state["auto_created"] = bool(auto_created)
    state["order_submission_mode"] = _agent_a_order_submission_mode(plan, trading_cfg)
    if fallback_log:
        state["agent_a_fallback_attempts"] = fallback_log
    _save_job_state(output_dir, trading_cfg, state_file, state)
    try:
        log_agent_a_plan_feedback(output_dir=output_dir, trading_cfg=trading_cfg, state=state)
    except Exception:
        pass

    decision = str(plan.get("decision", "hold")).lower()
    if decision not in {"buy", "sell"}:
        no_trade_story = (
            "Agent A plan completed with NO TRADE. "
            f"decision={plan.get('decision')}, model={plan.get('model')}, "
            "reason=signal blocked by confidence/confusion thresholds."
        )
        state.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_a",
                kind="no_trade",
                message=no_trade_story,
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={"signal_json_path": signal_json_path},
                force_telegram=True,
                job_id=resolved_job_id,
            )
        )
        state["status"] = "completed"
        state["closed_reason"] = "agent_a_no_trade"
        state["ended_at"] = _iso(_now_utc())
        _save_job_state(output_dir, trading_cfg, state_file, state)
        return state

    submission_mode = str(state.get("order_submission_mode") or _agent_a_order_submission_mode(plan, trading_cfg))
    approval_required, approval_reason, active_agent_b_count, approval_threshold = _agent_a_approval_decision(
        output_dir,
        trading_cfg,
        auto_created=bool(auto_created),
        autonomous_trigger=autonomous_trigger,
        submission_mode=submission_mode,
    )
    deadline_a = _now_utc() + timedelta(seconds=int(policy.get("normal_timeout_seconds", 7200)))
    programmed_expiration_minutes = _programmed_order_expiration_minutes(
        trading_cfg,
        autonomous_trigger,
    )
    order_label = "market MT5 order" if submission_mode == "market" else "programmed MT5 order"
    bypass_suffix = "" if submission_mode == "market" else f" If accepted as a programmed order, it will use the configured pending-order expiry window of {programmed_expiration_minutes} minutes."
    approval_suffix = "" if submission_mode == "market" else f" If MT5 accepts it as pending, the order will use the configured expiry window of {programmed_expiration_minutes} minutes."
    approval_details = (
        f"job_id={resolved_job_id}; signal={plan.get('decision')}/{plan.get('model')}; "
        f"entry={plan.get('entry')}; sl={plan.get('stop_loss')}; tp={plan.get('take_profit')}; "
        f"submission_mode={submission_mode}; "
        f"approval_reason={approval_reason}; active_agent_b_count={active_agent_b_count}; approval_threshold={approval_threshold}; "
        f"programmed_order_expires_in_minutes={programmed_expiration_minutes if submission_mode == 'programmed' else 'n/a'}; "
        f"confidence={plan.get('confidence')}; approve_cmd=/tsmm trading approve --job-id {resolved_job_id}; "
        f"reject_cmd=/tsmm trading reject --job-id {resolved_job_id}"
    )
    state["approval_required"] = bool(approval_required)
    state["approval_reason"] = approval_reason
    state["active_agent_b_count"] = active_agent_b_count
    _save_job_state(output_dir, trading_cfg, state_file, state)

    if not approval_required:
        state["agent_a_approved"] = True
        state.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_a",
                kind="approval_bypassed",
                message=(
                    f"Agent A order bypassed manual approval and will be submitted to MT5 immediately. "
                    f"reason={approval_reason}; active_agent_b_count={active_agent_b_count}; approval_threshold={approval_threshold}."
                    f"{bypass_suffix}"
                ),
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={
                    "signal_json_path": signal_json_path,
                    "auto_created": True,
                    "approval_reason": approval_reason,
                    "active_agent_b_count": active_agent_b_count,
                    "approval_threshold": approval_threshold,
                },
                force_telegram=True,
                job_id=resolved_job_id,
            )
        )
        _save_job_state(output_dir, trading_cfg, state_file, state)
        return _execute_approved_order(
            app_config=app_config,
            trading_cfg=trading_cfg,
            output_dir=output_dir,
            logger=logger,
            state=state,
            state_file=state_file,
        )

    approval_story = (
        f"Agent A requests approval to place {order_label} for this session. "
        f"Plan summary: decision={plan.get('decision')}, entry={plan.get('entry')}, "
        f"sl={plan.get('stop_loss')}, tp={plan.get('take_profit')}. "
        f"{approval_details}"
    )
    notify_res_appr = _notify(
        output_dir=output_dir,
        trading_cfg=trading_cfg,
        channel="agent_a",
        kind="approval_request",
        message=approval_story,
        requires_approval=True,
        emergency=False,
        approval_deadline_utc=_iso(deadline_a),
        metadata={
            "signal_json_path": signal_json_path,
            "decision": plan.get("decision"),
            "model": plan.get("model"),
            "entry": plan.get("entry"),
            "stop_loss": plan.get("stop_loss"),
            "take_profit": plan.get("take_profit"),
                "order_submission_mode": submission_mode,
                "programmed_order_expiration_minutes": programmed_expiration_minutes,
            "confidence": plan.get("confidence"),
            "approve_cmd": f"/tsmm trading approve --job-id {resolved_job_id}",
            "reject_cmd": f"/tsmm trading reject --job-id {resolved_job_id}",
        },
        job_id=resolved_job_id,
    )
    state.setdefault("notifications", []).append(notify_res_appr)
    _save_job_state(output_dir, trading_cfg, state_file, state)
    approved = request_approval(
        title="TSMM Agent A Approval",
        message=(
            f"Agent A generated a trading plan with one {order_label}. "
            "Approve order placement in MT5? "
            f"{approval_details}"
        ),
        timeout_sec=int(policy.get("normal_timeout_seconds", 7200)),
        channels=channels,
        output_dir=output_dir,
        trading_cfg=trading_cfg,
        approval_metadata={
            "channel": "agent_a",
            "kind": "approval_request",
            "signal_json_path": signal_json_path,
            "decision": plan.get("decision"),
            "model": plan.get("model"),
            "entry": plan.get("entry"),
            "stop_loss": plan.get("stop_loss"),
            "take_profit": plan.get("take_profit"),
            "order_submission_mode": submission_mode,
            "programmed_order_expiration_minutes": programmed_expiration_minutes,
            "confidence": plan.get("confidence"),
            "approve_cmd": f"/tsmm trading approve --job-id {resolved_job_id}",
            "reject_cmd": f"/tsmm trading reject --job-id {resolved_job_id}",
        },
        job_id=resolved_job_id,
    )
    state["agent_a_approved"] = bool(approved)

    if not approved:
        state.setdefault("notifications", []).append(
            _notify(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_a",
                kind="approval_rejected",
                message="Agent A order placement was rejected or timed out. No MT5 order was sent.",
                requires_approval=False,
                emergency=False,
                approval_deadline_utc=None,
                metadata={"signal_json_path": signal_json_path},
                force_telegram=True,
                job_id=resolved_job_id,
            )
        )
        state["status"] = "completed"
        state["closed_reason"] = "agent_a_not_approved"
        state["ended_at"] = _iso(_now_utc())
        _save_job_state(output_dir, trading_cfg, state_file, state)
        return state

    state.setdefault("notifications", []).append(
        _notify(
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            channel="agent_a",
            kind="approval_confirmed",
            message=f"Agent A approval received. Submitting {order_label} now.{approval_suffix}",
            requires_approval=False,
            emergency=False,
            approval_deadline_utc=None,
            metadata={"signal_json_path": signal_json_path},
            force_telegram=True,
            job_id=resolved_job_id,
        )
    )

    return _execute_approved_order(
        app_config=app_config,
        trading_cfg=trading_cfg,
        output_dir=output_dir,
        logger=logger,
        state=state,
        state_file=state_file,
    )


def _execute_approved_order(
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    logger,
    state: Dict[str, Any],
    state_file: str,
) -> Dict[str, Any]:
    plan = dict(state.get("plan") or {})
    decision = str(plan.get("decision") or "hold").strip().lower()
    signal_json_path = str(state.get("signal_json_path") or "").strip()
    resolved_job_id = str(state.get("job_id") or "").strip()
    tj = (trading_cfg.get("trading_job") or {})
    wait_fill_sec = int(tj.get("fill_check_seconds", 30) or 30)
    request_ctx = dict(state.get("request_context") or {})
    autonomous_trigger = str(request_ctx.get("autonomous_trigger") or "").strip().lower()
    max_wait_fill_minutes = _programmed_order_expiration_minutes(
        trading_cfg,
        autonomous_trigger,
    )
    submission_mode = str(state.get("order_submission_mode") or _agent_a_order_submission_mode(plan, trading_cfg))

    shadow_cfg = dict(trading_cfg.get("shadow_mode") or {})
    if bool(shadow_cfg.get("enabled", False)):
        state["status"] = "completed"
        state["closed_reason"] = "shadow_mode_no_broker_submission"
        state["shadow_evaluation"] = {
            "enabled": True,
            "recorded_at": _iso(_now_utc()),
            "decision": decision,
            "plan": plan,
        }
        state["ended_at"] = _iso(_now_utc())
        _save_job_state(output_dir, trading_cfg, state_file, state)
        return state

    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, msg_conn = adapter.connect()
    if not ok_conn:
        state["status"] = "failed"
        state["closed_reason"] = f"mt5_connect_failed: {msg_conn}"
        state["ended_at"] = _iso(_now_utc())
        _save_job_state(output_dir, trading_cfg, state_file, state)
        return state

    try:
        exec_cfg = (trading_cfg.get("execution") or {})
        schedule_cfg = (exec_cfg.get("market_schedule") or {})
        schedule_symbol = str(exec_cfg.get("symbol") or app_config.get("symbol") or "XAUUSD")
        max_wait_seconds = int(schedule_cfg.get("max_wait_seconds", 7200) or 7200)
        reopen_at_utc = _next_market_open_utc(trading_cfg, schedule_symbol)
        if reopen_at_utc is not None:
            wait_seconds = max(int((reopen_at_utc - _now_utc()).total_seconds()), 1)
            wait_seconds = min(wait_seconds, max_wait_seconds)
            state["status"] = "waiting_market_open"
            state["market_reopen_at_utc"] = _iso(reopen_at_utc)
            state.setdefault("notifications", []).append(
                _notify(
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    channel="agent_a",
                    kind="market_closed_wait",
                    message=(
                        "Agent A approval was received while the market is in a configured closed window. "
                        f"Waiting to submit the MT5 order until market reopen at {_iso(reopen_at_utc)} UTC."
                    ),
                    requires_approval=False,
                    emergency=False,
                    approval_deadline_utc=None,
                    metadata={"signal_json_path": signal_json_path, "market_reopen_at_utc": _iso(reopen_at_utc)},
                    force_telegram=True,
                    job_id=resolved_job_id,
                )
            )
            _save_job_state(output_dir, trading_cfg, state_file, state)
            time.sleep(wait_seconds)
            state["status"] = "agent_a_completed"

        duplicate_guard_enabled = bool(tj.get("prevent_duplicate_programmed_orders", True))
        duplicate_entry_tolerance = _safe_float(tj.get("duplicate_entry_tolerance", 0.15), 0.15)
        duplicate_volume_tolerance = _safe_float(tj.get("duplicate_volume_tolerance", 1e-9), 1e-9)
        duplicate_tsmm_only = bool(tj.get("duplicate_tsmm_only", True))
        strict_open_guard_enabled = bool(tj.get("prevent_new_programmed_when_open_position", False))
        strict_pending_guard_enabled = bool(tj.get("prevent_new_programmed_when_pending_exists", False))
        target_side = str(plan.get("decision", "hold")).strip().lower()
        target_entry = _safe_float(plan.get("entry"), 0.0)
        target_volume = _order_volume_for_plan(plan, trading_cfg)
        stack_policy = _intentional_same_side_stack_policy(plan, trading_cfg)
        order_stop_loss, order_take_profit, stop_protection = _order_risk_levels_for_plan(plan, trading_cfg)
        stop_protection["requested_volume"] = _safe_float(plan.get("volume", exec_cfg.get("default_volume", 0.01)), 0.01)
        stop_protection["effective_volume"] = target_volume
        plan["stop_loss_protection"] = stop_protection
        state["plan"] = plan
        state["stop_loss_protection"] = stop_protection

        def _finalize_duplicate_guard_block(blocked_order: Dict[str, Any]) -> Dict[str, Any]:
            dedup_payload = dict(blocked_order.get("dedup") or {})
            pending_orders = list(dedup_payload.get("pending_orders") or [])
            open_positions = list(dedup_payload.get("open_positions") or [])
            existing_similar_orders = len(pending_orders) + len(open_positions)
            pending_ticket = int(((pending_orders[0] or {}).get("order_ticket", 0) or 0)) if pending_orders else 0
            position_ticket = int(((open_positions[0] or {}).get("ticket", 0) or 0)) if open_positions else 0

            policy = dict((blocked_order.get("stack_policy") or {}))
            policy_allowed_existing = max(_safe_int(policy.get("allowed_existing_similar_orders", 0), 0), 0)
            policy_target_orders = max(_safe_int(policy.get("target_orders", 1), 1), 1)

            blocked_refs: List[str] = []
            if pending_ticket > 0:
                blocked_refs.append(f"pending_ticket={pending_ticket}")
            if position_ticket > 0:
                blocked_refs.append(f"position_ticket={position_ticket}")
            blocked_ref_text = ", ".join(blocked_refs) if blocked_refs else "existing similar exposure"

            state.pop("programmed_order_expiration_utc", None)
            state["order"] = {
                "ok": False,
                "blocked": True,
                "message": str(blocked_order.get("message") or "duplicate_order_prevented"),
                "dedup": dedup_payload,
                "stack_policy": policy,
                "existing_similar_orders": existing_similar_orders,
            }
            state.setdefault("notifications", []).append(
                _notify(
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    channel="agent_a",
                    kind="duplicate_order_prevented",
                    message=(
                        "Agent A duplicate guard blocked a new programmed MT5 order because a similar exposure already exists "
                        f"({blocked_ref_text}) for {schedule_symbol} {target_side} near entry {target_entry:.5f}. "
                        f"existing_similar_orders={existing_similar_orders}; "
                        f"allowed_existing={policy_allowed_existing}; target_orders={policy_target_orders}."
                    ),
                    requires_approval=False,
                    emergency=False,
                    approval_deadline_utc=None,
                    metadata={"signal_json_path": signal_json_path, "dedup": dedup_payload, "stack_policy": policy, "existing_similar_orders": existing_similar_orders},
                    force_telegram=True,
                    job_id=resolved_job_id,
                )
            )
            state["status"] = "completed"
            state["closed_reason"] = "duplicate_order_prevented"
            state["ended_at"] = _iso(_now_utc())
            _save_job_state(output_dir, trading_cfg, state_file, state)
            return state

        def _finalize_prop_firm_guard_block(blocked_order: Dict[str, Any]) -> Dict[str, Any]:
            guard = dict(blocked_order.get("guard") or {})
            reason = str(guard.get("reason") or blocked_order.get("message") or "prop_firm_guard_blocked")
            state["prop_firm_guard"] = guard
            state["order"] = {
                "ok": False,
                "blocked": True,
                "message": "prop_firm_guard_blocked",
                "reason": reason,
            }
            state.setdefault("notifications", []).append(
                _notify(
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    channel="agent_a",
                    kind="prop_firm_guard_blocked",
                    message=(
                        "The prop-firm safety guard blocked this MT5 order before submission. "
                        f"reason={reason}. No broker order was sent."
                    ),
                    requires_approval=False,
                    emergency=False,
                    approval_deadline_utc=None,
                    metadata={"signal_json_path": signal_json_path, "guard": guard},
                    force_telegram=True,
                    job_id=resolved_job_id,
                )
            )
            state["status"] = "completed"
            state["closed_reason"] = f"prop_firm_guard_blocked:{reason}"
            state["ended_at"] = _iso(_now_utc())
            _save_job_state(output_dir, trading_cfg, state_file, state)
            return state

        def _submit_order_attempt() -> Dict[str, Any]:
            guard = _prop_firm_guard_preflight(
                adapter=adapter,
                trading_cfg=trading_cfg,
                symbol=schedule_symbol,
                side=target_side,
                volume=target_volume,
                entry=target_entry,
                stop_loss=order_stop_loss,
            )
            state["prop_firm_guard"] = guard
            _save_job_state(output_dir, trading_cfg, state_file, state)
            if not guard.get("ok"):
                return {
                    "ok": False,
                    "prop_firm_guard_blocked": True,
                    "message": "prop_firm_guard_blocked",
                    "guard": guard,
                }

            effective_volume = _safe_float(guard.get("sized_volume"), target_volume)
            effective_plan = dict(plan)
            effective_plan["volume"] = effective_volume
            state["plan"] = effective_plan
            state["account_risk_sizing"] = guard.get("sizing")

            if submission_mode == "programmed":
                state.pop("programmed_order_expiration_utc", None)

                if strict_open_guard_enabled or strict_pending_guard_enabled:
                    exposure_res = _find_symbol_tsmm_exposure(
                        adapter=adapter,
                        symbol=schedule_symbol,
                        tsmm_only=duplicate_tsmm_only,
                    )
                    has_open = bool(exposure_res.get("open_positions"))
                    has_pending = bool(exposure_res.get("pending_orders"))
                    if (strict_open_guard_enabled and has_open) or (strict_pending_guard_enabled and has_pending):
                        reason_bits: List[str] = []
                        if strict_open_guard_enabled and has_open:
                            reason_bits.append("existing_open_position")
                        if strict_pending_guard_enabled and has_pending:
                            reason_bits.append("existing_pending_order")
                        blocked = dict(exposure_res)
                        blocked["strict_guard_reasons"] = reason_bits
                        return {
                            "ok": False,
                            "dedup_blocked": True,
                            "message": "strict_programmed_exposure_guard_blocked",
                            "dedup": blocked,
                        }

                if duplicate_guard_enabled and target_side in {"buy", "sell"} and target_entry > 0.0:
                    dedup_res = _find_similar_mt5_exposure(
                        adapter=adapter,
                        symbol=schedule_symbol,
                        side=target_side,
                        entry=target_entry,
                        volume=effective_volume,
                        entry_tolerance=duplicate_entry_tolerance,
                        volume_tolerance=duplicate_volume_tolerance,
                        tsmm_only=duplicate_tsmm_only,
                    )
                    existing_similar_orders = len(dedup_res.get("pending_orders") or []) + len(dedup_res.get("open_positions") or [])
                    allowed_existing_similar_orders = max(_safe_int(stack_policy.get("allowed_existing_similar_orders", 0), 0), 0)
                    target_orders = max(_safe_int(stack_policy.get("target_orders", 1), 1), 1)
                    if existing_similar_orders > 0 and existing_similar_orders <= allowed_existing_similar_orders:
                        state.setdefault("notifications", []).append(
                            _notify(
                                output_dir=output_dir,
                                trading_cfg=trading_cfg,
                                channel="agent_a",
                                kind="duplicate_guard_stack_allowed",
                                message=(
                                    "Agent A intentional stacking policy allowed a same-side programmed order that would normally "
                                    "be treated as duplicate. "
                                    f"existing_similar_orders={existing_similar_orders}; "
                                    f"allowed_existing={allowed_existing_similar_orders}; target_orders={target_orders}; "
                                    f"reason={stack_policy.get('reason', 'n/a')}."
                                ),
                                requires_approval=False,
                                emergency=False,
                                approval_deadline_utc=None,
                                metadata={
                                    "signal_json_path": signal_json_path,
                                    "dedup": dedup_res,
                                    "stack_policy": stack_policy,
                                    "existing_similar_orders": existing_similar_orders,
                                },
                                force_telegram=False,
                                job_id=resolved_job_id,
                            )
                        )
                    if existing_similar_orders > allowed_existing_similar_orders:
                        dedup_payload = dict(dedup_res)
                        dedup_payload["stack_policy"] = stack_policy
                        dedup_payload["existing_similar_orders"] = existing_similar_orders
                        return {
                            "ok": False,
                            "dedup_blocked": True,
                            "message": "duplicate_programmed_order_prevented",
                            "dedup": dedup_payload,
                            "stack_policy": stack_policy,
                        }

                programmed_order_expiration_utc = _programmed_order_expiration_utc(
                    trading_cfg,
                    autonomous_trigger=autonomous_trigger,
                )
                state["programmed_order_expiration_utc"] = _iso(programmed_order_expiration_utc)
                return _place_programmed_order(
                    adapter,
                    app_config,
                    trading_cfg,
                    effective_plan,
                    expiration_utc=programmed_order_expiration_utc,
                )

            state.pop("programmed_order_expiration_utc", None)
            return adapter.place_market_order(
                symbol=schedule_symbol,
                side=str(plan.get("decision", "hold")).lower(),
                volume=effective_volume,
                stop_loss=order_stop_loss,
                take_profit=order_take_profit,
            )

        order_res = _submit_order_attempt()
        if order_res.get("prop_firm_guard_blocked"):
            return _finalize_prop_firm_guard_block(order_res)
        if order_res.get("dedup_blocked"):
            return _finalize_duplicate_guard_block(order_res)
        if not order_res.get("ok") and _is_market_closed_retcode(order_res):
            reopen_at_utc = _next_market_open_utc(trading_cfg, schedule_symbol)
            if reopen_at_utc is not None:
                wait_seconds = max(int((reopen_at_utc - _now_utc()).total_seconds()), 1)
                wait_seconds = min(wait_seconds, max_wait_seconds)
                state["status"] = "waiting_market_open"
                state["market_reopen_at_utc"] = _iso(reopen_at_utc)
                state["order"] = order_res
                state.setdefault("notifications", []).append(
                    _notify(
                        output_dir=output_dir,
                        trading_cfg=trading_cfg,
                        channel="agent_a",
                        kind="market_closed_retry",
                        message=(
                            "Agent A approved order hit an MT5 market-closed response. "
                            f"Retrying order placement at market reopen {_iso(reopen_at_utc)} UTC."
                        ),
                        requires_approval=False,
                        emergency=False,
                        approval_deadline_utc=None,
                        metadata={"signal_json_path": signal_json_path, "order": order_res, "market_reopen_at_utc": _iso(reopen_at_utc)},
                        force_telegram=True,
                        job_id=resolved_job_id,
                    )
                )
                _save_job_state(output_dir, trading_cfg, state_file, state)
                time.sleep(wait_seconds)
                state["status"] = "agent_a_completed"
                order_res = _submit_order_attempt()
                if order_res.get("prop_firm_guard_blocked"):
                    return _finalize_prop_firm_guard_block(order_res)
                if order_res.get("dedup_blocked"):
                    return _finalize_duplicate_guard_block(order_res)

        if submission_mode == "market":
            return _finalize_agent_a_market_order_submission(
                adapter=adapter,
                app_config=app_config,
                trading_cfg=trading_cfg,
                output_dir=output_dir,
                logger=logger,
                state=state,
                state_file=state_file,
                order_res=order_res,
            )
        return _finalize_agent_a_order_submission(
            adapter=adapter,
            app_config=app_config,
            trading_cfg=trading_cfg,
            output_dir=output_dir,
            logger=logger,
            state=state,
            state_file=state_file,
            order_res=order_res,
            wait_fill_sec=wait_fill_sec,
            max_wait_fill_minutes=max_wait_fill_minutes,
        )
    finally:
        adapter.shutdown()


def resume_trading_job(
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    logger,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    requested_job_id = str(job_id or "").strip()
    resolved_job_id = requested_job_id or _resolve_job_id(output_dir, trading_cfg, None)
    state_file = _state_path(output_dir, trading_cfg, resolved_job_id) if resolved_job_id else _state_path(output_dir, trading_cfg)
    state = _load_state(state_file)
    if not state and not requested_job_id:
        legacy_state_file = _state_path(output_dir, trading_cfg)
        state = _load_state(legacy_state_file)
        if state:
            resolved_job_id = requested_job_id or _new_job_id(trading_cfg)
            state_file = _state_path(output_dir, trading_cfg, resolved_job_id)
    if not state:
        return {"ok": False, "error": "No trading job state found to resume"}

    loaded_job_id = str(state.get("job_id") or "").strip()
    if requested_job_id and loaded_job_id and loaded_job_id != resolved_job_id:
        return {
            "ok": False,
            "error": (
                "Job state identity mismatch: "
                f"requested_job_id={resolved_job_id}; state_job_id={loaded_job_id}; state_path={state_file}"
            ),
            "state": state,
        }

    state["job_id"] = loaded_job_id or resolved_job_id
    state["state_path"] = state_file
    state["runner_pid"] = _runner_pid()
    state["runner_started_at"] = _iso(_now_utc())
    _save_job_state(output_dir, trading_cfg, state_file, state)

    recovered = _recover_agent_b_state_from_live_position(state, trading_cfg, output_dir)
    if recovered.get("ok"):
        state = recovered["state"]
        _save_job_state(output_dir, trading_cfg, state_file, state)
        out = _run_agent_b_loop(state, trading_cfg, output_dir, state_file, logger)
        out["followup_agent_a"] = _launch_followup_agent_a_start(
            app_config=app_config,
            trading_cfg=trading_cfg,
            output_dir=output_dir,
            logger=logger,
            completed_state=out,
        )
        _save_job_state(output_dir, trading_cfg, state_file, out)
        return {"ok": True, "state": out, "message": "Recovered live MT5 position and resumed Agent B supervision"}

    if str(state.get("status")) in {"completed", "closed", "failed", "stopped"}:
        return {"ok": True, "state": state, "message": "Job already finished"}

    if str(state.get("stage") or "").strip() == "agent_a" and bool(state.get("agent_a_approved")):
        recovered_order = _recover_agent_a_state_from_pending_order(state, trading_cfg, output_dir)
        if recovered_order.get("ok"):
            state = recovered_order["state"]
            _save_job_state(output_dir, trading_cfg, state_file, state)
        order_res = dict(state.get("order") or {})
        if int(order_res.get("order_ticket", 0) or 0) > 0:
            state.pop("ended_at", None)
            state.pop("closed_reason", None)
            _save_job_state(output_dir, trading_cfg, state_file, state)

            mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
            adapter = MT5Adapter(mt5_cfg)
            ok_conn, msg_conn = adapter.connect()
            if not ok_conn:
                return {"ok": False, "error": f"MT5 connect failed: {msg_conn}", "state": state}
            try:
                out = _finalize_agent_a_order_submission(
                    adapter=adapter,
                    app_config=app_config,
                    trading_cfg=trading_cfg,
                    output_dir=output_dir,
                    logger=logger,
                    state=state,
                    state_file=state_file,
                    order_res=order_res,
                    wait_fill_sec=int(((trading_cfg.get("trading_job") or {}).get("fill_check_seconds", 30) or 30)),
                    max_wait_fill_minutes=_programmed_order_expiration_minutes(
                        trading_cfg,
                        str(
                            ((state.get("request_context") or {}).get("autonomous_trigger") or "")
                        ).strip().lower(),
                    ),
                )
            finally:
                adapter.shutdown()
            return {"ok": True, "state": out, "message": "Resumed approved Agent A fill monitoring"}

    if str(state.get("status") or "").strip() == "waiting_market_open" and str(state.get("stage") or "").strip() == "agent_a" and bool(state.get("agent_a_approved")):
        state.pop("ended_at", None)
        state.pop("closed_reason", None)
        _save_job_state(output_dir, trading_cfg, state_file, state)
        out = _execute_approved_order(
            app_config=app_config,
            trading_cfg=trading_cfg,
            output_dir=output_dir,
            logger=logger,
            state=state,
            state_file=state_file,
        )
        return {"ok": True, "state": out, "message": "Resumed approved Agent A order placement"}

    if str(state.get("stage") or "").strip() == "agent_a" and bool(state.get("agent_a_approved")):
        state.pop("ended_at", None)
        state.pop("closed_reason", None)
        _save_job_state(output_dir, trading_cfg, state_file, state)
        out = _execute_approved_order(
            app_config=app_config,
            trading_cfg=trading_cfg,
            output_dir=output_dir,
            logger=logger,
            state=state,
            state_file=state_file,
        )
        return {"ok": True, "state": out, "message": "Resumed approved Agent A order placement"}

    if str(state.get("stage")) != "agent_b":
        return {
            "ok": False,
            "error": "Resume currently supports Agent B stage only. Re-run trading-job start for Agent A.",
            "state": state,
        }

    out = _run_agent_b_loop(state, trading_cfg, output_dir, state_file, logger)
    out["followup_agent_a"] = _launch_followup_agent_a_start(
        app_config=app_config,
        trading_cfg=trading_cfg,
        output_dir=output_dir,
        logger=logger,
        completed_state=out,
    )
    _save_job_state(output_dir, trading_cfg, state_file, out)
    return {"ok": True, "state": out}
