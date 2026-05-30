from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .runtime_scope import resolve_runtime_dir

_TERMINAL_STATUSES = {"closed", "completed", "failed", "stopped"}


def _now_utc() -> datetime:
    return datetime.utcnow()


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _parse_utc(raw_value: Any) -> Optional[datetime]:
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    return None


def _feedback_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict((trading_cfg.get("operation_feedback") or {}) if isinstance(trading_cfg, dict) else {})


def feedback_enabled(trading_cfg: Dict[str, Any]) -> bool:
    cfg = _feedback_cfg(trading_cfg)
    return bool(cfg.get("enabled", True))


def _feedback_root_name(trading_cfg: Dict[str, Any]) -> str:
    cfg = _feedback_cfg(trading_cfg)
    root_name = str(cfg.get("root_dir") or cfg.get("root_subdir") or "operation_feedback").strip()
    return root_name or "operation_feedback"


def resolve_operation_feedback_root(output_dir: str, trading_cfg: Dict[str, Any]) -> Path:
    runtime_root = resolve_runtime_dir(output_dir=output_dir, trading_cfg=trading_cfg)
    root_name = _feedback_root_name(trading_cfg)
    root_candidate = Path(root_name)
    if root_candidate.is_absolute():
        return root_candidate
    return runtime_root / root_candidate


def _daily_file_path(root: Path, timestamp_utc: datetime) -> Path:
    day_key = timestamp_utc.strftime("%Y%m%d")
    return root / "daily" / timestamp_utc.strftime("%Y") / timestamp_utc.strftime("%m") / timestamp_utc.strftime("%d") / f"operations_{day_key}.jsonl"


def _safe_job_id(job_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(job_id or "").strip())
    return cleaned.strip("_")


def _job_file_path(root: Path, job_id: str) -> Path:
    safe_job_id = _safe_job_id(job_id)
    return root / "by_job" / f"{safe_job_id}.jsonl"


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, default=str) + "\n")


def _safe_float(raw_value: Any) -> Optional[float]:
    try:
        if raw_value is None:
            return None
        return float(raw_value)
    except Exception:
        return None


def _rounded(raw_value: Any, digits: int = 6) -> Optional[float]:
    value = _safe_float(raw_value)
    if value is None:
        return None
    return round(value, digits)


def _close_reason_family(raw_reason: Any) -> str:
    reason = str(raw_reason or "").strip().lower()
    if not reason:
        return ""
    reason = reason.split("(", 1)[0]
    reason = reason.split(":", 1)[0]
    return reason


def _extract_profit(state: Dict[str, Any]) -> Optional[float]:
    close_outcome = (state.get("close_outcome") or {}) if isinstance(state, dict) else {}
    close_profit = _safe_float(close_outcome.get("profit")) if isinstance(close_outcome, dict) else None
    if close_profit is not None:
        return close_profit

    position = (state.get("position") or {}) if isinstance(state, dict) else {}
    position_profit = _safe_float(position.get("profit")) if isinstance(position, dict) else None
    if position_profit is not None:
        return position_profit

    close_result = (state.get("close_result") or {}) if isinstance(state, dict) else {}
    close_result_profit = _safe_float(close_result.get("profit")) if isinstance(close_result, dict) else None
    return close_result_profit


def _extract_mode_b_timeframes(sig: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    timeframes = (sig.get("timeframes") or {}) if isinstance(sig, dict) else {}
    if not isinstance(timeframes, dict):
        return out

    for timeframe, entry in timeframes.items():
        if not isinstance(entry, dict):
            continue
        out[str(timeframe)] = {
            "signal": _safe_float(entry.get("signal")),
            "confidence": _safe_float(entry.get("confidence")),
            "model": str(entry.get("model") or "").strip(),
            "error": str(entry.get("error") or "").strip(),
        }
    return out


def _extract_performance_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    mode_a = (state.get("mode_a") or {}) if isinstance(state, dict) else {}
    backtest = (mode_a.get("backtest") or {}) if isinstance(mode_a, dict) else {}

    confidence = _safe_float(plan.get("confidence"))
    cm_accuracy = _safe_float(plan.get("cm_accuracy"))
    success_probability = _safe_float(plan.get("success_probability"))
    signal_score = _safe_float(plan.get("signal_score"))
    input_fooling_risk = _safe_float(plan.get("input_fooling_risk"))
    win_rate = _safe_float(backtest.get("win_rate"))

    return {
        "model_name": str(plan.get("model") or backtest.get("model_name") or "").strip(),
        "decision": str(plan.get("decision") or "").strip().lower(),
        "confidence": confidence,
        "cm_accuracy": cm_accuracy,
        "success_probability": success_probability,
        "signal_score": signal_score,
        "input_fooling_risk": input_fooling_risk,
        "backtest_win_rate": win_rate,
        "backtest_total_return_pct": _safe_float(backtest.get("total_return_pct")),
        "backtest_max_drawdown_pct": _safe_float(backtest.get("max_drawdown_pct")),
        "backtest_n_trades": int(backtest.get("n_trades", 0) or 0),
        "drift_indicators": {
            "confidence_minus_success_probability": None
            if confidence is None or success_probability is None
            else round(confidence - success_probability, 6),
            "cm_accuracy_minus_confidence": None
            if cm_accuracy is None or confidence is None
            else round(cm_accuracy - confidence, 6),
            "backtest_win_rate_minus_confidence": None
            if win_rate is None or confidence is None
            else round(win_rate - confidence, 6),
            "input_fooling_risk": input_fooling_risk,
        },
    }


def _extract_execution_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    order = (state.get("order") or {}) if isinstance(state, dict) else {}
    position = (state.get("position") or {}) if isinstance(state, dict) else {}
    close_result = (state.get("close_result") or {}) if isinstance(state, dict) else {}
    close_outcome = (state.get("close_outcome") or {}) if isinstance(state, dict) else {}

    return {
        "order_submission_mode": str(state.get("order_submission_mode") or "").strip().lower(),
        "order_ticket": int(order.get("order_ticket", 0) or 0) if isinstance(order, dict) else 0,
        "order_ok": bool(order.get("ok", False)) if isinstance(order, dict) else False,
        "order_retcode": int(order.get("retcode", 0) or 0) if isinstance(order, dict) else 0,
        "position_ticket": int(position.get("ticket", 0) or 0) if isinstance(position, dict) else 0,
        "position_symbol": str(position.get("symbol") or "").strip() if isinstance(position, dict) else "",
        "position_volume": _safe_float(position.get("volume")) if isinstance(position, dict) else None,
        "position_price_open": _safe_float(position.get("price_open")) if isinstance(position, dict) else None,
        "position_price_current": _safe_float(position.get("price_current")) if isinstance(position, dict) else None,
        "position_profit": _safe_float(position.get("profit")) if isinstance(position, dict) else None,
        "close_retcode": int(close_result.get("retcode", 0) or 0) if isinstance(close_result, dict) else 0,
        "close_deal_ticket": int(close_result.get("deal_ticket", 0) or 0) if isinstance(close_result, dict) else 0,
        "close_outcome_found": bool(close_outcome.get("found", False)) if isinstance(close_outcome, dict) else False,
        "close_outcome_profit": _safe_float(close_outcome.get("profit")) if isinstance(close_outcome, dict) else None,
        "close_outcome_reason": str(close_outcome.get("reason_label") or "").strip() if isinstance(close_outcome, dict) else "",
    }


def _extract_mode_b_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    mode_b = (state.get("mode_b") or {}) if isinstance(state, dict) else {}
    plan = (state.get("agent_b_plan") or {}) if isinstance(state, dict) else {}
    risk = (state.get("last_risk_adjustment") or {}) if isinstance(state, dict) else {}

    return {
        "sample_tick_utc": str(state.get("last_mode_b_tick") or "").strip(),
        "consensus": str(mode_b.get("consensus") or plan.get("consensus") or "").strip().lower(),
        "consensus_score": _safe_float(mode_b.get("consensus_score") if isinstance(mode_b, dict) else None),
        "recommendation": str(plan.get("recommendation") or "").strip().lower(),
        "should_close": bool(plan.get("should_close", False)),
        "close_reason": str(plan.get("close_reason") or "").strip(),
        "risk_action": str(((risk.get("result") or {}).get("action") if isinstance(risk, dict) and isinstance(risk.get("result"), dict) else "") or "").strip().lower(),
        "timeframe_signals": _extract_mode_b_timeframes(mode_b),
    }


def _state_signature(state: Dict[str, Any]) -> Dict[str, Any]:
    order = (state.get("order") or {}) if isinstance(state, dict) else {}
    position = (state.get("position") or {}) if isinstance(state, dict) else {}
    plan = (state.get("plan") or {}) if isinstance(state, dict) else {}
    mode_b_plan = (state.get("agent_b_plan") or {}) if isinstance(state, dict) else {}
    close_outcome = (state.get("close_outcome") or {}) if isinstance(state, dict) else {}
    close_result = (state.get("close_result") or {}) if isinstance(state, dict) else {}
    risk = (state.get("last_risk_adjustment") or {}) if isinstance(state, dict) else {}

    return {
        "status": str(state.get("status") or "").strip().lower(),
        "stage": str(state.get("stage") or "").strip().lower(),
        "closed_reason": str(state.get("closed_reason") or "").strip().lower(),
        "agent_a_approved": bool(state.get("agent_a_approved", False)),
        "order_ticket": int(order.get("order_ticket", 0) or 0) if isinstance(order, dict) else 0,
        "order_ok": bool(order.get("ok", False)) if isinstance(order, dict) else False,
        "order_retcode": int(order.get("retcode", 0) or 0) if isinstance(order, dict) else 0,
        "position_ticket": int(position.get("ticket", 0) or 0) if isinstance(position, dict) else 0,
        "position_profit": _rounded(position.get("profit") if isinstance(position, dict) else None, digits=4),
        "model": str(plan.get("model") or "").strip().lower(),
        "decision": str(plan.get("decision") or "").strip().lower(),
        "agent_b_recommendation": str(mode_b_plan.get("recommendation") or "").strip().lower(),
        "agent_b_consensus": str(mode_b_plan.get("consensus") or "").strip().lower(),
        "agent_b_should_close": bool(mode_b_plan.get("should_close", False)),
        "risk_signature": str(risk.get("signature") or "").strip(),
        "close_retcode": int(close_result.get("retcode", 0) or 0) if isinstance(close_result, dict) else 0,
        "close_outcome_found": bool(close_outcome.get("found", False)) if isinstance(close_outcome, dict) else False,
        "close_outcome_profit": _rounded(close_outcome.get("profit") if isinstance(close_outcome, dict) else None, digits=4),
    }


def _state_transition_diff(previous_state: Dict[str, Any], current_state: Dict[str, Any]) -> List[str]:
    previous_signature = _state_signature(previous_state)
    current_signature = _state_signature(current_state)
    changed_fields: List[str] = []

    for key, current_value in current_signature.items():
        if previous_signature.get(key) != current_value:
            changed_fields.append(key)

    return changed_fields


def _infer_outcome_label(state: Dict[str, Any], event_kind: str) -> str:
    event_kind_norm = str(event_kind or "").strip().lower()
    status = str((state or {}).get("status") or "").strip().lower()
    close_reason = _close_reason_family((state or {}).get("closed_reason"))

    if event_kind_norm in {"notify_agent_a_order_failed", "notify_agent_a_approval_rejected"}:
        return "bad"

    if status in _TERMINAL_STATUSES:
        profit = _extract_profit(state)
        if profit is not None:
            if profit > 0:
                return "good"
            if profit < 0:
                return "bad"

        if status == "failed":
            return "bad"

        neutral_reasons = {
            "agent_a_no_trade",
            "agent_a_not_approved",
            "duplicate_order_prevented",
            "order_not_filled_in_session",
        }
        if close_reason in neutral_reasons:
            return "neutral"

        if close_reason.endswith("failed") or "failed" in close_reason:
            return "bad"

        return "neutral"

    return "pending"


def _profile_label(trading_cfg: Dict[str, Any]) -> str:
    runtime_cfg = dict((trading_cfg.get("runtime") or {}) if isinstance(trading_cfg, dict) else {})
    label = str(runtime_cfg.get("profile_label") or runtime_cfg.get("job_id_prefix") or "").strip()
    return label or "TSMM"


def _base_event(
    *,
    source: str,
    event_kind: str,
    output_dir: str,
    trading_cfg: Dict[str, Any],
    state: Dict[str, Any],
    message: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    timestamp_utc = _now_utc()
    timestamp_iso = _iso(timestamp_utc)
    state_obj = dict(state or {})
    job_id = str(state_obj.get("job_id") or metadata.get("job_id") or "").strip()

    sampled_at_utc = str(state_obj.get("last_mode_b_tick") or state_obj.get("started_at") or timestamp_iso).strip() or timestamp_iso
    started_at = _parse_utc(state_obj.get("started_at"))
    operation_age_seconds = int((timestamp_utc - started_at).total_seconds()) if started_at is not None else None

    hash_seed = (
        f"{timestamp_iso}|{job_id}|{source}|{event_kind}|{time.time_ns()}|{int(state_obj.get('runner_pid', 0) or 0)}"
    )
    event_id = hashlib.sha1(hash_seed.encode("utf-8")).hexdigest()[:20]

    event = {
        "event_id": event_id,
        "timestamp_utc": timestamp_iso,
        "date_utc": timestamp_utc.strftime("%Y%m%d"),
        "source": str(source or "").strip().lower() or "runtime",
        "event_kind": str(event_kind or "").strip().lower() or "event",
        "profile": _profile_label(trading_cfg),
        "runtime_dir": str(resolve_runtime_dir(output_dir=output_dir, trading_cfg=trading_cfg)),
        "job_id": job_id,
        "stage": str(state_obj.get("stage") or "").strip().lower(),
        "status": str(state_obj.get("status") or "").strip().lower(),
        "sampled_at_utc": sampled_at_utc,
        "operation_age_seconds": operation_age_seconds,
        "message": str(message or "").strip(),
        "metadata": metadata,
        "outcome_label": _infer_outcome_label(state_obj, event_kind),
        "performance_snapshot": _extract_performance_snapshot(state_obj),
        "execution_snapshot": _extract_execution_snapshot(state_obj),
        "mode_b_snapshot": _extract_mode_b_snapshot(state_obj),
        "close_reason_family": _close_reason_family(state_obj.get("closed_reason")),
    }
    return event


def _write_feedback_event(output_dir: str, trading_cfg: Dict[str, Any], event: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    if not feedback_enabled(trading_cfg):
        return {"ok": False, "skipped": True, "reason": "feedback_disabled"}

    root = resolve_operation_feedback_root(output_dir, trading_cfg)
    event_dt = _parse_utc(event.get("timestamp_utc")) or _now_utc()
    daily_file = _daily_file_path(root, event_dt)
    _append_jsonl(daily_file, event)

    job_file: Optional[Path] = None
    cfg = _feedback_cfg(trading_cfg)
    if bool(cfg.get("write_by_job_logs", True)) and str(job_id or "").strip():
        job_file = _job_file_path(root, str(job_id))
        _append_jsonl(job_file, event)

    return {
        "ok": True,
        "daily_file": str(daily_file),
        "job_file": str(job_file) if job_file is not None else "",
        "event_id": str(event.get("event_id") or ""),
    }


def log_state_transition_feedback(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    previous_state: Dict[str, Any],
    current_state: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = _feedback_cfg(trading_cfg)
    if not bool(cfg.get("capture_state_transitions", True)):
        return {"ok": False, "skipped": True, "reason": "state_transition_capture_disabled"}

    current = dict(current_state or {})
    previous = dict(previous_state or {})
    changed_fields = _state_transition_diff(previous, current)
    if previous and not changed_fields:
        return {"ok": False, "skipped": True, "reason": "no_significant_state_change"}

    if not previous:
        event_kind = "state_initialized"
    elif "status" in changed_fields:
        event_kind = "state_status_changed"
    elif "stage" in changed_fields:
        event_kind = "state_stage_changed"
    elif any(x.startswith("order_") for x in changed_fields):
        event_kind = "state_order_updated"
    elif any(x.startswith("close_") for x in changed_fields):
        event_kind = "state_close_updated"
    else:
        event_kind = "state_updated"

    event = _base_event(
        source="state_transition",
        event_kind=event_kind,
        output_dir=output_dir,
        trading_cfg=trading_cfg,
        state=current,
        message="Trading job state transition recorded.",
        metadata={"changed_fields": changed_fields},
    )
    return _write_feedback_event(output_dir, trading_cfg, event, str(current.get("job_id") or ""))


def log_notification_feedback(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    *,
    channel: str,
    kind: str,
    message: str,
    metadata: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    job_id: str = "",
) -> Dict[str, Any]:
    cfg = _feedback_cfg(trading_cfg)
    if not bool(cfg.get("capture_notifications", True)):
        return {"ok": False, "skipped": True, "reason": "notification_capture_disabled"}

    state_obj = dict(state or {})
    if job_id and not state_obj.get("job_id"):
        state_obj["job_id"] = str(job_id)

    event = _base_event(
        source="notification",
        event_kind=f"notify_{str(channel or '').strip().lower()}_{str(kind or '').strip().lower()}",
        output_dir=output_dir,
        trading_cfg=trading_cfg,
        state=state_obj,
        message=message,
        metadata=dict(metadata or {}),
    )
    resolved_job_id = str(state_obj.get("job_id") or job_id or "").strip()
    return _write_feedback_event(output_dir, trading_cfg, event, resolved_job_id)


def log_agent_a_plan_feedback(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = _feedback_cfg(trading_cfg)
    if not bool(cfg.get("capture_agent_a_samples", True)):
        return {"ok": False, "skipped": True, "reason": "agent_a_sample_capture_disabled"}

    state_obj = dict(state or {})
    event = _base_event(
        source="agent_a_sample",
        event_kind="agent_a_plan_sample",
        output_dir=output_dir,
        trading_cfg=trading_cfg,
        state=state_obj,
        message="Agent A plan and model snapshot captured.",
        metadata={
            "signal_json_path": str(state_obj.get("signal_json_path") or "").strip(),
            "fallback_attempts": len(state_obj.get("agent_a_fallback_attempts") or []),
            "approval_required": bool(state_obj.get("approval_required", False)),
        },
    )
    return _write_feedback_event(output_dir, trading_cfg, event, str(state_obj.get("job_id") or ""))


def log_agent_b_sample_feedback(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    state: Dict[str, Any],
    signals: Dict[str, Any],
    current_plan: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = _feedback_cfg(trading_cfg)
    if not bool(cfg.get("capture_agent_b_samples", True)):
        return {"ok": False, "skipped": True, "reason": "agent_b_sample_capture_disabled"}

    state_obj = dict(state or {})
    state_obj["mode_b"] = dict(signals or {})
    state_obj["agent_b_plan"] = dict(current_plan or {})

    event = _base_event(
        source="agent_b_sample",
        event_kind="mode_b_assessment_sample",
        output_dir=output_dir,
        trading_cfg=trading_cfg,
        state=state_obj,
        message="Agent B assessment sample captured.",
        metadata={
            "consensus": str((signals or {}).get("consensus") or "").strip().lower(),
            "consensus_score": _safe_float((signals or {}).get("consensus_score")),
            "recommendation": str((current_plan or {}).get("recommendation") or "").strip().lower(),
            "should_close": bool((current_plan or {}).get("should_close", False)),
        },
    )
    return _write_feedback_event(output_dir, trading_cfg, event, str(state_obj.get("job_id") or ""))
