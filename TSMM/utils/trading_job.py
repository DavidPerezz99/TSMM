"""
Trading job orchestration for Agent A -> Agent B lifecycle.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from .investing_agent import (
    MT5Adapter,
    _collect_mode_b_signals,
    run_investing_agent,
)
from .llm_connector import load_llm_providers_config, call_llm
from .market_sentiment import aggregate_market_sentiment
from .agent_memory import AgentMemoryStore
from .agent_channel import publish_channel_message
from .notification_email import send_email_notification
from .notification_telegram import send_telegram_notification


def _now_utc() -> datetime:
    return datetime.utcnow()


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _state_path(output_dir: str, trading_cfg: Dict[str, Any]) -> str:
    tj = (trading_cfg.get("trading_job") or {})
    return str(tj.get("state_path") or os.path.join(output_dir, "runtime", "trading_job_state.json"))


def _stop_flag_path(output_dir: str, trading_cfg: Dict[str, Any]) -> str:
    tj = (trading_cfg.get("trading_job") or {})
    return str(tj.get("stop_flag_path") or os.path.join(output_dir, "runtime", "trading_job_stop.flag"))


def _save_state(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_state(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


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
) -> bool:
    channels = [str(c).strip().lower() for c in (channels or ["popup", "terminal"])]

    if "popup" in channels:
        res = _windows_yes_no(title, message)
        if res is True:
            return True
        if res is False and "terminal" not in channels:
            return False

    if "terminal" in channels:
        ans = _input_with_timeout(f"{message} [yes/no]: ", timeout_sec=timeout_sec)
        if ans is None:
            return False
        return str(ans).strip().lower() in {"y", "yes", "ok", "approve"}

    return False


def stop_trading_job(output_dir: str, trading_cfg: Dict[str, Any]) -> str:
    stop_path = _stop_flag_path(output_dir, trading_cfg)
    os.makedirs(os.path.dirname(stop_path), exist_ok=True)
    with open(stop_path, "w", encoding="utf-8") as f:
        f.write(_iso(_now_utc()))
    return stop_path


def _clear_stop_flag(output_dir: str, trading_cfg: Dict[str, Any]) -> None:
    stop_path = _stop_flag_path(output_dir, trading_cfg)
    if os.path.exists(stop_path):
        os.remove(stop_path)


def _should_stop(output_dir: str, trading_cfg: Dict[str, Any]) -> bool:
    return os.path.exists(_stop_flag_path(output_dir, trading_cfg))


def _place_programmed_order(
    adapter: MT5Adapter,
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    exec_cfg = (trading_cfg.get("execution") or {})
    symbol = str(exec_cfg.get("symbol") or app_config.get("symbol") or "XAUUSD")
    return adapter.place_programmed_order(
        symbol=symbol,
        side=str(plan.get("decision", "hold")).lower(),
        volume=float(plan.get("volume", exec_cfg.get("default_volume", 0.01))),
        entry=float(plan.get("entry", 0.0)),
        stop_loss=float(plan.get("stop_loss", 0.0)),
        take_profit=float(plan.get("take_profit", 0.0)),
    )


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
    return {
        "normal_timeout_seconds": int(ap.get("normal_timeout_seconds", 7200) or 7200),
        "fast_timeout_seconds": int(ap.get("fast_timeout_seconds", 120) or 120),
        "emergency_lead_minutes": int(ap.get("emergency_lead_minutes", 5) or 5),
        "channel_user_triggered_only": bool(ap.get("channel_user_triggered_only", True)),
    }


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
) -> Dict[str, Any]:
    ch = publish_channel_message(
        output_dir=output_dir,
        trading_cfg=trading_cfg,
        channel=channel,
        kind=kind,
        message=message,
        requires_approval=requires_approval,
        emergency=emergency,
        approval_deadline_utc=approval_deadline_utc,
        metadata=metadata or {},
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

    if bool(requires_approval) or bool(emergency):
        subject = f"TSMM {channel.upper()} {'EMERGENCY ' if emergency else ''}Approval Request"
        body = (
            f"Time (UTC): {_iso(_now_utc())}\n"
            f"Channel: {channel}\n"
            f"Type: {kind}\n"
            f"Emergency: {bool(emergency)}\n"
            f"Requires approval: {bool(requires_approval)}\n"
            f"Approval deadline (UTC): {approval_deadline_utc}\n\n"
            f"Message:\n{message}\n\n"
            f"Metadata:\n{json.dumps(metadata or {}, indent=2, default=str)}\n"
        )
        email_out = send_email_notification(email_cfg=email_cfg, subject=subject, body=body)

    if should_push:
        tg_message = (
            f"*TSMM Notification*\n"
            f"time_utc: {_iso(_now_utc())}\n"
            f"channel: {channel}\n"
            f"type: {kind}\n"
            f"emergency: {bool(emergency)}\n"
            f"requires_approval: {bool(requires_approval)}\n"
            f"approval_deadline_utc: {approval_deadline_utc}\n\n"
            f"{message}"
        )
        telegram_out = send_telegram_notification(telegram_cfg=telegram_cfg, message=tg_message)

    return {"channel": ch, "email": email_out, "telegram": telegram_out}


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
    logger,
) -> Dict[str, Any]:
    mb_cfg = (trading_cfg.get("mode_b") or {})
    tj = (trading_cfg.get("trading_job") or {})
    agent_cfg = (trading_cfg.get("agent") or {})

    poll_seconds = int(mb_cfg.get("poll_seconds", tj.get("poll_seconds", 60)) or 60)
    session_hours = float(tj.get("session_hours", 7.0) or 7.0)
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

    started_at = datetime.strptime(state["started_at"], "%Y-%m-%d %H:%M:%S")
    base_deadline = started_at + timedelta(hours=session_hours)
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
            if _should_stop(output_dir, trading_cfg):
                logger.warning("Manual stop flag detected, closing active position.")
                close_res = adapter.close_position_by_ticket(pos_ticket)
                state["status"] = "stopped"
                state["close_result"] = close_res
                state["closed_reason"] = "manual_stop"
                state["ended_at"] = _iso(_now_utc())
                return state

            now = _now_utc()
            if now >= hard_deadline:
                close_res = adapter.close_position_by_ticket(pos_ticket)
                state["status"] = "closed"
                state["close_result"] = close_res
                state["closed_reason"] = "hard_deadline_reached"
                state["ended_at"] = _iso(_now_utc())
                return state

            pos = adapter.get_position_by_ticket(pos_ticket)
            if not pos.get("ok") or not pos.get("position"):
                state["status"] = "closed"
                state["closed_reason"] = "position_not_found_assumed_closed"
                state["ended_at"] = _iso(_now_utc())
                return state

            sig = _collect_mode_b_signals(trading_cfg.get("model_endpoints", {}), trading_cfg=trading_cfg)
            consensus = str(sig.get("consensus", "hold"))
            score = float(sig.get("consensus_score", 0.0) or 0.0)
            state["mode_b"] = sig
            state["last_mode_b_tick"] = _iso(now)

            # Emergency approval path: sharp consensus/signature shift with short decision window.
            emergency_cfg = (trading_cfg.get("emergency") or {})
            emergency_enabled = bool(emergency_cfg.get("enabled", True))
            score_jump = float(emergency_cfg.get("score_jump_threshold", 0.55) or 0.55)
            require_approval = bool(emergency_cfg.get("require_approval", True))
            consensus_flip = prev_consensus is not None and consensus != prev_consensus and {consensus, prev_consensus} == {"buy", "sell"}
            jump = prev_score is not None and abs(score - float(prev_score)) >= score_jump
            if emergency_enabled and require_approval and (consensus_flip or jump):
                fast_timeout = int(policy.get("fast_timeout_seconds", 120))
                deadline = _now_utc() + timedelta(seconds=fast_timeout)
                emergency_story = (
                    "Emergency approval requested: Agent B detected a key signal-setting shift requiring rapid intervention. "
                    f"Previous consensus={prev_consensus}, current consensus={consensus}, previous score={prev_score}, current score={score:.4f}. "
                    f"Estimated impact lead <= {policy.get('emergency_lead_minutes', 5)} minutes."
                )
                notify_res = _notify(
                    output_dir=output_dir,
                    trading_cfg=trading_cfg,
                    channel="agent_b",
                    kind="emergency_approval",
                    message=emergency_story,
                    requires_approval=True,
                    emergency=True,
                    approval_deadline_utc=_iso(deadline),
                    metadata={"consensus": consensus, "score": score, "previous": {"consensus": prev_consensus, "score": prev_score}},
                )
                state.setdefault("notifications", []).append(notify_res)
                approved_fast = request_approval(
                    title="TSMM Agent B Emergency Approval",
                    message=emergency_story,
                    timeout_sec=fast_timeout,
                    channels=channels,
                )
                state["last_emergency_approval"] = {
                    "approved": bool(approved_fast),
                    "deadline_utc": _iso(deadline),
                    "timestamp_utc": _iso(_now_utc()),
                }
                if not approved_fast:
                    close_res = adapter.close_position_by_ticket(pos_ticket)
                    state["status"] = "closed"
                    state["close_result"] = close_res
                    state["closed_reason"] = "emergency_approval_rejected_or_timeout"
                    state["ended_at"] = _iso(_now_utc())
                    return state

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
                close_res = adapter.close_position_by_ticket(pos_ticket)
                state["status"] = "closed"
                state["close_result"] = close_res
                state["closed_reason"] = f"mode_b_consensus_close({consensus},{score:.3f})"
                state["ended_at"] = _iso(_now_utc())
                return state

            if now >= base_deadline and not extension_approved:
                if not extension_requested:
                    extension_requested = True
                    state["extension_requested"] = True
                    state["extension_requested_at"] = _iso(now)

                extension_window_deadline = base_deadline + timedelta(minutes=extension_window_minutes)
                if now > extension_window_deadline:
                    close_res = adapter.close_position_by_ticket(pos_ticket)
                    state["status"] = "closed"
                    state["close_result"] = close_res
                    state["closed_reason"] = "extension_not_approved_in_window"
                    state["ended_at"] = _iso(_now_utc())
                    return state

                approved = request_approval(
                    title="TSMM Agent B Extension Request",
                    message=(
                        "Agent B requests extending this trading job beyond the 7h session. "
                        "Approve extension (up to 1 week max)?"
                    ),
                    timeout_sec=int(policy.get("normal_timeout_seconds", approval_timeout_seconds)),
                    channels=channels,
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
                )
                state.setdefault("notifications", []).append(notify_res)
                if approved:
                    extension_approved = True
                    state["extension_approved"] = True
                    state["extension_approved_at"] = _iso(_now_utc())
                else:
                    close_res = adapter.close_position_by_ticket(pos_ticket)
                    state["status"] = "closed"
                    state["close_result"] = close_res
                    state["closed_reason"] = "extension_rejected_or_timeout"
                    state["ended_at"] = _iso(_now_utc())
                    return state

            _save_state(_state_path(output_dir, trading_cfg), state)
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
) -> Dict[str, Any]:
    _clear_stop_flag(output_dir, trading_cfg)
    state_file = _state_path(output_dir, trading_cfg)

    agent_cfg = (trading_cfg.get("agent") or {})
    channels = agent_cfg.get("approval_channels", ["popup", "terminal"])
    tj = (trading_cfg.get("trading_job") or {})
    policy = _approval_policy(trading_cfg)

    wait_fill_sec = int(tj.get("fill_check_seconds", 30) or 30)
    max_wait_fill_minutes = int(tj.get("max_wait_fill_minutes", 420) or 420)

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
                    plan["decision"] = run_decision
                    plan["confidence"] = run_conf
                    if timeframe:
                        plan["fallback_timeframe"] = timeframe
                    if model_name:
                        plan["model"] = model_name
                    plan["fallback_config_path"] = cfg_path
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
    story = (
        "Agent A completed signal analysis for a single-session operation. "
        f"Decision={plan.get('decision')}, model={plan.get('model')}, confidence={plan.get('confidence')}, "
        f"cm_accuracy={plan.get('cm_accuracy')}, signal_score={plan.get('signal_score')}, "
        f"entry={plan.get('entry')}, sl={plan.get('stop_loss')}, tp={plan.get('take_profit')}."
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
        "job_type": "trading_job",
        "status": "agent_a_completed",
        "started_at": _iso(_now_utc()),
        "stage": "agent_a",
        "mode": "mode_a",
        "plan": plan,
        "report_path": mode_a_result.get("report_path"),
        "signal_json_path": signal_json_path,
        "state_path": mode_a_result.get("state_path"),
        "mode_a": mode_a_result,
        "sentiment": sentiment,
        "llm_analysis": llm_analysis,
        "memory_context": memory_context,
        "notifications": [notify_res],
    }
    if fallback_log:
        state["agent_a_fallback_attempts"] = fallback_log
    _save_state(state_file, state)

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
            )
        )
        state["status"] = "completed"
        state["closed_reason"] = "agent_a_no_trade"
        state["ended_at"] = _iso(_now_utc())
        _save_state(state_file, state)
        return state

    approved = request_approval(
        title="TSMM Agent A Approval",
        message=(
            "Agent A generated a trading plan with one programmed order. "
            "Approve order placement in MT5?"
        ),
        timeout_sec=int(policy.get("normal_timeout_seconds", 7200)),
        channels=channels,
    )
    deadline_a = _now_utc() + timedelta(seconds=int(policy.get("normal_timeout_seconds", 7200)))
    approval_story = (
        "Agent A requests approval to place programmed MT5 order for this session. "
        f"Plan summary: decision={plan.get('decision')}, entry={plan.get('entry')}, "
        f"sl={plan.get('stop_loss')}, tp={plan.get('take_profit')}."
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
        metadata={"signal_json_path": signal_json_path},
    )
    state.setdefault("notifications", []).append(notify_res_appr)
    state["agent_a_approved"] = bool(approved)

    if not approved:
        state["status"] = "completed"
        state["closed_reason"] = "agent_a_not_approved"
        state["ended_at"] = _iso(_now_utc())
        _save_state(state_file, state)
        return state

    # Place programmed order in MT5.
    mt5_cfg = (((trading_cfg.get("broker") or {}).get("mt5") or {}))
    adapter = MT5Adapter(mt5_cfg)
    ok_conn, msg_conn = adapter.connect()
    if not ok_conn:
        state["status"] = "failed"
        state["closed_reason"] = f"mt5_connect_failed: {msg_conn}"
        state["ended_at"] = _iso(_now_utc())
        _save_state(state_file, state)
        return state

    try:
        order_res = _place_programmed_order(adapter, app_config, trading_cfg, plan)
        state["order"] = order_res
        if not order_res.get("ok"):
            state["status"] = "failed"
            state["closed_reason"] = f"order_place_failed: {order_res.get('message', 'unknown')}"
            state["ended_at"] = _iso(_now_utc())
            _save_state(state_file, state)
            return state

        order_ticket = int(order_res.get("order_ticket", 0) or 0)
        filled = _wait_fill_and_get_position(
            adapter=adapter,
            order_ticket=order_ticket,
            side=decision,
            max_wait_sec=max_wait_fill_minutes * 60,
            poll_sec=wait_fill_sec,
        )

        if not filled.get("filled"):
            state["status"] = "completed"
            state["closed_reason"] = "order_not_filled_in_session"
            state["fill_status"] = filled
            state["ended_at"] = _iso(_now_utc())
            _save_state(state_file, state)
            return state

        state["stage"] = "agent_b"
        state["mode"] = "mode_b"
        state["status"] = "agent_b_running"
        state["position"] = filled.get("position")
        state["agent_b_started_at"] = _iso(_now_utc())
        _save_state(state_file, state)
    finally:
        adapter.shutdown()

    # Agent B starts once programmed operation is done.
    out = _run_agent_b_loop(state, trading_cfg, output_dir, logger)
    _save_state(state_file, out)
    return out


def resume_trading_job(
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    logger,
) -> Dict[str, Any]:
    state_file = _state_path(output_dir, trading_cfg)
    state = _load_state(state_file)
    if not state:
        return {"ok": False, "error": "No trading job state found to resume"}

    if str(state.get("status")) in {"completed", "closed", "failed", "stopped"}:
        return {"ok": True, "state": state, "message": "Job already finished"}

    if str(state.get("stage")) != "agent_b":
        return {
            "ok": False,
            "error": "Resume currently supports Agent B stage only. Re-run trading-job start for Agent A.",
            "state": state,
        }

    out = _run_agent_b_loop(state, trading_cfg, output_dir, logger)
    _save_state(state_file, out)
    return {"ok": True, "state": out}
