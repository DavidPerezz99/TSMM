from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.market_db import query_ohlc


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _tf_to_minutes(label: str) -> int:
    s = str(label or "").strip().lower()
    m = re.match(r"^(\d+)([mhw])$", s)
    if not m:
        return 1
    n = int(m.group(1))
    u = m.group(2)
    if u == "m":
        return n
    if u == "h":
        return n * 60
    if u == "w":
        return n * 7 * 24 * 60
    return 1


def _parse_r2_from_filename(path: str) -> float:
    stem = os.path.splitext(os.path.basename(path))[0].lower()

    # Example: top1_08098 => 0.8098
    m = re.search(r"_(\d{4,6})$", stem)
    if m:
        digits = m.group(1)
        return float(int(digits) / (10 ** (len(digits) - 1)))

    # Generic decimal fallback.
    m2 = re.search(r"(\d+\.\d+)", stem)
    if m2:
        return float(m2.group(1))

    return 0.0


def discover_top_model_configs(config_root: str = "config") -> Dict[str, Dict[str, Any]]:
    """Discover top model config per timeframe from folders like high10mResults/*/top*.yaml."""
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(config_root):
        return out

    for name in os.listdir(config_root):
        p = os.path.join(config_root, name)
        if not os.path.isdir(p):
            continue

        m = re.match(r"^high(\d+)([mhw])results$", str(name).lower())
        if not m:
            continue

        tf_label = f"{m.group(1)}{m.group(2)}"
        best = None

        for model_dir_name in os.listdir(p):
            model_dir = os.path.join(p, model_dir_name)
            if not os.path.isdir(model_dir):
                continue

            for fn in os.listdir(model_dir):
                if not fn.lower().endswith(('.yaml', '.yml')):
                    continue
                full = os.path.join(model_dir, fn)
                r2 = _parse_r2_from_filename(full)
                rec = {
                    "timeframe": tf_label,
                    "timeframe_minutes": _tf_to_minutes(tf_label),
                    "model": str(model_dir_name),
                    "path": full,
                    "r2": float(r2),
                }
                if best is None or rec["r2"] > best["r2"]:
                    best = rec

        if best is not None:
            out[tf_label] = best

    return out


def _load_master_df(
    master_table_path: str,
    n_days: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: str = "XAUUSD",
) -> pd.DataFrame:
    """Load master candles from CSV or SQLite for validation/backtesting."""
    p = str(master_table_path or "").strip()
    if not p:
        return pd.DataFrame()

    if p.lower().endswith(".db") or p.lower().endswith(".sqlite"):
        # Approx rows per day with margin for minute-level series.
        if start_date and end_date:
            try:
                ds = pd.to_datetime(start_date)
                de = pd.to_datetime(end_date)
                span_days = max(int((de - ds).days) + 2, 2)
            except Exception:
                span_days = max(int(n_days or 1), 2)
        else:
            span_days = max(int(n_days or 1), 2)

        latest_records = max(span_days * 2000, 50000)
        return query_ohlc(
            db_path=p,
            timeframe_minutes=1,
            latest_records=latest_records,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol,
        )

    df = pd.read_csv(p)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        if start_date:
            df = df[df["DATE"] >= pd.to_datetime(start_date, errors="coerce")]
        if end_date:
            df = df[df["DATE"] <= pd.to_datetime(end_date, errors="coerce")]
    return df


def _simulate_day(day_df: pd.DataFrame, trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Simple deterministic per-day simulation on minute candles."""
    risk = trading_cfg.get("risk", {}) or {}

    sl_pct = _safe_float(risk.get("stop_loss_pct", 0.8)) / 100.0
    tp_pct = _safe_float(risk.get("take_profit_pct", 1.6)) / 100.0
    risk_per_trade_pct = _safe_float(risk.get("risk_per_trade_pct", 0.5))
    max_hold = int(((trading_cfg.get("mode_a") or {}).get("max_hold_bars", 60)) or 60)

    df = day_df.copy().sort_values("DATE").reset_index(drop=True)
    if "y_diff" in df.columns:
        sig = np.sign(pd.to_numeric(df["y_diff"], errors="coerce").fillna(0.0).values)
    else:
        close = pd.to_numeric(df["CLOSE"], errors="coerce").ffill().bfill()
        sig = np.sign(close.diff().fillna(0.0).values)

    close = pd.to_numeric(df["CLOSE"], errors="coerce").ffill().bfill().values

    trades: List[Dict[str, Any]] = []
    in_pos = False
    side = ""
    entry_i = 0
    entry = 0.0
    sl = 0.0
    tp = 0.0

    for i in range(len(df)):
        px = float(close[i])
        if not np.isfinite(px):
            continue

        if not in_pos:
            if sig[i] > 0:
                side = "buy"
            elif sig[i] < 0:
                side = "sell"
            else:
                continue

            in_pos = True
            entry_i = i
            entry = px
            if side == "buy":
                sl = entry * (1 - sl_pct)
                tp = entry * (1 + tp_pct)
            else:
                sl = entry * (1 + sl_pct)
                tp = entry * (1 - tp_pct)
            continue

        bars = i - entry_i
        exit_reason = None
        if side == "buy":
            if px <= sl:
                exit_reason = "stop_loss"
            elif px >= tp:
                exit_reason = "take_profit"
            elif bars >= max_hold:
                exit_reason = "time_stop"
        else:
            if px >= sl:
                exit_reason = "stop_loss"
            elif px <= tp:
                exit_reason = "take_profit"
            elif bars >= max_hold:
                exit_reason = "time_stop"

        if exit_reason is None:
            continue

        pnl_abs = (px - entry) if side == "buy" else (entry - px)
        pnl_pct = ((px / entry) - 1.0) * 100.0 if side == "buy" else ((entry / px) - 1.0) * 100.0
        trades.append(
            {
                "side": side,
                "entry_index": int(entry_i),
                "exit_index": int(i),
                "entry_time": str(df["DATE"].iloc[entry_i]),
                "exit_time": str(df["DATE"].iloc[i]),
                "entry_price": float(entry),
                "exit_price": float(px),
                "stop_loss": float(sl),
                "take_profit": float(tp),
                "bars_held": int(bars),
                "exit_reason": exit_reason,
                "pnl_abs": float(pnl_abs),
                "pnl_pct": float(pnl_pct),
                "risk_pct": float(risk_per_trade_pct),
                "exposure_abs": float(abs(entry)),
            }
        )
        in_pos = False

    pnl_list = np.array([t["pnl_abs"] for t in trades], dtype=float) if trades else np.array([])
    green = int(np.sum(pnl_list > 0)) if pnl_list.size else 0
    red = int(np.sum(pnl_list <= 0)) if pnl_list.size else 0
    exposure = float(np.mean([t["exposure_abs"] for t in trades])) if trades else 0.0

    day_summary = {
        "date": str(pd.to_datetime(df["DATE"].iloc[0]).date()) if len(df) else "N/A",
        "n_operations": int(len(trades)),
        "n_green": green,
        "n_red": red,
        "profit_abs": float(np.sum(pnl_list[pnl_list > 0])) if pnl_list.size else 0.0,
        "loss_abs": float(np.sum(pnl_list[pnl_list <= 0])) if pnl_list.size else 0.0,
        "net_abs": float(np.sum(pnl_list)) if pnl_list.size else 0.0,
        "exposure_avg_abs": exposure,
        "avg_risk_pct": float(np.mean([t["risk_pct"] for t in trades])) if trades else 0.0,
    }
    return {"summary": day_summary, "operations": trades}


def run_agent_validation_days(
    master_table_path: str,
    n_days: int,
    output_dir: str,
    trading_cfg: Dict[str, Any],
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    if not master_table_path or not os.path.exists(master_table_path):
        return {"ok": False, "error": f"Master table not found: {master_table_path}"}

    os.makedirs(output_dir, exist_ok=True)

    market_symbol = str(
        ((trading_cfg.get("dashboard") or {}).get("sql_symbol"))
        or ((trading_cfg.get("execution") or {}).get("symbol"))
        or "XAUUSD"
    ).strip()

    df = _load_master_df(master_table_path, n_days, symbol=market_symbol)
    if "DATE" not in df.columns or "CLOSE" not in df.columns:
        return {"ok": False, "error": "Master table must contain DATE and CLOSE columns"}

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE")

    if df.empty:
        return {"ok": False, "error": "Master table has no valid rows"}

    unique_days = sorted(df["DATE"].dt.date.unique())
    if not unique_days:
        return {"ok": False, "error": "No trading days found"}

    selected_days = unique_days[-max(int(n_days or 1), 1):]

    sessions: List[Dict[str, Any]] = []
    for idx, day in enumerate(selected_days, start=1):
        day_df = df[df["DATE"].dt.date == day].copy()
        session = _simulate_day(day_df, trading_cfg)
        session_path = os.path.join(output_dir, f"session_{str(day).replace('-', '')}.json")
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)

        sessions.append({"day": str(day), "file": session_path, **session["summary"]})
        if progress_cb:
            progress_cb(idx, len(selected_days), str(day))

    total_ops = int(sum(s["n_operations"] for s in sessions))
    total_green = int(sum(s["n_green"] for s in sessions))
    total_red = int(sum(s["n_red"] for s in sessions))
    total_profit = float(sum(s["profit_abs"] for s in sessions))
    total_loss = float(sum(s["loss_abs"] for s in sessions))
    total_net = float(sum(s["net_abs"] for s in sessions))
    avg_exposure = float(np.mean([s["exposure_avg_abs"] for s in sessions])) if sessions else 0.0

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "master_table_path": master_table_path,
        "symbol": market_symbol,
        "mode": "simple_daily",
        "n_days": int(len(selected_days)),
        "overall": {
            "n_operations": total_ops,
            "n_green": total_green,
            "n_red": total_red,
            "profit_abs": total_profit,
            "loss_abs": total_loss,
            "net_abs": total_net,
            "green_rate": float(total_green / max(total_ops, 1)),
            "avg_exposure_abs": avg_exposure,
        },
        "sessions": sessions,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"ok": True, "summary_path": summary_path, "summary": summary}


def _aggregate_timeframe(df_1m: pd.DataFrame, tf_minutes: int) -> pd.DataFrame:
    w = df_1m.copy()
    w = w.sort_values("DATE")
    w = w.set_index("DATE")
    out = (
        w.resample(f"{int(max(tf_minutes, 1))}min")
        .agg({"OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last", "VOLUME": "sum"})
        .dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
        .reset_index()
    )
    return out


def _signal_at(df_tf: pd.DataFrame, ts: pd.Timestamp) -> Dict[str, Any]:
    if df_tf is None or df_tf.empty:
        return {"signal": "hold", "score": 0.0, "confidence": 0.5}

    w = df_tf[df_tf["DATE"] <= ts].tail(30)
    if len(w) < 3:
        return {"signal": "hold", "score": 0.0, "confidence": 0.5}

    close = pd.to_numeric(w["CLOSE"], errors="coerce").ffill().bfill().values
    rets = np.diff(close)
    last_ret = float(rets[-1]) if len(rets) else 0.0
    vol = float(np.std(rets[-20:], ddof=1)) if len(rets) >= 5 else max(abs(last_ret), 1e-6)
    score = float(last_ret / max(vol, 1e-6))

    if score > 0.15:
        sig = "buy"
    elif score < -0.15:
        sig = "sell"
    else:
        sig = "hold"

    conf = float(np.clip(abs(score) / 3.0, 0.05, 0.95))
    return {"signal": sig, "score": score, "confidence": conf}


def _consensus_from_timeframes(tf_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    weighted = 0.0
    total_w = 0.0
    for _, s in tf_signals.items():
        sig = str(s.get("signal", "hold"))
        v = 1.0 if sig == "buy" else (-1.0 if sig == "sell" else 0.0)
        w = float(s.get("confidence", 0.5) or 0.5)
        weighted += v * w
        total_w += max(w, 0.01)

    score = weighted / total_w if total_w > 0 else 0.0
    consensus = "buy" if score > 0.1 else ("sell" if score < -0.1 else "hold")
    return {"consensus": consensus, "consensus_score": float(score)}


def _simulate_advanced_session(
    day_df_1m: pd.DataFrame,
    tf_data: Dict[str, pd.DataFrame],
    session_start: pd.Timestamp,
    session_hours: float,
    trading_cfg: Dict[str, Any],
    available_timeframes: List[str],
) -> Dict[str, Any]:
    risk = trading_cfg.get("risk", {}) or {}
    mb_cfg = (trading_cfg.get("mode_b") or {})

    sl_pct = _safe_float(risk.get("stop_loss_pct", 0.8)) / 100.0
    tp_pct = _safe_float(risk.get("take_profit_pct", 1.6)) / 100.0
    close_threshold = float(mb_cfg.get("close_consensus_threshold", 0.25) or 0.25)
    poll_min = max(int((mb_cfg.get("poll_seconds", 600) or 600) / 60), 1)

    session_end = session_start + timedelta(hours=float(session_hours))
    bars = day_df_1m[(day_df_1m["DATE"] >= session_start) & (day_df_1m["DATE"] <= session_end)].copy()
    if bars.empty:
        return {"trigger_at": str(session_start), "executed": False, "reason": "no_bars_in_session"}

    # Agent A technical 7h signal.
    sig7 = _signal_at(tf_data.get("7h", pd.DataFrame()), session_start)
    decision = str(sig7.get("signal", "hold"))
    if decision not in {"buy", "sell"}:
        return {
            "trigger_at": str(session_start),
            "executed": False,
            "reason": "agent_a_hold",
            "agent_a": sig7,
        }

    entry = float(bars.iloc[0]["CLOSE"])
    if decision == "buy":
        sl = entry * (1 - sl_pct)
        tp = entry * (1 + tp_pct)
    else:
        sl = entry * (1 + sl_pct)
        tp = entry * (1 - tp_pct)

    agent_b_enabled = any(tf != "7h" for tf in available_timeframes)

    exit_price = entry
    exit_reason = "time_stop"
    exit_time = pd.to_datetime(bars.iloc[-1]["DATE"])

    next_poll = session_start + timedelta(minutes=poll_min)
    tf_used = [tf for tf in available_timeframes if tf != "7h"]

    for _, row in bars.iloc[1:].iterrows():
        ts = pd.to_datetime(row["DATE"])
        px = float(row["CLOSE"])

        if decision == "buy":
            if px <= sl:
                exit_price = px
                exit_reason = "stop_loss"
                exit_time = ts
                break
            if px >= tp:
                exit_price = px
                exit_reason = "take_profit"
                exit_time = ts
                break
        else:
            if px >= sl:
                exit_price = px
                exit_reason = "stop_loss"
                exit_time = ts
                break
            if px <= tp:
                exit_price = px
                exit_reason = "take_profit"
                exit_time = ts
                break

        if agent_b_enabled and ts >= next_poll:
            tf_signals = {tf: _signal_at(tf_data.get(tf, pd.DataFrame()), ts) for tf in tf_used}
            cons = _consensus_from_timeframes(tf_signals)
            c = str(cons.get("consensus", "hold"))
            s = float(cons.get("consensus_score", 0.0) or 0.0)
            if (decision == "buy" and c == "sell" and s <= -abs(close_threshold)) or (
                decision == "sell" and c == "buy" and s >= abs(close_threshold)
            ):
                exit_price = px
                exit_reason = f"mode_b_consensus_close({c},{s:.3f})"
                exit_time = ts
                break
            next_poll = ts + timedelta(minutes=poll_min)

    pnl_abs = (exit_price - entry) if decision == "buy" else (entry - exit_price)
    pnl_pct = ((exit_price / entry) - 1.0) * 100.0 if decision == "buy" else ((entry / exit_price) - 1.0) * 100.0

    return {
        "trigger_at": str(session_start),
        "executed": True,
        "agent_a": sig7,
        "agent_b_enabled": bool(agent_b_enabled),
        "entry": float(entry),
        "stop_loss": float(sl),
        "take_profit": float(tp),
        "exit_price": float(exit_price),
        "exit_time": str(exit_time),
        "exit_reason": exit_reason,
        "side": decision,
        "pnl_abs": float(pnl_abs),
        "pnl_pct": float(pnl_pct),
    }


def run_agent_backtest_advanced(
    master_table_path: str,
    output_dir: str,
    trading_cfg: Dict[str, Any],
    n_days: int = 14,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config_root: str = "config",
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> Dict[str, Any]:
    if not master_table_path or not os.path.exists(master_table_path):
        return {"ok": False, "error": f"Master table not found: {master_table_path}"}

    os.makedirs(output_dir, exist_ok=True)

    discovered = discover_top_model_configs(config_root=config_root)

    market_symbol = str(
        ((trading_cfg.get("dashboard") or {}).get("sql_symbol"))
        or ((trading_cfg.get("execution") or {}).get("symbol"))
        or "XAUUSD"
    ).strip()

    df = _load_master_df(
        master_table_path,
        n_days=n_days,
        start_date=start_date,
        end_date=end_date,
        symbol=market_symbol,
    )
    if "DATE" not in df.columns or "CLOSE" not in df.columns:
        return {"ok": False, "error": "Master table must contain DATE and CLOSE columns"}

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE")
    if df.empty:
        return {"ok": False, "error": "No valid master data rows"}

    if start_date:
        df = df[df["DATE"] >= pd.to_datetime(start_date, errors="coerce")]
    if end_date:
        df = df[df["DATE"] <= pd.to_datetime(end_date, errors="coerce")]
    if df.empty:
        return {"ok": False, "error": "No rows inside selected period"}

    unique_days = sorted(df["DATE"].dt.date.unique())
    if not unique_days:
        return {"ok": False, "error": "No trading days found"}

    if not start_date and not end_date:
        unique_days = unique_days[-max(int(n_days or 1), 1):]

    selected_df = df[df["DATE"].dt.date.isin(unique_days)].copy()

    # Build timeframe series for discovered configs + mandatory 7h Agent A series.
    timeframe_labels = sorted(set(list(discovered.keys()) + ["7h"]))
    tf_data: Dict[str, pd.DataFrame] = {}
    for tf in timeframe_labels:
        tf_data[tf] = _aggregate_timeframe(selected_df[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]].copy(), _tf_to_minutes(tf))

    session_hours = float(((trading_cfg.get("trading_job") or {}).get("session_hours", 7.0) or 7.0))

    day_reports: List[Dict[str, Any]] = []
    all_ops: List[Dict[str, Any]] = []

    for idx, day in enumerate(unique_days, start=1):
        day_df = selected_df[selected_df["DATE"].dt.date == day].copy().sort_values("DATE")
        if day_df.empty:
            continue

        day_start = pd.to_datetime(day_df["DATE"].iloc[0]).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = pd.to_datetime(day_df["DATE"].iloc[-1])

        sessions: List[Dict[str, Any]] = []
        t = day_start
        while t <= day_end:
            ses = _simulate_advanced_session(
                day_df_1m=day_df,
                tf_data=tf_data,
                session_start=t,
                session_hours=session_hours,
                trading_cfg=trading_cfg,
                available_timeframes=timeframe_labels,
            )
            sessions.append(ses)
            if bool(ses.get("executed", False)):
                all_ops.append(ses)
            t = t + timedelta(hours=session_hours)

        n_ops = int(sum(1 for s in sessions if bool(s.get("executed", False))))
        pnl_vals = [float(s.get("pnl_abs", 0.0)) for s in sessions if bool(s.get("executed", False))]
        day_report = {
            "day": str(day),
            "n_sessions": int(len(sessions)),
            "n_operations": n_ops,
            "n_green": int(sum(1 for p in pnl_vals if p > 0)),
            "n_red": int(sum(1 for p in pnl_vals if p <= 0)),
            "net_abs": float(sum(pnl_vals)) if pnl_vals else 0.0,
            "sessions": sessions,
        }
        day_path = os.path.join(output_dir, f"advanced_day_{str(day).replace('-', '')}.json")
        with open(day_path, "w", encoding="utf-8") as f:
            json.dump(day_report, f, indent=2)
        day_reports.append({"day": str(day), "file": day_path, **{k: v for k, v in day_report.items() if k != "sessions"}})

        if progress_cb:
            progress_cb(idx, len(unique_days), str(day))

    pnl_all = [float(s.get("pnl_abs", 0.0)) for s in all_ops]
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "advanced_timeframe_backtest",
        "master_table_path": master_table_path,
        "selected_days": [str(d) for d in unique_days],
        "period": {
            "start": str(min(unique_days)) if unique_days else None,
            "end": str(max(unique_days)) if unique_days else None,
            "n_days": int(len(unique_days)),
        },
        "discovered_top_configs": discovered,
        "overall": {
            "n_operations": int(len(all_ops)),
            "n_green": int(sum(1 for p in pnl_all if p > 0)),
            "n_red": int(sum(1 for p in pnl_all if p <= 0)),
            "profit_abs": float(sum(p for p in pnl_all if p > 0)),
            "loss_abs": float(sum(p for p in pnl_all if p <= 0)),
            "net_abs": float(sum(pnl_all)) if pnl_all else 0.0,
            "green_rate": float((sum(1 for p in pnl_all if p > 0) / max(len(pnl_all), 1))),
            "session_hours": session_hours,
            "agent_a_timeframe": "7h",
            "agent_b_technical_only": True,
            "llm_used": False,
        },
        "sessions": day_reports,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"ok": True, "summary_path": summary_path, "summary": summary}
