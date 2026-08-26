from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.market_db import query_ohlc


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


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
    """Run the model-backed point-in-time strategy replay used by the dashboard."""
    if not master_table_path or not os.path.exists(master_table_path):
        return {"ok": False, "error": f"Master table not found: {master_table_path}"}
    effective_start = start_date
    effective_end = end_date
    if not effective_start and not effective_end:
        market_symbol = str(
            ((trading_cfg.get("dashboard") or {}).get("sql_symbol"))
            or ((trading_cfg.get("execution") or {}).get("symbol"))
            or "XAUUSD"
        ).strip()
        preview = _load_master_df(master_table_path, n_days=max(int(n_days), 1), symbol=market_symbol)
        if "DATE" not in preview.columns:
            return {"ok": False, "error": "Master table must contain DATE"}
        preview["DATE"] = pd.to_datetime(preview["DATE"], errors="coerce")
        unique_days = sorted(preview.dropna(subset=["DATE"])["DATE"].dt.date.unique())
        unique_days = unique_days[-max(int(n_days or 1), 1):]
        if not unique_days:
            return {"ok": False, "error": "No trading days found"}
        effective_start = str(unique_days[0])
        effective_end = str(unique_days[-1])

    from utils.strategy_backtest import run_historical_strategy_backtest

    return run_historical_strategy_backtest(
        market_source=master_table_path,
        trading_cfg=trading_cfg,
        output_dir=output_dir,
        start_date=effective_start,
        end_date=effective_end,
        previous_month=False,
        progress_cb=progress_cb,
    )
