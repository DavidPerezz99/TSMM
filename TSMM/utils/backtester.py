"""
Deterministic backtesting utilities for operation lifecycle validation.

Backtests open/hold/close logic on validation-window predictions already
produced by the forecasting pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    side: str
    entry_index: int
    exit_index: int
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl_abs: float
    pnl_pct: float
    bars_held: int
    exit_reason: str


@dataclass
class BacktestResult:
    model_name: str
    n_trades: int
    win_rate: float
    total_pnl_abs: float
    total_return_pct: float
    max_drawdown_pct: float
    avg_trade_return_pct: float
    trades: List[Dict[str, Any]]
    equity_curve: List[float]


def _choose_best_model(evaluation: Dict[str, Any], preferred_model: Optional[str] = None) -> Optional[str]:
    if preferred_model and preferred_model in (evaluation or {}):
        return preferred_model

    best = None
    best_mae = float("inf")
    for model_name, ev in (evaluation or {}).items():
        mae = (ev.get("metrics") or {}).get("MAE")
        if isinstance(mae, (int, float)) and mae < best_mae:
            best_mae = mae
            best = model_name
    return best


def _max_drawdown_pct(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = (equity - peaks) / np.maximum(peaks, 1e-12)
    return float(abs(np.min(dd)) * 100.0)


def run_backtest_from_validation(
    df: pd.DataFrame,
    config: Dict[str, Any],
    evaluation: Dict[str, Any],
    future_forecasts: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    preferred_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run deterministic backtest using validation forecasts vs actual prices.

    Notes:
    - Uses selected best model by MAE.
    - Uses validation forecast primary dimension as directional signal.
    - Simulates one position at a time with SL/TP/trailing and max hold bars.
    """
    model_name = _choose_best_model(evaluation, preferred_model=preferred_model)
    if model_name is None or model_name not in future_forecasts:
        return {"enabled": True, "error": "No valid model forecasts for backtesting"}

    pred_block = future_forecasts[model_name].get("validation")
    if not isinstance(pred_block, list) or len(pred_block) < 5:
        return {"enabled": True, "error": "Insufficient validation predictions"}

    # primary forecast signal
    pred = np.array(pred_block, dtype=float)
    if pred.ndim > 1:
        pred_sig = pred[:, 0]
    else:
        pred_sig = pred

    target_col = config.get("target_col")
    if target_col not in df.columns:
        return {"enabled": True, "error": f"Missing target_col '{target_col}' in dataframe"}

    n = len(pred_sig)
    actual = df[target_col].iloc[-n:].astype(float).to_numpy()

    risk = trading_cfg.get("risk", {}) or {}
    exec_cfg = trading_cfg.get("execution", {}) or {}

    sl_pct = float(risk.get("stop_loss_pct", 0.8)) / 100.0
    tp_pct = float(risk.get("take_profit_pct", 1.6)) / 100.0
    max_hold = int(trading_cfg.get("mode_a", {}).get("max_hold_bars", config.get("horizon", 6)))
    conf_min = float(risk.get("min_confidence_to_trade", 0.55))

    cm_acc = float(((evaluation.get(model_name, {}) or {}).get("confusion_matrix", {}) or {}).get("accuracy", 0.0) or 0.0)
    cm_min = float(risk.get("min_cm_accuracy_to_trade", 0.52))
    if cm_acc < cm_min:
        return {
            "enabled": True,
            "model_name": model_name,
            "error": f"CM accuracy gate failed ({cm_acc:.3f} < {cm_min:.3f})",
        }

    conf_levels = (evaluation.get(model_name, {}) or {}).get("confidence_levels", []) or []

    spread_bps = float(exec_cfg.get("spread_bps", 2.0))
    slip_bps = float(exec_cfg.get("slippage_bps", 2.0))
    commission = float(exec_cfg.get("commission_per_trade", 0.0))

    trail_cfg = (risk.get("trailing", {}) or {})
    trail_enabled = bool(trail_cfg.get("enabled", True))
    trail_base = float(trail_cfg.get("trail_pct_base", 0.5)) / 100.0
    trail_conf_gate = float(trail_cfg.get("min_confidence_to_extend", 0.65))

    trades: List[Trade] = []
    equity = [1.0]

    in_pos = False
    side = ""
    entry_i = 0
    entry = 0.0
    sl = 0.0
    tp = 0.0

    def _apply_cost(px: float, is_entry: bool) -> float:
        # conservative cost model using bps and side-agnostic friction
        cost = (spread_bps + slip_bps) / 10000.0
        return px * (1 + cost if is_entry else 1 - cost)

    i = 0
    while i < n:
        px = actual[i]
        conf = float(conf_levels[i]) if i < len(conf_levels) and isinstance(conf_levels[i], (int, float)) else 1.0

        if not in_pos:
            signal = 1 if pred_sig[i] > 0 else -1
            if conf >= conf_min:
                in_pos = True
                side = "long" if signal > 0 else "short"
                entry_i = i
                entry = _apply_cost(px, is_entry=True)
                if side == "long":
                    sl = entry * (1 - sl_pct)
                    tp = entry * (1 + tp_pct)
                else:
                    sl = entry * (1 + sl_pct)
                    tp = entry * (1 - tp_pct)
            i += 1
            continue

        # Manage open position
        bars = i - entry_i
        exit_reason = None
        exit_px = None

        if side == "long":
            if px <= sl:
                exit_reason, exit_px = "stop_loss", _apply_cost(px, is_entry=False)
            elif px >= tp:
                exit_reason, exit_px = "take_profit", _apply_cost(px, is_entry=False)
            elif bars >= max_hold:
                exit_reason, exit_px = "time_stop", _apply_cost(px, is_entry=False)
            elif trail_enabled and conf >= trail_conf_gate:
                sl = max(sl, px * (1 - trail_base))
        else:
            if px >= sl:
                exit_reason, exit_px = "stop_loss", _apply_cost(px, is_entry=False)
            elif px <= tp:
                exit_reason, exit_px = "take_profit", _apply_cost(px, is_entry=False)
            elif bars >= max_hold:
                exit_reason, exit_px = "time_stop", _apply_cost(px, is_entry=False)
            elif trail_enabled and conf >= trail_conf_gate:
                sl = min(sl, px * (1 + trail_base))

        if exit_reason is not None and exit_px is not None:
            if side == "long":
                pnl_abs = (exit_px - entry) - commission
                pnl_pct = (exit_px / max(entry, 1e-12) - 1.0) * 100.0
            else:
                pnl_abs = (entry - exit_px) - commission
                pnl_pct = (entry / max(exit_px, 1e-12) - 1.0) * 100.0

            trades.append(
                Trade(
                    side=side,
                    entry_index=entry_i,
                    exit_index=i,
                    entry_price=float(entry),
                    exit_price=float(exit_px),
                    stop_loss=float(sl),
                    take_profit=float(tp),
                    pnl_abs=float(pnl_abs),
                    pnl_pct=float(pnl_pct),
                    bars_held=bars,
                    exit_reason=exit_reason,
                )
            )

            equity.append(equity[-1] * (1.0 + pnl_pct / 100.0))
            in_pos = False
            side = ""
        i += 1

    # Close any remaining position at end
    if in_pos and n > 0:
        exit_px = _apply_cost(actual[-1], is_entry=False)
        bars = (n - 1) - entry_i
        if side == "long":
            pnl_abs = (exit_px - entry) - commission
            pnl_pct = (exit_px / max(entry, 1e-12) - 1.0) * 100.0
        else:
            pnl_abs = (entry - exit_px) - commission
            pnl_pct = (entry / max(exit_px, 1e-12) - 1.0) * 100.0
        trades.append(
            Trade(
                side=side,
                entry_index=entry_i,
                exit_index=n - 1,
                entry_price=float(entry),
                exit_price=float(exit_px),
                stop_loss=float(sl),
                take_profit=float(tp),
                pnl_abs=float(pnl_abs),
                pnl_pct=float(pnl_pct),
                bars_held=bars,
                exit_reason="forced_close",
            )
        )
        equity.append(equity[-1] * (1.0 + pnl_pct / 100.0))

    trade_returns = np.array([t.pnl_pct for t in trades], dtype=float) if trades else np.array([])
    win_rate = float(np.mean(trade_returns > 0)) if trade_returns.size else 0.0
    total_return = float((equity[-1] - 1.0) * 100.0) if equity else 0.0

    result = BacktestResult(
        model_name=model_name,
        n_trades=len(trades),
        win_rate=win_rate,
        total_pnl_abs=float(np.sum([t.pnl_abs for t in trades])) if trades else 0.0,
        total_return_pct=total_return,
        max_drawdown_pct=_max_drawdown_pct(np.array(equity, dtype=float)),
        avg_trade_return_pct=float(np.mean(trade_returns)) if trade_returns.size else 0.0,
        trades=[asdict(t) for t in trades],
        equity_curve=[float(x) for x in equity],
    )

    return {"enabled": True, **asdict(result)}
