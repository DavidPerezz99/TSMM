"""
Deterministic investing agent (Mode A / Mode B scaffold, MT5-first).

Mode A: Generate a single-session deterministic trading plan/report.
Mode B: Optional live-management scaffold with MT5 adapter and model endpoint checks.
"""

from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import yaml

from .backtester import run_backtest_from_validation
from .trading_reporter import generate_trading_plan_report


class MT5Adapter:
    """MT5 execution adapter (graceful fallback if MetaTrader5 is unavailable)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self._mt5 = None

    def connect(self) -> tuple[bool, str]:
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception:
            return False, "MetaTrader5 package not installed"

        self._mt5 = mt5
        path = self.cfg.get("path") or None
        ok_init = mt5.initialize(path=path) if path else mt5.initialize()
        if not ok_init:
            return False, f"MT5 initialize failed: {mt5.last_error()}"

        login = int(self.cfg.get("login", 0) or 0)
        password = self.cfg.get("password", "")
        server = self.cfg.get("server", "")
        if login and password and server:
            ok_login = mt5.login(login=login, password=password, server=server)
            if not ok_login:
                return False, f"MT5 login failed: {mt5.last_error()}"

        return True, "connected"

    def shutdown(self):
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()
            except Exception:
                pass


def load_trading_config(path: str = "config/trading_agent.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"agent": {"enabled": False}, "error": f"Trading config not found: {path}"}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def _latest_price(df, target_col: str) -> float:
    return float(df[target_col].iloc[-1])


def _build_mode_a_plan(
    df,
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    evaluation: Dict[str, Any],
    future_forecasts: Dict[str, Any],
    preferred_model: Optional[str] = None,
) -> Dict[str, Any]:
    risk = trading_cfg.get("risk", {}) or {}
    mode_a = trading_cfg.get("mode_a", {}) or {}
    exec_cfg = trading_cfg.get("execution", {}) or {}

    best_model = _choose_best_model(evaluation, preferred_model=preferred_model)
    if best_model is None or best_model not in future_forecasts:
        return {
            "decision": "hold",
            "rationale": "No valid forecast model available",
            "risk_notes": ["Model selection failed; no operation recommended."],
        }

    forecast_block = future_forecasts[best_model] or {}
    pred = forecast_block.get("future") or []
    if not pred:
        return {
            "decision": "hold",
            "rationale": "Missing future forecast block",
            "risk_notes": ["Forecast output missing; no operation recommended."],
        }

    feature_paths = forecast_block.get("future_by_feature", {}) or {}
    target_features = app_config.get("target_features", []) or []
    tf0 = target_features[0] if target_features else None

    first_pred = pred[0][0] if isinstance(pred[0], list) and pred[0] else (pred[0] if isinstance(pred[0], (int, float)) else 0.0)

    # Feature-level first-step values (if available)
    f_open = feature_paths.get("OPEN", [None])[0] if feature_paths.get("OPEN") else None
    f_high = feature_paths.get("HIGH", [None])[0] if feature_paths.get("HIGH") else None
    f_low = feature_paths.get("LOW", [None])[0] if feature_paths.get("LOW") else None
    f_close = feature_paths.get("CLOSE", [None])[0] if feature_paths.get("CLOSE") else None
    f_pr = feature_paths.get("Price_return", [None])[0] if feature_paths.get("Price_return") else None
    f_or = feature_paths.get("Open_return", [None])[0] if feature_paths.get("Open_return") else None
    f_yd = feature_paths.get("y_diff", [None])[0] if feature_paths.get("y_diff") else None

    # Consensus score across OHLC/return dimensions
    signal_score = 0.0
    score_parts = []

    for name, val, w in [
        ("y_diff", f_yd, 1.0),
        ("Price_return", f_pr, 1.0),
        ("Open_return", f_or, 0.5),
    ]:
        if isinstance(val, (int, float)):
            s = 1.0 if float(val) > 0 else -1.0
            signal_score += w * s
            score_parts.append(f"{name}:{float(val):.4f}")

    # If direct OHLC targets are present, use close-open directional cue
    if isinstance(f_close, (int, float)) and isinstance(f_open, (int, float)):
        delta = float(f_close) - float(f_open)
        signal_score += 1.0 if delta > 0 else -1.0
        score_parts.append(f"CLOSE-OPEN:{delta:.4f}")

    # Fallback to primary prediction sign
    if abs(signal_score) < 1e-9:
        signal_score = 1.0 if float(first_pred) > 0 else -1.0

    direction = "buy" if signal_score > 0 else "sell"

    cm_acc = float(((evaluation.get(best_model, {}) or {}).get("confusion_matrix", {}) or {}).get("accuracy", 0.0) or 0.0)
    conf_levels = (evaluation.get(best_model, {}) or {}).get("confidence_levels", []) or []
    confidence = float(np.mean(conf_levels[:3])) if conf_levels else 0.5

    min_conf = float(risk.get("min_confidence_to_trade", 0.55))
    min_cm = float(risk.get("min_cm_accuracy_to_trade", 0.52))

    allow_long = bool(mode_a.get("allow_long", True))
    allow_short = bool(mode_a.get("allow_short", True))

    entry = _latest_price(df, app_config["target_col"])
    sl_pct = float(risk.get("stop_loss_pct", 0.8)) / 100.0
    tp_pct = float(risk.get("take_profit_pct", 1.6)) / 100.0

    if direction == "buy":
        stop_loss = entry * (1 - sl_pct)
        take_profit = entry * (1 + tp_pct)
    else:
        stop_loss = entry * (1 + sl_pct)
        take_profit = entry * (1 - tp_pct)

    decision = direction
    rationale = (
        f"Model={best_model}, score={signal_score:.2f}, "
        f"forecast_sign={float(first_pred):.4f}, cm_accuracy={cm_acc:.3f}, confidence={confidence:.3f}, "
        f"features=[{', '.join(score_parts[:6])}]"
    )
    risk_notes: List[str] = [
        f"Risk per trade={risk.get('risk_per_trade_pct', 0.5)}%",
        f"Daily max loss={risk.get('daily_max_loss_pct', 2.0)}%",
        f"Max open positions={risk.get('max_open_positions', 3)}",
    ]

    if confidence < min_conf or cm_acc < min_cm:
        decision = "hold"
        risk_notes.append("Signal blocked by confidence/confusion thresholds.")

    if decision == "buy" and not allow_long:
        decision = "hold"
        risk_notes.append("Long operations disabled by config.")
    if decision == "sell" and not allow_short:
        decision = "hold"
        risk_notes.append("Short operations disabled by config.")

    return {
        "decision": decision,
        "model": best_model,
        "entry": round(entry, 6),
        "stop_loss": round(float(stop_loss), 6),
        "take_profit": round(float(take_profit), 6),
        "volume": float(exec_cfg.get("default_volume", 0.01)),
        "confidence": round(confidence, 4),
        "cm_accuracy": round(cm_acc, 4),
        "signal_score": round(signal_score, 4),
        "target_anchor": tf0,
        "feature_forecasts_step1": {
            "OPEN": f_open,
            "HIGH": f_high,
            "LOW": f_low,
            "CLOSE": f_close,
            "Price_return": f_pr,
            "Open_return": f_or,
            "y_diff": f_yd,
        },
        "rationale": rationale,
        "risk_notes": risk_notes,
    }


def _estimate_signal_success_probability(plan: Dict[str, Any], backtest: Dict[str, Any]) -> float:
    """Estimate signal success probability using confidence + CM quality + backtest prior."""
    conf = float(plan.get("confidence", 0.5) or 0.5)
    cm_acc = float(plan.get("cm_accuracy", 0.5) or 0.5)

    win_rate = float(backtest.get("win_rate", 0.5) or 0.5)
    n_trades = int(backtest.get("n_trades", 0) or 0)
    prior_strength = 20.0
    empirical = (win_rate * n_trades + 0.5 * prior_strength) / (n_trades + prior_strength)

    p = 0.4 * conf + 0.3 * cm_acc + 0.3 * empirical
    return float(np.clip(p, 0.01, 0.99))


def _build_probability_heatmaps(
    df,
    app_config: Dict[str, Any],
    future_forecasts: Dict[str, Any],
    model_name: str,
    output_dir: str,
    n_paths: int = 1200,
    n_price_bins: int = 70,
) -> Dict[str, Any]:
    """Create 2D/3D probability concentration maps for future price trajectories."""
    try:
        if model_name not in future_forecasts:
            return {"enabled": False, "error": "Model forecast block missing"}

        forecast_block = future_forecasts.get(model_name, {}) or {}
        future = np.asarray(forecast_block.get("future", []), dtype=float)
        validation = np.asarray(forecast_block.get("validation", []), dtype=float)
        if future.size == 0:
            return {"enabled": False, "error": "Future forecast unavailable"}

        if future.ndim == 2:
            yhat = future[:, 0]
        else:
            yhat = future.reshape(-1)

        target_features = app_config.get("target_features", []) or []
        target_col = app_config.get("target_col")

        resid_std = 1.0
        if validation.size > 0 and target_features:
            n_val = validation.shape[0] if validation.ndim > 1 else validation.shape[0]
            y_true = df[target_features].iloc[-n_val:].values
            y_pred = validation if validation.ndim > 1 else validation.reshape(-1, 1)
            min_len = min(len(y_true), len(y_pred))
            if min_len > 5:
                residuals = y_true[:min_len, 0] - y_pred[:min_len, 0]
                resid_std = float(np.std(residuals, ddof=1)) if np.std(residuals) > 0 else 1.0

        if target_features and target_features[0] == "y_diff" and target_col in df.columns:
            anchor = float(df[target_col].iloc[-1])
            mean_path = anchor + np.cumsum(yhat)
            shocks = np.random.normal(loc=0.0, scale=max(resid_std, 1e-6), size=(n_paths, len(yhat)))
            paths = anchor + np.cumsum((yhat.reshape(1, -1) + shocks), axis=1)
        else:
            mean_path = yhat.copy()
            increments = np.diff(np.r_[mean_path[0], mean_path])
            shocks = np.random.normal(loc=0.0, scale=max(resid_std, 1e-6), size=(n_paths, len(yhat)))
            paths = mean_path.reshape(1, -1) + np.cumsum(shocks + increments.reshape(1, -1) * 0.0, axis=1)

        p_min, p_max = float(np.nanmin(paths)), float(np.nanmax(paths))
        if p_min == p_max:
            p_min -= 1.0
            p_max += 1.0
        bins = np.linspace(p_min, p_max, n_price_bins + 1)

        density = np.zeros((len(yhat), n_price_bins), dtype=float)
        for t in range(len(yhat)):
            hist, _ = np.histogram(paths[:, t], bins=bins, density=True)
            density[t, :] = hist

        centers = 0.5 * (bins[:-1] + bins[1:])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)

        heat2d_path = os.path.join(output_dir, f"probability_heatmap_2d_{model_name}_{ts}.png")
        fig2d, ax2d = plt.subplots(figsize=(10, 5))
        im = ax2d.imshow(
            density.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[1, len(yhat), centers[0], centers[-1]],
        )
        ax2d.plot(np.arange(1, len(yhat) + 1), mean_path, color="white", linewidth=1.5, label="Mean path")
        ax2d.set_title("Time-Price Probability Heatmap (2D)")
        ax2d.set_xlabel("Horizon step")
        ax2d.set_ylabel("Price")
        ax2d.legend(loc="upper right")
        fig2d.colorbar(im, ax=ax2d, label="Density")
        fig2d.tight_layout()
        fig2d.savefig(heat2d_path, dpi=200, bbox_inches="tight")
        plt.close(fig2d)

        heat3d_path = os.path.join(output_dir, f"probability_heatmap_3d_{model_name}_{ts}.png")
        fig3d = plt.figure(figsize=(10, 6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(np.arange(1, len(yhat) + 1), centers)
        Z = density.T
        ax3d.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
        ax3d.set_title("Time-Price Density Surface (3D)")
        ax3d.set_xlabel("Horizon step")
        ax3d.set_ylabel("Price")
        ax3d.set_zlabel("Density")
        fig3d.tight_layout()
        fig3d.savefig(heat3d_path, dpi=200, bbox_inches="tight")
        plt.close(fig3d)

        return {
            "enabled": True,
            "model": model_name,
            "n_paths": int(n_paths),
            "residual_std": float(resid_std),
            "heatmap_2d_path": heat2d_path,
            "heatmap_3d_path": heat3d_path,
        }
    except Exception as e:
        return {"enabled": False, "error": str(e)}


def _check_endpoints(model_endpoints: Dict[str, str], timeout_sec: float = 2.0) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for tf, url in (model_endpoints or {}).items():
        try:
            r = requests.get(url, timeout=timeout_sec)
            out[tf] = {"ok": r.status_code < 500, "status_code": r.status_code}
        except Exception as e:
            out[tf] = {"ok": False, "error": str(e)}
    return out


def _extract_endpoint_signal(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized signal from heterogeneous model endpoint payloads."""
    if not isinstance(payload, dict):
        return {"signal": 0, "confidence": 0.5, "raw": payload}

    if "signal" in payload:
        s = str(payload.get("signal", "hold")).lower()
        signal = 1 if s in ("buy", "long", "up") else (-1 if s in ("sell", "short", "down") else 0)
    elif "forecast_sign" in payload and isinstance(payload.get("forecast_sign"), (int, float)):
        fs = float(payload.get("forecast_sign"))
        signal = 1 if fs > 0 else (-1 if fs < 0 else 0)
    elif "prediction" in payload and isinstance(payload.get("prediction"), (int, float)):
        pv = float(payload.get("prediction"))
        signal = 1 if pv > 0 else (-1 if pv < 0 else 0)
    else:
        signal = 0

    confidence = payload.get("confidence", payload.get("probability", 0.5))
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.5

    return {
        "signal": signal,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "raw": payload,
    }


def _collect_mode_b_signals(model_endpoints: Dict[str, str], timeout_sec: float = 2.0) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    votes = []
    weighted = 0.0
    total_w = 0.0

    for tf, url in (model_endpoints or {}).items():
        try:
            r = requests.get(url, timeout=timeout_sec)
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            sig = _extract_endpoint_signal(data)
            sig["status_code"] = r.status_code
            out[tf] = sig
            votes.append(sig["signal"])
            w = max(sig["confidence"], 0.01)
            weighted += w * sig["signal"]
            total_w += w
        except Exception as e:
            out[tf] = {"signal": 0, "confidence": 0.5, "error": str(e)}

    consensus_score = weighted / total_w if total_w > 0 else 0.0
    consensus = "buy" if consensus_score > 0.1 else ("sell" if consensus_score < -0.1 else "hold")
    return {
        "timeframes": out,
        "consensus": consensus,
        "consensus_score": float(consensus_score),
        "n_timeframes": len(out),
    }


def _write_runtime_state(output_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, "agent_state_latest.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return state_path


def run_investing_agent(
    app_config: Dict[str, Any],
    results: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    selected_model: Optional[str] = None,
    mode_override: Optional[str] = None,
    interrupt_mode_b: bool = False,
) -> Dict[str, Any]:
    agent_cfg = (trading_cfg.get("agent") or {})
    if not agent_cfg.get("enabled", False):
        return {"enabled": False, "reason": "Trading agent disabled"}

    mode = str(mode_override or agent_cfg.get("mode", "mode_a")).lower()
    target_col = app_config.get("target_col", "asset")

    # External interrupt flag (for UI/dashboard stop control)
    mb_cfg = (trading_cfg.get("mode_b") or {})
    interrupt_flag_path = mb_cfg.get("interrupt_flag_path", os.path.join(output_dir, "runtime", "mode_b_interrupt.flag"))
    interrupt_mode_b = bool(interrupt_mode_b or (interrupt_flag_path and os.path.exists(interrupt_flag_path)))

    plan = _build_mode_a_plan(
        results.get("df"),
        app_config,
        trading_cfg,
        results.get("evaluation", {}),
        results.get("future_forecasts", {}),
        preferred_model=selected_model,
    )

    backtest = run_backtest_from_validation(
        results.get("df"),
        app_config,
        results.get("evaluation", {}),
        results.get("future_forecasts", {}),
        trading_cfg,
        preferred_model=plan.get("model"),
    )

    success_probability = _estimate_signal_success_probability(plan, backtest)
    plan["success_probability"] = round(success_probability, 4)

    pm_cfg = trading_cfg.get("probability_maps", {}) or {}
    if bool(pm_cfg.get("enabled", True)):
        prob_dir = os.path.join(output_dir, "probability_maps")
        heatmaps = _build_probability_heatmaps(
            results.get("df"),
            app_config,
            results.get("future_forecasts", {}),
            plan.get("model"),
            prob_dir,
            n_paths=int(pm_cfg.get("n_paths", 1200)),
            n_price_bins=int(pm_cfg.get("n_price_bins", 70)),
        )
    else:
        heatmaps = {"enabled": False, "reason": "Disabled in trading config"}

    warnings: List[str] = []

    mode_b_status = {"enabled": False}
    if mode == "mode_b" and (trading_cfg.get("mode_b") or {}).get("enabled", False):
        if interrupt_mode_b:
            warnings.append("Mode B execution interrupted by user flag.")
            mode_b_status = {
                "enabled": True,
                "interrupted": True,
                "reason": "interrupt_mode_b flag",
                "interrupt_flag_path": interrupt_flag_path,
            }
        else:
            endpoints = _check_endpoints(trading_cfg.get("model_endpoints", {}))
            if bool((trading_cfg.get("mode_b") or {}).get("allow_endpoint_signals", True)):
                mtf_signals = _collect_mode_b_signals(trading_cfg.get("model_endpoints", {}))
            else:
                mtf_signals = {"timeframes": {}, "consensus": "hold", "consensus_score": 0.0, "n_timeframes": 0}
            mode_b_status = {
                "enabled": True,
                "endpoints": endpoints,
                "timeframe_signals": mtf_signals,
                "live_execution_requested": not bool(agent_cfg.get("confirm_live_execution", True)),
            }

            broker_cfg = (trading_cfg.get("broker", {}) or {}).get("mt5", {}) or {}
            if (trading_cfg.get("broker", {}) or {}).get("active", "mt5") == "mt5" and broker_cfg.get("enabled", False):
                adapter = MT5Adapter(broker_cfg)
                ok, msg = adapter.connect()
                mode_b_status["mt5_connection"] = {"ok": ok, "message": msg}
                adapter.shutdown()
            else:
                warnings.append("Mode B selected but MT5 broker is disabled.")

    report_cfg = trading_cfg.get("reporting", {}) or {}
    rep_dir = report_cfg.get("output_dir") or os.path.join(output_dir, "trading_plans")
    os.makedirs(rep_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep_prefix = report_cfg.get("filename_prefix", "trading_plan")
    plan_path = os.path.join(rep_dir, f"{rep_prefix}_{target_col}_{ts}.pdf")

    generate_trading_plan_report(
        output_path=plan_path,
        target_col=target_col,
        mode=mode,
        plan=plan,
        backtest=backtest,
        warnings=warnings,
        heatmaps=heatmaps,
    )

    state_dir = os.path.join(output_dir, "runtime")
    state_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "plan": plan,
        "backtest": {
            "model_name": backtest.get("model_name"),
            "n_trades": backtest.get("n_trades"),
            "win_rate": backtest.get("win_rate"),
            "total_return_pct": backtest.get("total_return_pct"),
            "max_drawdown_pct": backtest.get("max_drawdown_pct"),
        },
        "mode_b": mode_b_status,
        "open_positions": (backtest.get("trades") or [])[-3:],
        "signal_success_probability": success_probability,
        "heatmaps": heatmaps,
        "report_path": plan_path,
        "warnings": warnings,
    }
    state_path = _write_runtime_state(state_dir, state_payload)

    return {
        "enabled": True,
        "mode": mode,
        "plan": plan,
        "backtest": backtest,
        "signal_success_probability": success_probability,
        "heatmaps": heatmaps,
        "mode_b": mode_b_status,
        "state_path": state_path,
        "report_path": plan_path,
        "warnings": warnings,
    }
