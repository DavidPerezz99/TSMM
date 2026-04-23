"""
Regime Classification Module (Phase-2 scaffold)

Provides a lightweight 2-state regime detector and policy mapper:
- risk_on
- risk_off
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd


def _returns(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    if "y_diff" in df.columns:
        return pd.Series(df["y_diff"], index=df.index, dtype=float)

    target_col = config.get("target_col")
    if target_col in df.columns:
        s = pd.Series(df[target_col], index=df.index, dtype=float)
        return s.pct_change().fillna(0.0)

    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if not numeric_cols:
        return pd.Series([], dtype=float)
    return pd.Series(df[numeric_cols[0]], index=df.index, dtype=float).pct_change().fillna(0.0)


def classify_market_regime(
    df: pd.DataFrame,
    config: Dict[str, Any],
    momentum_result: Dict[str, Any] | None = None,
    vol_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Classify market regime and attach policy guidance."""
    rcfg = config.get("regime", {}) or {}
    if not rcfg.get("enabled", False):
        return {"enabled": False}

    ret = _returns(df, config).replace([np.inf, -np.inf], np.nan).dropna()
    if len(ret) < 30:
        return {"enabled": True, "error": "Insufficient samples for regime classification"}

    growth_window = int(rcfg.get("growth_window", 20))
    vol_window = int(rcfg.get("vol_window", 30))
    vol_quantile = float(rcfg.get("vol_quantile", 0.75))

    short_growth = float(ret.iloc[-growth_window:].mean()) if len(ret) >= growth_window else float(ret.mean())
    realized_vol = float(ret.iloc[-vol_window:].std(ddof=0)) if len(ret) >= vol_window else float(ret.std(ddof=0))
    vol_threshold = float(ret.rolling(max(vol_window, 5)).std(ddof=0).dropna().quantile(vol_quantile)) if len(ret) >= vol_window * 2 else float(ret.std(ddof=0))

    mom_score = float((momentum_result or {}).get("momentum_score", 0.0))

    is_risk_off = (short_growth < 0 and realized_vol >= vol_threshold) or (mom_score < -0.2)
    regime_state = "risk_off" if is_risk_off else "risk_on"

    # Confidence proxy from boundary distance.
    vol_dist = (realized_vol - vol_threshold) / (abs(vol_threshold) + 1e-9)
    growth_dist = -short_growth if is_risk_off else short_growth
    raw_conf = abs(0.7 * vol_dist + 0.3 * growth_dist + 0.4 * mom_score)
    confidence = float(np.clip(np.tanh(max(raw_conf, 0.0)), 0.0, 1.0))

    policy_map = (rcfg.get("policy_map", {}) or {})
    default_policy = {
        "models": ["ulr", "svr", "nbeats"],
        "risk_scale": 1.0,
    }
    policy = policy_map.get(regime_state, default_policy)

    # Optional adjustment with vol targeting output.
    suggested_scale = float(policy.get("risk_scale", 1.0))
    if vol_result and isinstance(vol_result.get("position_scale"), (int, float)):
        suggested_scale *= float(vol_result["position_scale"])

    return {
        "enabled": True,
        "as_of": datetime.utcnow().isoformat(),
        "state": regime_state,
        "confidence": confidence,
        "signals": {
            "short_growth": short_growth,
            "realized_vol": realized_vol,
            "vol_threshold": vol_threshold,
            "momentum_score": mom_score,
        },
        "policy": {
            "preferred_models": policy.get("models", default_policy["models"]),
            "risk_scale": float(policy.get("risk_scale", 1.0)),
            "suggested_scale": float(suggested_scale),
        },
    }
