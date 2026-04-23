"""
Volatility Targeting Module

Computes realized volatility and a clipped position scale for risk control.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd


def _get_return_series(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    if "y_diff" in df.columns:
        return pd.Series(df["y_diff"], index=df.index, dtype=float)

    target_col = config.get("target_col")
    if target_col in df.columns:
        target = pd.Series(df[target_col], index=df.index, dtype=float)
        return target.pct_change().fillna(0.0)

    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if not numeric_cols:
        return pd.Series([], dtype=float)

    return pd.Series(df[numeric_cols[0]], index=df.index, dtype=float).pct_change().fillna(0.0)


def compute_vol_target_overlay(
    df: pd.DataFrame,
    config: Dict[str, Any],
    momentum_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute realized vol and exposure scale with hard caps."""
    vcfg = config.get("vol_target", {}) or {}
    if not vcfg.get("enabled", True):
        return {"enabled": False}

    ret = _get_return_series(df, config).replace([np.inf, -np.inf], np.nan).dropna()
    window = int(vcfg.get("window", 30))
    bars_per_year = float(vcfg.get("bars_per_year", 252.0))

    if len(ret) < max(5, window):
        return {
            "enabled": True,
            "error": "Insufficient samples for volatility targeting",
            "window": window,
        }

    realized_vol = float(ret.iloc[-window:].std(ddof=0) * np.sqrt(max(bars_per_year, 1.0)))
    target_vol = float(vcfg.get("target_vol", 0.15))

    caps = vcfg.get("caps", {}) or {}
    min_scale = float(caps.get("min_scale", 0.25))
    max_scale = float(caps.get("max_scale", 1.5))
    max_leverage = float(caps.get("max_leverage", 2.0))
    min_exposure = float(caps.get("min_exposure", 0.0))

    eps = 1e-9
    raw_scale = target_vol / max(realized_vol, eps)
    clipped_scale = float(np.clip(raw_scale, min_scale, max_scale))

    # Optional momentum-aware sign. If missing, keep long-only +1.
    direction = 1
    if momentum_result and momentum_result.get("trend_state") == "down":
        direction = -1

    signed_exposure = float(np.clip(direction * clipped_scale, -max_leverage, max_leverage))
    if abs(signed_exposure) < min_exposure:
        signed_exposure = float(np.sign(signed_exposure) * min_exposure) if signed_exposure != 0 else float(min_exposure)

    return {
        "enabled": True,
        "as_of": datetime.utcnow().isoformat(),
        "window": window,
        "bars_per_year": bars_per_year,
        "realized_vol": realized_vol,
        "target_vol": target_vol,
        "raw_scale": float(raw_scale),
        "position_scale": clipped_scale,
        "recommended_exposure": signed_exposure,
        "caps": {
            "min_scale": min_scale,
            "max_scale": max_scale,
            "max_leverage": max_leverage,
            "min_exposure": min_exposure,
        },
    }
