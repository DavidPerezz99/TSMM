"""
Momentum Overlay Module

Computes a compact multi-horizon momentum signal and trend state for
risk-aware forecast overlays.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _clip_score(x: float) -> float:
    """Map a raw score to [-1, 1] with smooth saturation."""
    return float(np.tanh(x))


def _get_signal_series(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """Pick the best available series for momentum calculations."""
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


def compute_momentum_overlay(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute multi-window momentum score and trend state.

    Output keys:
    - momentum_score in [-1, 1]
    - trend_state: up/down/flat
    - confidence_bucket: low/medium/high
    """
    mcfg = config.get("momentum", {}) or {}
    if not mcfg.get("enabled", True):
        return {"enabled": False}

    windows: List[int] = [int(w) for w in mcfg.get("windows", [20, 60, 120]) if int(w) > 1]
    if not windows:
        windows = [20, 60, 120]

    signal = _get_signal_series(df, config)
    signal = signal.replace([np.inf, -np.inf], np.nan).dropna()

    if len(signal) < max(windows) + 2:
        return {
            "enabled": True,
            "error": "Insufficient samples for momentum windows",
            "windows": windows,
        }

    # Use cumulative return over each lookback and normalize by local volatility.
    by_window: Dict[str, float] = {}
    per_window_scores: List[float] = []

    eps = 1e-9
    for w in windows:
        recent = signal.iloc[-w:]
        cum_ret = float((1.0 + recent).prod() - 1.0)
        local_vol = float(recent.std(ddof=0))
        raw = cum_ret / (local_vol + eps)
        score = _clip_score(raw)
        by_window[str(w)] = score
        per_window_scores.append(score)

    weights = np.array(mcfg.get("weights", [1.0] * len(per_window_scores)), dtype=float)
    if len(weights) != len(per_window_scores) or np.allclose(weights.sum(), 0.0):
        weights = np.ones(len(per_window_scores), dtype=float)
    weights = weights / weights.sum()

    momentum_score = float(np.dot(weights, np.array(per_window_scores, dtype=float)))

    flat_thr = float(mcfg.get("flat_threshold", 0.12))
    if momentum_score > flat_thr:
        trend_state = "up"
    elif momentum_score < -flat_thr:
        trend_state = "down"
    else:
        trend_state = "flat"

    strength = abs(momentum_score)
    if strength < 0.2:
        confidence_bucket = "low"
    elif strength < 0.5:
        confidence_bucket = "medium"
    else:
        confidence_bucket = "high"

    return {
        "enabled": True,
        "as_of": datetime.utcnow().isoformat(),
        "windows": windows,
        "by_window": by_window,
        "momentum_score": momentum_score,
        "trend_state": trend_state,
        "confidence_bucket": confidence_bucket,
    }
