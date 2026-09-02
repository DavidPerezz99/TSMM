"""Shared OHLC confirmation and volatility policy for live and replay trading."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


DEFAULT_FAMILY_WEIGHTS = {
    "high": 0.45,
    "low": 0.45,
    "close": 0.10,
    "open": 0.00,
}


def _finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def signal_policy_config(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict((trading_cfg or {}).get("signal_policy") or {})
    cfg.setdefault("enabled", False)
    cfg.setdefault("direction_timeframe", "7h")
    cfg.setdefault(
        "direction_timeframes", ["10m", "30m", "1h", "3h", "7h", "12h", "24h"]
    )
    cfg.setdefault("select_best_direction_timeframe", True)
    cfg.setdefault("confirmation_timeframes", ["10m", "30m", "1h"])
    cfg.setdefault("required_confirmations", 2)
    cfg.setdefault("min_direction_score", 0.12)
    cfg.setdefault("min_confirmation_score", 0.08)
    cfg.setdefault("minimum_coverage", 0.70)
    cfg.setdefault("family_weights", dict(DEFAULT_FAMILY_WEIGHTS))
    cfg.setdefault("require_projected_range", True)
    cfg.setdefault("entry_range_fraction", 0.20)
    cfg.setdefault("max_entry_offset_pct", 0.40)
    return cfg


def _timeframe_score(
    signals: Dict[str, Dict[str, Any]],
    timeframe: str,
    family_weights: Dict[str, float],
) -> Dict[str, Any]:
    weighted = 0.0
    total = 0.0
    families: Dict[str, Any] = {}
    for family, configured_weight in family_weights.items():
        item = dict(signals.get(f"{family}:{timeframe}") or {})
        signal = int(item.get("signal", 0) or 0)
        if item.get("error") or signal == 0:
            families[family] = {"available": False, "error": item.get("error")}
            continue
        confidence = max(min(float(item.get("confidence", 0.5) or 0.5), 1.0), 0.0)
        quality_weight = 1.0
        quality = item.get("quality")
        if isinstance(quality, dict):
            quality_weight = max(float(quality.get("weight", 1.0) or 0.0), 0.0)
            if not bool(quality.get("qualified", True)):
                quality_weight = 0.0
        weight = max(float(configured_weight), 0.0) * max(confidence, 0.01) * quality_weight
        if weight <= 0.0:
            families[family] = {"available": False, "reason": "model_quality_not_qualified"}
            continue
        weighted += weight * signal
        total += weight
        families[family] = {
            "available": True,
            "signal": signal,
            "confidence": confidence,
            "weight": weight,
        }
    score = weighted / total if total > 0.0 else 0.0
    return {
        "timeframe": timeframe,
        "score": float(score),
        "direction": "buy" if score > 0.0 else ("sell" if score < 0.0 else "hold"),
        "available_families": sum(1 for value in families.values() if value.get("available")),
        "families": families,
    }


def _qualified_signal(signals: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    item = dict(signals.get(key) or {})
    quality = item.get("quality")
    if item.get("error") or (isinstance(quality, dict) and not quality.get("qualified", True)):
        return {}
    return item


def _projected_range(signals: Dict[str, Dict[str, Any]], timeframe: str) -> Dict[str, Any]:
    high_item = _qualified_signal(signals, f"high:{timeframe}")
    low_item = _qualified_signal(signals, f"low:{timeframe}")
    high = _finite_float(high_item.get("forecast_price"))
    low = _finite_float(low_item.get("forecast_price"))
    if high is None or low is None:
        return {
            "available": False,
            "reason": "qualified_high_low_forecast_prices_unavailable",
        }
    if high <= low:
        return {
            "available": False,
            "reason": "independent_high_low_forecasts_crossed_or_collapsed",
            "forecast_high": float(high),
            "forecast_low": float(low),
        }
    projected_low = low
    projected_high = high
    return {
        "available": projected_high > projected_low,
        "projected_low": float(projected_low),
        "projected_high": float(projected_high),
        "range_width": float(projected_high - projected_low),
        "independent_forecasts_crossed": False,
    }


def evaluate_joint_ohlc_policy(
    bundle: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    *,
    market_price: Optional[float] = None,
) -> Dict[str, Any]:
    """Choose a side from OHLC families and require shorter-timeframe timing."""
    cfg = signal_policy_config(trading_cfg)
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "decision": "hold", "reason": "signal_policy_disabled"}

    coverage = float(bundle.get("coverage", 1.0) or 0.0)
    minimum_coverage = float(cfg.get("minimum_coverage", 0.70) or 0.70)
    if coverage < minimum_coverage:
        return {
            "enabled": True,
            "decision": "hold",
            "reason": "model_coverage_below_policy_minimum",
            "coverage": coverage,
        }

    signals = dict(bundle.get("signals") or {})
    family_weights = {
        str(key).strip().lower(): max(float(value), 0.0)
        for key, value in dict(cfg.get("family_weights") or DEFAULT_FAMILY_WEIGHTS).items()
    }
    configured_direction_timeframe = str(cfg.get("direction_timeframe") or "7h")
    direction_timeframes = [
        str(value) for value in (cfg.get("direction_timeframes") or []) if str(value)
    ]
    if configured_direction_timeframe not in direction_timeframes:
        direction_timeframes.append(configured_direction_timeframe)
    candidate_votes = [
        _timeframe_score(signals, timeframe, family_weights)
        for timeframe in direction_timeframes
    ]

    def _direction_rank(vote: Dict[str, Any]) -> tuple[float, float, int]:
        family_details = dict(vote.get("families") or {})
        high_low_quality = 0.0
        high_low_available = 0
        for family in ("high", "low"):
            details = dict(family_details.get(family) or {})
            if details.get("available"):
                high_low_available += 1
                high_low_quality += float(details.get("weight", 0.0) or 0.0)
        score_strength = abs(float(vote.get("score", 0.0) or 0.0))
        return high_low_quality * score_strength, high_low_quality, high_low_available

    if bool(cfg.get("select_best_direction_timeframe", True)):
        direction_vote = max(candidate_votes, key=_direction_rank, default={})
    else:
        direction_vote = next(
            (vote for vote in candidate_votes if vote.get("timeframe") == configured_direction_timeframe),
            {},
        )
    direction_timeframe = str(direction_vote.get("timeframe") or configured_direction_timeframe)
    min_direction = abs(float(cfg.get("min_direction_score", 0.12) or 0.12))
    if direction_vote["available_families"] < 2 or abs(float(direction_vote["score"])) < min_direction:
        return {
            "enabled": True,
            "decision": "hold",
            "reason": "joint_ohlc_direction_not_strong_enough",
            "direction_vote": direction_vote,
        }

    decision = str(direction_vote["direction"])
    min_confirmation = abs(float(cfg.get("min_confirmation_score", 0.08) or 0.08))
    confirmation_votes = []
    confirmations = 0
    oppositions = 0
    for timeframe in [str(value) for value in (cfg.get("confirmation_timeframes") or [])]:
        if timeframe == direction_timeframe:
            continue
        vote = _timeframe_score(signals, timeframe, family_weights)
        score = float(vote["score"])
        aligned = vote["available_families"] >= 2 and abs(score) >= min_confirmation and vote["direction"] == decision
        opposed = vote["available_families"] >= 2 and abs(score) >= min_confirmation and vote["direction"] not in {decision, "hold"}
        vote["aligned"] = bool(aligned)
        vote["opposed"] = bool(opposed)
        confirmation_votes.append(vote)
        confirmations += int(aligned)
        oppositions += int(opposed)

    required = max(int(cfg.get("required_confirmations", 2) or 2), 0)
    if confirmations < required:
        return {
            "enabled": True,
            "decision": "hold",
            "reason": "short_timeframe_confirmation_failed",
            "direction_vote": direction_vote,
            "confirmation_votes": confirmation_votes,
            "confirmations": confirmations,
            "required_confirmations": required,
            "oppositions": oppositions,
        }

    projected = _projected_range(signals, direction_timeframe)
    if bool(cfg.get("require_projected_range", True)) and not projected.get("available"):
        return {
            "enabled": True,
            "decision": "hold",
            "reason": "qualified_high_low_range_unavailable",
            "direction_vote": direction_vote,
            "confirmation_votes": confirmation_votes,
            "confirmations": confirmations,
            "required_confirmations": required,
            "oppositions": oppositions,
            "projected_range": projected,
            "selected_direction_timeframe": direction_timeframe,
            "direction_candidates": candidate_votes,
        }
    entry = _finite_float(market_price)
    if projected.get("available"):
        low = float(projected["projected_low"])
        high = float(projected["projected_high"])
        fraction = min(max(float(cfg.get("entry_range_fraction", 0.20) or 0.20), 0.0), 0.5)
        entry = low + (high - low) * fraction if decision == "buy" else high - (high - low) * fraction
        reference = _finite_float(market_price)
        if reference and reference > 0.0:
            max_offset = reference * max(float(cfg.get("max_entry_offset_pct", 0.40) or 0.40), 0.0) / 100.0
            entry = min(max(float(entry), reference - max_offset), reference + max_offset)

    confidence_values = []
    for item in signals.values():
        quality = item.get("quality")
        qualified = not isinstance(quality, dict) or bool(quality.get("qualified", True))
        if qualified and not item.get("error") and int(item.get("signal", 0) or 0) != 0:
            confidence_values.append(float(item.get("confidence", 0.5) or 0.5))
    return {
        "enabled": True,
        "decision": decision,
        "reason": "joint_ohlc_and_short_timeframe_confirmation_passed",
        "score": float(direction_vote["score"]),
        "confidence": float(np.mean(confidence_values)) if confidence_values else 0.5,
        "entry": float(entry) if entry is not None else None,
        "direction_vote": direction_vote,
        "confirmation_votes": confirmation_votes,
        "confirmations": confirmations,
        "required_confirmations": required,
        "oppositions": oppositions,
        "projected_range": projected,
        "selected_direction_timeframe": direction_timeframe,
        "direction_candidates": candidate_votes,
    }


def calculate_atr(frame: pd.DataFrame, period: int = 14) -> Optional[float]:
    if frame is None or frame.empty or not {"HIGH", "LOW", "CLOSE"}.issubset(frame.columns):
        return None
    high = pd.to_numeric(frame["HIGH"], errors="coerce")
    low = pd.to_numeric(frame["LOW"], errors="coerce")
    close = pd.to_numeric(frame["CLOSE"], errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    values = true_range.dropna().tail(max(int(period), 1))
    if values.empty:
        return None
    atr = float(values.mean())
    return atr if math.isfinite(atr) and atr > 0.0 else None


def apply_volatility_protection(
    plan: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    frame: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Replace fixed percentage SL/TP with bounded ATR/realized-volatility levels."""
    out = dict(plan or {})
    cfg = dict((signal_policy_config(trading_cfg).get("volatility") or {}))
    if not bool(cfg.get("enabled", False)) or str(out.get("decision")) not in {"buy", "sell"}:
        return out
    entry = _finite_float(out.get("entry"))
    if entry is None or entry <= 0.0:
        return out

    period = max(int(cfg.get("atr_period", 14) or 14), 2)
    atr = calculate_atr(frame, period=period) if frame is not None else None
    realized = None
    if frame is not None and not frame.empty and "CLOSE" in frame.columns:
        returns = pd.to_numeric(frame["CLOSE"], errors="coerce").pct_change().dropna().tail(period)
        if not returns.empty:
            realized = float(returns.std(ddof=0) * entry)

    min_stop = entry * max(float(cfg.get("min_stop_pct", 0.10) or 0.10), 0.0) / 100.0
    max_stop = entry * max(float(cfg.get("max_stop_pct", 0.45) or 0.45), 0.01) / 100.0
    candidates = [min_stop]
    if atr is not None:
        candidates.append(atr * max(float(cfg.get("stop_atr_multiplier", 1.4) or 1.4), 0.1))
    if realized is not None and math.isfinite(realized):
        candidates.append(realized * max(float(cfg.get("realized_vol_multiplier", 2.0) or 2.0), 0.1))
    stop_distance = min(max(candidates), max_stop)
    reward_risk = max(float(cfg.get("reward_risk_ratio", 1.6) or 1.6), 0.1)
    target_distance = stop_distance * reward_risk
    if atr is not None:
        target_distance = max(target_distance, atr * max(float(cfg.get("target_atr_multiplier", 2.0) or 2.0), 0.1))
    max_target = entry * max(float(cfg.get("max_target_pct", 0.90) or 0.90), 0.01) / 100.0
    target_distance = min(target_distance, max_target)

    if out["decision"] == "buy":
        out["stop_loss"] = round(entry - stop_distance, 6)
        out["take_profit"] = round(entry + target_distance, 6)
    else:
        out["stop_loss"] = round(entry + stop_distance, 6)
        out["take_profit"] = round(entry - target_distance, 6)
    out["volatility_protection"] = {
        "enabled": True,
        "timeframe": str(cfg.get("timeframe") or "10m"),
        "atr": atr,
        "realized_volatility_price": realized,
        "stop_distance": float(stop_distance),
        "target_distance": float(target_distance),
        "reward_risk_ratio": float(target_distance / stop_distance) if stop_distance > 0.0 else None,
    }
    return out


def all_training_cutoffs_precede(manifest: Iterable[Dict[str, Any]], period_start: Any) -> bool:
    """Require explicit training cutoffs; file mtimes are not training lineage."""
    start = pd.Timestamp(period_start)
    checks = []
    for item in manifest:
        if item.get("load_error"):
            checks.append(False)
            continue
        raw = item.get("training_data_last_index")
        if not raw:
            checks.append(False)
            continue
        try:
            value = pd.Timestamp(raw)
            if value.tzinfo is not None:
                value = value.tz_convert("UTC").tz_localize(None)
            checks.append(value < start)
        except Exception:
            checks.append(False)
    return bool(checks) and all(checks)
