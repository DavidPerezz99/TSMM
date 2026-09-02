"""Model-quality weighting and conservative hybrid trade admission gates."""

from __future__ import annotations

from typing import Any, Dict, Optional


def finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def model_quality_weight(
    static_r2: Any,
    refreshed_r2: Any,
    minimum_r2: float = 0.0,
    legacy_static_discount: float = 0.35,
) -> Dict[str, Any]:
    """Prefer refreshed R2 and discount legacy scores without reliable validation."""
    refreshed = finite_float(refreshed_r2)
    static = finite_float(static_r2)
    score = refreshed if refreshed is not None else static
    source = "refreshed_r2" if refreshed is not None else "selected_r2"
    if score is None or score < float(minimum_r2):
        return {"qualified": False, "score": score, "source": source, "weight": 0.0}
    weight = min(max((score - float(minimum_r2)) / max(1.0 - float(minimum_r2), 1e-9), 0.0), 1.0)
    reliability = 1.0
    if refreshed is None:
        reliability = min(max(float(legacy_static_discount), 0.0), 1.0)
        weight *= reliability
    return {
        "qualified": weight > 0.0,
        "score": score,
        "source": source,
        "weight": weight,
        "reliability_factor": reliability,
    }


def apply_hybrid_trade_gate(plan: Dict[str, Any], trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Combine forecast quality, regime sanity, costs, and expected value; abstain on failure."""
    out = dict(plan or {})
    gate_cfg = dict(trading_cfg.get("hybrid_strategy") or {})
    out.setdefault("risk_notes", [])
    if not bool(gate_cfg.get("enabled", True)):
        out["hybrid_gate"] = {"enabled": False, "passed": True}
        return out

    decision = str(out.get("decision") or "hold").lower()
    reasons = []
    enrichment = dict(out.get("enrichment") or {})
    min_models = max(int(gate_cfg.get("min_qualified_models", 2) or 2), 1)
    qualified = int(enrichment.get("n_qualified_signals", 0) or 0)
    if qualified < min_models:
        reasons.append(f"qualified_models:{qualified}<{min_models}")

    min_consensus = float(gate_cfg.get("min_abs_consensus", 0.15) or 0.15)
    consensus = abs(float(enrichment.get("consensus_score", 0.0) or 0.0))
    if consensus < min_consensus:
        reasons.append(f"consensus:{consensus:.4f}<{min_consensus:.4f}")

    entry = finite_float(out.get("entry")) or 0.0
    stop = finite_float(out.get("stop_loss")) or 0.0
    take = finite_float(out.get("take_profit")) or 0.0
    probability = finite_float(out.get("success_probability"))
    if probability is None:
        probability = finite_float(out.get("confidence")) or 0.5
    risk_distance = abs(entry - stop) if entry > 0 and stop > 0 else 0.0
    reward_distance = abs(take - entry) if entry > 0 and take > 0 else 0.0
    execution = dict(trading_cfg.get("execution") or {})
    round_trip_bps = 2.0 * (
        float(execution.get("spread_bps", 0.0) or 0.0)
        + float(execution.get("slippage_bps", 0.0) or 0.0)
    )
    cost_distance = entry * round_trip_bps / 10000.0
    expected_value = probability * reward_distance - (1.0 - probability) * risk_distance - cost_distance
    min_ev = float(gate_cfg.get("minimum_expected_value_price", 0.0) or 0.0)
    if risk_distance <= 0.0:
        reasons.append("missing_hard_stop")
    if expected_value <= min_ev:
        reasons.append(f"expected_value:{expected_value:.6f}<={min_ev:.6f}")

    features = dict(out.get("feature_forecasts_step1") or {})
    high = finite_float(features.get("HIGH"))
    low = finite_float(features.get("LOW"))
    if entry > 0 and high is not None and low is not None:
        range_pct = abs(high - low) / entry * 100.0
        max_range_pct = float(gate_cfg.get("max_forecast_range_pct", 3.0) or 3.0)
        if range_pct > max_range_pct:
            reasons.append(f"forecast_range_pct:{range_pct:.4f}>{max_range_pct:.4f}")
    else:
        range_pct = None

    passed = decision in {"buy", "sell"} and not reasons
    if decision in {"buy", "sell"} and not passed:
        out["decision"] = "hold"
        out["risk_notes"].append("Hybrid trade gate abstained: " + ", ".join(reasons))
    out["hybrid_gate"] = {
        "enabled": True,
        "passed": passed,
        "reasons": reasons,
        "qualified_models": qualified,
        "consensus_score_abs": consensus,
        "success_probability": probability,
        "risk_distance": risk_distance,
        "reward_distance": reward_distance,
        "estimated_cost_distance": cost_distance,
        "expected_value_price": expected_value,
        "forecast_range_pct": range_pct,
    }
    return out
