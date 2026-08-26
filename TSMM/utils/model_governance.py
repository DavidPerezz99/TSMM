"""Champion/challenger admission, atomic registry updates, and rollback."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Optional


DEFAULT_POLICY = {
    "minimum_holdout_r2": 0.0,
    "minimum_median_walk_forward_r2": 0.0,
    "minimum_worst_fold_r2": -0.25,
    "minimum_directional_accuracy": 0.52,
    "minimum_profit_factor": 1.10,
    "minimum_expectancy": 0.0,
    "maximum_drawdown_pct": 15.0,
    "minimum_trades": 30,
    "minimum_r2_improvement": 0.01,
}


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def assess_challenger(
    candidate: Dict[str, Any], champion: Optional[Dict[str, Any]] = None,
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply multi-metric, fail-closed promotion gates."""
    rules = {**DEFAULT_POLICY, **(policy or {})}
    failures = []
    holdout_r2 = _number(candidate.get("holdout_r2"))
    fold_r2 = [_number(value) for value in (candidate.get("walk_forward_r2") or [])]
    fold_r2 = [value for value in fold_r2 if value is not None]
    directional = _number(candidate.get("directional_accuracy"))
    profit_factor = _number(candidate.get("profit_factor"))
    expectancy = _number(candidate.get("expectancy"))
    drawdown = _number(candidate.get("max_drawdown_pct"))
    trades = int(candidate.get("trades", 0) or 0)

    def require(name: str, value: Optional[float], predicate, expectation: str) -> None:
        if value is None:
            failures.append(f"missing_{name}")
        elif not predicate(value):
            failures.append(f"{name}:{value}:{expectation}")

    require("holdout_r2", holdout_r2, lambda x: x >= rules["minimum_holdout_r2"],
            f">={rules['minimum_holdout_r2']}")
    if not fold_r2:
        failures.append("missing_walk_forward_r2")
    else:
        if median(fold_r2) < rules["minimum_median_walk_forward_r2"]:
            failures.append("median_walk_forward_r2_below_gate")
        if min(fold_r2) < rules["minimum_worst_fold_r2"]:
            failures.append("worst_walk_forward_fold_below_gate")
    require("directional_accuracy", directional,
            lambda x: x >= rules["minimum_directional_accuracy"],
            f">={rules['minimum_directional_accuracy']}")
    require("profit_factor", profit_factor, lambda x: x >= rules["minimum_profit_factor"],
            f">={rules['minimum_profit_factor']}")
    require("expectancy", expectancy, lambda x: x > rules["minimum_expectancy"],
            f">{rules['minimum_expectancy']}")
    require("max_drawdown_pct", drawdown, lambda x: x <= rules["maximum_drawdown_pct"],
            f"<={rules['maximum_drawdown_pct']}")
    if trades < int(rules["minimum_trades"]):
        failures.append(f"trades:{trades}:>={int(rules['minimum_trades'])}")

    champion_r2 = _number((champion or {}).get("holdout_r2"))
    if champion_r2 is not None and holdout_r2 is not None:
        if holdout_r2 < champion_r2 + float(rules["minimum_r2_improvement"]):
            failures.append("does_not_beat_champion_r2")
    return {
        "approved": not failures,
        "failures": failures,
        "policy": rules,
        "candidate": candidate,
        "champion": champion,
        "derived": {
            "median_walk_forward_r2": median(fold_r2) if fold_r2 else None,
            "worst_walk_forward_r2": min(fold_r2) if fold_r2 else None,
        },
    }


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "endpoints": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def promote(registry_path: Path, endpoint: str, bundle: str, metrics: Dict[str, Any],
            assessment: Dict[str, Any]) -> Dict[str, Any]:
    if not assessment.get("approved"):
        raise ValueError("Challenger did not pass promotion gates")
    registry = load_registry(registry_path)
    record = registry.setdefault("endpoints", {}).setdefault(endpoint, {"history": []})
    current = record.get("champion")
    if current:
        record.setdefault("history", []).append(current)
    record["champion"] = {
        "bundle": str(Path(bundle).resolve()),
        "metrics": metrics,
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    record["last_assessment"] = assessment
    _atomic_json(registry_path, registry)
    return record["champion"]


def rollback(registry_path: Path, endpoint: str) -> Dict[str, Any]:
    registry = load_registry(registry_path)
    record = (registry.get("endpoints") or {}).get(endpoint)
    if not record or not record.get("history"):
        raise ValueError(f"No rollback generation is available for {endpoint}")
    current = record.get("champion")
    previous = record["history"].pop()
    if current:
        record.setdefault("rolled_back", []).append(current)
    record["champion"] = previous
    record["rolled_back_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(registry_path, registry)
    return previous
