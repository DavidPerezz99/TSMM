"""Assemble the multi-stage evidence required by champion/challenger gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--walk-forward", required=True)
    parser.add_argument("--backtest-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    bundle = Path(args.bundle).resolve()
    evaluation = json.loads((bundle / "evaluation.json").read_text(encoding="utf-8"))
    walk = json.loads(Path(args.walk_forward).read_text(encoding="utf-8"))
    backtest = json.loads(Path(args.backtest_summary).read_text(encoding="utf-8"))
    model_eval = next(iter(evaluation.values()))
    overall = (backtest.get("summary") or backtest).get("overall") or {}
    directional_values = [
        float(fold["directional_accuracy"])
        for fold in (walk.get("folds") or []) if fold.get("directional_accuracy") is not None
    ]
    payload = {
        "bundle": str(bundle),
        "holdout_r2": (model_eval.get("metrics") or {}).get("R2"),
        "walk_forward_r2": walk.get("walk_forward_r2") or [],
        "directional_accuracy": sum(directional_values) / len(directional_values) if directional_values else None,
        "profit_factor": overall.get("profit_factor"),
        "expectancy": overall.get("expectancy_per_trade"),
        "max_drawdown_pct": overall.get("max_drawdown_pct"),
        "trades": overall.get("n_trades"),
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
