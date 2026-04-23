"""
Quick validator from source dataset to disruption overlays.

Usage:
    python scripts/validate_source_dataset.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import copy

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data_loader import load_data
from utils.momentum import compute_momentum_overlay
from utils.vol_target import compute_vol_target_overlay
from utils.regime import classify_market_regime
from utils.rupture_forecaster import forecast_market_rupture


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml", help="YAML config path")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_path = ROOT / args.config
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = ROOT / config["data_path"]
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = load_data(str(data_path), config["date_col"], config["target_col"], config)

    # Keep this validator fast and deterministic on larger datasets.
    max_rows = int(config.get("records", 2000))
    if len(df) > max_rows:
        df = df.tail(max_rows).copy()

    momentum = compute_momentum_overlay(df, config)
    vol_target = compute_vol_target_overlay(df, config, momentum)
    regime = classify_market_regime(df, config, momentum, vol_target)
    rupture_cfg = copy.deepcopy(config)
    rupture_cfg.setdefault("rupture_forecast", {})
    rupture_cfg["rupture_forecast"]["n_estimators"] = int(
        min(100, rupture_cfg["rupture_forecast"].get("n_estimators", 100))
    )
    rupture = forecast_market_rupture(df, rupture_cfg)

    summary = {
        "rows": len(df),
        "date_start": str(df.index.min()),
        "date_end": str(df.index.max()),
        "momentum": {
            "trend_state": momentum.get("trend_state"),
            "momentum_score": momentum.get("momentum_score"),
            "confidence_bucket": momentum.get("confidence_bucket"),
        },
        "vol_target": {
            "realized_vol": vol_target.get("realized_vol"),
            "target_vol": vol_target.get("target_vol"),
            "position_scale": vol_target.get("position_scale"),
            "recommended_exposure": vol_target.get("recommended_exposure"),
        },
        "regime": {
            "state": regime.get("state"),
            "confidence": regime.get("confidence"),
            "policy": regime.get("policy", {}),
        },
        "rupture": {
            "enabled": rupture.get("enabled"),
            "next_step": rupture.get("next_step", {}),
            "binary_metrics": (rupture.get("metrics", {}) or {}).get("binary", {}),
        },
    }

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
