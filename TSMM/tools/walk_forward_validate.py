"""Run genuine expanding-window retraining over an experiment configuration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.multivariate_models import train_multivariate_models
from models.univariate_models import train_univariate_models
from utils.data_loader import load_data_cached
from utils.evaluator import evaluate_models
from utils.logger import setup_logger
from utils.walk_forward_validation import expanding_window_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Leakage-safe expanding-window model validation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--test-rows", type=int, default=60)
    parser.add_argument("--gap-rows", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    logger = setup_logger(str((ROOT / "logs" / "walk_forward.log").resolve()))
    frame = load_data_cached(config["data_path"], config["date_col"], config["target_col"], config)
    if config.get("records"):
        frame = frame.tail(int(config["records"]))
    gap = args.gap_rows if args.gap_rows is not None else max(int(config.get("m_steps", 1) or 1), 1)
    minimum_train = max(int(config.get("n_steps", 1) or 1) * 3, 100)
    splits = expanding_window_splits(len(frame), args.folds, args.test_rows, gap, minimum_train)
    fold_results = []
    for split in splits:
        # The gap is deliberately excluded. The fold dataframe ends after the
        # test block and its test_size reserves that block as untouched holdout.
        train = frame.iloc[:split["train_end"]]
        test = frame.iloc[split["test_start"]:split["test_end"]]
        fold_frame = pd.concat([train, test], axis=0).copy()
        fold_cfg = dict(config)
        fold_cfg["test_size"] = int(args.test_rows)
        fold_cfg["records"] = int(len(fold_frame))
        if fold_cfg.get("problem_type") == "univariate":
            models = train_univariate_models(
                fold_frame, fold_cfg, logger, fold_cfg["input_features"], fold_cfg["target_features"],
                fold_cfg.get("exclude_cols", []), fold_cfg["n_steps"], fold_cfg["m_steps"], fold_cfg["split_ratio"],
            )
        else:
            models = train_multivariate_models(fold_frame, fold_cfg, logger)
        evaluation, _ = evaluate_models(models, fold_frame, fold_cfg)
        best_name, best_payload = max(
            evaluation.items(), key=lambda item: float(((item[1].get("metrics") or {}).get("R2", -1e99)))
        )
        metrics = dict(best_payload.get("metrics") or {})
        fold_results.append({"split": split, "model": best_name, "metrics": metrics,
                             "directional_accuracy": (best_payload.get("confusion_matrix") or {}).get("accuracy")})
    r2_values = [float(item["metrics"]["R2"]) for item in fold_results]
    output = {
        "config": str(config_path), "policy": "expanding_window_train_only_scaling_with_gap",
        "folds": fold_results, "walk_forward_r2": r2_values,
        "median_walk_forward_r2": float(np.median(r2_values)), "worst_walk_forward_r2": min(r2_values),
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
