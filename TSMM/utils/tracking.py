"""
Tracking Module

Provides utilities for tracking experiment results.
"""

import json
import time
import os
from pathlib import Path
import numpy as np


def _jsonify(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_run_summary(
    config_path: str,
    metrics: dict,
    summary_dir: str = "experiments",
    suffix: str = None
):
    """
    Save one JSON file describing the run.
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file used for this run
    metrics : dict
        Dictionary of metrics from the run
    summary_dir : str
        Directory to save the summary
    suffix : str, optional
        Suffix to add to the filename
    """
    summary_dir = Path(summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(config_path).stem
    if suffix:
        stem = f"{stem}__{suffix}"
    summary_file = summary_dir / f"{stem}__summary.json"

    # Heuristic status: SUCCESS only if at least one model has
    # non-empty numeric metrics; otherwise mark as NO_METRICS.
    status = "NO_METRICS"
    try:
        for model_name, model_metrics in (metrics or {}).items():
            if isinstance(model_metrics, dict):
                # Look for common regression keys
                if any(k in model_metrics for k in ("MAE", "RMSE", "R2", "MAPE")):
                    status = "SUCCESS"
                    break
    except Exception:
        # Fallback: keep default status if inspection fails
        pass

    payload = {
        "config_path": str(Path(config_path).resolve()),
        "status": status,
        "metric": _jsonify(metrics),
        "wall_time": time.time()
    }
    
    with open(summary_file, "w") as fp:
        json.dump(payload, fp, indent=2)
    
    return str(summary_file)
