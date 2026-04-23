"""
Rupture Forecaster Module

Provides a classifier-based market-structure rupture forecaster.
It predicts:
1) whether the next step is a rupture event
2) the likely rupture direction (up/down) when rupture risk is high
"""

from __future__ import annotations

import logging
import tempfile
from typing import Dict, Any, List
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


_logger = logging.getLogger(__name__)


def _export_rupture_plot_data(config: Dict[str, Any], plot_name: str, payload: Dict[str, Any]) -> None:
    cfg = config.get('plot_data_export', {}) or {}
    if not bool(cfg.get('enabled', True)):
        return
    base_dir = cfg.get('directory', 'output/plot_data')
    out_dir = os.path.join(base_dir, 'rupture')
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(out_dir, f"{plot_name}_{ts}.json"), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)


def _window_stats(window: np.ndarray) -> Dict[str, float]:
    """Extract compact statistical features from a window (n_steps, n_features)."""
    feats: Dict[str, float] = {}
    if window.ndim == 1:
        window = window.reshape(-1, 1)

    for i in range(window.shape[1]):
        col = window[:, i]
        feats[f"mean_dim_{i}"] = float(np.mean(col))
        feats[f"std_dim_{i}"] = float(np.std(col))
        feats[f"min_dim_{i}"] = float(np.min(col))
        feats[f"max_dim_{i}"] = float(np.max(col))
        feats[f"median_dim_{i}"] = float(np.median(col))
        feats[f"range_dim_{i}"] = float(np.max(col) - np.min(col))

    feats["overall_mean"] = float(np.mean(window))
    feats["overall_std"] = float(np.std(window))
    feats["overall_min"] = float(np.min(window))
    feats["overall_max"] = float(np.max(window))
    feats["overall_median"] = float(np.median(window))
    feats["overall_range"] = float(np.max(window) - np.min(window))
    return feats


def _save_cm_plot(cm: np.ndarray, labels: List[str], title: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def forecast_market_rupture(df: pd.DataFrame, config: Dict[str, Any], logger=None) -> Dict[str, Any]:
    """Train and score rupture classifiers using rolling windows.

    Returns a dict with training metrics and next-step rupture prediction.
    """
    log = logger or _logger
    out: Dict[str, Any] = {
        "enabled": False,
        "metrics": {},
        "next_step": {},
        "figures": [],
    }

    rcfg = config.get("rupture_forecast", {})
    if not rcfg.get("enabled", True):
        return out

    input_features = rcfg.get("input_features", config.get("input_features", []))
    n_steps = int(rcfg.get("n_steps", config.get("n_steps", 30)))
    split_ratio = float(rcfg.get("split_ratio", 0.8))

    # Prefer y_diff if available as rupture signal; fallback to target_col diff
    if "y_diff" in df.columns:
        signal = df["y_diff"].astype(float).values
    else:
        target_col = config.get("target_col")
        if target_col not in df.columns:
            out["error"] = "No rupture signal available (missing y_diff and target_col)"
            return out
        signal = np.diff(df[target_col].astype(float).values, prepend=df[target_col].iloc[0])

    if len(df) < n_steps + 20:
        out["error"] = "Not enough samples for rupture forecasting"
        return out

    # Rupture threshold from absolute move quantile
    q = float(rcfg.get("quantile", 0.9))
    thr = float(np.quantile(np.abs(signal), q))

    X_rows: List[Dict[str, float]] = []
    y_bin: List[int] = []
    y_dir: List[int] = []  # -1 down rupture, 0 no rupture, 1 up rupture

    vals = df[input_features].values
    for i in range(len(df) - n_steps - 1):
        w = vals[i:i + n_steps]
        next_move = float(signal[i + n_steps])
        is_rupture = int(abs(next_move) >= thr)
        direction = 0
        if is_rupture:
            direction = 1 if next_move > 0 else -1

        X_rows.append(_window_stats(w))
        y_bin.append(is_rupture)
        y_dir.append(direction)

    X = pd.DataFrame(X_rows)
    y_bin_arr = np.array(y_bin)
    y_dir_arr = np.array(y_dir)

    split = int(len(X) * split_ratio)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    yb_train, yb_test = y_bin_arr[:split], y_bin_arr[split:]
    yd_train, yd_test = y_dir_arr[:split], y_dir_arr[split:]

    if len(np.unique(yb_train)) < 2 or len(np.unique(yb_test)) < 2:
        out["error"] = "Insufficient class diversity for rupture classification"
        return out

    clf_bin = RandomForestClassifier(
        n_estimators=int(rcfg.get("n_estimators", 200)),
        max_depth=rcfg.get("max_depth", 10),
        random_state=42,
        class_weight="balanced",
    )
    clf_bin.fit(X_train, yb_train)
    yb_pred = clf_bin.predict(X_test)

    # Direction model: train only on rupture samples
    rup_train_idx = np.where(yb_train == 1)[0]
    rup_test_idx = np.where(yb_test == 1)[0]
    clf_dir = None
    yd_pred_full = np.zeros_like(yd_test)
    direction_metrics = {
        "accuracy": None,
        "precision_macro": None,
        "recall_macro": None,
        "f1_macro": None,
    }

    if len(rup_train_idx) > 20 and len(rup_test_idx) > 5:
        clf_dir = RandomForestClassifier(
            n_estimators=int(rcfg.get("n_estimators", 200)),
            max_depth=rcfg.get("max_depth", 10),
            random_state=42,
            class_weight="balanced",
        )
        clf_dir.fit(X_train.iloc[rup_train_idx], yd_train[rup_train_idx])
        yd_pred_rup = clf_dir.predict(X_test.iloc[rup_test_idx])
        yd_pred_full[rup_test_idx] = yd_pred_rup

        direction_metrics = {
            "accuracy": float(accuracy_score(yd_test[rup_test_idx], yd_pred_rup)),
            "precision_macro": float(precision_score(yd_test[rup_test_idx], yd_pred_rup, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(yd_test[rup_test_idx], yd_pred_rup, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(yd_test[rup_test_idx], yd_pred_rup, average="macro", zero_division=0)),
        }

    # Binary metrics
    bin_metrics = {
        "threshold_abs_move": thr,
        "accuracy": float(accuracy_score(yb_test, yb_pred)),
        "precision": float(precision_score(yb_test, yb_pred, zero_division=0)),
        "recall": float(recall_score(yb_test, yb_pred, zero_division=0)),
        "f1": float(f1_score(yb_test, yb_pred, zero_division=0)),
    }

    # Next-step inference from last window
    last_window = vals[-n_steps:]
    X_last = pd.DataFrame([_window_stats(last_window)]).reindex(columns=X.columns, fill_value=0.0)
    p_bin = clf_bin.predict_proba(X_last)[0]
    pred_bin = int(clf_bin.predict(X_last)[0])

    next_dir = 0
    next_dir_probs = {}
    if clf_dir is not None and pred_bin == 1:
        p_dir = clf_dir.predict_proba(X_last)[0]
        classes = list(clf_dir.classes_)
        next_dir = int(clf_dir.predict(X_last)[0])
        next_dir_probs = {str(int(c)): float(p) for c, p in zip(classes, p_dir)}

    # Figures
    try:
        cm_bin = confusion_matrix(yb_test, yb_pred)
        path_bin = _save_cm_plot(cm_bin, ["NoRupture", "Rupture"], "Rupture Detection Confusion Matrix")
        out["figures"].append(path_bin)
        _export_rupture_plot_data(
            config,
            'rupture_detection_confusion_matrix',
            {
                'labels': ["NoRupture", "Rupture"],
                'matrix': cm_bin.tolist(),
                'metrics': bin_metrics
            }
        )

        if clf_dir is not None and len(rup_test_idx) > 0:
            cm_dir = confusion_matrix(yd_test[rup_test_idx], yd_pred_full[rup_test_idx], labels=[-1, 1])
            path_dir = _save_cm_plot(cm_dir, ["Down", "Up"], "Rupture Direction Confusion Matrix")
            out["figures"].append(path_dir)
            _export_rupture_plot_data(
                config,
                'rupture_direction_confusion_matrix',
                {
                    'labels': ["Down", "Up"],
                    'matrix': cm_dir.tolist(),
                    'metrics': direction_metrics
                }
            )
    except Exception as e:
        log.warning(f"Could not generate rupture confusion matrix plots: {e}")

    out["enabled"] = True
    out["metrics"] = {
        "binary": bin_metrics,
        "direction": direction_metrics,
    }
    out["next_step"] = {
        "is_rupture": bool(pred_bin),
        "rupture_probability": float(p_bin[1] if len(p_bin) > 1 else p_bin[0]),
        "direction": int(next_dir),
        "direction_probabilities": next_dir_probs,
    }

    return out
