"""
Interpretability Utilities Module

Provides SHAP, permutation importance, and integrated gradients helpers
for selected models. Designed to be lightweight and resilient: if a
particular method or library is unavailable, it will silently skip that
method for the affected model and log a warning.
"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Ensure non-interactive backend
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from .sequence_utils import prepare_sequences_cached


_logger = logging.getLogger(__name__)


def add_interpretability(
    models_data: Dict[str, Dict[str, Any]],
    df,
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Attach interpretability information to models_data in-place.

    For each model, this may add a top-level ``interpretability`` key with
    sub-entries for:

    - ``permutation``: permutation-based feature importance (ULR, SVR)
    - ``integrated_gradients``: attributions for N-BEATS (PyTorch)
    - ``shap``: SHAP-based feature importance for XGBoost

    Any failures are logged and do not stop the main pipeline.
    """

    log = logger if logger is not None else _logger

    for model_name, model_data in models_data.items():
        if not isinstance(model_data, dict):
            continue
        if "model" not in model_data or model_data.get("error"):
            continue

        try:
            if model_name in {"ulr", "svr"}:
                perm_res = _sequence_permutation_importance(model_name, model_data, df, config, log)
                if perm_res:
                    model_data.setdefault("interpretability", {})["permutation"] = perm_res

            if model_name == "nbeats":
                ig_res = _nbeats_integrated_gradients(model_data, df, config, log)
                if ig_res:
                    model_data.setdefault("interpretability", {})["integrated_gradients"] = ig_res

            if model_name == "xgboost":
                shap_res = _xgboost_shap_importance(model_data, df, config, log)
                if shap_res:
                    model_data.setdefault("interpretability", {})["shap"] = shap_res
        except Exception as e:  # pragma: no cover - defensive
            log.warning(f"Interpretability computation failed for {model_name}: {e}")


def _get_sequence_parameters(model_data: Dict[str, Any], config: Dict[str, Any]):
    params = model_data.get("parameters", {}) or {}
    input_features = params.get("input_features", config.get("input_features", []))
    target_features = params.get("target_features", config.get("target_features", []))
    n_steps = params.get("n_steps", config.get("n_steps"))
    m_steps = params.get("m_steps", config.get("m_steps"))
    split_ratio = params.get("split_ratio", config.get("split_ratio", 0.8))
    return input_features, target_features, n_steps, m_steps, split_ratio


def _sequence_permutation_importance(
    model_name: str,
    model_data: Dict[str, Any],
    df,
    config: Dict[str, Any],
    logger: logging.Logger,
    max_samples: int = 100,
) -> Optional[Dict[str, Any]]:
    """Permutation importance for sequence models that use flattened inputs.

    Currently applied to ``ulr`` and ``svr`` models, which are trained via
    multiVrecurrent_LR / multiVrecurrent_SVR with StandardScaler.
    """

    scalers = model_data.get("scalers") or {}
    scaler_X = scalers.get("X")
    scaler_y = scalers.get("y")
    if scaler_X is None or scaler_y is None:
        return None

    input_features, target_features, n_steps, m_steps, _ = _get_sequence_parameters(model_data, config)
    if not input_features or not target_features or n_steps is None or m_steps is None:
        return None

    try:
        X_seq, y_seq = prepare_sequences_cached(df, input_features, target_features, n_steps, m_steps)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Permutation importance skipped for {model_name}: sequence prep failed ({e})")
        return None

    if X_seq.size == 0:
        return None

    # Restrict to last samples for efficiency
    n_samples = min(max_samples, X_seq.shape[0])
    X_seq = X_seq[-n_samples:]
    y_seq = y_seq[-n_samples:]

    # Match training layout: scaler_X was fit on 2D arrays of shape
    # (-1, n_base_features), not on flattened windows.
    n_base_feats = len(input_features)
    X_flat = X_seq.reshape(-1, n_base_feats)
    X_scaled_flat = scaler_X.transform(X_flat)
    # Reshape back to per-window layout and then flatten per sample for model
    X_scaled_windows = X_scaled_flat.reshape(n_samples, -1)

    # True values for primary target: first timestep of the primary target feature
    main_target = "y_diff" if "y_diff" in target_features else target_features[0]
    main_idx = target_features.index(main_target)
    y_true_main = y_seq[:, 0, main_idx].astype(float)

    model = model_data.get("model")
    if model is None:
        return None

    # Helper to get main-target predictions in original scale
    def _predict_main(X_scaled_arr: np.ndarray) -> np.ndarray:
        y_pred_scaled = model.predict(X_scaled_arr)
        y_pred_2d = y_pred_scaled.reshape(-1, len(target_features))
        y_pred_inv = scaler_y.inverse_transform(y_pred_2d)
        y_pred_3d = y_pred_inv.reshape(len(X_scaled_arr), m_steps, len(target_features))
        return y_pred_3d[:, 0, main_idx].astype(float)

    try:
        y_pred_main = _predict_main(X_scaled_windows)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Permutation importance skipped for {model_name}: prediction failed ({e})")
        return None

    baseline_mse = mean_squared_error(y_true_main, y_pred_main)

    n_features_total = X_scaled_windows.shape[1]
    raw_importances = np.zeros(n_features_total, dtype=float)

    rng = np.random.default_rng(42)

    for j in range(n_features_total):
        X_perm = X_scaled_windows.copy()
        rng.shuffle(X_perm[:, j])
        y_perm_main = _predict_main(X_perm)
        mse_perm = mean_squared_error(y_true_main, y_perm_main)
        raw_importances[j] = mse_perm - baseline_mse

    # Aggregate absolute importances per base feature (summing over timesteps)
    n_base_feats = len(input_features)
    agg: Dict[str, float] = {}
    for j, delta in enumerate(raw_importances):
        feat_idx = j % n_base_feats
        feat_name = input_features[feat_idx]
        agg[feat_name] = agg.get(feat_name, 0.0) + float(abs(delta))

    if not agg:
        return None

    top_items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:10]

    # Create a bar plot for the top features
    try:
        features = [k for k, _ in top_items]
        values = [v for _, v in top_items]
        plt.figure(figsize=(8, 4))
        plt.barh(features, values)
        plt.xlabel("Permutation Importance (ΔMSE)")
        plt.title(f"{model_name.upper()} - Permutation Importance (Top Features)")
        plt.gca().invert_yaxis()
        with tempfile.NamedTemporaryFile(prefix=f"interp_perm_{model_name}_", suffix=".png", delete=False) as tmp:
            fig_path = tmp.name
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        model_data.setdefault("figures", []).append(fig_path)
    except Exception as e:  # pragma: no cover - plotting is best-effort
        logger.warning(f"Could not create permutation importance plot for {model_name}: {e}")

    return {
        "top_features": [
            {"feature": feat, "importance": float(val)} for feat, val in top_items
        ],
        "figure_path": fig_path if 'fig_path' in locals() else None
    }


def _nbeats_integrated_gradients(
    model_data: Dict[str, Any],
    df,
    config: Dict[str, Any],
    logger: logging.Logger,
    steps: int = 50,
) -> Optional[Dict[str, Any]]:
    """Integrated gradients for N-BEATS models.

    Uses a single representative window (last available) and a zero baseline
    to attribute the flattened input dimensions, then aggregates per
    original feature across timesteps.
    """

    try:
        import torch
    except ImportError:  # pragma: no cover - environment without torch
        logger.warning("Integrated gradients for N-BEATS skipped: torch not available")
        return None

    scalers = model_data.get("scalers") or {}
    scaler_X = scalers.get("X")
    if scaler_X is None:
        return None

    params = model_data.get("parameters", {}) or {}
    input_features = params.get("input_features", config.get("input_features", []))
    target_features = params.get("target_features", config.get("target_features", []))
    n_steps = params.get("n_steps", config.get("n_steps"))
    m_steps = params.get("m_steps", config.get("m_steps"))
    if not input_features or n_steps is None or m_steps is None:
        return None

    # Recreate N-BEATS training sequences
    from models.multivariate_models import prepare_sequences_nbeats  # local import to avoid cycles

    try:
        df_clean = df.dropna().reset_index(drop=True)
        X_seq, _ = prepare_sequences_nbeats(df_clean, input_features, target_features, n_steps, m_steps)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Integrated gradients skipped for N-BEATS: sequence prep failed ({e})")
        return None

    if X_seq.size == 0:
        return None

    # Use the last window as representative input
    sample = X_seq[-1:]
    X_flat = sample.reshape(sample.shape[0], -1)
    X_scaled = scaler_X.transform(X_flat)

    model = model_data.get("model")
    if model is None:
        return None

    device_str = model_data.get("device", "cpu")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    input_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
    baseline = torch.zeros_like(input_tensor)

    ig_attr = _integrated_gradients_tensor(model, input_tensor, baseline, steps)
    if ig_attr is None:
        return None

    # Aggregate attributions per base feature across timesteps
    n_base_feats = len(input_features)
    attributions = ig_attr.reshape(-1)  # shape: input_size
    agg: Dict[str, float] = {}
    for j, val in enumerate(attributions):
        feat_idx = j % n_base_feats
        feat_name = input_features[feat_idx]
        agg[feat_name] = agg.get(feat_name, 0.0) + float(abs(val))

    if not agg:
        return None

    top_items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)[:10]

    # Plot attributions
    try:
        feats = [k for k, _ in top_items]
        vals = [v for _, v in top_items]
        plt.figure(figsize=(8, 4))
        plt.barh(feats, vals)
        plt.xlabel("Integrated Gradients |attribution|")
        plt.title("N-BEATS - Integrated Gradients (Top Features)")
        plt.gca().invert_yaxis()
        with tempfile.NamedTemporaryFile(prefix="interp_ig_nbeats_", suffix=".png", delete=False) as tmp:
            fig_path = tmp.name
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close()
        model_data.setdefault("figures", []).append(fig_path)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not create IG plot for N-BEATS: {e}")

    return {
        "top_features": [
            {"feature": feat, "attribution": float(val)} for feat, val in top_items
        ],
        "figure_path": fig_path if 'fig_path' in locals() else None
    }


def _integrated_gradients_tensor(model, input_tensor, baseline, steps: int = 50):
    """Basic integrated gradients implementation for a PyTorch model.

    Returns a 1D numpy array of attributions with the same size as the
    flattened input sample.
    """

    import torch

    try:
        scaled_inputs = [
            baseline + (float(i) / steps) * (input_tensor - baseline)
            for i in range(1, steps + 1)
        ]

        grads_sum = torch.zeros_like(input_tensor)

        for scaled in scaled_inputs:
            scaled = scaled.clone().detach().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            output = model(scaled)
            # Aggregate all outputs to a scalar
            total = output.sum()
            total.backward()
            grads_sum += scaled.grad.detach()

        avg_grads = grads_sum / float(steps)
        attributions = (input_tensor - baseline) * avg_grads
        return attributions.detach().cpu().numpy().reshape(-1)
    except Exception:
        return None


def _xgboost_shap_importance(
    model_data: Dict[str, Any],
    df,
    config: Dict[str, Any],
    logger: logging.Logger,
    max_samples: int = 200,
) -> Optional[Dict[str, Any]]:
    """SHAP-based feature importance for XGBoost models.

    Uses TreeExplainer when available. Falls back silently if ``shap``
    is not installed or if explanation fails.
    """

    try:
        import shap  # type: ignore
    except ImportError:  # pragma: no cover - shap not installed
        logger.warning("SHAP importance for XGBoost skipped: shap not available")
        return None

    model = model_data.get("model")
    if model is None:
        return None

    from pandas import to_datetime

    params = model_data.get("parameters", {}) or {}
    target_features = params.get("target_features", config.get("target_features", []))
    target_col = target_features[0] if target_features else config.get("target_col")
    if target_col is None:
        return None

    date_col = config.get("date_col", "DATE")

    try:
        df_xgb = df.reset_index().copy()
        df_xgb[date_col] = to_datetime(df_xgb[date_col])
        df_xgb = df_xgb.set_index(date_col).sort_index()

        # Recreate engineered features similar to training
        df_xgb["year"] = df_xgb.index.year
        df_xgb["month"] = df_xgb.index.month
        df_xgb["day"] = df_xgb.index.day
        df_xgb["dayofweek"] = df_xgb.index.dayofweek
        df_xgb["weekofyear"] = df_xgb.index.isocalendar().week
        df_xgb["quarter"] = df_xgb.index.quarter

        lags = config.get("lags", [2, 7, 30, 60])
        for lag in lags:
            df_xgb[f"{target_col}_lag{lag}"] = df_xgb[target_col].shift(lag)

        rolling_windows = config.get("rolling_windows", [2, 7, 30, 60])
        for window in rolling_windows:
            df_xgb[f"{target_col}_roll{window}"] = df_xgb[target_col].rolling(window).mean()

        df_xgb = df_xgb.dropna()
        exclude_cols = config.get("exclude_cols", [])
        feature_cols = [
            col
            for col in df_xgb.columns
            if col
            not in exclude_cols
            + [target_col]
            + [config.get("target_col", target_col)]
        ]
        feature_cols = [c for c in feature_cols if c in df_xgb.columns]
        if not feature_cols:
            return None

        X = df_xgb[feature_cols]
        if len(X) == 0:
            return None

        X_sample = X.sample(n=min(max_samples, len(X)), random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # shap_values can be list (multiclass); for regression it's ndarray
        if isinstance(shap_values, list):
            shap_arr = np.mean([np.abs(s) for s in shap_values], axis=0)
        else:
            shap_arr = np.abs(shap_values)

        mean_abs = shap_arr.mean(axis=0)
        if mean_abs.ndim != 1 or mean_abs.shape[0] != len(feature_cols):
            return None

        top_idx = np.argsort(mean_abs)[::-1][:10]
        top_feats = [(feature_cols[i], float(mean_abs[i])) for i in top_idx]

        # Plot SHAP importances
        try:
            feats = [k for k, _ in top_feats]
            vals = [v for _, v in top_feats]
            plt.figure(figsize=(8, 4))
            plt.barh(feats, vals)
            plt.xlabel("Mean |SHAP value|")
            plt.title("XGBoost - SHAP Feature Importance (Top Features)")
            plt.gca().invert_yaxis()
            with tempfile.NamedTemporaryFile(prefix="interp_shap_xgboost_", suffix=".png", delete=False) as tmp:
                fig_path = tmp.name
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches="tight")
            plt.close()
            model_data.setdefault("figures", []).append(fig_path)
        except Exception as e:  # pragma: no cover
            logger.warning(f"Could not create SHAP plot for XGBoost: {e}")

        return {
            "top_features": [
                {"feature": feat, "importance": float(val)} for feat, val in top_feats
            ],
            "figure_path": fig_path if 'fig_path' in locals() else None
        }

    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"SHAP importance for XGBoost failed: {e}")
        return None
