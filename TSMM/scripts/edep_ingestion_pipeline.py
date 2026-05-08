import asyncio
import os
from pathlib import Path

import joblib
import numpy as np

try:
    import torch
except Exception:
    torch = None

__version__ = "1.0.0"


def _resolve_bundle_path() -> Path:
    model_name = str(os.getenv("MODEL_FILE", "")).strip()
    model_dir = Path("/app/model")
    if model_name:
        path = model_dir / model_name
        if path.exists():
            return path

    files = sorted(p for p in model_dir.glob("*") if p.is_file())
    if not files:
        raise FileNotFoundError("No model bundle file found under /app/model")
    return files[0]


def _load_bundle() -> dict:
    bundle_path = _resolve_bundle_path()
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError("Bundle must be a dict with model, artifacts, and spec")
    if "model" not in bundle:
        raise ValueError("Bundle is missing 'model'")
    if "artifacts" not in bundle:
        raise ValueError("Bundle is missing 'artifacts'")
    if "spec" not in bundle:
        raise ValueError("Bundle is missing 'spec'")
    return bundle


BUNDLE = _load_bundle()
SPEC = dict(BUNDLE["spec"])
MODEL = BUNDLE["model"]
ARTIFACTS = dict(BUNDLE["artifacts"])
SCALER_X = ARTIFACTS["scaler_X"]
SCALER_Y = ARTIFACTS["scaler_y"]


def _confidence_from_value(value: float) -> float:
    magnitude = abs(float(value))
    conf = 1.0 - np.exp(-magnitude)
    return float(np.clip(conf, 0.05, 0.95))


def _normalize_batch(data) -> np.ndarray:
    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        raise ValueError("eDep process_input expected a list of objects")

    payload_field = str(SPEC.get("payload_field", "window"))
    vector_field = str(SPEC.get("vector_field", "vector"))
    n_steps = int(SPEC["n_steps"])
    n_features = len(SPEC["input_features"])
    expected_flat = n_steps * n_features

    batch = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Each batch item must be an object")

        if payload_field in item:
            arr = np.asarray(item[payload_field], dtype=float)
            if arr.shape != (n_steps, n_features):
                raise ValueError(
                    f"Expected '{payload_field}' shape {(n_steps, n_features)}, got {tuple(arr.shape)}"
                )
            batch.append(arr.reshape(-1))
            continue

        if vector_field in item:
            arr = np.asarray(item[vector_field], dtype=float).reshape(-1)
            if arr.size != expected_flat:
                raise ValueError(
                    f"Expected '{vector_field}' length {expected_flat}, got {int(arr.size)}"
                )
            batch.append(arr)
            continue

        raise ValueError(
            f"Each item must include either '{payload_field}' or '{vector_field}'"
        )

    if not batch:
        raise ValueError("Received empty batch")

    return np.vstack(batch)


def _predict_scaled(X_scaled: np.ndarray) -> np.ndarray:
    model_type = str(SPEC.get("model_type", "")).lower()
    if model_type == "nbeats" and torch is not None and hasattr(MODEL, "eval"):
        MODEL.eval()
        with torch.no_grad():
            return MODEL(torch.as_tensor(X_scaled, dtype=torch.float32)).cpu().numpy()

    if hasattr(MODEL, "predict"):
        return MODEL.predict(X_scaled)

    raise ValueError("Loaded model does not expose a supported predict interface")


async def process_input(data):
    try:
        X_flat = _normalize_batch(data)
        X_scaled = SCALER_X.transform(X_flat)
        y_scaled = _predict_scaled(X_scaled)

        if y_scaled.ndim == 1:
            y_scaled = y_scaled.reshape(1, -1)

        y_inv = SCALER_Y.inverse_transform(y_scaled)
        n_targets = len(SPEC["target_features"])
        y_pred = y_inv.reshape(X_flat.shape[0], -1, n_targets)
        signal_feature = str(SPEC.get("signal_feature", "y_diff"))
        signal_idx = SPEC["target_features"].index(signal_feature)

        out = []
        for idx in range(X_flat.shape[0]):
            first_signal = float(y_pred[idx, 0, signal_idx])
            forecast_sign = float(np.sign(first_signal))
            out.append(
                {
                    "forecast_sign": forecast_sign,
                    "prediction": first_signal,
                    "confidence": _confidence_from_value(first_signal),
                    "signal": "buy" if forecast_sign > 0 else ("sell" if forecast_sign < 0 else "hold"),
                    "targets_step1": {
                        name: float(y_pred[idx, 0, j])
                        for j, name in enumerate(SPEC["target_features"])
                    },
                    "timeframe": SPEC.get("timeframe"),
                    "model_type": SPEC.get("model_type"),
                }
            )

        await asyncio.sleep(0)
        return out if len(out) > 1 else out[0]
    except Exception as exc:
        await asyncio.sleep(0)
        return {
            "error": str(exc),
            "forecast_sign": 0.0,
            "confidence": 0.5,
            "signal": "hold",
        }