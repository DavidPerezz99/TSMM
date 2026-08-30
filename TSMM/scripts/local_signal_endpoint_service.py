"""Local endpoint service for Agent B timeframe signals.

Routes:
- GET /health
- POST /predict/{timeframe}

This server is intentionally lightweight and only serves inference payloads
built by utils.investing_agent._build_endpoint_payloads.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import torch
    try:
        import torch._utils as _torch_utils  # type: ignore[attr-defined]
        if not hasattr(torch, "_utils"):
            torch._utils = _torch_utils  # type: ignore[attr-defined]
    except Exception:
        pass
    _TORCH_IMPORT_ERROR: str | None = None
except Exception as exc:
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = str(exc)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.investing_agent import _discover_endpoint_specs, _latest_inference_window
from utils.model_deployment import resolve_active_deployment
from utils.recursive_inference import recursive_forecast_matrix

TRADING_CFG_PATH = Path(os.environ.get("TRADING_CONFIG_PATH", str(ROOT / "config" / "trading_agent.yaml")))
MODEL_DIR = ROOT / "model_files"


class PredictPayload(BaseModel):
    rows: List[Dict[str, Any]]
    timeframe: str | None = None
    model: str | None = None
    config_path: str | None = None
    input_features: List[str] | None = None
    n_steps: int | None = None


class LoadedModel:
    def __init__(self, timeframe: str, spec: Dict[str, Any], model_path: Path, artifacts_path: Path | None):
        self.timeframe = timeframe
        self.spec = spec
        self.model_path = model_path
        self.artifacts_path = artifacts_path
        self.model_mtime_ns = model_path.stat().st_mtime_ns
        self.artifacts_mtime_ns = artifacts_path.stat().st_mtime_ns if artifacts_path and artifacts_path.exists() else None
        self.model = joblib.load(model_path)
        artifacts: Dict[str, Any] = {}
        if artifacts_path and artifacts_path.exists():
            try:
                artifacts = joblib.load(artifacts_path)
            except Exception:
                artifacts = {}
        self.scaler_x = artifacts.get("scaler_X") or ((artifacts.get("scalers") or {}).get("X") if isinstance(artifacts.get("scalers"), dict) else None)
        self.scaler_y = artifacts.get("scaler_y") or ((artifacts.get("scalers") or {}).get("y") if isinstance(artifacts.get("scalers"), dict) else None)


def _normalize_config_path(config_path: str) -> str:
    raw = str(config_path or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if not p.is_absolute():
        p = (ROOT / raw).resolve()
    return str(p)


def _latest_file(pattern: str, base: Path, include_artifacts: bool = False) -> Path | None:
    candidates = list(base.glob(pattern))
    if not include_artifacts:
        candidates = [p for p in candidates if "_artifacts_" not in p.name.lower()]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _resolve_target_slug(spec: Dict[str, Any]) -> str:
    target = str(spec.get("target_col") or "").strip().lower()
    if target in {"high", "low", "open", "close"}:
        return target

    cfg_path = str(spec.get("config_path") or "").replace("\\", "/").lower()
    for candidate in ("high", "low", "open", "close"):
        if f"/{candidate}" in cfg_path and "results/" in cfg_path:
            return candidate
    return "high"


def _resolve_model_paths(spec: Dict[str, Any]) -> Tuple[Path | None, Path | None]:
    deployed_model = str(spec.get("model_path") or "").strip()
    deployed_artifacts = str(spec.get("artifacts_path") or "").strip()
    if deployed_model:
        model_path = Path(deployed_model)
        artifacts_path = Path(deployed_artifacts) if deployed_artifacts else None
        if model_path.is_file() and (artifacts_path is None or artifacts_path.is_file()):
            return model_path, artifacts_path

    timeframe = str(spec.get("timeframe") or "").strip()
    model_name = str(spec.get("model") or "").strip().lower()
    target_slug = _resolve_target_slug(spec)

    exact_model_pattern = f"{model_name}_{target_slug}_{timeframe}_*.joblib"
    exact_artifacts_pattern = f"{model_name}_artifacts_{target_slug}_{timeframe}_*.joblib"

    m = _latest_file(exact_model_pattern, MODEL_DIR)
    a = _latest_file(exact_artifacts_pattern, MODEL_DIR, include_artifacts=True)
    if m:
        return m, a

    # Generic fallback by model prefix only if the exact target/timeframe artifact is absent.
    m = _latest_file(f"{model_name}_*.joblib", MODEL_DIR)
    a = _latest_file(f"{model_name}_artifacts_*.joblib", MODEL_DIR, include_artifacts=True)
    return m, a


def _load_specs() -> Dict[str, Dict[str, Any]]:
    if not TRADING_CFG_PATH.exists():
        return {}
    with TRADING_CFG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    endpoints = (cfg.get("model_endpoints") or {})
    return _discover_endpoint_specs(endpoints, config_root=str(ROOT / "config"))


def _load_spec_from_config(timeframe: str, config_path: str, fallback: Dict[str, Any] | None = None) -> Dict[str, Any]:
    spec = dict(fallback or {})
    resolved_config = _normalize_config_path(config_path)
    if resolved_config and Path(resolved_config).exists():
        with Path(resolved_config).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        spec.update(
            {
                "timeframe": timeframe,
                "config_path": resolved_config,
                "model": str(spec.get("model") or Path(resolved_config).parents[0].name),
                "r2": spec.get("r2"),
                "n_steps": int(cfg.get("n_steps", spec.get("n_steps", 1)) or 1),
                "m_steps": int(cfg.get("m_steps", spec.get("m_steps", 1)) or 1),
                "horizon": int(cfg.get("horizon", spec.get("horizon", cfg.get("m_steps", 1))) or 1),
                "input_features": list(cfg.get("input_features") or spec.get("input_features") or []),
                "target_features": list(cfg.get("target_features") or spec.get("target_features") or []),
                "target_col": str(cfg.get("target_col") or spec.get("target_col") or "HIGH"),
                "rolling_windows": list(cfg.get("rolling_windows") or spec.get("rolling_windows") or [2, 7, 30, 60]),
            }
        )
    return spec


def _spec_cache_key(spec: Dict[str, Any]) -> str:
    config_path = _normalize_config_path(str(spec.get("config_path") or ""))
    target_slug = _resolve_target_slug(spec)
    return "|".join(
        [
            str(spec.get("timeframe") or ""),
            str(spec.get("model") or "").lower(),
            target_slug,
            config_path,
            str(spec.get("deployment_id") or ""),
            str(spec.get("model_path") or ""),
        ]
    )


def _apply_active_deployment(spec: Dict[str, Any]) -> Dict[str, Any]:
    timeframe = str(spec.get("timeframe") or "").strip().lower()
    family = _resolve_target_slug(spec)
    if not timeframe:
        return spec
    deployment = resolve_active_deployment(f"{timeframe}_{family}")
    if deployment is None:
        return spec
    deployed = _load_spec_from_config(
        timeframe,
        str(deployment.get("config_path") or ""),
        fallback=spec,
    )
    deployed.update(
        {
            "model": str(deployment.get("model") or deployed.get("model") or "").lower(),
            "model_path": str(deployment.get("model_path") or ""),
            "artifacts_path": deployment.get("artifacts_path"),
            "deployment_id": deployment.get("deployment_id"),
            "deployment_endpoint": deployment.get("endpoint"),
            "training_data_first_index": deployment.get("training_data_first_index"),
            "training_data_last_index": deployment.get("training_data_last_index"),
        }
    )
    metrics = deployment.get("metrics") or {}
    qualification = deployment.get("qualification") or {}
    deployed["r2"] = metrics.get("holdout_r2", qualification.get("score", deployed.get("r2")))
    return deployed


def _resolve_request_spec(timeframe: str, payload: PredictPayload) -> Dict[str, Any]:
    base = dict(DEFAULT_SPECS.get(timeframe) or {})
    config_path = _normalize_config_path(str(payload.config_path or base.get("config_path") or ""))
    if config_path:
        base = _load_spec_from_config(timeframe, config_path, fallback=base)
    base["timeframe"] = timeframe
    if payload.model:
        base["model"] = str(payload.model).strip().lower()
    if payload.input_features:
        base["input_features"] = [str(x) for x in payload.input_features]
    if payload.n_steps:
        base["n_steps"] = int(payload.n_steps)
    if config_path:
        base["config_path"] = config_path
    return _apply_active_deployment(base)


def _get_loaded_model(timeframe: str, payload: PredictPayload) -> LoadedModel | None:
    spec = _resolve_request_spec(timeframe, payload)
    if not spec.get("model"):
        return DEFAULT_LOADED.get(timeframe)

    cache_key = _spec_cache_key(spec)
    cached = LOADED_CACHE.get(cache_key)
    m_path, a_path = _resolve_model_paths(spec)
    if not m_path or not m_path.exists():
        return None

    model_mtime_ns = m_path.stat().st_mtime_ns
    artifacts_mtime_ns = a_path.stat().st_mtime_ns if a_path and a_path.exists() else None
    if (
        cached is not None
        and cached.model_path == m_path
        and cached.artifacts_path == a_path
        and cached.model_mtime_ns == model_mtime_ns
        and cached.artifacts_mtime_ns == artifacts_mtime_ns
    ):
        return cached

    lm = LoadedModel(timeframe=timeframe, spec=spec, model_path=m_path, artifacts_path=a_path)
    LOADED_CACHE[cache_key] = lm
    return lm


def _load_models() -> Dict[str, LoadedModel]:
    specs = _load_specs()
    loaded: Dict[str, LoadedModel] = {}
    for tf, spec in specs.items():
        spec = dict(spec)
        spec["timeframe"] = tf
        m_path, a_path = _resolve_model_paths(spec)
        if not m_path or not m_path.exists():
            continue
        try:
            loaded[tf] = LoadedModel(timeframe=tf, spec=spec, model_path=m_path, artifacts_path=a_path)
        except Exception:
            continue
    return loaded


def _confidence_from_rows(rows: List[Dict[str, Any]], pred: float) -> float:
    vals = []
    for r in rows[-64:]:
        v = r.get("y_diff")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    scale = float(np.std(vals)) if vals else 1.0
    scale = max(scale, 1e-6)
    score = min(abs(pred) / scale, 4.0)
    return float(np.clip(0.5 + 0.12 * score, 0.5, 0.95))


def _predict_with_loaded(lm: LoadedModel, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    spec = lm.spec
    n_steps = int(spec.get("n_steps", 1) or 1)
    m_steps = int(spec.get("m_steps", 1) or 1)
    forecast_steps = int(spec.get("horizon", m_steps) or m_steps)
    feats = [str(c) for c in (spec.get("input_features") or [])]
    if len(rows) < n_steps:
        raise ValueError(f"insufficient_rows: need={n_steps} got={len(rows)}")

    # Training maps n_steps observations to m_steps future outputs. At
    # inference time the input must therefore be the newest n_steps rows.
    tail = _latest_inference_window(rows, n_steps)
    x2 = np.array([[float(r.get(c, 0.0) or 0.0) for c in feats] for r in tail], dtype=np.float64)

    model_name = str(spec.get("model") or "").strip().lower()
    target_features = [str(value) for value in (spec.get("target_features") or ["y_diff"])]
    target_col = str(spec.get("target_col") or "HIGH")

    def predict_window(window: np.ndarray) -> np.ndarray:
        if model_name == "nbeats":
            model_input = window.reshape(1, -1)
            if lm.scaler_x is not None:
                model_input = lm.scaler_x.transform(model_input)
            if torch is None:
                raise RuntimeError(f"torch_unavailable:{_TORCH_IMPORT_ERROR}")
            tensor = torch.tensor(model_input, dtype=torch.float32)
            with torch.no_grad():
                prediction = lm.model(tensor).cpu().numpy()
        else:
            model_input = window
            if lm.scaler_x is not None:
                model_input = lm.scaler_x.transform(model_input)
            prediction = lm.model.predict(model_input.reshape(1, -1))
        prediction = np.asarray(prediction)
        if lm.scaler_y is not None:
            prediction = lm.scaler_y.inverse_transform(prediction)
        return prediction.reshape(-1, len(target_features))

    rolling_windows = [int(value) for value in (spec.get("rolling_windows") or [2, 7, 30, 60])]
    y_matrix = recursive_forecast_matrix(
        predict_window=predict_window,
        initial_window=x2,
        steps=forecast_steps,
        m_steps=m_steps,
        input_features=feats,
        target_features=target_features,
        target_col=target_col,
        max_window=max(rolling_windows + [1]),
    )
    feature_horizons = {
        target: [float(value) for value in y_matrix[:, idx].tolist()]
        for idx, target in enumerate(target_features)
    }
    primary_target = "y_diff" if "y_diff" in feature_horizons else target_features[0]
    horizon = feature_horizons[primary_target]
    lead = float(horizon[0]) if horizon else 0.0
    sig = "buy" if lead > 0 else ("sell" if lead < 0 else "hold")
    conf = _confidence_from_rows(rows, lead)

    return {
        "signal": sig,
        "confidence": conf,
        "inference_strength": conf,
        "forecast_sign": lead,
        "prediction": lead,
        "horizon": horizon,
        "target_feature": primary_target,
        "feature_horizons": feature_horizons,
        "configured_horizon_steps": forecast_steps,
        "model_output_steps": m_steps,
        "timeframe": lm.timeframe,
        "model": model_name,
        "r2_train": spec.get("r2"),
        "data_as_of_utc": str((rows[-1] or {}).get("DATE") or "") if rows else "",
        "input_window_start_utc": str((tail[0] or {}).get("DATE") or "") if tail else "",
        "input_window_end_utc": str((tail[-1] or {}).get("DATE") or "") if tail else "",
        "inference_generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": str(lm.model_path),
        "artifacts_path": str(lm.artifacts_path) if lm.artifacts_path else None,
    }


app = FastAPI(title="TSMM Local Signal Endpoint Service", version="0.1.0")
DEFAULT_SPECS: Dict[str, Dict[str, Any]] = _load_specs()
DEFAULT_LOADED: Dict[str, LoadedModel] = _load_models()
LOADED_CACHE: Dict[str, LoadedModel] = {
    _spec_cache_key(lm.spec): lm
    for lm in DEFAULT_LOADED.values()
}


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "torch_available": torch is not None,
        "torch_import_error": _TORCH_IMPORT_ERROR,
        "loaded_timeframes": sorted(list(DEFAULT_LOADED.keys())),
        "cached_models": len(LOADED_CACHE),
        "loaded_models": {
            tf: {
                "model": lm.spec.get("model"),
                "config_path": lm.spec.get("config_path"),
                "model_path": str(lm.model_path),
                "artifacts_path": str(lm.artifacts_path) if lm.artifacts_path else None,
                "deployment_id": lm.spec.get("deployment_id"),
                "deployment_endpoint": lm.spec.get("deployment_endpoint"),
                "training_data_last_index": lm.spec.get("training_data_last_index"),
            }
            for tf, lm in DEFAULT_LOADED.items()
        },
    }


@app.post("/predict/{timeframe}")
def predict(timeframe: str, payload: PredictPayload) -> Dict[str, Any]:
    tf = str(timeframe or "").strip()
    try:
        lm = _get_loaded_model(tf, payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"model_load_failed:{e}")
    if lm is None:
        raise HTTPException(status_code=404, detail=f"timeframe_not_loaded:{tf}")

    try:
        return _predict_with_loaded(lm, payload.rows or [])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("TSMM_SIGNAL_HOST", "127.0.0.1")
    port = int(os.environ.get("TSMM_SIGNAL_PORT", "8000") or 8000)
    uvicorn.run("scripts.local_signal_endpoint_service:app", host=host, port=port, reload=False)
