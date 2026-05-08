"""Local endpoint service for Agent B timeframe signals.

Routes:
- GET /health
- POST /predict/{timeframe}

This server is intentionally lightweight and only serves inference payloads
built by utils.investing_agent._build_endpoint_payloads.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.investing_agent import _discover_endpoint_specs

TRADING_CFG_PATH = ROOT / "config" / "trading_agent.yaml"
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
        self.model = joblib.load(model_path)
        artifacts: Dict[str, Any] = {}
        if artifacts_path and artifacts_path.exists():
            try:
                artifacts = joblib.load(artifacts_path)
            except Exception:
                artifacts = {}
        self.scaler_x = artifacts.get("scaler_X") or ((artifacts.get("scalers") or {}).get("X") if isinstance(artifacts.get("scalers"), dict) else None)
        self.scaler_y = artifacts.get("scaler_y") or ((artifacts.get("scalers") or {}).get("y") if isinstance(artifacts.get("scalers"), dict) else None)


def _latest_file(pattern: str, base: Path, include_artifacts: bool = False) -> Path | None:
    candidates = list(base.glob(pattern))
    if not include_artifacts:
        candidates = [p for p in candidates if "_artifacts_" not in p.name.lower()]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _resolve_model_paths(timeframe: str, model_name: str) -> Tuple[Path | None, Path | None]:
    model_name = str(model_name or "").strip().lower()

    # Timeframe-dedicated historical bundles.
    if timeframe == "3h" and model_name == "nbeats":
        folder = MODEL_DIR / "high_3h_nbeats"
        m = _latest_file("nbeats_*.joblib", folder)
        a = _latest_file("nbeats_artifacts_*.joblib", folder, include_artifacts=True)
        if m:
            return m, a

    # Latest freshly trained ULR artifacts (used for 7h in this workspace flow).
    if timeframe == "7h" and model_name == "ulr":
        m = _latest_file("ulr_*.joblib", MODEL_DIR)
        a = _latest_file("ulr_artifacts_*.joblib", MODEL_DIR, include_artifacts=True)
        if m:
            return m, a

    # Generic fallback by model prefix.
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


def _load_models() -> Dict[str, LoadedModel]:
    specs = _load_specs()
    loaded: Dict[str, LoadedModel] = {}
    for tf, spec in specs.items():
        model_name = str(spec.get("model") or "").strip().lower()
        m_path, a_path = _resolve_model_paths(tf, model_name)
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
    feats = [str(c) for c in (spec.get("input_features") or [])]
    if len(rows) < n_steps:
        raise ValueError(f"insufficient_rows: need={n_steps} got={len(rows)}")

    tail = rows[-n_steps:]
    x2 = np.array([[float(r.get(c, 0.0) or 0.0) for c in feats] for r in tail], dtype=np.float32)
    x = x2.reshape(1, -1)

    if lm.scaler_x is not None:
        x = lm.scaler_x.transform(x)

    model_name = str(spec.get("model") or "").strip().lower()
    if model_name == "nbeats":
        xt = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            yp = lm.model(xt).cpu().numpy()
    else:
        yp = lm.model.predict(x)

    y = np.asarray(yp)
    if lm.scaler_y is not None:
        y = lm.scaler_y.inverse_transform(y)

    y_flat = y.reshape(-1)
    lead = float(y_flat[0]) if y_flat.size else 0.0
    sig = "buy" if lead > 0 else ("sell" if lead < 0 else "hold")
    conf = _confidence_from_rows(rows, lead)

    return {
        "signal": sig,
        "confidence": conf,
        "forecast_sign": lead,
        "prediction": lead,
        "timeframe": lm.timeframe,
        "model": model_name,
        "model_path": str(lm.model_path),
        "artifacts_path": str(lm.artifacts_path) if lm.artifacts_path else None,
    }


app = FastAPI(title="TSMM Local Signal Endpoint Service", version="0.1.0")
LOADED: Dict[str, LoadedModel] = _load_models()


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "loaded_timeframes": sorted(list(LOADED.keys())),
        "loaded_models": {
            tf: {
                "model": lm.spec.get("model"),
                "model_path": str(lm.model_path),
                "artifacts_path": str(lm.artifacts_path) if lm.artifacts_path else None,
            }
            for tf, lm in LOADED.items()
        },
    }


@app.post("/predict/{timeframe}")
def predict(timeframe: str, payload: PredictPayload) -> Dict[str, Any]:
    tf = str(timeframe or "").strip()
    lm = LOADED.get(tf)
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
