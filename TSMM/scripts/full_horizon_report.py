"""Generate full horizon predictions report for all models and families.
Run on demand: python scripts/full_horizon_report.py

Outputs a formatted table of all 28 models (7 timeframes × 4 families)
with complete future arrays, actual training R², confidence,
and saves to reports/runtime/full_horizon_predictions.json
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import warnings
from pathlib import Path

# Import torch FIRST, before anything else tries to load DLLs
try:
    import torch

    _ = torch.tensor([1.0])  # force DLL init
    _TORCH_OK = True
except Exception:
    torch = None  # type: ignore[assignment]
    _TORCH_OK = False

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
from utils.market_db import query_ohlc
from utils.investing_agent import _enrich_endpoint_features, _tf_to_minutes


def _parse_r2_from_name(path: Path) -> float:
    """Parse training R² from config filename like top1_08098.yaml -> 0.8098."""
    stem = path.stem.lower()
    m = re.search(r"_(\d{4,6})$", stem)
    if m:
        digits = m.group(1)
        return float(int(digits) / (10 ** (len(digits) - 1)))
    m2 = re.search(r"(\d+\.\d+)", stem)
    if m2:
        return float(m2.group(1))
    return 0.0


def _confidence_from_rows(enriched: pd.DataFrame, pred: float) -> float:
    """Compute confidence from y_diff volatility, matching endpoint logic."""
    vals = []
    for _, r in enriched.tail(64).iterrows():
        v = r.get("y_diff")
        if isinstance(v, (int, float)) and not pd.isna(v):
            vals.append(float(v))
    scale = float(np.std(vals)) if vals else 1.0
    scale = max(scale, 1e-6)
    score = min(abs(pred) / scale, 4.0)
    return float(np.clip(0.5 + 0.12 * score, 0.5, 0.95))


def _load_training_r2_map() -> dict:
    """Build map of (timeframe, family) -> training R² from forecast run logs.

    Parses the first line of each forecast log to extract family/timeframe
    from the config path, then extracts R² from the end of the log.
    """
    import re
    logs_dir = ROOT / "reports/runtime/forecast_runs"

    r2_map = {}
    for log_path in sorted(logs_dir.glob("top1_*.log"), reverse=True):
        try:
            with open(log_path) as f:
                first_line = f.readline().strip()
                content = f.read()
        except Exception:
            continue

        # Extract family and timeframe from first line:
        # e.g. "config=...\close10mResults\nbeats\top1_08098.yaml"
        m = re.search(r"config=.+[\\/]([a-z]+)(\d+[a-z]*)Results", first_line, re.IGNORECASE)
        if not m:
            continue
        family = m.group(1).lower()
        timeframe_raw = m.group(2)
        timeframe = timeframe_raw.lower()

        # Extract R² from "Model Performance Summary" section at the end
        m2 = re.search(r"R.\s*:\s*([-\d.]+)", content)
        if m2:
            r2_map[(timeframe, family)] = float(m2.group(1))

    return r2_map


def _load_model_and_predict(
    cfg_path: Path, family: str, timeframe: str, model_name: str
) -> tuple[list[float] | None, float | None, str | None]:
    """Returns (horizon, confidence, error)."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    n_steps = int(cfg.get("n_steps", 10))
    m_steps = int(cfg.get("m_steps", 1))
    rolling_windows = list(cfg.get("rolling_windows", [2, 7, 30, 60]))
    input_features = list(cfg.get("input_features", ["HIGH", "y_diff", "Low_return", "Price_return"]))
    target_col = str(cfg.get("target_col", family.upper()))

    # Load market data for inference
    tf_minutes = _tf_to_minutes(timeframe)
    db_path = str(ROOT / "data" / "market_data.sqlite")
    latest_records = max(n_steps + max(rolling_windows) + 5, 200)

    df = query_ohlc(db_path, tf_minutes, latest_records=latest_records, symbol="XAUUSD")
    if df is None or df.empty:
        return None, None, "no_market_data"

    enriched = _enrich_endpoint_features(df, target_col=target_col, rolling_windows=rolling_windows)
    if len(enriched) < n_steps + m_steps:
        return None, None, f"insufficient_rows:{len(enriched)}"

    # Build input array with same window alignment as endpoint
    tail = enriched[-(n_steps + m_steps):-m_steps]
    x2 = np.array(
        [[float(r.get(c, 0.0) or 0.0) for c in input_features] for _, r in tail.iterrows()],
        dtype=np.float64,
    )

    # Find model file (newest matching)
    model_file = None
    for pat in [f"{model_name}_{family.lower()}_{timeframe}_*.joblib"]:
        cands = sorted(glob.glob(str(ROOT / "model_files" / pat)))
        if cands:
            model_file = cands[-1]
            break
    if not model_file or not os.path.exists(model_file):
        return None, None, "model_file_not_found"

    # Load artifacts for scalers
    arts = sorted(
        glob.glob(str(ROOT / "model_files" / f"*artifacts*{family.lower()}*{timeframe}*.joblib"))
    )
    scaler_x, scaler_y = None, None
    if arts:
        try:
            artifacts = joblib.load(arts[-1])
            scaler_x = artifacts.get("scaler_X") or (
                (artifacts.get("scalers") or {}).get("X")
                if isinstance(artifacts.get("scalers"), dict)
                else None
            )
            scaler_y = artifacts.get("scaler_y") or (
                (artifacts.get("scalers") or {}).get("y")
                if isinstance(artifacts.get("scalers"), dict)
                else None
            )
        except Exception:
            pass

    # For nbeats models, ensure torch was loaded before joblib load
    if model_name.lower() == "nbeats":
        if not _TORCH_OK:
            return None, None, "torch_unavailable"

    # Predict
    model = joblib.load(model_file)
    try:
        if model_name.lower() == "nbeats":
            x = x2.reshape(1, -1)
            if scaler_x is not None:
                x = scaler_x.transform(x)
            xt = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                yp = model(xt).cpu().numpy()
        else:
            x = x2
            if scaler_x is not None:
                x = scaler_x.transform(x)
            x = x.reshape(1, -1)
            yp = model.predict(x)

        y = np.asarray(yp)
        if scaler_y is not None:
            y = scaler_y.inverse_transform(y)

        y_full = y.reshape(-1).tolist()

        # Compute confidence from the first prediction
        first_pred = y_full[0] if y_full else 0.0
        confidence = _confidence_from_rows(enriched, first_pred)

        return y_full, confidence, None
    except Exception as e:
        return None, None, str(e)


def main() -> int:
    families = ["high", "low", "open", "close"]
    timeframes = ["10m", "30m", "1h", "3h", "7h", "12h", "24h"]
    models_by_tf = {
        "10m": "nbeats", "30m": "nbeats", "1h": "nbeats",
        "3h": "nbeats", "7h": "ulr", "12h": "nbeats", "24h": "nbeats",
    }

    # Load actual training R² from forecast logs
    training_r2 = _load_training_r2_map()

    results: dict = {}
    errors: list[str] = []

    for tf in timeframes:
        model_name = models_by_tf[tf]
        results[tf] = {}

        for family in families:
            cfg_dir = ROOT / "config" / f"{family}{tf}Results" / model_name
            if not cfg_dir.exists():
                results[tf][family] = {"error": "config_dir_not_found"}
                continue

            cands = sorted(
                list(cfg_dir.glob("top1*.yaml")) + list(cfg_dir.glob("top1*.yml"))
            )
            if not cands:
                cands = sorted(
                    list(cfg_dir.glob("top*.yaml")) + list(cfg_dir.glob("top*.yml"))
                )
            if not cands:
                results[tf][family] = {"error": "no_config"}
                continue

            # Pick best config by R² from filename
            cands.sort(key=_parse_r2_from_name, reverse=True)
            cfg_path = cands[0]

            pred, confidence, err = _load_model_and_predict(cfg_path, family, tf, model_name)
            if err:
                results[tf][family] = {"error": err}
                errors.append(f"{tf}/{family}/{model_name}: {err}")
            else:
                # Get actual training R² from forecast logs
                train_r2 = training_r2.get((tf, family))
                results[tf][family] = {
                    "horizon": pred,
                    "r2_train": round(train_r2, 4) if train_r2 is not None else None,
                    "confidence": round(confidence, 4) if confidence is not None else None,
                }

    # Print formatted table
    print(f"\n{'=' * 170}")
    header = (
        f"{'TF':<8} {'FAMILY':<8} {'MODEL':<10} {'STEPS':<6}"
        f" {'R2':<10} {'CONF':<8} {'SIGNAL':<8}"
        f" {'FULL HORIZON (price change)':<95}"
    )
    print(header)
    print(f"{'=' * 170}")

    for tf in timeframes:
        for family in families:
            r = results[tf].get(family, {})
            if "error" in r:
                print(f"{tf:<8} {family:<8} {models_by_tf[tf]:<10} {'ERR':<6} {'':<10} {'':<8} {'':<8} {r['error'][:95]}")
            else:
                h = r.get("horizon", [])
                n = len(h)
                h_str = ", ".join(f"{v:>+9.2f}" for v in h)
                sig = "BUY" if (h and h[0] > 0) else ("SELL" if (h and h[0] < 0) else "HOLD")
                r2_str = f"{r.get('r2_train', 0):.4f}" if r.get("r2_train") is not None else "N/A"
                conf_str = f"{r.get('confidence', 0):.4f}" if r.get("confidence") is not None else "N/A"
                print(f"{tf:<8} {family:<8} {models_by_tf[tf]:<10} {n:<6} {r2_str:<10} {conf_str:<8} {sig:<8} [{h_str}]")

    print(f"\n{'=' * 170}")
    print("\nR²: Extraído del log de entrenamiento (evaluación real del último retrain).")
    if errors:
        print(f"\nWarnings ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    # Save to JSON
    out_path = ROOT / "reports" / "runtime" / "full_horizon_predictions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
