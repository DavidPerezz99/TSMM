"""Generate full horizon predictions report for all models and families.
Run on demand: python scripts/full_horizon_report.py

Outputs a formatted table of all 28 models (7 timeframes × 4 families)
with complete future arrays, actual training R², confidence,
and saves to reports/runtime/full_horizon_predictions.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import warnings
from datetime import datetime, timezone
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
STATUS_PATH = ROOT / "reports" / "runtime" / "full_horizon_report_status.json"

import joblib
from utils.market_db import query_ohlc
from utils.investing_agent import _enrich_endpoint_features, _latest_inference_window, _tf_to_minutes
from utils.inference_performance import InferencePerformanceStore
from utils.live_data import bootstrap_master_on_backend_start, resolve_tiingo_token_candidates
from utils.model_deployment import resolve_active_deployment
from utils.recursive_inference import recursive_forecast_matrix


def _utc_iso(timestamp: float | None = None) -> str:
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc) if timestamp is not None else datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _resolve_root_path(raw_path: str) -> Path:
    path = Path(str(raw_path or "").strip())
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_trading_config() -> dict:
    cfg_path = _resolve_root_path(os.environ.get("TRADING_CONFIG_PATH", "config/trading_agent.yaml"))
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp_path, path)


def _refresh_market_source() -> dict:
    cfg_path = _resolve_root_path(os.environ.get("TRADING_CONFIG_PATH", "config/trading_agent.yaml"))
    if not cfg_path.exists():
        return {"ok": False, "skipped": True, "reason": "trading_config_not_found", "config_path": str(cfg_path)}
    trading_cfg = _load_trading_config()
    dashboard_cfg = trading_cfg.get("dashboard") or {}
    if not bool(dashboard_cfg.get("startup_sync_enabled", True)):
        return {"ok": True, "skipped": True, "reason": "market_sync_disabled"}

    master_path = _resolve_root_path(
        str(dashboard_cfg.get("master_table_path") or dashboard_cfg.get("raw_data_path") or "data/market_data.sqlite")
    )
    token_env = str(dashboard_cfg.get("tiingo_token_env") or "TIINGO_API_TOKEN")
    token_envs = dashboard_cfg.get("tiingo_token_envs")
    token = os.environ.get(token_env, "")
    if not resolve_tiingo_token_candidates(token_env=token_env, token_envs=token_envs, token=token):
        return {"ok": False, "skipped": True, "reason": "missing_tiingo_token", "master_path": str(master_path)}

    result = bootstrap_master_on_backend_start(
        master_table_path=str(master_path),
        rate=str(dashboard_cfg.get("tiingo_rate") or "1min"),
        symbol=str(dashboard_cfg.get("tiingo_symbol") or "xauusd"),
        token=token,
        max_pulls=1,
        freshness_lag_minutes=int(dashboard_cfg.get("startup_freshness_lag_minutes", 20) or 20),
        token_env=token_env,
        token_envs=token_envs,
        token_rotation_state_path=str(dashboard_cfg.get("tiingo_token_rotation_state_path") or "") or None,
    )
    result["master_path"] = str(master_path)
    return result


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
    log_paths = sorted(logs_dir.glob("top1_*.log"), key=lambda path: path.stat().st_mtime, reverse=True)
    for log_path in log_paths:
        try:
            with open(log_path) as f:
                first_line = f.readline().strip()
                content = f.read()
        except Exception:
            continue

        # Extract family and timeframe from first line:
        # e.g. "config=...\close10mResults\nbeats\top1_08098.yaml"
        m = re.search(
            r"config=.+[\\/]([a-z]+)(\d+[a-z]*)Results[\\/]([^\\/]+)[\\/]",
            first_line,
            re.IGNORECASE,
        )
        if not m:
            continue
        family = m.group(1).lower()
        timeframe_raw = m.group(2)
        timeframe = timeframe_raw.lower()
        model_name = m.group(3).lower()
        key = (timeframe, family, model_name)
        if key in r2_map:
            continue

        # Extract R² from "Model Performance Summary" section at the end
        m2 = re.search(r"R.\s*:\s*([-\d.]+)", content)
        if m2:
            r2_map[key] = {
                "value": float(m2.group(1)),
                "source_log": str(log_path),
                "evaluated_at_utc": _utc_iso(log_path.stat().st_mtime),
            }

    # Direct production refreshes record their current holdout result in the
    # endpoint version manifest. Those values are newer and more specific than
    # historical bulk-run logs, so they must override the older lineage score.
    versions_path = ROOT / "config" / "model_endpoint_versions.yaml"
    try:
        versions = yaml.safe_load(versions_path.read_text(encoding="utf-8")) or {}
        for endpoint_key, endpoint_entry in (versions.get("endpoints") or {}).items():
            current = (endpoint_entry or {}).get("current") or {}
            refreshed_r2 = current.get("refreshed_r2")
            if refreshed_r2 is None or "_" not in str(endpoint_key):
                continue
            timeframe, family = str(endpoint_key).split("_", 1)
            model_name = str(current.get("model") or "").strip().lower()
            if not model_name:
                continue
            r2_map[(timeframe.lower(), family.lower(), model_name)] = {
                "value": float(refreshed_r2),
                "source_log": str(versions_path),
                "evaluated_at_utc": str(current.get("refreshed_at_utc") or "") or None,
            }
    except Exception:
        pass

    return r2_map


def _load_model_and_predict(
    cfg_path: Path, family: str, timeframe: str, model_name: str,
    deployment: dict | None = None,
) -> tuple[list[float] | None, float | None, dict, str | None]:
    """Return the latest horizon, dynamic confidence, inference metadata, and error."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f) or {}

    n_steps = int(cfg.get("n_steps", 10))
    m_steps = int(cfg.get("m_steps", 1) or 1)
    forecast_steps = int(cfg.get("horizon", m_steps) or m_steps)
    rolling_windows = list(cfg.get("rolling_windows", [2, 7, 30, 60]))
    input_features = list(cfg.get("input_features", ["HIGH", "y_diff", "Low_return", "Price_return"]))
    target_features = [str(value) for value in (cfg.get("target_features") or ["y_diff"])]
    target_col = str(cfg.get("target_col", family.upper()))

    # Load market data for inference
    tf_minutes = _tf_to_minutes(timeframe)
    db_path = str(ROOT / "data" / "market_data.sqlite")
    latest_records = max(n_steps + max(rolling_windows) + 5, 200)

    df = query_ohlc(db_path, tf_minutes, latest_records=latest_records, symbol="XAUUSD")
    if df is None or df.empty:
        return None, None, {}, "no_market_data"

    enriched = _enrich_endpoint_features(df, target_col=target_col, rolling_windows=rolling_windows)
    if len(enriched) < n_steps:
        return None, None, {}, f"insufficient_rows:{len(enriched)}"

    # m_steps is the output horizon. Inference must consume the newest n_steps.
    tail = _latest_inference_window(enriched, n_steps)
    x2 = np.array(
        [[float(r.get(c, 0.0) or 0.0) for c in input_features] for _, r in tail.iterrows()],
        dtype=np.float64,
    )

    # Find model file (newest matching)
    model_file: Path | None = None
    deployed_model_path = str((deployment or {}).get("model_path") or "").strip()
    if deployed_model_path:
        model_file = Path(deployed_model_path)
    else:
        for pat in [f"{model_name}_{family.lower()}_{timeframe}_*.joblib"]:
            cands = [Path(path) for path in glob.glob(str(ROOT / "model_files" / pat))]
            if cands:
                model_file = max(cands, key=lambda path: path.stat().st_mtime)
                break
    if not model_file or not model_file.exists():
        return None, None, {}, "model_file_not_found"

    # Load artifacts for scalers
    deployed_artifacts_path = str((deployment or {}).get("artifacts_path") or "").strip()
    if deployment is not None:
        arts = [Path(deployed_artifacts_path)] if deployed_artifacts_path else []
    else:
        arts = [
            Path(path)
            for path in glob.glob(
                str(ROOT / "model_files" / f"{model_name}_artifacts_{family.lower()}_{timeframe}_*.joblib")
            )
        ]
    scaler_x, scaler_y = None, None
    if arts:
        try:
            artifacts = joblib.load(max(arts, key=lambda path: path.stat().st_mtime))
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
            return None, None, {}, "torch_unavailable"

    # Predict the configured future horizon recursively when the model emits
    # fewer steps per call than the report requests.
    model = joblib.load(model_file)
    try:
        def predict_window(window: np.ndarray) -> np.ndarray:
            if model_name.lower() == "nbeats":
                model_input = window.reshape(1, -1)
                if scaler_x is not None:
                    model_input = scaler_x.transform(model_input)
                tensor = torch.tensor(model_input, dtype=torch.float32)
                with torch.no_grad():
                    prediction = model(tensor).cpu().numpy()
            else:
                model_input = window
                if scaler_x is not None:
                    model_input = scaler_x.transform(model_input)
                prediction = model.predict(model_input.reshape(1, -1))
            prediction = np.asarray(prediction)
            if scaler_y is not None:
                prediction = scaler_y.inverse_transform(prediction)
            return prediction.reshape(-1, len(target_features))

        y_matrix = recursive_forecast_matrix(
            predict_window=predict_window,
            initial_window=x2,
            steps=forecast_steps,
            m_steps=m_steps,
            input_features=input_features,
            target_features=target_features,
            target_col=target_col,
            max_window=max([int(value) for value in rolling_windows] + [1]),
        )
        feature_horizons = {
            target: [float(value) for value in y_matrix[:, idx].tolist()]
            for idx, target in enumerate(target_features)
        }
        primary_target = "y_diff" if "y_diff" in feature_horizons else target_features[0]
        y_full = feature_horizons[primary_target]

        # Compute confidence from the first prediction
        first_pred = y_full[0] if y_full else 0.0
        confidence = _confidence_from_rows(enriched, first_pred)
        fingerprint_payload = tail[[col for col in input_features if col in tail.columns]].to_json(
            orient="split", date_format="iso", double_precision=12
        )
        metadata = {
            "inference_generated_at_utc": _utc_iso(),
            "timeframe_bucket_utc": pd.to_datetime(df.iloc[-1]["DATE"]).strftime("%Y-%m-%d %H:%M:%S"),
            "input_window_start_utc": pd.to_datetime(tail.iloc[0]["DATE"]).strftime("%Y-%m-%d %H:%M:%S"),
            "input_window_end_utc": pd.to_datetime(tail.iloc[-1]["DATE"]).strftime("%Y-%m-%d %H:%M:%S"),
            "input_fingerprint": hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest(),
            "model_path": str(model_file),
            "model_updated_at_utc": _utc_iso(model_file.stat().st_mtime),
            "config_path": str(cfg_path),
            "deployment_id": (deployment or {}).get("deployment_id"),
            "deployment_endpoint": (deployment or {}).get("endpoint"),
            "target_feature": primary_target,
            "feature_horizons": feature_horizons,
            "configured_horizon_steps": forecast_steps,
            "model_output_steps": m_steps,
        }
        return y_full, confidence, metadata, None
    except Exception as e:
        return None, None, {}, str(e)


def _latest_source_data_timestamp() -> str | None:
    frame = query_ohlc(str(ROOT / "data" / "market_data.sqlite"), 1, latest_records=1, symbol="XAUUSD")
    if frame is None or frame.empty:
        return None
    return pd.to_datetime(frame.iloc[-1]["DATE"]).strftime("%Y-%m-%d %H:%M:%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-market-refresh",
        action="store_true",
        help="Use the current local minute database without requesting a Tiingo refresh first.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_started_at = _utc_iso()
    trading_cfg = _load_trading_config()
    _write_json_atomic(
        STATUS_PATH,
        {"ok": True, "status": "running", "started_at_utc": report_started_at, "pid": os.getpid()},
    )
    refresh_result = {"ok": True, "skipped": True, "reason": "cli_skip_market_refresh"}
    if not args.skip_market_refresh:
        try:
            refresh_result = _refresh_market_source()
        except Exception as exc:
            refresh_result = {"ok": False, "error": str(exc)}
    source_data_as_of = _latest_source_data_timestamp()
    families = ["high", "low", "open", "close"]
    timeframes = ["10m", "30m", "1h", "3h", "7h", "12h", "24h"]
    models_by_tf = {
        "10m": "nbeats", "30m": "nbeats", "1h": "nbeats",
        "3h": "nbeats", "7h": "ulr", "12h": "nbeats", "24h": "nbeats",
    }
    models_by_family_tf = {
        ("7h", "high"): "nbeats",
        ("7h", "low"): "nbeats",
    }

    def selected_model(timeframe: str, family: str) -> str:
        return models_by_family_tf.get((timeframe, family), models_by_tf[timeframe])

    # Load actual training R² from forecast logs
    training_r2 = _load_training_r2_map()

    results: dict = {}
    errors: list[str] = []

    for tf in timeframes:
        results[tf] = {}

        for family in families:
            deployment = resolve_active_deployment(f"{tf}_{family}")
            model_name = str((deployment or {}).get("model") or selected_model(tf, family)).lower()
            if deployment is not None:
                cfg_path = Path(str(deployment["config_path"]))
            else:
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

                # Pick best config by R² from filename only for the legacy fallback.
                cands.sort(key=_parse_r2_from_name, reverse=True)
                cfg_path = cands[0]

            pred, confidence, inference_meta, err = _load_model_and_predict(
                cfg_path, family, tf, model_name, deployment=deployment
            )
            if err:
                results[tf][family] = {"error": err}
                errors.append(f"{tf}/{family}/{model_name}: {err}")
            else:
                # Get actual training R² from forecast logs
                deployment_metrics = (deployment or {}).get("metrics") or {}
                qualification = (deployment or {}).get("qualification") or {}
                deployed_r2 = deployment_metrics.get("holdout_r2", qualification.get("score"))
                training_metric = (
                    {"value": deployed_r2, "source_log": str((deployment or {}).get("deployment_dir") or ""),
                     "evaluated_at_utc": (deployment or {}).get("activated_at_utc")}
                    if deployed_r2 is not None else (training_r2.get((tf, family, model_name)) or {})
                )
                train_r2 = training_metric.get("value") if isinstance(training_metric, dict) else training_metric
                results[tf][family] = {
                    "model": model_name,
                    "horizon": pred,
                    "r2_train": round(train_r2, 4) if train_r2 is not None else None,
                    "r2_train_source_log": training_metric.get("source_log") if isinstance(training_metric, dict) else None,
                    "r2_train_evaluated_at_utc": training_metric.get("evaluated_at_utc") if isinstance(training_metric, dict) else None,
                    "confidence": round(confidence, 6) if confidence is not None else None,
                    "inference_strength": round(confidence, 6) if confidence is not None else None,
                    **inference_meta,
                }

    performance_cfg = trading_cfg.get("inference_performance") or {}
    online_evaluation: Dict[str, Any] = {"enabled": bool(performance_cfg.get("enabled", True))}
    if online_evaluation["enabled"]:
        try:
            performance_path = _resolve_root_path(
                str(performance_cfg.get("database_path") or "reports/runtime/full_horizon_metrics.sqlite")
            )
            store = InferencePerformanceStore(performance_path)
            forecast_rows: list[dict] = []
            for tf in timeframes:
                for family in families:
                    result = results.get(tf, {}).get(family, {})
                    if "error" in result:
                        continue
                    for step, prediction in enumerate(result.get("horizon") or [], start=1):
                        forecast_rows.append(
                            {
                                "generated_at_utc": report_started_at,
                                "origin_bucket_utc": result.get("timeframe_bucket_utc"),
                                "timeframe": tf,
                                "timeframe_minutes": _tf_to_minutes(tf),
                                "family": family,
                                "model": result.get("model") or selected_model(tf, family),
                                "model_path": result.get("model_path"),
                                "model_updated_at_utc": result.get("model_updated_at_utc"),
                                "target_feature": result.get("target_feature") or "y_diff",
                                "step": step,
                                "predicted_value": prediction,
                                "inference_strength": result.get("inference_strength", result.get("confidence")),
                                "r2_train": result.get("r2_train"),
                                "input_fingerprint": result.get("input_fingerprint"),
                            }
                        )
            inserted = store.record_forecasts(forecast_rows)
            matured = store.mature_pending(
                market_db_path=str(_resolve_root_path("data/market_data.sqlite")),
                source_data_as_of_utc=str(source_data_as_of or ""),
                symbol="XAUUSD",
            )
            window_samples = max(int(performance_cfg.get("rolling_window_samples", 100) or 100), 2)
            min_samples = max(int(performance_cfg.get("min_samples", 10) or 10), 2)
            for tf in timeframes:
                for family in families:
                    result = results.get(tf, {}).get(family, {})
                    if "error" in result:
                        continue
                    metrics = store.rolling_metrics(
                        timeframe=tf,
                        family=family,
                        model=str(result.get("model") or selected_model(tf, family)),
                        window_samples=window_samples,
                        min_samples=min_samples,
                    )
                    current_model_metrics = store.rolling_metrics(
                        timeframe=tf,
                        family=family,
                        model=str(result.get("model") or selected_model(tf, family)),
                        model_path=str(result.get("model_path") or ""),
                        window_samples=window_samples,
                        min_samples=min_samples,
                    )
                    drift = store.record_metric_snapshot(
                        generated_at_utc=report_started_at,
                        timeframe=tf,
                        family=family,
                        model=str(result.get("model") or selected_model(tf, family)),
                        model_path=str(result.get("model_path") or ""),
                        metrics=metrics,
                    )
                    result.update(metrics)
                    result.update(drift)
                    result["r2_live_current_model"] = current_model_metrics.get("r2_live_rolling")
                    result["r2_live_current_model_samples"] = current_model_metrics.get("r2_live_samples")
            retention = store.prune(
                forecast_retention_days=int(performance_cfg.get("forecast_retention_days", 180) or 180),
                snapshot_retention_days=int(performance_cfg.get("snapshot_retention_days", 365) or 365),
            )
            online_evaluation.update(
                {
                    "ok": True,
                    "database_path": str(performance_path),
                    "forecasts_inserted": inserted,
                    "forecasts_matured": matured,
                    "rolling_window_samples": window_samples,
                    "min_samples": min_samples,
                    "retention": retention,
                }
            )
        except Exception as exc:
            online_evaluation.update({"ok": False, "error": str(exc)})

    # Print formatted table
    print(f"\n{'=' * 202}")
    header = (
        f"{'TF':<8} {'FAMILY':<8} {'MODEL':<10} {'STEPS':<6}"
        f" {'R2 TRAIN':<10} {'STRENGTH':<10} {'R2 LIVE':<10} {'DELTA':<10} {'N':<5} {'SIGNAL':<8}"
        f" {'FULL HORIZON (price change)':<95}"
    )
    print(header)
    print(f"{'=' * 202}")

    for tf in timeframes:
        for family in families:
            r = results[tf].get(family, {})
            if "error" in r:
                print(f"{tf:<8} {family:<8} {str(r.get('model') or selected_model(tf, family)):<10} {'ERR':<6} {'':<10} {'':<8} {'':<8} {r['error'][:95]}")
            else:
                h = r.get("horizon", [])
                n = len(h)
                h_str = ", ".join(f"{v:>+9.2f}" for v in h)
                sig = "BUY" if (h and h[0] > 0) else ("SELL" if (h and h[0] < 0) else "HOLD")
                r2_str = f"{r.get('r2_train', 0):.4f}" if r.get("r2_train") is not None else "N/A"
                strength_str = f"{r.get('inference_strength', 0):.4f}" if r.get("inference_strength") is not None else "N/A"
                live_r2_str = f"{r.get('r2_live_rolling', 0):.4f}" if r.get("r2_live_rolling") is not None else "N/A"
                live_delta_str = f"{r.get('r2_live_delta', 0):+.4f}" if r.get("r2_live_delta") is not None else "N/A"
                live_n = int(r.get("r2_live_samples") or 0)
                print(f"{tf:<8} {family:<8} {str(r.get('model') or selected_model(tf, family)):<10} {n:<6} {r2_str:<10} {strength_str:<10} {live_r2_str:<10} {live_delta_str:<10} {live_n:<5} {sig:<8} [{h_str}]")

    print(f"\n{'=' * 202}")
    print("\nR²: evaluación del último retrain, tomada del manifiesto de producción o del log más reciente.")
    if errors:
        print(f"\nWarnings ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    # Save to JSON
    out_path = ROOT / "reports" / "runtime" / "full_horizon_predictions.json"
    output = dict(results)
    output["_meta"] = {
        "schema_version": 2,
        "report_started_at_utc": report_started_at,
        "report_completed_at_utc": _utc_iso(),
        "source_data_as_of_utc": source_data_as_of,
        "market_refresh": refresh_result,
        "online_evaluation": online_evaluation,
        "r2_semantics": "Static holdout evaluation from the production version manifest or newest retraining log for each timeframe/family/model.",
        "inference_strength_semantics": "Dynamic magnitude-versus-volatility heuristic; not a calibrated probability.",
        "r2_live_semantics": "Rolling R2 over matured forecasts versus realized target-family y_diff, scoped to the timeframe/family/model lineage across retraining.",
        "r2_live_current_model_semantics": "Rolling R2 over matured forecasts scoped to the exact current model artifact.",
    }
    _write_json_atomic(out_path, output)
    _write_json_atomic(
        STATUS_PATH,
        {
            "ok": True,
            "status": "completed",
            "started_at_utc": report_started_at,
            "completed_at_utc": output["_meta"]["report_completed_at_utc"],
            "source_data_as_of_utc": source_data_as_of,
            "output_path": str(out_path),
            "error_count": len(errors),
        },
    )
    print(f"\nSaved: {out_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        _write_json_atomic(
            STATUS_PATH,
            {"ok": False, "status": "failed", "failed_at_utc": _utc_iso(), "error": str(exc)},
        )
        raise
