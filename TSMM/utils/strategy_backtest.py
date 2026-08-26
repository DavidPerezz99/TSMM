"""Point-in-time market replay for the TSMM trading strategy.

The replay deliberately keeps broker access out of the process.  It feeds only
market rows available at each simulated timestamp to the current local model
artifacts, evaluates Agent A at configured session/follow-up times, and runs
Agent B consensus checks at the configured polling interval.

Using today's fitted artifacts against an older period is a useful retrospective
stress test, but it is not an unbiased walk-forward result when those artifacts
were trained after (or through) the evaluated period.  Every report makes that
distinction explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, TextIO
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import yaml

from utils.investing_agent import _discover_agent_a_enrichment_candidates
from utils.market_db import master_table_name


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIMEFRAME_WEIGHTS = {
    "7h": 2.4,
    "3h": 1.8,
    "1h": 1.4,
    "30m": 1.1,
    "10m": 0.9,
    "12h": 1.6,
    "24h": 1.5,
    "1w": 1.7,
}


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(float(seconds)) or float(seconds) < 0:
        return "estimating"
    remaining = int(round(float(seconds)))
    hours, remainder = divmod(remaining, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


class ConsoleProgressBar:
    """Throttled console renderer for long historical replay runs."""

    def __init__(
        self,
        *,
        stream: Optional[TextIO] = None,
        width: int = 30,
        min_interval_seconds: float = 0.25,
    ):
        self.stream = stream or sys.stdout
        self.width = max(int(width), 10)
        self.min_interval_seconds = max(float(min_interval_seconds), 0.0)
        self.started_at: Optional[float] = None
        self.last_rendered_at = 0.0
        self.last_milestone = -1
        self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())

    def __call__(self, current: int, total: int, simulated_at: str) -> None:
        now = time.monotonic()
        current = max(int(current), 0)
        total = max(int(total), 1)
        if current == 0 or self.started_at is None:
            self.started_at = now
            self.last_rendered_at = 0.0
            self.last_milestone = -1

        fraction = min(max(current / total, 0.0), 1.0)
        percentage = fraction * 100.0
        elapsed = max(now - float(self.started_at), 0.0)
        eta = (elapsed / current) * (total - current) if current > 0 else None

        if self.is_tty:
            if current not in {0, total} and now - self.last_rendered_at < self.min_interval_seconds:
                return
        else:
            milestone = min(int(percentage // 5) * 5, 100)
            if current not in {0, total} and milestone <= self.last_milestone:
                return
            self.last_milestone = milestone

        completed = min(int(round(self.width * fraction)), self.width)
        if current >= total:
            bar = "=" * self.width
        elif completed <= 0:
            bar = ">" + "-" * (self.width - 1)
        else:
            bar = "=" * max(completed - 1, 0) + ">" + "-" * (self.width - completed)
        label = str(simulated_at or "initializing")
        message = (
            f"Backtest [{bar}] {percentage:6.2f}% | {current:,}/{total:,} ticks | "
            f"elapsed {_format_duration(elapsed)} | ETA {_format_duration(eta)} | simulated {label}"
        )
        if self.is_tty:
            suffix = "\n" if current >= total else ""
            self.stream.write("\r" + message + suffix)
        else:
            self.stream.write(message + "\n")
        self.stream.flush()
        self.last_rendered_at = now


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except Exception:
        return default


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=True)
    os.replace(temp, path)


def _parse_hhmm(raw: Any) -> tuple[int, int]:
    text = str(raw or "00:00").strip()
    hour, minute = text.split(":", 1)
    return int(hour), int(minute)


def _timeframe_minutes(label: str) -> int:
    text = str(label or "").strip().lower()
    if text.endswith("m"):
        return max(int(text[:-1]), 1)
    if text.endswith("h"):
        return max(int(text[:-1]) * 60, 1)
    if text.endswith("w"):
        return max(int(text[:-1]) * 7 * 24 * 60, 1)
    return 1


def previous_calendar_month(reference: Optional[datetime] = None, timezone_name: str = "UTC") -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return inclusive minute bounds for the previous local calendar month."""
    zone = ZoneInfo(str(timezone_name or "UTC"))
    now = reference or datetime.now(zone)
    if now.tzinfo is None:
        now = now.replace(tzinfo=zone)
    else:
        now = now.astimezone(zone)
    first_this_month = datetime(now.year, now.month, 1, tzinfo=zone)
    last_previous = first_this_month - timedelta(minutes=1)
    start_previous = datetime(last_previous.year, last_previous.month, 1, tzinfo=zone)
    return (
        pd.Timestamp(start_previous.astimezone(timezone.utc).replace(tzinfo=None)),
        pd.Timestamp(last_previous.astimezone(timezone.utc).replace(tzinfo=None)),
    )


def normalize_period(
    start_date: Optional[str],
    end_date: Optional[str],
    *,
    previous_month: bool,
    timezone_name: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if previous_month or (not start_date and not end_date):
        return previous_calendar_month(timezone_name=timezone_name)
    if not start_date or not end_date:
        raise ValueError("Both start_date and end_date are required unless previous_month is selected")

    zone = ZoneInfo(str(timezone_name or "UTC"))

    def _local_to_utc_naive(raw: str, end: bool) -> pd.Timestamp:
        parsed = pd.Timestamp(raw)
        date_only = len(str(raw).strip()) <= 10
        if date_only and end:
            parsed = parsed + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
        if parsed.tzinfo is None:
            parsed = parsed.tz_localize(zone)
        return parsed.tz_convert("UTC").tz_localize(None)

    start = _local_to_utc_naive(start_date, False)
    end = _local_to_utc_naive(end_date, True)
    if end < start:
        raise ValueError(f"End date {end_date} is before start date {start_date}")
    return start, end


def _load_market_minutes(
    source_path: str,
    symbol: str,
    warmup_start: pd.Timestamp,
    period_end: pd.Timestamp,
) -> pd.DataFrame:
    path = Path(source_path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Market source not found: {path}")

    if path.suffix.lower() in {".sqlite", ".db"}:
        table = master_table_name(symbol)
        uri = f"file:{path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            frame = pd.read_sql_query(
                f"""
                SELECT DATE, OPEN, HIGH, LOW, CLOSE, VOLUME
                FROM {table}
                WHERE DATE >= ? AND DATE <= ?
                ORDER BY DATE
                """,
                connection,
                params=[warmup_start.strftime("%Y-%m-%d %H:%M:%S"), period_end.strftime("%Y-%m-%d %H:%M:%S")],
            )
    else:
        frame = pd.read_csv(path)

    frame.columns = [str(col).strip().upper() for col in frame.columns]
    required = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Market source is missing columns: {missing}")
    if "VOLUME" not in frame.columns:
        frame["VOLUME"] = 0.0
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce").dt.tz_localize(None)
    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=required).drop_duplicates(subset=["DATE"], keep="last").sort_values("DATE")
    return frame.reset_index(drop=True)


class AsOfMarketTape:
    """Serve timeframe candles without exposing future rows in a live bucket."""

    def __init__(self, minute_frame: pd.DataFrame, timeframes: Iterable[str]):
        self.minutes = minute_frame.copy().sort_values("DATE").reset_index(drop=True)
        self._minute_dates = self.minutes["DATE"].to_numpy(dtype="datetime64[ns]")
        self.frames: Dict[str, pd.DataFrame] = {}
        for label in sorted(set(str(value) for value in timeframes)):
            tf_minutes = _timeframe_minutes(label)
            indexed = self.minutes.set_index("DATE")
            grouped = (
                indexed.resample(f"{tf_minutes}min", origin="epoch")
                .agg({"OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last", "VOLUME": "sum"})
                .dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
                .reset_index()
            )
            self.frames[label] = grouped

    @staticmethod
    def _bucket_start(timestamp: pd.Timestamp, minutes: int) -> pd.Timestamp:
        epoch_ns = int(pd.Timestamp("1970-01-01").value)
        step_ns = int(minutes) * 60 * 1_000_000_000
        value = int(pd.Timestamp(timestamp).value)
        return pd.Timestamp(epoch_ns + ((value - epoch_ns) // step_ns) * step_ns)

    def timeframe_as_of(self, label: str, as_of: pd.Timestamp, max_rows: int = 220) -> pd.DataFrame:
        tf_minutes = _timeframe_minutes(label)
        bucket = self._bucket_start(pd.Timestamp(as_of), tf_minutes)
        complete = self.frames[label]
        historical = complete[complete["DATE"] < bucket].tail(max(max_rows - 1, 1)).copy()

        left = int(np.searchsorted(self._minute_dates, np.datetime64(bucket), side="left"))
        right = int(np.searchsorted(self._minute_dates, np.datetime64(as_of), side="right"))
        partial = self.minutes.iloc[left:right]
        if partial.empty:
            return historical.tail(max_rows).reset_index(drop=True)

        current = pd.DataFrame(
            [
                {
                    "DATE": bucket,
                    "OPEN": float(partial.iloc[0]["OPEN"]),
                    "HIGH": float(partial["HIGH"].max()),
                    "LOW": float(partial["LOW"].min()),
                    "CLOSE": float(partial.iloc[-1]["CLOSE"]),
                    "VOLUME": float(partial["VOLUME"].fillna(0.0).sum()),
                }
            ]
        )
        return pd.concat([historical, current], ignore_index=True).tail(max_rows).reset_index(drop=True)


def _enrich_features(
    frame: pd.DataFrame,
    target_col: str,
    rolling_windows: List[int],
    required_features: Optional[List[str]] = None,
) -> pd.DataFrame:
    out = frame.copy().sort_values("DATE").reset_index(drop=True)
    if out.empty:
        return out
    target = pd.to_numeric(out[target_col], errors="coerce")
    out["Price_return"] = pd.to_numeric(out["CLOSE"], errors="coerce").diff()
    out["Open_return"] = pd.to_numeric(out["OPEN"], errors="coerce").diff()
    out["High_return"] = pd.to_numeric(out["HIGH"], errors="coerce").diff()
    out["Low_return"] = pd.to_numeric(out["LOW"], errors="coerce").diff()
    out["daily_return"] = pd.to_numeric(out["CLOSE"], errors="coerce") - pd.to_numeric(out["OPEN"], errors="coerce")
    out["y_diff"] = target.diff()
    windows = sorted(set(int(value) for value in rolling_windows if int(value) > 0))
    required = set(required_features or [])
    rolling_required = any(
        feature.startswith(("SMA_", "EMA_", "Volatility_"))
        for feature in required
    )
    if rolling_required:
        for window in windows:
            out[f"SMA_{window}"] = target.rolling(window=window).mean()
            out[f"EMA_{window}"] = target.ewm(span=window, adjust=False).mean()
            out[f"Volatility_{window}"] = target.rolling(window=window).std()
            out[f"SMA_{window}_diff"] = out["y_diff"].rolling(window=window).mean()
            out[f"EMA_{window}_diff"] = out["y_diff"].ewm(span=window, adjust=False).mean()
            out[f"Volatility_{window}_diff"] = out["y_diff"].rolling(window=window).std()
        return out.dropna().reset_index(drop=True)

    # Production enrichment creates all rolling columns and drops rows until
    # the largest window is mature, even when a model consumes only OHLC and
    # first differences. Preserve that row eligibility without constructing
    # dozens of unused pandas columns at every five-minute tick.
    mature_from = max(windows + [1])
    subset = ["DATE"] + [feature for feature in required if feature in out.columns]
    return out.iloc[mature_from:].dropna(subset=subset).reset_index(drop=True)


def discover_replay_model_specs(trading_cfg: Dict[str, Any], project_root: Path = PROJECT_ROOT) -> List[Dict[str, Any]]:
    candidates = _discover_agent_a_enrichment_candidates(trading_cfg)
    specs: List[Dict[str, Any]] = []
    for candidate in candidates:
        config_path = Path(str(candidate.get("config_path") or ""))
        if not config_path.is_absolute():
            config_path = (project_root / config_path).resolve()
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        specs.append(
            {
                **candidate,
                "config_path": str(config_path),
                "n_steps": int(config.get("n_steps", 1) or 1),
                "m_steps": int(config.get("m_steps", 1) or 1),
                "horizon": int(config.get("horizon", config.get("m_steps", 1)) or 1),
                "input_features": [str(value) for value in (config.get("input_features") or [])],
                "target_features": [str(value) for value in (config.get("target_features") or ["y_diff"])],
                "target_col": str(config.get("target_col") or str(candidate.get("family") or "HIGH")).upper(),
                "rolling_windows": [int(value) for value in (config.get("rolling_windows") or [2, 7, 30, 60])],
            }
        )
    return specs


@dataclass
class LoadedReplayModel:
    spec: Dict[str, Any]
    model_path: Path
    artifacts_path: Optional[Path]
    model: Any
    scaler_x: Any = None
    scaler_y: Any = None


class FrozenModelSignalProvider:
    """Load each current fitted artifact once and reuse it across replay ticks."""

    def __init__(self, specs: List[Dict[str, Any]], model_dir: Optional[Path] = None):
        self.model_dir = (model_dir or (PROJECT_ROOT / "model_files")).resolve()
        self.loaded: Dict[str, LoadedReplayModel] = {}
        self.errors: Dict[str, str] = {}
        self._torch = None
        self._torch_error = ""
        if any(str(spec.get("model") or "").lower() == "nbeats" for spec in specs):
            try:
                import torch

                self._torch = torch
            except Exception as exc:  # pragma: no cover - environment-specific
                self._torch_error = str(exc)
        for spec in specs:
            key = self.key(spec)
            try:
                self.loaded[key] = self._load(spec)
            except Exception as exc:
                self.errors[key] = str(exc)

    @staticmethod
    def key(spec: Dict[str, Any]) -> str:
        return f"{spec.get('family')}:{spec.get('timeframe')}"

    def _latest(self, pattern: str) -> Optional[Path]:
        files = sorted(self.model_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        return files[0] if files else None

    def _load(self, spec: Dict[str, Any]) -> LoadedReplayModel:
        model_name = str(spec.get("model") or "").strip().lower()
        family = str(spec.get("family") or "high").strip().lower()
        timeframe = str(spec.get("timeframe") or "").strip().lower()
        model_path = self._latest(f"{model_name}_{family}_{timeframe}_*.joblib")
        artifacts_path = self._latest(f"{model_name}_artifacts_{family}_{timeframe}_*.joblib")
        if model_path is None:
            raise FileNotFoundError(f"model artifact not found for {family}:{timeframe}:{model_name}")
        if model_name == "nbeats" and self._torch is None:
            raise RuntimeError(f"torch unavailable: {self._torch_error}")
        model = joblib.load(model_path)
        artifacts: Dict[str, Any] = {}
        if artifacts_path is not None:
            try:
                artifacts = joblib.load(artifacts_path) or {}
            except Exception:
                artifacts = {}
        scaler_x = artifacts.get("scaler_X") or (
            (artifacts.get("scalers") or {}).get("X") if isinstance(artifacts.get("scalers"), dict) else None
        )
        scaler_y = artifacts.get("scaler_y") or (
            (artifacts.get("scalers") or {}).get("y") if isinstance(artifacts.get("scalers"), dict) else None
        )
        return LoadedReplayModel(spec=dict(spec), model_path=model_path, artifacts_path=artifacts_path, model=model, scaler_x=scaler_x, scaler_y=scaler_y)

    @staticmethod
    def _confidence(rows: List[Dict[str, Any]], prediction: float) -> float:
        values = [_safe_float(row.get("y_diff")) for row in rows[-64:] if isinstance(row.get("y_diff"), (int, float))]
        scale = float(np.std(values)) if values else 1.0
        score = min(abs(float(prediction)) / max(scale, 1e-6), 4.0)
        return float(np.clip(0.5 + 0.12 * score, 0.5, 0.95))

    def predict(self, spec: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        key = self.key(spec)
        if key not in self.loaded:
            raise RuntimeError(self.errors.get(key) or f"model not loaded: {key}")
        loaded = self.loaded[key]
        n_steps = int(spec.get("n_steps", 1) or 1)
        if len(rows) < n_steps:
            raise ValueError(f"insufficient_rows: need={n_steps} got={len(rows)}")
        tail = rows[-n_steps:]
        features = [str(value) for value in (spec.get("input_features") or [])]
        initial = np.asarray([[float(row.get(feature, 0.0) or 0.0) for feature in features] for row in tail], dtype=np.float64)
        targets = [str(value) for value in (spec.get("target_features") or ["y_diff"])]
        model_name = str(spec.get("model") or "").lower()

        def predict_window(window: np.ndarray) -> np.ndarray:
            if model_name == "nbeats":
                model_input = window.reshape(1, -1)
                if loaded.scaler_x is not None:
                    model_input = loaded.scaler_x.transform(model_input)
                tensor = self._torch.tensor(model_input, dtype=self._torch.float32)
                with self._torch.no_grad():
                    prediction = loaded.model(tensor).cpu().numpy()
            else:
                model_input = window
                if loaded.scaler_x is not None:
                    model_input = loaded.scaler_x.transform(model_input)
                prediction = loaded.model.predict(model_input.reshape(1, -1))
            prediction = np.asarray(prediction)
            if loaded.scaler_y is not None:
                prediction = loaded.scaler_y.inverse_transform(prediction)
            return prediction.reshape(-1, len(targets))

        # Agent B consumes only the first forecast sign and confidence. Running
        # the full six-step recursive path at every five-minute replay tick is
        # unnecessary and makes a month-long evaluation several times slower;
        # the first row is identical to the first row of recursive inference.
        matrix = predict_window(initial)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, len(targets))
        matrix = matrix[:1]
        feature_horizons = {
            target: [float(value) for value in matrix[:, index].tolist()]
            for index, target in enumerate(targets)
        }
        primary = "y_diff" if "y_diff" in feature_horizons else targets[0]
        lead = float(feature_horizons[primary][0]) if feature_horizons.get(primary) else 0.0
        return {
            "raw_signal": 1 if lead > 0 else (-1 if lead < 0 else 0),
            "confidence": self._confidence(rows, lead),
            "forecast_sign": lead,
            "feature_horizons": feature_horizons,
            "input_window_start": str(tail[0].get("DATE") or ""),
            "input_window_end": str(tail[-1].get("DATE") or ""),
            "model_path": str(loaded.model_path),
            "artifacts_path": str(loaded.artifacts_path) if loaded.artifacts_path else None,
        }

    def manifest(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key, loaded in sorted(self.loaded.items()):
            rows.append(
                {
                    "key": key,
                    "family": loaded.spec.get("family"),
                    "timeframe": loaded.spec.get("timeframe"),
                    "model": loaded.spec.get("model"),
                    "r2_from_config_name": loaded.spec.get("r2"),
                    "config_path": loaded.spec.get("config_path"),
                    "model_path": str(loaded.model_path),
                    "model_modified_at_utc": datetime.fromtimestamp(loaded.model_path.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "artifacts_path": str(loaded.artifacts_path) if loaded.artifacts_path else None,
                }
            )
        for key, error in sorted(self.errors.items()):
            rows.append({"key": key, "load_error": error})
        return rows


class ReplaySignalEngine:
    def __init__(
        self,
        tape: AsOfMarketTape,
        specs: List[Dict[str, Any]],
        provider: Any,
        trading_cfg: Dict[str, Any],
    ):
        self.tape = tape
        self.specs = list(specs)
        self.provider = provider
        self.trading_cfg = trading_cfg
        self.interpretation = str(((trading_cfg.get("agent") or {}).get("signal_interpretation") or "momentum")).lower()
        worker_count = max(int(os.environ.get("TSMM_BACKTEST_INFERENCE_WORKERS", "4") or 4), 1)
        self._executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="tsmm-backtest-model")

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    def evaluate(self, as_of: pd.Timestamp) -> Dict[str, Any]:
        signals: Dict[str, Dict[str, Any]] = {}
        by_timeframe: Dict[str, pd.DataFrame] = {}
        by_family_frame: Dict[tuple[str, str], pd.DataFrame] = {}
        prepared: List[tuple[Dict[str, Any], str, str, str, List[Dict[str, Any]]]] = []
        for spec in self.specs:
            timeframe = str(spec.get("timeframe") or "")
            family = str(spec.get("family") or "").lower()
            key = f"{family}:{timeframe}"
            try:
                if timeframe not in by_timeframe:
                    max_rows = max(int(spec.get("n_steps", 1)) + max(spec.get("rolling_windows") or [1]) + 5, 200)
                    by_timeframe[timeframe] = self.tape.timeframe_as_of(timeframe, as_of, max_rows=max_rows)
                family_key = (timeframe, family)
                if family_key not in by_family_frame:
                    input_features = [str(value) for value in (spec.get("input_features") or [])]
                    by_family_frame[family_key] = _enrich_features(
                        by_timeframe[timeframe],
                        str(spec.get("target_col") or family).upper(),
                        list(spec.get("rolling_windows") or [2, 7, 30, 60]),
                        required_features=input_features,
                    )
                enriched = by_family_frame[family_key]
                input_features = [str(value) for value in (spec.get("input_features") or [])]
                n_steps = int(spec.get("n_steps", 1) or 1)
                if len(enriched) < n_steps:
                    raise ValueError(f"insufficient enriched rows: need={n_steps} got={len(enriched)}")
                columns = list(dict.fromkeys(["DATE", "y_diff"] + input_features))
                rows = []
                for _, record in enriched[columns].tail(n_steps).iterrows():
                    item: Dict[str, Any] = {"DATE": pd.Timestamp(record["DATE"]).strftime("%Y-%m-%d %H:%M:%S")}
                    for column in columns:
                        if column != "DATE":
                            item[column] = float(record[column])
                    rows.append(item)
                prepared.append((spec, key, family, timeframe, rows))
            except Exception as exc:
                signals[key] = {
                    "family": family,
                    "timeframe": timeframe,
                    "model": spec.get("model"),
                    "signal": 0,
                    "confidence": 0.0,
                    "vote_weight": 0.0,
                    "error": str(exc),
                }

        def _predict(item: tuple[Dict[str, Any], str, str, str, List[Dict[str, Any]]]) -> tuple[Any, ...]:
            spec, key, family, timeframe, rows = item
            try:
                return spec, key, family, timeframe, self.provider.predict(spec, rows), None
            except Exception as exc:
                return spec, key, family, timeframe, None, str(exc)

        weighted = 0.0
        total_weight = 0.0
        for spec, key, family, timeframe, prediction, prediction_error in self._executor.map(_predict, prepared):
            if prediction_error:
                signals[key] = {
                    "family": family,
                    "timeframe": timeframe,
                    "model": spec.get("model"),
                    "signal": 0,
                    "confidence": 0.0,
                    "vote_weight": 0.0,
                    "error": prediction_error,
                }
                continue
            try:
                raw_signal = int(prediction.get("raw_signal", 0) or 0)
                signal = -raw_signal if self.interpretation in {"contrarian", "mean_reversion", "mean-reversion", "fade"} else raw_signal
                confidence = float(prediction.get("confidence", 0.5) or 0.5)
                vote_weight = max(confidence, 0.01) * float(TIMEFRAME_WEIGHTS.get(timeframe, 1.0))
                signals[key] = {
                    "family": family,
                    "timeframe": timeframe,
                    "model": spec.get("model"),
                    "r2": spec.get("r2"),
                    "signal": signal,
                    "raw_signal": raw_signal,
                    "confidence": confidence,
                    "vote_weight": vote_weight,
                    "forecast_sign": prediction.get("forecast_sign"),
                    "feature_horizons": prediction.get("feature_horizons") or {},
                    "input_window_start": prediction.get("input_window_start"),
                    "input_window_end": prediction.get("input_window_end"),
                }
                weighted += vote_weight * signal
                total_weight += vote_weight
            except Exception as exc:
                signals[key] = {
                    "family": family,
                    "timeframe": timeframe,
                    "model": spec.get("model"),
                    "signal": 0,
                    "confidence": 0.0,
                    "vote_weight": 0.0,
                    "error": str(exc),
                }

        score = weighted / total_weight if total_weight > 0 else 0.0
        consensus = "buy" if score > 0.1 else ("sell" if score < -0.1 else "hold")
        usable = [signal for signal in signals.values() if not signal.get("error")]
        return {
            "as_of_utc": pd.Timestamp(as_of).strftime("%Y-%m-%d %H:%M:%S"),
            "consensus": consensus,
            "consensus_score": float(score),
            "avg_confidence": float(np.mean([float(value.get("confidence", 0.5)) for value in usable])) if usable else 0.0,
            "n_signals": len(signals),
            "n_usable": len(usable),
            "coverage": float(len(usable) / max(len(signals), 1)),
            "signals": signals,
        }


def _current_session(local_time: datetime, trading_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    windows = ((trading_cfg.get("autonomous_trading") or {}).get("session_windows") or [])
    for raw in windows:
        start_hour, start_minute = _parse_hhmm(raw.get("start"))
        end_hour, end_minute = _parse_hhmm(raw.get("end"))
        for day_offset in (0, -1):
            day = (local_time + timedelta(days=day_offset)).date()
            start = datetime(day.year, day.month, day.day, start_hour, start_minute, tzinfo=local_time.tzinfo)
            end_day = day + timedelta(days=1) if (end_hour, end_minute) <= (start_hour, start_minute) else day
            end = datetime(end_day.year, end_day.month, end_day.day, end_hour, end_minute, tzinfo=local_time.tzinfo)
            if start <= local_time < end:
                return {
                    "name": str(raw.get("name") or "session"),
                    "start_local": start,
                    "end_local": end,
                    "session_id": f"{str(raw.get('name') or 'session')}:{start.strftime('%Y%m%d%H%M')}",
                }
    return None


def _build_plan(
    bundle: Dict[str, Any],
    primary_key: str,
    entry: float,
    trading_cfg: Dict[str, Any],
    account_equity: float,
    contract_size: float,
) -> Dict[str, Any]:
    signals = bundle.get("signals") or {}
    primary = dict(signals.get(primary_key) or {})
    if primary.get("error") or int(primary.get("signal", 0) or 0) == 0:
        return {"decision": "hold", "reason": f"primary_signal_unavailable:{primary_key}", "entry": float(entry)}
    if float(bundle.get("coverage", 0.0) or 0.0) < 0.70:
        return {"decision": "hold", "reason": "model_coverage_below_70pct", "entry": float(entry)}

    decision = "buy" if int(primary.get("signal", 0)) > 0 else "sell"
    consensus = str(bundle.get("consensus") or "hold")
    consensus_score = float(bundle.get("consensus_score", 0.0) or 0.0)
    alignment = "aligned" if consensus == decision else ("opposed" if consensus in {"buy", "sell"} else "neutral")
    if alignment == "opposed" and abs(consensus_score) >= 0.2:
        decision = consensus
        alignment = "overridden"

    mode_a = trading_cfg.get("mode_a") or {}
    if decision == "buy" and not bool(mode_a.get("allow_long", True)):
        return {"decision": "hold", "reason": "long_disabled", "entry": float(entry)}
    if decision == "sell" and not bool(mode_a.get("allow_short", True)):
        return {"decision": "hold", "reason": "short_disabled", "entry": float(entry)}

    base_confidence = float(primary.get("confidence", 0.5) or 0.5)
    avg_confidence = float(bundle.get("avg_confidence", 0.5) or 0.5)
    confidence = float(np.clip(0.7 * base_confidence + 0.3 * avg_confidence, 0.0, 1.0))
    risk = trading_cfg.get("risk") or {}
    conviction_cfg = trading_cfg.get("conviction") or {}

    # This reproduces the deterministic portion of production conviction.  The
    # live memory term is intentionally excluded because today's memory would
    # leak later trade outcomes into an older timestamp.
    cm_proxy = float(risk.get("min_cm_accuracy_to_trade", 0.5) or 0.5)
    fooling_proxy = float(risk.get("max_input_fooling_risk", 0.45) or 0.45)
    signal_factor = 0.15 * max(0.0, min(1.0, (confidence - 0.40) / 0.30))
    signal_factor += 0.10 * max(0.0, min(1.0, (cm_proxy - 0.40) / 0.30))
    signal_factor += 0.10 * max(0.0, min(1.0, (0.50 - 0.35) / 0.30))
    signal_factor += 0.10 * max(0.0, min(1.0, (1.0 - fooling_proxy) / 0.30))
    enrich_factor = 0.12 * abs(consensus_score)
    enrich_factor += 0.10 * max(0.0, min(1.0, (avg_confidence - 0.45) / 0.25))
    if consensus == decision:
        enrich_factor += 0.08
    enrich_factor = max(0.0, min(0.30, enrich_factor))
    breadth = max(0.0, min(0.10, int(bundle.get("n_usable", 0) or 0) / 240.0))
    conviction = min(1.0, signal_factor + enrich_factor + breadth)

    thresholds = [
        ("standard", float(conviction_cfg.get("min_conviction_for_no_sl", 0.80) or 0.80), 1.0, 1.0, 1.0),
        ("standard", float(conviction_cfg.get("min_conviction_for_wide", 0.65) or 0.65), 1.0, 1.0, 0.75),
        ("standard", float(conviction_cfg.get("min_conviction_for_standard", 0.45) or 0.45), 1.0, 1.0, 0.5),
        ("tight", float(conviction_cfg.get("min_conviction_for_tight", 0.30) or 0.30), 1.0, 1.0, 0.25),
    ]
    risk_mode, sl_multiplier, tp_multiplier, volume_multiplier = "skip", 0.0, 0.0, 0.0
    for candidate, threshold, sl_mult, tp_mult, vol_mult in thresholds:
        if conviction >= threshold:
            risk_mode, sl_multiplier, tp_multiplier, volume_multiplier = candidate, sl_mult, tp_mult, vol_mult
            break
    if risk_mode == "skip":
        return {
            "decision": "hold",
            "reason": "conviction_below_entry_threshold",
            "entry": float(entry),
            "conviction": float(conviction),
        }

    sl_pct = float(risk.get("stop_loss_pct", 0.8) or 0.8) / 100.0
    tp_pct = float(risk.get("take_profit_pct", 1.6) or 1.6) / 100.0
    if decision == "buy":
        stop_loss = entry * (1.0 - sl_pct * sl_multiplier)
        take_profit = entry * (1.0 + tp_pct * tp_multiplier)
    else:
        stop_loss = entry * (1.0 + sl_pct * sl_multiplier)
        take_profit = entry * (1.0 - tp_pct * tp_multiplier)
    base_volume = float(((trading_cfg.get("execution") or {}).get("default_volume", 0.01)) or 0.01)
    requested_volume = max(round(base_volume * volume_multiplier, 6), 0.0)
    sizing_cfg = dict(risk.get("account_sizing") or {})
    if bool(sizing_cfg.get("enabled", False)):
        allowance = max(float(account_equity), 0.0) * float(risk.get("risk_per_trade_pct", 0.5) or 0.5) / 100.0
        loss_per_lot = abs(float(entry) - float(stop_loss)) * float(contract_size)
        raw_volume = allowance / loss_per_lot if loss_per_lot > 0.0 else 0.0
        volume_step = float(sizing_cfg.get("simulation_volume_step", 0.01) or 0.01)
        volume_min = float(sizing_cfg.get("simulation_volume_min", 0.01) or 0.01)
        sized_volume = int((raw_volume + 1e-12) / volume_step) * volume_step
        volume = min(requested_volume, round(sized_volume, 8))
        if volume < volume_min:
            return {
                "decision": "hold", "reason": "minimum_broker_volume_exceeds_risk_budget",
                "entry": float(entry), "raw_risk_sized_volume": raw_volume,
                "risk_allowance": allowance,
            }
    else:
        volume = requested_volume
    return {
        "decision": decision,
        "entry": float(entry),
        "stop_loss": float(stop_loss) if stop_loss is not None else None,
        "take_profit": float(take_profit),
        "volume": float(volume),
        "confidence": confidence,
        "consensus": consensus,
        "consensus_score": consensus_score,
        "alignment": alignment,
        "conviction": float(conviction),
        "risk_mode": risk_mode,
        "primary_key": primary_key,
        "account_equity": float(account_equity),
    }


def _adverse_fill(price: float, side: str, entry: bool, cost_rate: float) -> float:
    if side == "buy":
        return float(price) * (1.0 + cost_rate if entry else 1.0 - cost_rate)
    return float(price) * (1.0 - cost_rate if entry else 1.0 + cost_rate)


def _pending_triggered(order: Dict[str, Any], bar: pd.Series) -> bool:
    entry = float(order["planned_entry"])
    order_type = str(order.get("order_type") or "")
    if order_type in {"buy_stop", "sell_limit"}:
        return float(bar["HIGH"]) >= entry
    return float(bar["LOW"]) <= entry


def _intrabar_exit(position: Dict[str, Any], bar: pd.Series) -> Optional[tuple[str, float]]:
    side = str(position["side"])
    stop = position.get("stop_loss")
    take = position.get("take_profit")
    low = float(bar["LOW"])
    high = float(bar["HIGH"])
    # When both barriers are crossed in the same minute, assume the stop was
    # touched first.  This is conservative and is recorded in report policy.
    if side == "buy":
        if stop is not None and low <= float(stop):
            return "stop_loss", float(stop)
        if take is not None and high >= float(take):
            return "take_profit", float(take)
    else:
        if stop is not None and high >= float(stop):
            return "stop_loss", float(stop)
        if take is not None and low <= float(take):
            return "take_profit", float(take)
    return None


def _max_drawdown_pct(equity_points: List[Dict[str, Any]]) -> float:
    if not equity_points:
        return 0.0
    values = np.asarray([float(point["equity"]) for point in equity_points], dtype=float)
    peaks = np.maximum.accumulate(values)
    drawdowns = (values - peaks) / np.maximum(peaks, 1e-9)
    return float(abs(np.min(drawdowns)) * 100.0)


def _report_markdown(summary: Dict[str, Any]) -> str:
    overall = summary.get("overall") or {}
    validity = summary.get("validity") or {}
    lines = [
        "# TSMM Trading Strategy Evaluation",
        "",
        f"Generated: {summary.get('generated_at_utc')} UTC  ",
        f"Period: {summary.get('period', {}).get('start_utc')} to {summary.get('period', {}).get('end_utc')} UTC  ",
        f"Result grade: **{validity.get('result_grade')}**",
        "",
        "## Important interpretation",
        "",
        str(validity.get("plain_language_warning") or ""),
        "",
        "Market rows were replayed point-in-time, and unfinished timeframe candles contained only minutes already available at each decision. "
        "The live-memory conviction term, LLM commentary, approval latency, opposing countertrades, and broker stop-distance normalization are excluded.",
        "",
        "## Overall performance",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Closed trades | {overall.get('n_trades', 0)} |",
        f"| Win rate | {100.0 * float(overall.get('win_rate', 0.0)):.2f}% |",
        f"| Net P/L | {float(overall.get('net_pnl', 0.0)):.2f} |",
        f"| Return | {float(overall.get('total_return_pct', 0.0)):.3f}% |",
        f"| Max drawdown | {float(overall.get('max_drawdown_pct', 0.0)):.3f}% |",
        f"| Profit factor | {overall.get('profit_factor')} |",
        f"| Ending equity | {float(overall.get('ending_equity', 0.0)):.2f} |",
        f"| Signal coverage | {100.0 * float(overall.get('signal_coverage', 0.0)):.2f}% |",
        "",
        "## Daily operations",
        "",
        "| Day | Attempts | Trades | Wins | Losses | Net P/L | End equity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for day in summary.get("daily") or []:
        lines.append(
            f"| {day.get('date_local')} | {day.get('entry_attempts', 0)} | {day.get('n_trades', 0)} | "
            f"{day.get('wins', 0)} | {day.get('losses', 0)} | {float(day.get('net_pnl', 0.0)):.2f} | "
            f"{float(day.get('end_equity', 0.0)):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Trade lifecycle",
            "",
            "| ID | Session | Trigger | Side | Entry (UTC) | Exit (UTC) | Exit reason | Net P/L | MAE | MFE |",
            "|---|---|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for trade in summary.get("operations") or []:
        lines.append(
            f"| {trade.get('operation_id')} | {trade.get('session_name')} | {trade.get('trigger')} | {trade.get('side')} | "
            f"{trade.get('entry_time_utc')} | {trade.get('exit_time_utc')} | {trade.get('exit_reason')} | "
            f"{float(trade.get('net_pnl', 0.0)):.2f} | {float(trade.get('mae_abs', 0.0)):.2f} | {float(trade.get('mfe_abs', 0.0)):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Execution assumptions",
            "",
            "- Latest model artifacts are frozen for the entire replay.",
            "- Agent B forecasts and pending-order maintenance run at the configured poll interval.",
            "- Programmed entries fill when a later one-minute candle crosses the requested price.",
            "- Spread and slippage are charged adversely on entry and exit.",
            "- If stop and take-profit are both crossed within one minute, the stop is assumed first.",
            "- Base-session extensions are treated as rejected, so remaining positions close at the configured session deadline.",
            "",
        ]
    )
    return "\n".join(lines)


def _finalize_outputs(
    output_dir: Path,
    summary: Dict[str, Any],
    attempts: List[Dict[str, Any]],
    signal_rows: List[Dict[str, Any]],
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    operations_path = output_dir / "operations.csv"
    attempts_path = output_dir / "entry_attempts.csv"
    daily_path = output_dir / "daily_summary.csv"
    signals_path = output_dir / "signal_timeline.csv"
    report_path = output_dir / "report.md"
    _write_json_atomic(summary_path, summary)
    pd.DataFrame(summary.get("operations") or []).to_csv(operations_path, index=False)
    pd.DataFrame(attempts).to_csv(attempts_path, index=False)
    pd.DataFrame(summary.get("daily") or []).to_csv(daily_path, index=False)
    pd.DataFrame(signal_rows).to_csv(signals_path, index=False)
    report_path.write_text(_report_markdown(summary), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "operations_path": str(operations_path),
        "attempts_path": str(attempts_path),
        "daily_path": str(daily_path),
        "signals_path": str(signals_path),
        "report_path": str(report_path),
    }


def run_historical_strategy_backtest(
    *,
    market_source: str,
    trading_cfg: Dict[str, Any],
    output_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    previous_month: bool = False,
    initial_balance: float = 100000.0,
    contract_size: float = 100.0,
    poll_minutes: Optional[int] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    tick_progress_cb: Optional[Callable[[int, int, str], None]] = None,
    specs: Optional[List[Dict[str, Any]]] = None,
    provider: Any = None,
    market_frame: Optional[pd.DataFrame] = None,
    max_ticks: Optional[int] = None,
) -> Dict[str, Any]:
    """Run accelerated TSMM strategy replay and write audit-friendly reports."""
    autonomy = trading_cfg.get("autonomous_trading") or {}
    timezone_name = str(autonomy.get("timezone") or ((trading_cfg.get("agent") or {}).get("timezone") or "UTC"))
    start, end = normalize_period(start_date, end_date, previous_month=previous_month, timezone_name=timezone_name)
    symbol = str(((trading_cfg.get("execution") or {}).get("symbol") or "XAUUSD"))
    replay_specs = list(specs or discover_replay_model_specs(trading_cfg))
    if not replay_specs:
        return {"ok": False, "error": "No replay model configurations were discovered"}

    max_tf = max(_timeframe_minutes(str(spec.get("timeframe") or "1m")) for spec in replay_specs)
    max_window = max(max(spec.get("rolling_windows") or [1]) + int(spec.get("n_steps", 1) or 1) + 5 for spec in replay_specs)
    warmup_start = start - pd.Timedelta(minutes=max_tf * max_window)
    minutes = market_frame.copy() if market_frame is not None else _load_market_minutes(market_source, symbol, warmup_start, end)
    minutes["DATE"] = pd.to_datetime(minutes["DATE"], errors="coerce").dt.tz_localize(None)
    minutes = minutes.dropna(subset=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]).sort_values("DATE").reset_index(drop=True)
    period_minutes = minutes[(minutes["DATE"] >= start) & (minutes["DATE"] <= end)].copy()
    if period_minutes.empty:
        return {"ok": False, "error": f"No market rows in requested period {start} through {end}"}

    timeframes = sorted(set(str(spec.get("timeframe") or "") for spec in replay_specs))
    tape = AsOfMarketTape(minutes, timeframes)
    signal_provider = provider or FrozenModelSignalProvider(replay_specs)
    engine = ReplaySignalEngine(tape, replay_specs, signal_provider, trading_cfg)
    provider_manifest = signal_provider.manifest() if hasattr(signal_provider, "manifest") else []

    configured_poll = max(int(((trading_cfg.get("mode_b") or {}).get("poll_seconds", 300) or 300) / 60), 1)
    tick_minutes = max(int(poll_minutes or configured_poll), 1)
    risk = trading_cfg.get("risk") or {}
    execution = trading_cfg.get("execution") or {}
    trading_job = trading_cfg.get("trading_job") or {}
    session_hours = float(trading_job.get("session_hours", 7.0) or 7.0)
    cost_rate = (float(execution.get("spread_bps", 2.0) or 2.0) + float(execution.get("slippage_bps", 2.0) or 2.0)) / 10000.0
    commission = float(execution.get("commission_per_trade", 0.0) or 0.0)
    expiration_minutes = int(trading_job.get("programmed_order_expiration_minutes", 420) or 420)
    max_positions = int(risk.get("max_open_positions", 5) or 5)
    daily_loss_limit = float(risk.get("daily_max_loss_pct", 2.0) or 2.0)
    weekly_loss_limit = float(risk.get("weekly_max_loss_pct", 5.0) or 5.0)
    max_drawdown_guard = float(risk.get("max_drawdown_pct", 15.0) or 15.0)
    close_threshold = abs(float(((trading_cfg.get("mode_b") or {}).get("close_consensus_threshold", 0.25) or 0.25)))
    cancel_threshold = abs(float(((autonomy.get("pending_order_maintenance") or {}).get("cancel_opposed_consensus_threshold", close_threshold) or close_threshold)))
    session_capacity = min(
        int(autonomy.get("max_jobs_per_session", 3) or 3),
        int(((trading_cfg.get("mode_a") or {}).get("max_operations_per_session", 3) or 3)),
    )
    max_followups = int(autonomy.get("max_followup_launches_per_session", session_capacity) or session_capacity)
    followup_cooldown = int(autonomy.get("followup_cooldown_seconds", 600) or 600)
    followup_enabled = bool(autonomy.get("followup_enabled", True))

    local_zone = ZoneInfo(timezone_name)
    pending: List[Dict[str, Any]] = []
    positions: List[Dict[str, Any]] = []
    operations: List[Dict[str, Any]] = []
    attempts: List[Dict[str, Any]] = []
    signal_rows: List[Dict[str, Any]] = []
    equity_points: List[Dict[str, Any]] = [{"timestamp_utc": start.strftime("%Y-%m-%d %H:%M:%S"), "equity": float(initial_balance)}]
    balance = float(initial_balance)
    operation_seq = 0
    order_seq = 0
    sessions: Dict[str, Dict[str, Any]] = {}
    last_bundle: Optional[Dict[str, Any]] = None
    tick_count = 0
    daily_realized: Dict[str, float] = {}
    weekly_realized: Dict[str, float] = {}
    progress_days = sorted(
        set(
            pd.Timestamp(value).tz_localize("UTC").tz_convert(local_zone).date()
            for value in period_minutes["DATE"]
        )
    )
    day_index = {day: index + 1 for index, day in enumerate(progress_days)}
    progress_seen: set[Any] = set()
    period_minute_numbers = period_minutes["DATE"].astype("int64") // (60 * 1_000_000_000)
    total_ticks = int((period_minute_numbers % tick_minutes == 0).sum())
    if max_ticks is not None:
        total_ticks = min(total_ticks, max(int(max_ticks), 0))
    if total_ticks <= 0:
        engine.close()
        return {"ok": False, "error": "No replay ticks are available in the requested period"}
    if tick_progress_cb:
        tick_progress_cb(0, total_ticks, period_minutes.iloc[0]["DATE"].strftime("%Y-%m-%d %H:%M:%S"))

    def local_label(timestamp: pd.Timestamp) -> str:
        aware = pd.Timestamp(timestamp).tz_localize("UTC").tz_convert(local_zone)
        return aware.strftime("%Y-%m-%d")

    def local_week_label(timestamp: pd.Timestamp) -> str:
        aware = pd.Timestamp(timestamp).tz_localize("UTC").tz_convert(local_zone)
        iso = aware.isocalendar()
        return f"{int(iso.year):04d}-W{int(iso.week):02d}"

    def close_position(position: Dict[str, Any], timestamp: pd.Timestamp, exit_price: float, reason: str) -> None:
        nonlocal balance
        side = str(position["side"])
        effective_exit = _adverse_fill(float(exit_price), side, False, cost_rate)
        gross = ((effective_exit - float(position["entry_price"])) if side == "buy" else (float(position["entry_price"]) - effective_exit))
        gross *= float(position["volume"]) * float(contract_size)
        net = gross - commission
        balance_before = balance
        balance += net
        operation = {
            **{key: value for key, value in position.items() if not key.startswith("_")},
            "exit_time_utc": pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "exit_price": float(effective_exit),
            "exit_reason": reason,
            "gross_pnl": float(gross),
            "commission": float(commission),
            "net_pnl": float(net),
            "return_on_balance_pct": float(net / max(balance_before, 1e-9) * 100.0),
            "duration_minutes": int((pd.Timestamp(timestamp) - pd.Timestamp(position["_entry_ts"])).total_seconds() / 60),
            "mae_abs": float(position.get("_mae_abs", 0.0)),
            "mfe_abs": float(position.get("_mfe_abs", 0.0)),
            "balance_after": float(balance),
        }
        operations.append(operation)
        day = local_label(timestamp)
        daily_realized[day] = daily_realized.get(day, 0.0) + net
        week = local_week_label(timestamp)
        weekly_realized[week] = weekly_realized.get(week, 0.0) + net
        equity_points.append({"timestamp_utc": pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"), "equity": float(balance)})
        state = sessions.get(str(position.get("session_id")))
        if state is not None:
            state["last_terminal_ts"] = pd.Timestamp(timestamp)

    def open_position(order: Dict[str, Any], timestamp: pd.Timestamp, raw_price: float) -> None:
        nonlocal operation_seq
        operation_seq += 1
        side = str(order["side"])
        effective_entry = _adverse_fill(float(raw_price), side, True, cost_rate)
        position = {
            "operation_id": f"op_{operation_seq:05d}",
            "order_id": order.get("order_id"),
            "session_id": order.get("session_id"),
            "session_name": order.get("session_name"),
            "trigger": order.get("trigger"),
            "submission_mode": order.get("submission_mode"),
            "side": side,
            "order_created_time_utc": order.get("created_time_utc"),
            "entry_time_utc": pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "planned_entry": float(order.get("planned_entry", raw_price)),
            "entry_price": float(effective_entry),
            "stop_loss": order.get("stop_loss"),
            "take_profit": order.get("take_profit"),
            "volume": float(order.get("volume", 0.01)),
            "confidence": float(order.get("confidence", 0.0)),
            "consensus_score_at_entry": float(order.get("consensus_score", 0.0)),
            "conviction": float(order.get("conviction", 0.0)),
            "risk_mode": order.get("risk_mode"),
            "_entry_ts": pd.Timestamp(timestamp),
            "_deadline": pd.Timestamp(order["deadline"]),
            "_mae_abs": 0.0,
            "_mfe_abs": 0.0,
        }
        positions.append(position)

    def create_entry(
        timestamp: pd.Timestamp,
        session: Dict[str, Any],
        trigger: str,
        primary_key: str,
        submission_mode: str,
        bundle: Dict[str, Any],
        close_price: float,
    ) -> None:
        nonlocal order_seq
        session_state = sessions[session["session_id"]]
        day = local_label(timestamp)
        week = local_week_label(timestamp)
        drawdown = _max_drawdown_pct(equity_points)
        blocked_reason = ""
        if len(positions) >= max_positions:
            blocked_reason = "max_open_positions"
        elif daily_realized.get(day, 0.0) <= -(initial_balance * daily_loss_limit / 100.0):
            blocked_reason = "daily_loss_guard"
        elif weekly_realized.get(week, 0.0) <= -(initial_balance * weekly_loss_limit / 100.0):
            blocked_reason = "weekly_loss_guard"
        elif drawdown >= max_drawdown_guard:
            blocked_reason = "max_drawdown_guard"

        entry_anchor = float(close_price)
        tf_label = primary_key.split(":", 1)[1]
        tf_frame = tape.timeframe_as_of(tf_label, timestamp, max_rows=2)
        if not tf_frame.empty:
            entry_anchor = float(tf_frame.iloc[-1]["HIGH"])
        plan = _build_plan(bundle, primary_key, entry_anchor, trading_cfg, balance, contract_size)
        if blocked_reason:
            plan = {**plan, "decision": "hold", "reason": blocked_reason}
        attempt = {
            "timestamp_utc": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "date_local": day,
            "session_id": session["session_id"],
            "session_name": session["name"],
            "trigger": trigger,
            "submission_mode": submission_mode,
            "primary_key": primary_key,
            "decision": plan.get("decision"),
            "reason": plan.get("reason", "entry_created"),
            "planned_entry": plan.get("entry"),
            "confidence": plan.get("confidence"),
            "consensus": bundle.get("consensus"),
            "consensus_score": bundle.get("consensus_score"),
            "conviction": plan.get("conviction"),
            "risk_mode": plan.get("risk_mode"),
            "model_coverage": bundle.get("coverage"),
        }
        attempts.append(attempt)
        session_state["launches"] += 1
        if trigger == "followup":
            session_state["followups"] += 1
        if str(plan.get("decision")) not in {"buy", "sell"}:
            session_state["last_terminal_ts"] = timestamp
            return

        order_seq += 1
        side = str(plan["decision"])
        base = {
            "order_id": f"order_{order_seq:05d}",
            "session_id": session["session_id"],
            "session_name": session["name"],
            "trigger": trigger,
            "submission_mode": submission_mode,
            "side": side,
            "created_time_utc": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "planned_entry": float(plan["entry"]),
            "stop_loss": plan.get("stop_loss"),
            "take_profit": plan.get("take_profit"),
            "volume": float(plan.get("volume", 0.01)),
            "confidence": float(plan.get("confidence", 0.5)),
            "consensus_score": float(plan.get("consensus_score", 0.0)),
            "conviction": float(plan.get("conviction", 0.0)),
            "risk_mode": plan.get("risk_mode"),
            "deadline": timestamp + pd.Timedelta(hours=session_hours),
        }
        if submission_mode == "market":
            # A market follow-up uses the contemporaneous close and recalculates
            # the risk distances around the actual fill anchor.
            distance_sl = abs(float(plan["entry"]) - float(plan.get("stop_loss") or plan["entry"]))
            distance_tp = abs(float(plan["entry"]) - float(plan.get("take_profit") or plan["entry"]))
            base["planned_entry"] = float(close_price)
            if side == "buy":
                base["stop_loss"] = float(close_price - distance_sl) if plan.get("stop_loss") is not None else None
                base["take_profit"] = float(close_price + distance_tp)
            else:
                base["stop_loss"] = float(close_price + distance_sl) if plan.get("stop_loss") is not None else None
                base["take_profit"] = float(close_price - distance_tp)
            open_position(base, timestamp, close_price)
            return

        current = float(close_price)
        if side == "buy":
            order_type = "buy_limit" if float(plan["entry"]) <= current else "buy_stop"
        else:
            order_type = "sell_limit" if float(plan["entry"]) >= current else "sell_stop"
        base["order_type"] = order_type
        base["expires_at"] = timestamp + pd.Timedelta(minutes=expiration_minutes)
        pending.append(base)

    for _, minute_bar in period_minutes.iterrows():
        timestamp = pd.Timestamp(minute_bar["DATE"])
        local_time = timestamp.tz_localize("UTC").tz_convert(local_zone).to_pydatetime()
        local_day = local_time.date()
        if local_day not in progress_seen:
            progress_seen.add(local_day)
            if progress_cb:
                progress_cb(day_index.get(local_day, len(progress_seen)), len(progress_days), str(local_day))

        # Pending orders can fill only after creation; the current minute's full
        # high/low is then legitimately known to the accelerated broker replay.
        for order in list(pending):
            created = pd.Timestamp(order["created_time_utc"])
            if timestamp <= created:
                continue
            if timestamp >= pd.Timestamp(order["expires_at"]):
                pending.remove(order)
                state = sessions.get(str(order.get("session_id")))
                if state is not None:
                    state["last_terminal_ts"] = timestamp
                attempts.append(
                    {
                        "timestamp_utc": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "date_local": local_label(timestamp),
                        "session_id": order.get("session_id"),
                        "session_name": order.get("session_name"),
                        "trigger": "pending_terminal",
                        "submission_mode": "programmed",
                        "primary_key": "",
                        "decision": order.get("side"),
                        "reason": "order_not_filled_before_expiration",
                        "planned_entry": order.get("planned_entry"),
                    }
                )
                continue
            if _pending_triggered(order, minute_bar):
                pending.remove(order)
                open_position(order, timestamp, float(order["planned_entry"]))

        for position in list(positions):
            side = str(position["side"])
            entry_price = float(position["entry_price"])
            if side == "buy":
                adverse = max(entry_price - float(minute_bar["LOW"]), 0.0)
                favorable = max(float(minute_bar["HIGH"]) - entry_price, 0.0)
            else:
                adverse = max(float(minute_bar["HIGH"]) - entry_price, 0.0)
                favorable = max(entry_price - float(minute_bar["LOW"]), 0.0)
            multiplier = float(position["volume"]) * float(contract_size)
            position["_mae_abs"] = max(float(position.get("_mae_abs", 0.0)), adverse * multiplier)
            position["_mfe_abs"] = max(float(position.get("_mfe_abs", 0.0)), favorable * multiplier)
            exit_hit = _intrabar_exit(position, minute_bar)
            if exit_hit:
                positions.remove(position)
                close_position(position, timestamp, exit_hit[1], exit_hit[0])

        minute_number = int(timestamp.value // (60 * 1_000_000_000))
        is_tick = minute_number % tick_minutes == 0
        if not is_tick:
            continue
        if max_ticks is not None and tick_count >= max_ticks:
            break
        tick_count += 1
        bundle = engine.evaluate(timestamp)
        last_bundle = bundle
        signal_rows.append(
            {
                "timestamp_utc": bundle.get("as_of_utc"),
                "consensus": bundle.get("consensus"),
                "consensus_score": bundle.get("consensus_score"),
                "avg_confidence": bundle.get("avg_confidence"),
                "n_signals": bundle.get("n_signals"),
                "n_usable": bundle.get("n_usable"),
                "coverage": bundle.get("coverage"),
                "error_count": int(bundle.get("n_signals", 0)) - int(bundle.get("n_usable", 0)),
            }
        )
        if tick_progress_cb:
            tick_progress_cb(tick_count, total_ticks, timestamp.strftime("%Y-%m-%d %H:%M:%S"))

        for order in list(pending):
            side = str(order["side"])
            consensus = str(bundle.get("consensus") or "hold")
            score = float(bundle.get("consensus_score", 0.0) or 0.0)
            opposed = (side == "buy" and consensus == "sell" and score <= -cancel_threshold) or (
                side == "sell" and consensus == "buy" and score >= cancel_threshold
            )
            if opposed:
                pending.remove(order)
                state = sessions.get(str(order.get("session_id")))
                if state is not None:
                    state["last_terminal_ts"] = timestamp
                attempts.append(
                    {
                        "timestamp_utc": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "date_local": local_label(timestamp),
                        "session_id": order.get("session_id"),
                        "session_name": order.get("session_name"),
                        "trigger": "pending_terminal",
                        "submission_mode": "programmed",
                        "primary_key": "",
                        "decision": side,
                        "reason": "programmed_order_consensus_invalidated",
                        "planned_entry": order.get("planned_entry"),
                        "consensus": consensus,
                        "consensus_score": score,
                    }
                )

        for position in list(positions):
            side = str(position["side"])
            consensus = str(bundle.get("consensus") or "hold")
            score = float(bundle.get("consensus_score", 0.0) or 0.0)
            should_close = (side == "buy" and consensus == "sell" and score <= -close_threshold) or (
                side == "sell" and consensus == "buy" and score >= close_threshold
            )
            reason = ""
            if should_close:
                reason = f"mode_b_consensus_close({consensus},{score:.3f})"
            elif timestamp >= pd.Timestamp(position["_deadline"]):
                reason = "session_extension_not_approved"
            if reason:
                positions.remove(position)
                close_position(position, timestamp, float(minute_bar["CLOSE"]), reason)
                continue

            trailing = risk.get("trailing") or {}
            if bool(trailing.get("enabled", True)) and consensus == side:
                entry = float(position["entry_price"])
                current = float(minute_bar["CLOSE"])
                target = float(position.get("take_profit") or entry)
                activation = float(trailing.get("breakeven_activation_ratio", 0.75) or 0.75)
                favorable = (current - entry) if side == "buy" else (entry - current)
                target_distance = abs(target - entry)
                if target_distance > 0 and favorable >= activation * target_distance:
                    trail_pct = float(trailing.get("trail_pct_base", 0.5) or 0.5) / 100.0
                    if side == "buy":
                        candidate = max(entry, current * (1.0 - trail_pct))
                        position["stop_loss"] = max(float(position.get("stop_loss") or candidate), candidate)
                    else:
                        candidate = min(entry, current * (1.0 + trail_pct))
                        position["stop_loss"] = min(float(position.get("stop_loss") or candidate), candidate)

        session = _current_session(local_time, trading_cfg)
        if session is None:
            continue
        session_id = str(session["session_id"])
        if session_id not in sessions:
            sessions[session_id] = {
                **session,
                "launches": 0,
                "followups": 0,
                "last_terminal_ts": None,
            }
        state = sessions[session_id]
        session_pending = any(str(order.get("session_id")) == session_id for order in pending)
        session_positions = any(str(position.get("session_id")) == session_id for position in positions)
        if int(state["launches"]) == 0:
            create_entry(timestamp, session, "mandatory_session", "high:7h", "programmed", bundle, float(minute_bar["CLOSE"]))
            continue
        if not followup_enabled or session_pending or session_positions:
            continue
        last_terminal = state.get("last_terminal_ts")
        if last_terminal is None or (timestamp - pd.Timestamp(last_terminal)).total_seconds() < followup_cooldown:
            continue
        if int(state["launches"]) >= session_capacity or int(state["followups"]) >= max_followups:
            continue
        if str(bundle.get("consensus")) not in {"buy", "sell"}:
            continue
        create_entry(timestamp, session, "followup", "high:3h", "market", bundle, float(minute_bar["CLOSE"]))

    last_timestamp = pd.Timestamp(period_minutes.iloc[min(len(period_minutes) - 1, max(0, len(period_minutes) - 1))]["DATE"])
    if max_ticks is not None and signal_rows:
        last_timestamp = pd.Timestamp(signal_rows[-1]["timestamp_utc"])
    last_bar = period_minutes[period_minutes["DATE"] <= last_timestamp].iloc[-1]
    for position in list(positions):
        positions.remove(position)
        close_position(position, last_timestamp, float(last_bar["CLOSE"]), "forced_period_end")
    for order in list(pending):
        pending.remove(order)
        attempts.append(
            {
                "timestamp_utc": last_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "date_local": local_label(last_timestamp),
                "session_id": order.get("session_id"),
                "session_name": order.get("session_name"),
                "trigger": "pending_terminal",
                "submission_mode": "programmed",
                "primary_key": "",
                "decision": order.get("side"),
                "reason": "forced_period_end_unfilled",
                "planned_entry": order.get("planned_entry"),
            }
        )
    engine.close()

    artifact_times = []
    for item in provider_manifest:
        raw = item.get("model_modified_at_utc")
        if raw:
            artifact_times.append(pd.Timestamp(raw))
    point_in_time_safe = bool(artifact_times) and all(timestamp <= start for timestamp in artifact_times)
    if point_in_time_safe:
        result_grade = "point-in-time market replay with pre-period frozen models"
        warning = (
            "All loaded model artifacts predate the evaluated period. Market-data lookahead was prevented, but current config selection "
            "and omitted live-only behavior still mean this is not a perfect historical reconstruction."
        )
    else:
        result_grade = "exploratory retrospective current-model replay"
        warning = (
            "One or more fitted model artifacts were created after the evaluated period began. Those models may have learned from this "
            "same market history, so the result must not be treated as unbiased evidence of future strategy strength."
        )

    pnl_values = np.asarray([float(operation.get("net_pnl", 0.0)) for operation in operations], dtype=float)
    winners = pnl_values[pnl_values > 0]
    losers = pnl_values[pnl_values <= 0]
    profit_factor: Any = None
    if losers.size and abs(float(np.sum(losers))) > 1e-12:
        profit_factor = float(np.sum(winners) / abs(np.sum(losers)))
    elif winners.size:
        profit_factor = "infinite"
    coverage = float(np.mean([float(row.get("coverage", 0.0)) for row in signal_rows])) if signal_rows else 0.0

    all_local_days = sorted(set(local_label(value) for value in period_minutes["DATE"]))
    daily: List[Dict[str, Any]] = []
    running_equity = float(initial_balance)
    for day in all_local_days:
        day_ops = [operation for operation in operations if local_label(pd.Timestamp(operation["exit_time_utc"])) == day]
        day_attempts = [attempt for attempt in attempts if str(attempt.get("date_local")) == day and attempt.get("trigger") in {"mandatory_session", "followup"}]
        day_net = float(sum(float(operation.get("net_pnl", 0.0)) for operation in day_ops))
        running_equity += day_net
        daily.append(
            {
                "date_local": day,
                "entry_attempts": len(day_attempts),
                "n_trades": len(day_ops),
                "wins": sum(1 for operation in day_ops if float(operation.get("net_pnl", 0.0)) > 0),
                "losses": sum(1 for operation in day_ops if float(operation.get("net_pnl", 0.0)) <= 0),
                "net_pnl": day_net,
                "end_equity": running_equity,
            }
        )

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "model_backed_point_in_time_market_replay",
        "market_source": str(market_source),
        "symbol": symbol,
        "timezone": timezone_name,
        "period": {
            "start_utc": start.strftime("%Y-%m-%d %H:%M:%S"),
            "end_utc": end.strftime("%Y-%m-%d %H:%M:%S"),
            "first_market_row_utc": period_minutes.iloc[0]["DATE"].strftime("%Y-%m-%d %H:%M:%S"),
            "last_market_row_utc": period_minutes.iloc[-1]["DATE"].strftime("%Y-%m-%d %H:%M:%S"),
            "market_rows": int(len(period_minutes)),
            "trading_days": int(len(all_local_days)),
        },
        "validity": {
            "result_grade": result_grade,
            "point_in_time_market_data": True,
            "unfinished_candles_use_only_available_minutes": True,
            "point_in_time_model_artifacts": point_in_time_safe,
            "model_policy": "latest_current_artifacts_frozen_for_replay",
            "plain_language_warning": warning,
            "excluded_live_behaviors": [
                "LLM commentary and sentiment",
                "approval latency and manual decisions",
                "live trade-memory conviction term",
                "opposing countertrade automation",
                "MT5 broker stop/freeze-distance normalization",
                "network, terminal, and rejection failures",
            ],
        },
        "execution_policy": {
            "agent_b_poll_minutes": tick_minutes,
            "signal_forecast_scope": "first_model_output_only; identical to the first step of the configured recursive horizon",
            "session_hours": session_hours,
            "programmed_order_expiration_minutes": expiration_minutes,
            "spread_bps": float(execution.get("spread_bps", 2.0) or 2.0),
            "slippage_bps": float(execution.get("slippage_bps", 2.0) or 2.0),
            "commission_per_trade": commission,
            "contract_size": float(contract_size),
            "intrabar_barrier_policy": "stop_first_when_stop_and_take_profit_share_a_minute",
            "extension_policy": "reject_and_close_at_base_deadline",
        },
        "overall": {
            "initial_balance": float(initial_balance),
            "ending_equity": float(balance),
            "n_trades": int(len(operations)),
            "wins": int(np.sum(pnl_values > 0)) if pnl_values.size else 0,
            "losses": int(np.sum(pnl_values <= 0)) if pnl_values.size else 0,
            "win_rate": float(np.mean(pnl_values > 0)) if pnl_values.size else 0.0,
            "gross_profit": float(np.sum(winners)) if winners.size else 0.0,
            "gross_loss": float(np.sum(losers)) if losers.size else 0.0,
            "net_pnl": float(np.sum(pnl_values)) if pnl_values.size else 0.0,
            "total_return_pct": float((balance / max(initial_balance, 1e-9) - 1.0) * 100.0),
            "max_drawdown_pct": _max_drawdown_pct(equity_points),
            "profit_factor": profit_factor,
            "expectancy_per_trade": float(np.mean(pnl_values)) if pnl_values.size else 0.0,
            "signal_ticks": int(tick_count),
            "signal_coverage": coverage,
            "entry_attempts": sum(1 for attempt in attempts if attempt.get("trigger") in {"mandatory_session", "followup"}),
        },
        "daily": daily,
        "operations": operations,
        "model_manifest": provider_manifest,
        "signal_errors_last_tick": {
            key: value.get("error")
            for key, value in ((last_bundle or {}).get("signals") or {}).items()
            if value.get("error")
        },
    }
    output_paths = _finalize_outputs(Path(output_dir), summary, attempts, signal_rows)
    return {"ok": True, "summary": summary, **output_paths}
