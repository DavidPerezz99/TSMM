"""Persistent online evaluation for matured TSMM inference forecasts."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .market_db import query_ohlc


def _parse_utc(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw[:19], fmt)
        except Exception:
            continue
    return None


def _iso(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _r2_score(actual: List[float], predicted: List[float]) -> Optional[float]:
    if len(actual) != len(predicted) or len(actual) < 2:
        return None
    mean_actual = sum(actual) / len(actual)
    total = sum((value - mean_actual) ** 2 for value in actual)
    if total <= 1e-12:
        return None
    residual = sum((truth - estimate) ** 2 for truth, estimate in zip(actual, predicted))
    return 1.0 - (residual / total)


class InferencePerformanceStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=15000")
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS inference_forecasts (
                    forecast_key TEXT PRIMARY KEY,
                    generated_at_utc TEXT NOT NULL,
                    origin_bucket_utc TEXT NOT NULL,
                    target_bucket_utc TEXT,
                    timeframe TEXT NOT NULL,
                    timeframe_minutes INTEGER NOT NULL,
                    family TEXT NOT NULL,
                    model TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    model_updated_at_utc TEXT,
                    target_feature TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    predicted_value REAL NOT NULL,
                    inference_strength REAL,
                    r2_train REAL,
                    input_fingerprint TEXT,
                    actual_value REAL,
                    matured_at_utc TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_inference_pending
                    ON inference_forecasts(matured_at_utc, timeframe, origin_bucket_utc);
                CREATE INDEX IF NOT EXISTS idx_inference_metric
                    ON inference_forecasts(timeframe, family, model_path, matured_at_utc);
                CREATE INDEX IF NOT EXISTS idx_inference_model_lineage
                    ON inference_forecasts(timeframe, family, model, matured_at_utc);

                CREATE TABLE IF NOT EXISTS inference_metric_snapshots (
                    snapshot_key TEXT PRIMARY KEY,
                    generated_at_utc TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    family TEXT NOT NULL,
                    model TEXT NOT NULL,
                    model_path TEXT NOT NULL,
                    r2_live_rolling REAL,
                    sample_count INTEGER NOT NULL,
                    window_samples INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_metric_snapshot_series
                    ON inference_metric_snapshots(timeframe, family, model_path, generated_at_utc);
                CREATE INDEX IF NOT EXISTS idx_metric_snapshot_lineage
                    ON inference_metric_snapshots(timeframe, family, model, generated_at_utc);
                """
            )

    def record_forecasts(self, rows: Iterable[Dict[str, Any]]) -> int:
        inserted = 0
        sql = """
            INSERT OR IGNORE INTO inference_forecasts (
                forecast_key, generated_at_utc, origin_bucket_utc,
                timeframe, timeframe_minutes, family, model, model_path,
                model_updated_at_utc, target_feature, step, predicted_value,
                inference_strength, r2_train, input_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._connection() as conn:
            for row in rows:
                identity = {
                    "generated_at_utc": row.get("generated_at_utc"),
                    "timeframe": row.get("timeframe"),
                    "family": row.get("family"),
                    "model_path": row.get("model_path"),
                    "step": row.get("step"),
                    "input_fingerprint": row.get("input_fingerprint"),
                }
                key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()
                cursor = conn.execute(
                    sql,
                    (
                        key,
                        str(row.get("generated_at_utc") or ""),
                        str(row.get("origin_bucket_utc") or ""),
                        str(row.get("timeframe") or ""),
                        int(row.get("timeframe_minutes") or 1),
                        str(row.get("family") or "").lower(),
                        str(row.get("model") or "").lower(),
                        str(row.get("model_path") or ""),
                        str(row.get("model_updated_at_utc") or ""),
                        str(row.get("target_feature") or "y_diff"),
                        int(row.get("step") or 1),
                        float(row.get("predicted_value")),
                        float(row.get("inference_strength")) if row.get("inference_strength") is not None else None,
                        float(row.get("r2_train")) if row.get("r2_train") is not None else None,
                        str(row.get("input_fingerprint") or ""),
                    ),
                )
                inserted += int(cursor.rowcount or 0)
        return inserted

    def mature_pending(self, market_db_path: str, source_data_as_of_utc: str, symbol: str = "XAUUSD") -> int:
        source_as_of = _parse_utc(source_data_as_of_utc)
        if source_as_of is None:
            return 0

        with self._connection() as conn:
            pending = conn.execute(
                """
                SELECT forecast_key, origin_bucket_utc, timeframe,
                       timeframe_minutes, family, step
                FROM inference_forecasts
                WHERE matured_at_utc IS NULL
                ORDER BY timeframe_minutes, origin_bucket_utc
                """
            ).fetchall()

        by_timeframe: Dict[tuple[str, int], List[sqlite3.Row]] = {}
        for row in pending:
            by_timeframe.setdefault((str(row["timeframe"]), int(row["timeframe_minutes"])), []).append(row)

        updates: List[tuple[float, str, str, str]] = []
        for (_timeframe, timeframe_minutes), rows in by_timeframe.items():
            origins = [_parse_utc(row["origin_bucket_utc"]) for row in rows]
            valid_origins = [value for value in origins if value is not None]
            if not valid_origins:
                continue
            start = min(valid_origins) - timedelta(minutes=timeframe_minutes * 2)
            frame = query_ohlc(
                market_db_path,
                timeframe_minutes=timeframe_minutes,
                latest_records=10000,
                start_date=_iso(start),
                symbol=symbol,
            )
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            frame["DATE"] = frame["DATE"].map(lambda value: _parse_utc(value))
            frame = frame.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)
            date_to_index = {value: idx for idx, value in enumerate(frame["DATE"].tolist())}

            for row in rows:
                origin = _parse_utc(row["origin_bucket_utc"])
                origin_idx = date_to_index.get(origin)
                if origin_idx is None:
                    continue
                target_idx = origin_idx + int(row["step"])
                if target_idx >= len(frame.index) or target_idx <= 0:
                    continue
                target_bucket = frame.iloc[target_idx]["DATE"]
                if source_as_of < target_bucket + timedelta(minutes=timeframe_minutes):
                    continue
                family_column = str(row["family"] or "").upper()
                if family_column not in frame.columns:
                    continue
                current_value = float(frame.iloc[target_idx][family_column])
                previous_value = float(frame.iloc[target_idx - 1][family_column])
                actual = current_value - previous_value
                if not math.isfinite(actual):
                    continue
                updates.append((actual, _iso(target_bucket), _iso(source_as_of), str(row["forecast_key"])))

        if updates:
            with self._connection() as conn:
                conn.executemany(
                    """
                    UPDATE inference_forecasts
                    SET actual_value=?, target_bucket_utc=?, matured_at_utc=?
                    WHERE forecast_key=? AND matured_at_utc IS NULL
                    """,
                    updates,
                )
        return len(updates)

    def rolling_metrics(
        self,
        timeframe: str,
        family: str,
        model: str,
        model_path: Optional[str] = None,
        window_samples: int = 100,
        min_samples: int = 10,
    ) -> Dict[str, Any]:
        where = ["timeframe=?", "family=?", "model=?", "matured_at_utc IS NOT NULL", "actual_value IS NOT NULL"]
        params: List[Any] = [str(timeframe), str(family).lower(), str(model).lower()]
        if model_path:
            where.append("model_path=?")
            params.append(str(model_path))
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT generated_at_utc, origin_bucket_utc, step,
                       predicted_value, actual_value, matured_at_utc
                FROM inference_forecasts
                WHERE {' AND '.join(where)}
                ORDER BY generated_at_utc DESC
                """,
                params,
            ).fetchall()

        # Each timeframe candle represents one model observation. During a
        # partial candle TSMM may infer repeatedly, so score only its latest
        # issued forecast to avoid overweighting one realized target.
        selected: List[sqlite3.Row] = []
        seen: set[tuple[str, int]] = set()
        for row in rows:
            key = (str(row["origin_bucket_utc"]), int(row["step"]))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= max(int(window_samples), 1):
                break

        actual = [float(row["actual_value"]) for row in selected]
        predicted = [float(row["predicted_value"]) for row in selected]
        score = _r2_score(actual, predicted) if len(selected) >= max(int(min_samples), 2) else None
        return {
            "r2_live_rolling": round(float(score), 6) if score is not None else None,
            "r2_live_samples": len(selected),
            "r2_live_min_samples": max(int(min_samples), 2),
            "r2_live_window_samples": max(int(window_samples), 1),
            "r2_live_last_matured_at_utc": str(selected[0]["matured_at_utc"] or "") if selected else None,
        }

    def record_metric_snapshot(
        self,
        generated_at_utc: str,
        timeframe: str,
        family: str,
        model: str,
        model_path: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        with self._connection() as conn:
            previous = conn.execute(
                """
                SELECT r2_live_rolling, sample_count, generated_at_utc
                FROM inference_metric_snapshots
                WHERE timeframe=? AND family=? AND model=?
                ORDER BY generated_at_utc DESC LIMIT 1
                """,
                (str(timeframe), str(family).lower(), str(model).lower()),
            ).fetchone()
            identity = f"{generated_at_utc}|{timeframe}|{family}|{model_path}"
            key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            conn.execute(
                """
                INSERT OR REPLACE INTO inference_metric_snapshots (
                    snapshot_key, generated_at_utc, timeframe, family, model,
                    model_path, r2_live_rolling, sample_count, window_samples
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    generated_at_utc,
                    timeframe,
                    family.lower(),
                    model.lower(),
                    model_path,
                    metrics.get("r2_live_rolling"),
                    int(metrics.get("r2_live_samples") or 0),
                    int(metrics.get("r2_live_window_samples") or 0),
                ),
            )

        previous_score = float(previous["r2_live_rolling"]) if previous and previous["r2_live_rolling"] is not None else None
        current_score = metrics.get("r2_live_rolling")
        delta = float(current_score) - previous_score if current_score is not None and previous_score is not None else None
        return {
            "r2_live_previous": round(previous_score, 6) if previous_score is not None else None,
            "r2_live_delta": round(delta, 6) if delta is not None else None,
            "r2_live_previous_at_utc": str(previous["generated_at_utc"] or "") if previous else None,
        }

    def prune(self, forecast_retention_days: int = 180, snapshot_retention_days: int = 365) -> Dict[str, int]:
        now = datetime.utcnow()
        forecast_cutoff = _iso(now - timedelta(days=max(int(forecast_retention_days), 1)))
        snapshot_cutoff = _iso(now - timedelta(days=max(int(snapshot_retention_days), 1)))
        with self._connection() as conn:
            forecast_deleted = conn.execute(
                "DELETE FROM inference_forecasts WHERE generated_at_utc < ?", (forecast_cutoff,)
            ).rowcount
            snapshot_deleted = conn.execute(
                "DELETE FROM inference_metric_snapshots WHERE generated_at_utc < ?", (snapshot_cutoff,)
            ).rowcount
        return {"forecasts_deleted": int(forecast_deleted or 0), "snapshots_deleted": int(snapshot_deleted or 0)}
