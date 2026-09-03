"""Tiingo equity-minute ingestion and audited US500 proxy maintenance.

The historical TSMM US500 series is a broker/index-style instrument, whereas
Tiingo supplies SPY equity minutes.  This module deliberately stores the raw
SPY rows in their own symbol namespace and only continues US500 after a robust
overlap calibration.  Every synthetic interval is recorded in SQLite so it can
be excluded or replaced when a native broker feed becomes available.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import closing
from datetime import datetime, timezone
import math
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import requests

from utils.live_data import (
    _is_tiingo_quota_error,
    _ordered_tiingo_candidates,
    _persist_active_tiingo_env,
    resolve_tiingo_token_candidates,
)
from utils.market_db import (
    get_latest_date,
    init_market_db,
    master_table_name,
    normalize_market_symbol,
    upsert_ohlc_1m,
)


TIINGO_IEX_BASE_URL = "https://api.tiingo.com/iex"


@dataclass(frozen=True)
class ProxyCalibration:
    scale: float
    samples: int
    first_utc: str
    last_utc: str
    relative_mad: float
    match_method: str


def _utc_text(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def _chunk_ranges(start: Any, end: Any, days: int = 28) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    first = pd.Timestamp(start).normalize()
    last = pd.Timestamp(end).normalize()
    if first > last:
        return
    cursor = first
    width = pd.Timedelta(days=max(int(days), 1) - 1)
    while cursor <= last:
        chunk_end = min(cursor + width, last)
        yield cursor, chunk_end
        cursor = chunk_end + pd.Timedelta(days=1)


def _normalise_tiingo_equity_payload(payload: Any) -> pd.DataFrame:
    frame = pd.DataFrame(payload or [])
    if frame.empty:
        return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
    rename = {
        "date": "DATE",
        "open": "OPEN",
        "high": "HIGH",
        "low": "LOW",
        "close": "CLOSE",
        "volume": "VOLUME",
    }
    frame = frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})
    required = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Tiingo IEX response is missing columns: {missing}")
    if "VOLUME" not in frame.columns:
        frame["VOLUME"] = 0.0
    dates = pd.to_datetime(frame["DATE"], errors="coerce", utc=True)
    frame["DATE"] = dates.dt.tz_convert("UTC").dt.tz_localize(None)
    for column in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=required)
    return (
        frame[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
        .sort_values("DATE")
        .drop_duplicates(subset=["DATE"], keep="last")
        .reset_index(drop=True)
    )


def fetch_tiingo_iex_minutes(
    *,
    ticker: str,
    start_date: Any,
    end_date: Any,
    rate: str = "1min",
    token: str = "",
    token_env: str = "TIINGO_API_TOKEN",
    token_envs: Any = None,
    token_rotation_state_path: Optional[str] = None,
    timeout_seconds: int = 45,
    chunk_days: int = 28,
) -> Dict[str, Any]:
    """Fetch bounded IEX minute chunks without placing credentials in URLs."""
    candidates = resolve_tiingo_token_candidates(token_env, token_envs, token=token)
    if not candidates:
        return {
            "ok": False,
            "error": f"Missing Tiingo token in configured env vars: {token_env}",
            "attempts": [],
        }
    ordered, previous_env = _ordered_tiingo_candidates(candidates, token_rotation_state_path)
    frames: list[pd.DataFrame] = []
    attempts: list[Dict[str, Any]] = []
    active_index = 0
    active_env = ""

    for chunk_start, chunk_end in _chunk_ranges(start_date, end_date, chunk_days):
        chunk_ok = False
        last_error = ""
        for offset in range(len(ordered)):
            index = (active_index + offset) % len(ordered)
            candidate = ordered[index]
            used_env = str(candidate.get("env") or token_env)
            used_token = str(candidate.get("token") or "")
            url = f"{TIINGO_IEX_BASE_URL}/{str(ticker).strip().lower()}/prices"
            params = {
                "startDate": chunk_start.strftime("%Y-%m-%d"),
                "endDate": chunk_end.strftime("%Y-%m-%d"),
                "resampleFreq": str(rate or "1min"),
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Token {used_token}",
            }
            try:
                response = requests.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=max(int(timeout_seconds), 5),
                )
            except Exception as exc:
                attempts.append({"env": used_env, "status": None, "chunk": params, "error": str(exc)})
                last_error = str(exc)
                continue
            attempts.append({"env": used_env, "status": int(response.status_code), "chunk": params})
            if response.status_code == 200:
                try:
                    frames.append(_normalise_tiingo_equity_payload(response.json()))
                except Exception as exc:
                    return {"ok": False, "error": f"Invalid Tiingo IEX payload: {exc}", "attempts": attempts}
                active_index = index
                active_env = used_env
                chunk_ok = True
                break
            last_error = str(response.text or f"HTTP {response.status_code}")[:500]
            if not _is_tiingo_quota_error(response.status_code, last_error):
                return {"ok": False, "error": f"Tiingo IEX request failed: {last_error}", "attempts": attempts}
        if not chunk_ok:
            return {"ok": False, "error": f"Tiingo token pool exhausted: {last_error}", "attempts": attempts}

    if active_env:
        _persist_active_tiingo_env(
            token_rotation_state_path,
            used_env=active_env,
            previous_env=previous_env,
            rotation_reason="equity_iex_refresh",
        )
    combined = pd.concat(frames, ignore_index=True) if frames else _normalise_tiingo_equity_payload([])
    if not combined.empty:
        combined = combined.sort_values("DATE").drop_duplicates(subset=["DATE"], keep="last")
    return {
        "ok": True,
        "data": combined.reset_index(drop=True),
        "rows": int(len(combined)),
        "used_token_env": active_env,
        "rotated": bool(active_env and previous_env and active_env != previous_env),
        "attempts": attempts,
    }


def _read_symbol_rows(
    db_path: str,
    symbol: str,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    table = master_table_name(symbol)
    where: list[str] = []
    params: list[Any] = []
    if start:
        where.append("DATE >= ?")
        params.append(start)
    if end:
        where.append("DATE <= ?")
        params.append(end)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        frame = pd.read_sql_query(
            f"SELECT DATE, OPEN, HIGH, LOW, CLOSE, VOLUME FROM {table} {clause} ORDER BY DATE",
            connection,
            params=params,
        )
    frame["DATE"] = pd.to_datetime(frame["DATE"], errors="coerce")
    return frame.dropna(subset=["DATE"]).sort_values("DATE").reset_index(drop=True)


def calibrate_price_proxy(
    *,
    db_path: str,
    target_symbol: str = "US500",
    source_symbol: str = "SPY",
    lookback_days: int = 60,
    minimum_samples: int = 100,
    target_end_date: Any = None,
) -> ProxyCalibration:
    """Estimate a robust target/source price ratio from overlapping minutes."""
    target_latest = _utc_text(target_end_date) if target_end_date is not None else get_latest_date(
        db_path, symbol=target_symbol
    )
    if not target_latest:
        raise ValueError(f"Cannot calibrate {target_symbol}: target history is empty")
    end = pd.Timestamp(target_latest)
    start = end - pd.Timedelta(days=max(int(lookback_days), 5))
    target = _read_symbol_rows(db_path, target_symbol, start=_utc_text(start), end=_utc_text(end))
    source = _read_symbol_rows(db_path, source_symbol, start=_utc_text(start), end=_utc_text(end))
    if target.empty or source.empty:
        raise ValueError("Cannot calibrate proxy: overlapping target/source history is unavailable")

    exact = target[["DATE", "CLOSE"]].rename(columns={"CLOSE": "TARGET_CLOSE"}).merge(
        source[["DATE", "CLOSE"]].rename(columns={"CLOSE": "SOURCE_CLOSE"}),
        on="DATE",
        how="inner",
    )
    match_method = "exact_timestamp"
    matched = exact
    if len(matched) < int(minimum_samples):
        match_method = "nearest_within_2_minutes"
        matched = pd.merge_asof(
            target[["DATE", "CLOSE"]].rename(columns={"CLOSE": "TARGET_CLOSE"}).sort_values("DATE"),
            source[["DATE", "CLOSE"]].rename(columns={"CLOSE": "SOURCE_CLOSE"}).sort_values("DATE"),
            on="DATE",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=2),
        ).dropna(subset=["SOURCE_CLOSE"])
    ratios = pd.to_numeric(matched["TARGET_CLOSE"], errors="coerce") / pd.to_numeric(
        matched["SOURCE_CLOSE"], errors="coerce"
    )
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
    ratios = ratios[ratios > 0.0]
    if len(ratios) < int(minimum_samples):
        raise ValueError(
            f"Cannot calibrate proxy: {len(ratios)} matched minutes, need {int(minimum_samples)}"
        )
    lower, upper = ratios.quantile([0.01, 0.99])
    trimmed = ratios[(ratios >= lower) & (ratios <= upper)]
    scale = float(trimmed.median())
    mad = float((trimmed - scale).abs().median())
    relative_mad = mad / scale if scale > 0.0 else math.inf
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Proxy calibration produced an invalid scale")
    return ProxyCalibration(
        scale=scale,
        samples=int(len(trimmed)),
        first_utc=_utc_text(matched["DATE"].min()),
        last_utc=_utc_text(matched["DATE"].max()),
        relative_mad=float(relative_mad),
        match_method=match_method,
    )


def _native_target_end(db_path: str, target_symbol: str, latest: str) -> str:
    """Return the immutable native-series cutoff before any proxy continuation."""
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='market_data_provenance'"
        ).fetchone()
        if not table_exists:
            return latest
        row = connection.execute(
            """
            SELECT MIN(first_utc)
            FROM market_data_provenance
            WHERE target_symbol = ? AND method = 'overlap_anchored_price_proxy'
            """,
            (normalize_market_symbol(target_symbol),),
        ).fetchone()
    first_proxy = str((row or [None])[0] or "").strip()
    if not first_proxy:
        return latest
    return _utc_text(pd.Timestamp(first_proxy) - pd.Timedelta(minutes=1))


def _record_proxy_provenance(
    db_path: str,
    *,
    target_symbol: str,
    source_symbol: str,
    first_utc: str,
    last_utc: str,
    rows: int,
    calibration: ProxyCalibration,
) -> None:
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS market_data_provenance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target_symbol TEXT NOT NULL,
                source_symbol TEXT NOT NULL,
                provider TEXT NOT NULL,
                method TEXT NOT NULL,
                first_utc TEXT NOT NULL,
                last_utc TEXT NOT NULL,
                rows_written INTEGER NOT NULL,
                scale REAL NOT NULL,
                calibration_samples INTEGER NOT NULL,
                calibration_first_utc TEXT,
                calibration_last_utc TEXT,
                calibration_relative_mad REAL,
                calibration_match_method TEXT,
                created_at_utc TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO market_data_provenance (
                target_symbol, source_symbol, provider, method, first_utc, last_utc,
                rows_written, scale, calibration_samples, calibration_first_utc,
                calibration_last_utc, calibration_relative_mad,
                calibration_match_method, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalize_market_symbol(target_symbol),
                normalize_market_symbol(source_symbol),
                "tiingo_iex",
                "overlap_anchored_price_proxy",
                first_utc,
                last_utc,
                int(rows),
                float(calibration.scale),
                int(calibration.samples),
                calibration.first_utc,
                calibration.last_utc,
                float(calibration.relative_mad),
                calibration.match_method,
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        connection.commit()


def refresh_us500_proxy(
    *,
    db_path: str,
    source_ticker: str = "SPY",
    target_symbol: str = "US500",
    rate: str = "1min",
    token: str = "",
    token_env: str = "TIINGO_API_TOKEN",
    token_envs: Any = None,
    token_rotation_state_path: Optional[str] = None,
    calibration_lookback_days: int = 60,
    minimum_calibration_samples: int = 100,
    maximum_relative_mad: float = 0.02,
    maximum_seam_jump_pct: float = 8.0,
    end_date: Any = None,
) -> Dict[str, Any]:
    """Append an audited SPY-derived continuation after native US500 history."""
    resolved = str(Path(db_path).resolve())
    init_market_db(resolved, symbol=target_symbol)
    init_market_db(resolved, symbol=source_ticker)
    target_latest = get_latest_date(resolved, symbol=target_symbol)
    if not target_latest:
        return {"ok": False, "error": f"{target_symbol} history is empty; a native anchor is required"}

    target_latest_ts = pd.Timestamp(target_latest)
    native_target_end = _native_target_end(resolved, target_symbol, target_latest)
    native_target_end_ts = pd.Timestamp(native_target_end)
    source_latest = get_latest_date(resolved, symbol=source_ticker)
    overlap_start = native_target_end_ts - pd.Timedelta(
        days=max(int(calibration_lookback_days), 5)
    )
    if source_latest:
        existing_overlap = _read_symbol_rows(
            resolved,
            source_ticker,
            start=_utc_text(overlap_start),
            end=_utc_text(native_target_end_ts),
        )
        fetch_start = (
            pd.Timestamp(source_latest) - pd.Timedelta(days=2)
            if len(existing_overlap) >= int(minimum_calibration_samples)
            else overlap_start
        )
    else:
        fetch_start = overlap_start
    fetch_end = pd.Timestamp(end_date or datetime.now(timezone.utc))
    if fetch_end.tzinfo is not None:
        fetch_end = fetch_end.tz_convert("UTC").tz_localize(None)
    fetched = fetch_tiingo_iex_minutes(
        ticker=source_ticker,
        start_date=fetch_start,
        end_date=fetch_end,
        rate=rate,
        token=token,
        token_env=token_env,
        token_envs=token_envs,
        token_rotation_state_path=token_rotation_state_path,
    )
    if not fetched.get("ok"):
        return fetched
    raw = fetched.get("data")
    source_rows_written = upsert_ohlc_1m(resolved, raw, symbol=source_ticker) if isinstance(raw, pd.DataFrame) else 0

    try:
        calibration = calibrate_price_proxy(
            db_path=resolved,
            target_symbol=target_symbol,
            source_symbol=source_ticker,
            lookback_days=calibration_lookback_days,
            minimum_samples=minimum_calibration_samples,
            target_end_date=native_target_end,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source_rows_written": source_rows_written}
    if calibration.relative_mad > float(maximum_relative_mad):
        return {
            "ok": False,
            "error": (
                f"Proxy scale is unstable: relative MAD {calibration.relative_mad:.4%} "
                f"> {float(maximum_relative_mad):.4%}"
            ),
            "calibration": calibration.__dict__,
        }

    future_source = _read_symbol_rows(
        resolved,
        source_ticker,
        start=_utc_text(target_latest_ts + pd.Timedelta(minutes=1)),
    )
    if future_source.empty:
        return {
            "ok": True,
            "updated": False,
            "reason": "no_new_source_minutes",
            "target_latest": target_latest,
            "source_rows_written": source_rows_written,
            "calibration": calibration.__dict__,
        }
    proxy = future_source.copy()
    for column in ["OPEN", "HIGH", "LOW", "CLOSE"]:
        proxy[column] = pd.to_numeric(proxy[column], errors="coerce") * calibration.scale
    proxy = proxy.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])

    target_tail = _read_symbol_rows(resolved, target_symbol, start=target_latest, end=target_latest)
    if not target_tail.empty and not proxy.empty:
        previous_close = float(target_tail.iloc[-1]["CLOSE"])
        next_open = float(proxy.iloc[0]["OPEN"])
        seam_jump_pct = abs(next_open / previous_close - 1.0) * 100.0 if previous_close > 0.0 else math.inf
    else:
        seam_jump_pct = 0.0
    if seam_jump_pct > float(maximum_seam_jump_pct):
        return {
            "ok": False,
            "error": (
                f"US500 proxy seam jump {seam_jump_pct:.2f}% exceeds "
                f"{float(maximum_seam_jump_pct):.2f}%"
            ),
            "calibration": calibration.__dict__,
        }

    target_rows_written = upsert_ohlc_1m(resolved, proxy, symbol=target_symbol)
    first_utc = _utc_text(proxy["DATE"].min())
    last_utc = _utc_text(proxy["DATE"].max())
    _record_proxy_provenance(
        resolved,
        target_symbol=target_symbol,
        source_symbol=source_ticker,
        first_utc=first_utc,
        last_utc=last_utc,
        rows=target_rows_written,
        calibration=calibration,
    )
    return {
        "ok": True,
        "updated": bool(target_rows_written),
        "db_path": resolved,
        "target_symbol": normalize_market_symbol(target_symbol),
        "source_symbol": normalize_market_symbol(source_ticker),
        "source_rows_written": int(source_rows_written),
        "target_rows_written": int(target_rows_written),
        "first_proxy_utc": first_utc,
        "latest_proxy_utc": last_utc,
        "native_target_end_utc": native_target_end,
        "seam_jump_pct": float(seam_jump_pct),
        "calibration": calibration.__dict__,
        "provenance": "market_data_provenance",
        "used_token_env": fetched.get("used_token_env"),
        "token_rotated": bool(fetched.get("rotated", False)),
    }
