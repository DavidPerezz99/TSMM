"""
Live FX data utilities for Tiingo updates, aggregation, and enrichment.
"""

from __future__ import annotations

import os
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import requests
from utils.market_db import init_market_db, upsert_ohlc_1m, get_latest_date


def read_csv_tail(path: str, n_rows: int, usecols=None) -> pd.DataFrame:
    """Read only the latest ``n_rows`` rows from a CSV file.

    Uses a line-count + skiprows strategy to avoid loading the full file into memory.
    """
    n_rows = max(int(n_rows or 1), 1)
    if not os.path.exists(path):
        return pd.DataFrame()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        total_lines = sum(1 for _ in f)

    # total_lines includes header row
    data_lines = max(total_lines - 1, 0)
    if data_lines <= n_rows:
        return pd.read_csv(path, usecols=usecols)

    skip_until = data_lines - n_rows
    # keep header (line 0), skip first `skip_until` data lines
    skiprows = range(1, 1 + skip_until)
    return pd.read_csv(path, skiprows=skiprows, usecols=usecols)


def update_fx_data(data: pd.DataFrame, rate: str, symbol: str, token: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Update an in-memory FX dataframe with latest Tiingo candles.

    Parameters
    ----------
    data : pd.DataFrame
        Existing dataframe with DATE, OPEN, HIGH, LOW, CLOSE columns.
    rate : str
        Tiingo resample frequency (e.g. '1min').
    symbol : str
        FX symbol (e.g. 'xauusd').
    token : str
        Tiingo API token.
    """
    if data is None or data.empty:
        raise ValueError("Input 'data' is empty. Seed it with initial history first.")

    data = data.copy()
    if "Unnamed: 0" in data.columns:
        data.drop(columns=["Unnamed: 0"], inplace=True)

    if "DATE" not in data.columns:
        raise ValueError("Input data must contain DATE column")

    reference_date = pd.to_datetime(data["DATE"]).max()
    reference_date_str = reference_date.strftime("%Y-%m-%d %H:%M:%S")

    last_minute = int(reference_date_str[14:16])
    last_hour = int(reference_date_str[11:13])
    date = datetime.strptime(reference_date_str[:10], "%Y-%m-%d")

    if last_minute == 59:
        last_minute = 0
        last_hour += 1
        if last_hour == 24:
            last_hour = 0
            date += timedelta(days=1)
    else:
        last_minute += 1

    new_date = f"{date.strftime('%Y-%m-%d')} {last_hour:02}:{last_minute:02}:00"

    headers = {"Content-Type": "application/json"}
    url = (
        f"https://api.tiingo.com/tiingo/fx/{symbol}/prices"
        f"?startDate={new_date}&resampleFreq={rate}&token={token}"
    )

    response = requests.get(url, headers=headers, timeout=20)
    if response.status_code != 200:
        raise ValueError(f"Error fetching data: {response.text}")

    ds = pd.DataFrame(response.json())
    if ds.empty:
        return data, pd.DataFrame()

    ds.drop(columns=[c for c in ["ticker"] if c in ds.columns], inplace=True)
    ds.rename(
        columns={
            "date": "DATE",
            "open": "OPEN",
            "high": "HIGH",
            "low": "LOW",
            "close": "CLOSE",
        },
        inplace=True,
    )

    ds["DATE"] = pd.to_datetime(ds["DATE"]).dt.tz_localize(None)
    data["DATE"] = pd.to_datetime(data["DATE"], utc=True, format="mixed").dt.tz_localize(None)

    data = pd.concat([data, ds], ignore_index=True)
    data.drop_duplicates(subset=["DATE"], inplace=True)
    data.sort_values(by="DATE", inplace=True)

    return data, ds


def aggregate_fx_data(
    historical_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    interval_minutes: int,
) -> pd.DataFrame:
    """Aggregate minute-level OHLC into higher timeframe candles."""
    if historical_data is None or historical_data.empty:
        return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])

    df = historical_data.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    filtered_data = df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)].copy()
    if filtered_data.empty:
        return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE"])

    filtered_data.set_index("DATE", inplace=True)
    aggregated_data = filtered_data.resample(f"{interval_minutes}T").agg({
        "OPEN": "first",
        "HIGH": "max",
        "LOW": "min",
        "CLOSE": "last",
    }).dropna().reset_index()
    return aggregated_data


def enrich_ohlc_features(aggregated_data: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering used by TSMM on aggregated candles."""
    df = aggregated_data.copy()

    # Differential returns
    df["Price_return"] = df["CLOSE"].diff()
    df["Open_return"] = df["OPEN"].diff()
    df["High_return"] = df["HIGH"].diff()
    df["Low_return"] = df["LOW"].diff()

    # Moving averages
    df["SMA_10"] = df["CLOSE"].rolling(window=10).mean()
    df["SMA_30"] = df["CLOSE"].rolling(window=30).mean()
    df["EMA_10"] = df["CLOSE"].ewm(span=10, adjust=False).mean()
    df["EMA_30"] = df["CLOSE"].ewm(span=30, adjust=False).mean()

    # Volatility
    df["Volatility_10"] = df["CLOSE"].rolling(window=10).std()
    df["Volatility_30"] = df["CLOSE"].rolling(window=30).std()

    # Session return
    df["daily_return"] = df["CLOSE"] - df["OPEN"]

    # Drop NaNs introduced by rolling/diff
    df = df.dropna().reset_index(drop=True)
    return df


def refresh_dataset_from_tiingo(refresh_cfg: Dict[str, Any], output_path: str, logger=None) -> Dict[str, Any]:
    """Refresh raw minute data from Tiingo and regenerate aggregated/enriched dataset.

    Expected cfg keys:
    - enabled: bool
    - raw_data_path: str
    - symbol: str (e.g. xauusd)
    - rate: str (e.g. 1min)
    - interval_minutes: int (e.g. 481)
    - start_date: str
    - end_date: str
    - token_env: str (default TIINGO_API_TOKEN)
    """
    log = logger or logging.getLogger(__name__)
    cfg = refresh_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return {"enabled": False, "reason": "data_refresh disabled"}

    token_env = cfg.get("token_env", "TIINGO_API_TOKEN")
    token = os.environ.get(token_env, "")
    if not token:
        return {"enabled": True, "updated": False, "error": f"Missing API token env var: {token_env}"}

    raw_path = cfg.get("raw_data_path")
    if not raw_path:
        return {"enabled": True, "updated": False, "error": "Missing raw_data_path in data_refresh config"}

    symbol = cfg.get("symbol", "xauusd")
    rate = cfg.get("rate", "1min")
    interval_minutes = int(cfg.get("interval_minutes", 481))
    start_date = cfg.get("start_date", "2009-01-01 00:00:00")
    end_date = cfg.get("end_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path)
    else:
        # bootstrap from configured source if exists
        if os.path.exists(output_path):
            raw_df = pd.read_csv(output_path)
            keep_cols = [c for c in ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"] if c in raw_df.columns]
            raw_df = raw_df[keep_cols].copy()
        else:
            return {
                "enabled": True,
                "updated": False,
                "error": f"Raw path does not exist and output source not found: {raw_path}",
            }

    updated_raw, new_data = update_fx_data(raw_df, rate, symbol, token)
    updated_raw.to_csv(raw_path, index=False)

    agg = aggregate_fx_data(updated_raw, start_date, end_date, interval_minutes)
    enriched = enrich_ohlc_features(agg)
    enriched.to_csv(output_path, index=False)

    log.info(
        "Data refresh completed. raw_rows=%s new_rows=%s agg_rows=%s output=%s",
        len(updated_raw),
        len(new_data),
        len(enriched),
        output_path,
    )

    return {
        "enabled": True,
        "updated": True,
        "raw_rows": int(len(updated_raw)),
        "new_rows": int(len(new_data)),
        "aggregated_rows": int(len(enriched)),
        "raw_data_path": raw_path,
        "output_path": output_path,
        "latest_date": str(updated_raw["DATE"].iloc[-1]) if len(updated_raw) else None,
    }


def _infer_timeframe_minutes(target_path: str, explicit_minutes: Optional[int] = None) -> int:
    if explicit_minutes is not None:
        try:
            return max(int(explicit_minutes), 1)
        except Exception:
            pass

    s = str(target_path or "").lower()
    m = re.search(r"(\d+)(m|h)", s)
    if not m:
        return 1

    value = int(m.group(1))
    unit = m.group(2)
    if unit == "h":
        return value * 60
    return value


def _load_master_minute_frame(master_table_path: str, latest_records: int) -> pd.DataFrame:
    latest_records = max(int(latest_records or 1), 1)
    if str(master_table_path).lower().endswith(".db") or str(master_table_path).lower().endswith(".sqlite"):
        from utils.market_db import query_ohlc

        return query_ohlc(
            db_path=master_table_path,
            timeframe_minutes=1,
            latest_records=latest_records,
            start_date=None,
            end_date=None,
        )

    df = read_csv_tail(master_table_path, latest_records)
    if df.empty:
        return df
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
    return df


def sync_dataset_source_from_master(
    master_table_path: str,
    output_path: str,
    timeframe_minutes: Optional[int],
    records: int,
    rolling_windows: list[int] | None = None,
    n_steps: int = 1,
    horizon: int = 1,
    logger=None,
) -> Dict[str, Any]:
    """Regenerate the active model source file from the refreshed master source."""
    log = logger or logging.getLogger(__name__)
    tf_minutes = _infer_timeframe_minutes(output_path, explicit_minutes=timeframe_minutes)
    max_window = max([int(w) for w in (rolling_windows or [])] + [0])

    required_target_rows = max(int(records or 0), 1) + max_window + max(int(n_steps or 1), 1) + max(int(horizon or 1), 1) + 50
    if tf_minutes <= 1:
        required_master_rows = required_target_rows
    else:
        required_master_rows = (required_target_rows * tf_minutes) + 200

    master_df = _load_master_minute_frame(master_table_path, latest_records=required_master_rows)
    if master_df.empty:
        return {
            "updated": False,
            "error": f"No rows available in master source: {master_table_path}",
            "output_path": output_path,
        }

    if tf_minutes > 1:
        start_date = str(master_df["DATE"].min())
        end_date = str(master_df["DATE"].max())
        target_df = aggregate_fx_data(master_df, start_date, end_date, tf_minutes)
    else:
        target_df = master_df.copy()

    if target_df.empty:
        return {
            "updated": False,
            "error": f"No aggregated rows generated for timeframe {tf_minutes}m",
            "output_path": output_path,
        }

    enriched = enrich_ohlc_features(target_df)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    enriched.to_csv(output_path, index=False)

    log.info(
        "Active config dataset refreshed from master. master=%s output=%s timeframe=%sm rows=%s",
        master_table_path,
        output_path,
        tf_minutes,
        len(enriched),
    )
    return {
        "updated": True,
        "master_table_path": master_table_path,
        "output_path": output_path,
        "timeframe_minutes": tf_minutes,
        "rows": int(len(enriched)),
        "latest_date": str(enriched["DATE"].iloc[-1]) if len(enriched) else None,
    }


def update_fx_master_table_file(
    master_table_path: str,
    rate: str,
    symbol: str,
    token: str,
    date_col: str = "DATE",
    tail_probe_rows: int = 2000,
) -> Dict[str, Any]:
    """Update a master minute CSV in-place by appending only new Tiingo rows.

    This avoids loading the complete master table into memory.
    """
    if not master_table_path:
        return {"updated": False, "error": "Missing master_table_path"}
    if not os.path.exists(master_table_path):
        return {"updated": False, "error": f"Master table file not found: {master_table_path}"}
    if not token:
        return {"updated": False, "error": "Missing Tiingo token"}

    # Read tail rows with all columns and auto-detect which column currently holds datetime values.
    probe = read_csv_tail(master_table_path, max(int(tail_probe_rows), 10), usecols=None)
    if probe.empty:
        return {"updated": False, "error": "Could not read master table tail"}

    best_col = None
    best_series = None
    best_valid = -1

    # Prefer configured DATE column but gracefully recover from shifted/legacy schemas.
    candidate_cols = list(probe.columns)
    if date_col in candidate_cols:
        candidate_cols = [date_col] + [c for c in candidate_cols if c != date_col]

    for col in candidate_cols:
        as_text = probe[col].astype(str).str.strip()
        # Guard against pandas parsing pure numeric columns as timestamps.
        looks_like_date = as_text.str.contains(r"[-/:T]", regex=True, na=False)
        if int(looks_like_date.sum()) <= 0:
            continue

        parsed = pd.to_datetime(as_text.where(looks_like_date, None), errors="coerce")
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best_valid = valid
            best_col = col
            best_series = parsed

    if best_series is None or best_valid <= 0:
        return {
            "updated": False,
            "error": "No valid DATE values in master table tail",
            "columns": list(probe.columns),
        }

    reference_date = pd.to_datetime(best_series.dropna()).max()
    reference_date_str = reference_date.strftime("%Y-%m-%d %H:%M:%S")

    last_minute = int(reference_date_str[14:16])
    last_hour = int(reference_date_str[11:13])
    date = datetime.strptime(reference_date_str[:10], "%Y-%m-%d")

    if last_minute == 59:
        last_minute = 0
        last_hour += 1
        if last_hour == 24:
            last_hour = 0
            date += timedelta(days=1)
    else:
        last_minute += 1

    new_date = f"{date.strftime('%Y-%m-%d')} {last_hour:02}:{last_minute:02}:00"

    headers = {"Content-Type": "application/json"}
    url = (
        f"https://api.tiingo.com/tiingo/fx/{symbol}/prices"
        f"?startDate={new_date}&resampleFreq={rate}&token={token}"
    )

    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        return {"updated": False, "error": f"Error fetching data: {response.text}"}

    ds = pd.DataFrame(response.json())
    if ds.empty:
        return {
            "updated": True,
            "new_rows": 0,
            "master_table_path": master_table_path,
            "latest_date": reference_date_str,
        }

    ds.drop(columns=[c for c in ["ticker"] if c in ds.columns], inplace=True)
    ds.rename(
        columns={
            "date": "DATE",
            "open": "OPEN",
            "high": "HIGH",
            "low": "LOW",
            "close": "CLOSE",
        },
        inplace=True,
    )

    ds["DATE"] = pd.to_datetime(ds["DATE"], errors="coerce").dt.tz_localize(None)
    ds = ds.dropna(subset=["DATE"]).sort_values("DATE")
    ds = ds[ds["DATE"] > reference_date]
    if ds.empty:
        return {
            "updated": True,
            "new_rows": 0,
            "master_table_path": master_table_path,
            "latest_date": reference_date_str,
        }

    # Append using the existing header order to avoid schema drift.
    header_df = pd.read_csv(master_table_path, nrows=0)
    existing_cols = list(header_df.columns)

    write_df = ds.copy()
    write_df["DATE"] = write_df["DATE"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "VOLUME" in existing_cols and "VOLUME" not in write_df.columns:
        write_df["VOLUME"] = 0.0

    for col in existing_cols:
        if col not in write_df.columns:
            write_df[col] = ""

    write_df = write_df[existing_cols]
    write_df.to_csv(master_table_path, mode="a", header=False, index=False)

    return {
        "updated": True,
        "new_rows": int(len(write_df)),
        "master_table_path": master_table_path,
        "latest_date": str(ds["DATE"].iloc[-1]),
    }


def update_fx_master_table_db(
    db_path: str,
    rate: str,
    symbol: str,
    token: str,
) -> Dict[str, Any]:
    """Update SQLite ohlc_1m table by appending only new Tiingo rows."""
    if not db_path:
        return {"updated": False, "error": "Missing db_path"}
    if not token:
        return {"updated": False, "error": "Missing Tiingo token"}

    init_market_db(db_path)
    latest = get_latest_date(db_path)
    if latest:
        reference_date = pd.to_datetime(latest, errors="coerce")
    else:
        reference_date = pd.Timestamp("2009-01-01 00:00:00")

    reference_date_str = reference_date.strftime("%Y-%m-%d %H:%M:%S")
    last_minute = int(reference_date_str[14:16])
    last_hour = int(reference_date_str[11:13])
    date = datetime.strptime(reference_date_str[:10], "%Y-%m-%d")

    if last_minute == 59:
        last_minute = 0
        last_hour += 1
        if last_hour == 24:
            last_hour = 0
            date += timedelta(days=1)
    else:
        last_minute += 1

    new_date = f"{date.strftime('%Y-%m-%d')} {last_hour:02}:{last_minute:02}:00"

    headers = {"Content-Type": "application/json"}
    url = (
        f"https://api.tiingo.com/tiingo/fx/{symbol}/prices"
        f"?startDate={new_date}&resampleFreq={rate}&token={token}"
    )

    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code != 200:
        return {"updated": False, "error": f"Error fetching data: {response.text}"}

    ds = pd.DataFrame(response.json())
    if ds.empty:
        return {
            "updated": True,
            "new_rows": 0,
            "db_path": db_path,
            "latest_date": reference_date_str,
        }

    ds.drop(columns=[c for c in ["ticker"] if c in ds.columns], inplace=True)
    ds.rename(
        columns={
            "date": "DATE",
            "open": "OPEN",
            "high": "HIGH",
            "low": "LOW",
            "close": "CLOSE",
        },
        inplace=True,
    )
    if "VOLUME" not in ds.columns:
        ds["VOLUME"] = 0.0

    ds["DATE"] = pd.to_datetime(ds["DATE"], errors="coerce").dt.tz_localize(None)
    ds = ds.dropna(subset=["DATE"]).sort_values("DATE")
    ds = ds[ds["DATE"] > reference_date]
    if ds.empty:
        return {
            "updated": True,
            "new_rows": 0,
            "db_path": db_path,
            "latest_date": reference_date_str,
        }

    n = upsert_ohlc_1m(db_path, ds)
    return {
        "updated": True,
        "new_rows": int(n),
        "db_path": db_path,
        "latest_date": str(ds["DATE"].iloc[-1]),
    }


def _latest_ts_from_csv(master_table_path: str, tail_probe_rows: int = 5000) -> Optional[pd.Timestamp]:
    probe = read_csv_tail(master_table_path, max(int(tail_probe_rows), 50), usecols=None)
    if probe.empty:
        return None

    best_series = None
    best_valid = -1
    for col in list(probe.columns):
        as_text = probe[col].astype(str).str.strip()
        looks_like_date = as_text.str.contains(r"[-/:T]", regex=True, na=False)
        if int(looks_like_date.sum()) <= 0:
            continue
        parsed = pd.to_datetime(as_text.where(looks_like_date, None), errors="coerce")
        valid = int(parsed.notna().sum())
        if valid > best_valid:
            best_valid = valid
            best_series = parsed

    if best_series is None or best_valid <= 0:
        return None
    return pd.to_datetime(best_series.dropna()).max()


def get_master_latest_timestamp(master_table_path: str) -> Optional[pd.Timestamp]:
    if not master_table_path:
        return None
    if str(master_table_path).lower().endswith(".db") or str(master_table_path).lower().endswith(".sqlite"):
        latest = get_latest_date(master_table_path)
        if not latest:
            return None
        return pd.to_datetime(latest, errors="coerce")
    if not os.path.exists(master_table_path):
        return None
    return _latest_ts_from_csv(master_table_path)


def is_master_aligned(
    master_table_path: str,
    freshness_lag_minutes: int = 20,
    now_ts: Optional[datetime] = None,
) -> Dict[str, Any]:
    now_utc = now_ts or datetime.utcnow()
    is_weekend = now_utc.weekday() >= 5

    latest = get_master_latest_timestamp(master_table_path)
    if latest is None or pd.isna(latest):
        return {
            "is_aligned": False,
            "is_weekend": is_weekend,
            "latest_date": None,
            "lag_minutes": None,
            "reason": "no_latest_timestamp",
        }

    latest = pd.to_datetime(latest, errors="coerce")
    if pd.isna(latest):
        return {
            "is_aligned": False,
            "is_weekend": is_weekend,
            "latest_date": None,
            "lag_minutes": None,
            "reason": "invalid_latest_timestamp",
        }

    lag_minutes = (now_utc - latest.to_pydatetime()).total_seconds() / 60.0

    # On weekends do not enforce strict freshness to current minute.
    if is_weekend:
        return {
            "is_aligned": True,
            "is_weekend": True,
            "latest_date": latest.strftime("%Y-%m-%d %H:%M:%S"),
            "lag_minutes": float(lag_minutes),
            "reason": "weekend_relaxed",
        }

    return {
        "is_aligned": bool(lag_minutes <= max(int(freshness_lag_minutes), 1)),
        "is_weekend": False,
        "latest_date": latest.strftime("%Y-%m-%d %H:%M:%S"),
        "lag_minutes": float(lag_minutes),
        "reason": "freshness_check",
    }


def bootstrap_master_on_backend_start(
    master_table_path: str,
    rate: str,
    symbol: str,
    token: str,
    max_pulls: int = 2,
    freshness_lag_minutes: int = 20,
) -> Dict[str, Any]:
    """Attempt to sync master source on backend startup, up to max_pulls attempts."""
    if not master_table_path:
        return {"ok": False, "error": "Missing master table path"}
    if not token:
        return {"ok": False, "error": "Missing Tiingo token"}

    attempts = []
    pull_count = max(int(max_pulls), 1)

    for i in range(1, pull_count + 1):
        if str(master_table_path).lower().endswith(".db") or str(master_table_path).lower().endswith(".sqlite"):
            pull_res = update_fx_master_table_db(
                db_path=master_table_path,
                rate=rate,
                symbol=symbol,
                token=token,
            )
        else:
            pull_res = update_fx_master_table_file(
                master_table_path=master_table_path,
                rate=rate,
                symbol=symbol,
                token=token,
            )

        align = is_master_aligned(
            master_table_path=master_table_path,
            freshness_lag_minutes=freshness_lag_minutes,
        )
        attempts.append({"attempt": i, "pull": pull_res, "alignment": align})

        if bool(align.get("is_aligned", False)):
            return {
                "ok": True,
                "attempts": attempts,
                "aligned": True,
                "latest_date": align.get("latest_date"),
                "is_weekend": bool(align.get("is_weekend", False)),
            }

    final_align = attempts[-1]["alignment"] if attempts else {}
    return {
        "ok": bool(final_align.get("is_aligned", False)),
        "attempts": attempts,
        "aligned": bool(final_align.get("is_aligned", False)),
        "latest_date": final_align.get("latest_date"),
        "is_weekend": bool(final_align.get("is_weekend", False)),
    }
