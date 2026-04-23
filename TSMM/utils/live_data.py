"""
Live FX data utilities for Tiingo updates, aggregation, and enrichment.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple

import pandas as pd
import requests


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

    probe = read_csv_tail(master_table_path, max(int(tail_probe_rows), 10), usecols=[date_col])
    if probe.empty or date_col not in probe.columns:
        return {"updated": False, "error": f"Could not read tail date column '{date_col}'"}

    probe[date_col] = pd.to_datetime(probe[date_col], errors="coerce")
    probe = probe.dropna(subset=[date_col])
    if probe.empty:
        return {"updated": False, "error": "No valid DATE values in master table tail"}

    reference_date = pd.to_datetime(probe[date_col]).max()
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

    write_df = ds.copy()
    write_df["DATE"] = write_df["DATE"].dt.strftime("%Y-%m-%d %H:%M:%S")
    write_df.to_csv(master_table_path, mode="a", header=False, index=False)

    return {
        "updated": True,
        "new_rows": int(len(write_df)),
        "master_table_path": master_table_path,
        "latest_date": str(ds["DATE"].iloc[-1]),
    }
