"""
Data Loader Module

This module provides functionality for loading and preprocessing time series data.
"""

import pandas as pd
import yaml
import logging
import hashlib
import json
import math
import os
import re
import time
from utils.market_db import query_ohlc


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cross_asset_sources(config):
    raw = (config or {}).get('cross_asset_features')
    if not raw:
        return []
    if isinstance(raw, dict):
        return [raw] if bool(raw.get('enabled', True)) else []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict) and bool(item.get('enabled', True))]
    raise ValueError("cross_asset_features must be a mapping or list of mappings")


def merge_cross_asset_frame(target_frame, exogenous_frame, source_cfg, timeframe_minutes):
    """Attach prefixed OHLC features using a backward-only as-of merge."""
    target = target_frame.copy()
    exogenous = exogenous_frame.copy()
    if 'DATE' not in target.columns or 'DATE' not in exogenous.columns:
        raise ValueError("Cross-asset merge requires DATE in both frames")
    source_symbol = str(source_cfg.get('symbol') or '').strip().upper()
    prefix = str(source_cfg.get('prefix') or source_symbol).strip().upper()
    prefix = re.sub(r'[^A-Z0-9_]+', '_', prefix).strip('_')
    if not prefix:
        raise ValueError("cross_asset_features.prefix resolved to an empty value")
    max_staleness = max(
        int(source_cfg.get('max_staleness_minutes', timeframe_minutes) or 0),
        0,
    )
    target['DATE'] = pd.to_datetime(target['DATE'], errors='coerce')
    exogenous['DATE'] = pd.to_datetime(exogenous['DATE'], errors='coerce')
    target = target.dropna(subset=['DATE']).sort_values('DATE')
    exogenous = exogenous.dropna(subset=['DATE']).sort_values('DATE')
    for column in ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']:
        if column not in exogenous.columns:
            exogenous[column] = 0.0 if column == 'VOLUME' else float('nan')
        exogenous[column] = pd.to_numeric(exogenous[column], errors='coerce')
        exogenous[f'{prefix}_{column}'] = exogenous[column]
    exogenous[f'{prefix}_Price_return'] = exogenous['CLOSE'].diff()
    exogenous[f'{prefix}_Open_return'] = exogenous['OPEN'].diff()
    exogenous[f'{prefix}_High_return'] = exogenous['HIGH'].diff()
    exogenous[f'{prefix}_Low_return'] = exogenous['LOW'].diff()
    feature_columns = [
        f'{prefix}_OPEN', f'{prefix}_HIGH', f'{prefix}_LOW',
        f'{prefix}_CLOSE', f'{prefix}_VOLUME',
        f'{prefix}_Price_return', f'{prefix}_Open_return',
        f'{prefix}_High_return', f'{prefix}_Low_return',
    ]
    source_date_column = f'__{prefix}_SOURCE_DATE'
    exogenous[source_date_column] = exogenous['DATE']
    merged = pd.merge_asof(
        target,
        exogenous[[source_date_column] + feature_columns],
        left_on='DATE',
        right_on=source_date_column,
        direction='backward',
        allow_exact_matches=True,
        tolerance=pd.Timedelta(minutes=max_staleness),
    )
    matched = merged[source_date_column].notna()
    coverage = float(matched.mean()) if len(merged) else 0.0
    minimum_coverage = float(source_cfg.get('minimum_coverage', 0.0) or 0.0)
    if coverage < minimum_coverage:
        raise ValueError(
            f"Cross-asset coverage for {source_symbol} is {coverage:.2%}, "
            f"below {minimum_coverage:.2%}"
        )
    if bool(source_cfg.get('require_match', True)):
        merged = merged[matched].copy()
    source_age_minutes = (
        (merged['DATE'] - merged[source_date_column]).dt.total_seconds() / 60.0
    )
    stale = source_age_minutes > 0.0
    for column in (
        f'{prefix}_Price_return',
        f'{prefix}_Open_return',
        f'{prefix}_High_return',
        f'{prefix}_Low_return',
    ):
        merged.loc[stale, column] = 0.0
    merged[f'{prefix}_AGE_MINUTES'] = source_age_minutes
    merged = merged.drop(columns=[source_date_column])
    return merged.sort_values('DATE').reset_index(drop=True), coverage


def _merge_cross_asset_features(df, config, timeframe_minutes):
    """Merge only same/past exogenous candles with a bounded staleness rule."""
    sources = _cross_asset_sources(config)
    if not sources or df is None or df.empty:
        return df

    target = df.copy().sort_index()
    if not isinstance(target.index, pd.DatetimeIndex):
        target.index = pd.to_datetime(target.index, errors='coerce')
    target = target[~target.index.isna()].sort_index()
    target_name = target.index.name or str(config.get('date_col') or 'DATE')

    for source_cfg in sources:
        source_symbol = str(source_cfg.get('symbol') or '').strip().upper()
        if not source_symbol:
            raise ValueError("cross_asset_features.symbol is required")
        prefix = str(source_cfg.get('prefix') or source_symbol).strip().upper()
        prefix = re.sub(r'[^A-Z0-9_]+', '_', prefix).strip('_')
        requested_features = {
            str(value) for value in (config.get('input_features') or []) if str(value)
        }
        if not any(feature.startswith(f'{prefix}_') for feature in requested_features):
            # A sweep can compare baseline and cross-asset recipes in the same
            # session.  Do not silently restrict the baseline recipe to the
            # exogenous asset's trading hours when it consumes no such column.
            continue
        source_path = _resolve_repo_path(
            str(source_cfg.get('db_path') or config.get('data_path') or '').strip()
        )
        if not source_path.lower().endswith(('.db', '.sqlite')):
            raise ValueError("cross_asset_features.db_path must be a SQLite database")
        max_staleness = max(
            int(source_cfg.get('max_staleness_minutes', timeframe_minutes) or 0),
            0,
        )
        source_mode = str(source_cfg.get('source_mode') or 'auto').strip().lower()
        start = target.index.min() - pd.Timedelta(minutes=max_staleness)
        end = target.index.max()
        span_records = int(
            math.ceil(
                max((end - start).total_seconds(), 0.0)
                / (max(int(timeframe_minutes), 1) * 60.0)
            )
        ) + 5
        exogenous = query_ohlc(
            db_path=source_path,
            timeframe_minutes=int(max(timeframe_minutes, 1)),
            latest_records=max(int(len(target) * 3), span_records, 500),
            start_date=start.strftime('%Y-%m-%d %H:%M:%S'),
            end_date=end.strftime('%Y-%m-%d %H:%M:%S'),
            symbol=source_symbol,
            source_mode=source_mode,
        )
        if exogenous is None or exogenous.empty:
            raise ValueError(
                f"No cross-asset rows returned for {source_symbol} from {source_path}"
            )
        left = target.reset_index()
        date_column = left.columns[0]
        left = left.rename(columns={date_column: 'DATE'}).sort_values('DATE')
        merged, coverage = merge_cross_asset_frame(
            left,
            exogenous,
            source_cfg,
            timeframe_minutes,
        )
        target = merged.set_index('DATE').sort_index()
        target.index.name = target_name
        logging.info(
            "Merged cross-asset features. symbol=%s prefix=%s rows=%s coverage=%.4f "
            "direction=backward max_staleness_minutes=%s",
            source_symbol,
            prefix,
            len(target),
            coverage,
            max_staleness,
        )
    return target


def _resolve_repo_path(path_value: str) -> str:
    p = str(path_value or '').strip()
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(REPO_ROOT, p))


def get_memory():
    """Lazy import of memory to avoid circular imports."""
    from .cache import memory
    return memory


def get_data_hash(data_path, config):
    """Generate unique hash for dataset based on file and configuration."""
    try:
        with open(data_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        
        config_copy = config.copy()
        config_copy.pop('data_path', None)
        config_str = json.dumps(config_copy, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        return hashlib.md5(f"{file_hash}_{config_hash}".encode()).hexdigest()
    except Exception as e:
        logging.warning(f"Could not generate data hash: {str(e)}")
        return str(hash((data_path, json.dumps(config, sort_keys=True))))


def load_data_cached(data_path, date_col, target_col, config):
    """Cached version of load_data."""
    cfg = config or {}
    cache_cfg = cfg.get('cache', {}) if isinstance(cfg, dict) else {}
    cache_enabled = bool(cache_cfg.get('enabled', True))

    data_path_str = str(data_path or '').strip().lower()
    sql_master_first = bool(cfg.get('use_sql_master', True)) if isinstance(cfg, dict) else True
    is_sql_path = data_path_str.endswith('.db') or data_path_str.endswith('.sqlite')

    # SQL-master and SQL-path runs are dynamic; bypass joblib cache to avoid stale/locked cache states.
    if (not cache_enabled) or is_sql_path or sql_master_first:
        return load_data(data_path, date_col, target_col, cfg)

    memory = get_memory()
    try:
        return memory.cache(load_data)(data_path, date_col, target_col, cfg)
    except Exception as e:
        logging.warning(f"Cache read failed, loading data directly: {str(e)}")
        return load_data(data_path, date_col, target_col, cfg)


def load_data(data_path, date_col, target_col, config):
    """
    Load and preprocess time series data.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file
    date_col : str
        Name of date column
    target_col : str
        Name of target column
    config : dict
        Configuration dictionary
    
    Returns:
    --------
    pd.DataFrame
        Processed dataframe with datetime index
    """
    try:
        config = config or {}
        data_path_str = _resolve_repo_path(str(data_path or '').strip())
        db_default = os.path.join('data', 'market_data.sqlite')
        sql_master_path = _resolve_repo_path(str(config.get('sql_master_path') or db_default).strip())
        use_sql_master = bool(config.get('use_sql_master', True))

        def _infer_timeframe_minutes(path_value: str, fallback: int = 1) -> int:
            m = re.search(r"(\d+)(m|h|d|w)", str(path_value or '').lower())
            if not m:
                return int(max(fallback, 1))
            value = int(m.group(1))
            unit = m.group(2)
            if unit == 'm':
                return max(value, 1)
            if unit == 'h':
                return max(value * 60, 1)
            if unit == 'd':
                return max(value * 1440, 1)
            if unit == 'w':
                return max(value * 10080, 1)
            return int(max(fallback, 1))

        timeframe_minutes = int(config.get('data_timeframe_minutes', 0) or 0)
        if timeframe_minutes <= 0:
            timeframe_minutes = _infer_timeframe_minutes(data_path_str, fallback=1)

        should_use_sql = False
        sql_source_path = data_path_str
        if data_path_str.lower().endswith('.db') or data_path_str.lower().endswith('.sqlite'):
            should_use_sql = True
            sql_source_path = data_path_str
        elif use_sql_master:
            # Enforce SQL master as source of truth for legacy timeframe configs.
            should_use_sql = True
            sql_source_path = sql_master_path

        if should_use_sql:
            sql_symbol = str(
                config.get('sql_symbol')
                or config.get('symbol')
                or config.get('tiingo_symbol')
                or 'XAUUSD'
            ).strip()
            latest_records = int(
                config.get('sql_latest_records', config.get('records', 50000)) or 50000
            )
            sql_source_mode = str(config.get('sql_source_mode', 'auto') or 'auto').strip().lower()
            logging.info(
                "Loading data from SQL source. sql_source=%s requested_path=%s symbol=%s timeframe_minutes=%s latest_records=%s source_mode=%s start_date=%s end_date=%s",
                sql_source_path,
                data_path_str,
                sql_symbol,
                timeframe_minutes,
                latest_records,
                sql_source_mode,
                config.get('start_date') or None,
                config.get('end_date') or None,
            )
            started = time.time()
            df = query_ohlc(
                db_path=sql_source_path,
                timeframe_minutes=timeframe_minutes,
                latest_records=latest_records,
                start_date=(config.get('start_date') or None),
                end_date=(config.get('end_date') or None),
                symbol=sql_symbol,
                source_mode=sql_source_mode,
            )
            logging.info(
                "Finished SQL data load. rows=%s elapsed_seconds=%.3f source=%s symbol=%s timeframe_minutes=%s",
                len(df),
                time.time() - started,
                sql_source_path,
                sql_symbol,
                timeframe_minutes,
            )
            if df.empty:
                raise ValueError(f"No rows returned from SQLite source: {sql_source_path}")
        else:
            logging.info("Loading data from CSV source: %s", data_path_str)
            started = time.time()
            df = pd.read_csv(data_path_str, parse_dates=[date_col])
            logging.info(
                "Finished CSV data load. rows=%s elapsed_seconds=%.3f source=%s",
                len(df),
                time.time() - started,
                data_path_str,
            )
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        if df[date_col].isnull().any():
            missing_count = df[date_col].isnull().sum()
            logging.warning(f"Found {missing_count} invalid dates. Filling with linear interpolation")
            df[date_col] = df[date_col].interpolate(method='linear')
        
        df.set_index(date_col, inplace=True)
        df.index = pd.DatetimeIndex(df.index)
        df.sort_index(inplace=True)

        df = _merge_cross_asset_features(df, config, timeframe_minutes)
        
        logging.info("Adding engineered features")
        # Use CLOSE.diff() for Price_return to match inference pipeline
        # (inference computes Price_return from CLOSE, not from target_col).
        if 'CLOSE' in df.columns:
            df['Price_return'] = df['CLOSE'].diff()
        else:
            df['Price_return'] = df[target_col].diff()
        if 'OPEN' in df.columns:
            df['Open_return'] = df['OPEN'].diff()
        if 'HIGH' in df.columns:
            df['High_return'] = df['HIGH'].diff()
        if 'LOW' in df.columns:
            df['Low_return'] = df['LOW'].diff()
        if 'OPEN' in df.columns and target_col in df.columns:
            df['daily_return'] = df[target_col] - df['OPEN']
        df['y_diff'] = df[target_col].diff()
        
        rolling_windows = config.get('rolling_windows', [7, 30, 60])
        for window in rolling_windows:
            df[f'SMA_{window}_diff'] = df['y_diff'].rolling(window=window).mean()
            df[f'EMA_{window}_diff'] = df['y_diff'].ewm(span=window, adjust=False).mean()
            df[f'Volatility_{window}_diff'] = df['y_diff'].rolling(window=window).std()
            
            df[f'SMA_{window}'] = df[target_col].rolling(window=window).mean()
            df[f'EMA_{window}'] = df[target_col].ewm(span=window, adjust=False).mean()
            df[f'Volatility_{window}'] = df[target_col].rolling(window=window).std()
        
        df = df.dropna().copy()
        df.index = pd.DatetimeIndex(df.index)
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        logging.info(f"Data loaded successfully. Shape: {df.shape}, Date range: {df.index.min()} to {df.index.max()}")
        return df
    
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise
