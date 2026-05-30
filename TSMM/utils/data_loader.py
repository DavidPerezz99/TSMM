"""
Data Loader Module

This module provides functionality for loading and preprocessing time series data.
"""

import pandas as pd
import yaml
import logging
import hashlib
import json
import os
import re
import time
from utils.market_db import query_ohlc


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
            timeframe_minutes = int(config.get('data_timeframe_minutes', 0) or 0)
            if timeframe_minutes <= 0:
                timeframe_minutes = _infer_timeframe_minutes(data_path_str, fallback=1)
            latest_records = int(
                config.get('sql_latest_records', config.get('records', 50000)) or 50000
            )
            logging.info(
                "Loading data from SQL source. sql_source=%s requested_path=%s timeframe_minutes=%s latest_records=%s start_date=%s end_date=%s",
                sql_source_path,
                data_path_str,
                timeframe_minutes,
                latest_records,
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
            )
            logging.info(
                "Finished SQL data load. rows=%s elapsed_seconds=%.3f source=%s timeframe_minutes=%s",
                len(df),
                time.time() - started,
                sql_source_path,
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
        
        logging.info("Adding engineered features")
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
