"""
Data Loader Module

This module provides functionality for loading and preprocessing time series data.
"""

import pandas as pd
import yaml
import logging
import hashlib
import json


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
    memory = get_memory()
    return memory.cache(load_data)(data_path, date_col, target_col, config)


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
        df = pd.read_csv(data_path, parse_dates=[date_col], infer_datetime_format=True)
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        if df[date_col].isnull().any():
            missing_count = df[date_col].isnull().sum()
            logging.warning(f"Found {missing_count} invalid dates. Filling with linear interpolation")
            df[date_col] = df[date_col].interpolate(method='linear')
        
        df.set_index(date_col, inplace=True)
        df.index = pd.DatetimeIndex(df.index)
        df.sort_index(inplace=True)
        
        logging.info("Adding engineered features")
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
