"""
Sequence Utilities Module

Provides functions for preparing sequences for time series models.
"""

import numpy as np
import pandas as pd
import hashlib
import json


def get_memory():
    """Lazy import of memory to avoid circular imports."""
    from .cache import memory
    return memory


def get_sequences_hash(df, input_features, target_features, n_steps, m_steps):
    """Generate unique hash for sequences based on data and parameters."""
    data_hash = hashlib.md5(
        pd.util.hash_pandas_object(df[input_features + target_features]).values.tobytes()
    ).hexdigest()
    
    params_str = json.dumps({
        'input_features': sorted(input_features),
        'target_features': sorted(target_features),
        'n_steps': n_steps,
        'm_steps': m_steps
    }, sort_keys=True)
    
    return hashlib.md5(f"{data_hash}_{params_str}".encode()).hexdigest()


def prepare_sequences_cached(df, input_features, target_features, n_steps, m_steps):
    """Cached version of prepare_sequences."""
    memory = get_memory()
    return memory.cache(prepare_sequences)(df, input_features, target_features, n_steps, m_steps)


def prepare_sequences(df, input_features, target_features, n_steps, m_steps):
    """
    Prepare sequences for training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    input_features : list
        List of input feature names
    target_features : list
        List of target feature names
    n_steps : int
        Lookback window size
    m_steps : int
        Forecast horizon
    
    Returns:
    --------
    tuple
        (X, y) arrays for training
    """
    X, y = [], []
    for i in range(len(df) - n_steps - m_steps + 1):
        X.append(df[input_features].iloc[i:i + n_steps].values)
        y.append(df[target_features].iloc[i + n_steps:i + n_steps + m_steps].values)
    return np.array(X), np.array(y)
