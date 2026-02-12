"""
Metrics Saver Module

This module provides functionality to save evaluation metrics from model training
to various formats (JSON, CSV) for later analysis and comparison.
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np


def _jsonify(obj):
    """Convert numpy types and other non-serializable objects to Python native types."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_metrics_to_json(
    metrics: Dict[str, Any],
    output_path: str,
    config: Optional[Dict] = None,
    model_name: Optional[str] = None
) -> str:
    """
    Save metrics to a JSON file.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics to save
    output_path : str
        Path to save the JSON file
    config : dict, optional
        Configuration dictionary to include in the output
    model_name : str, optional
        Name of the model for the metrics
    
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Prepare the payload
    payload = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        'metrics': _jsonify(metrics),
    }
    
    if config:
        # Include relevant config parameters
        config_to_save = {
            'data_path': config.get('data_path'),
            'target_col': config.get('target_col'),
            'problem_type': config.get('problem_type'),
            'n_steps': config.get('n_steps'),
            'm_steps': config.get('m_steps'),
            'horizon': config.get('horizon'),
            'split_ratio': config.get('split_ratio'),
            'input_features': config.get('input_features'),
            'target_features': config.get('target_features'),
        }
        payload['config'] = _jsonify(config_to_save)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    
    return output_path


def save_all_models_metrics(
    evaluation: Dict[str, Dict],
    output_dir: str,
    config: Optional[Dict] = None
) -> str:
    """
    Save metrics for all models to a single JSON file.
    
    Parameters:
    -----------
    evaluation : dict
        Dictionary containing evaluation results for all models
    output_dir : str
        Directory to save the metrics file
    config : dict, optional
        Configuration dictionary
    
    Returns:
    --------
    str
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'all_models_metrics_{timestamp}.json')
    
    # Extract metrics from evaluation
    all_metrics = {}
    for model_name, eval_data in evaluation.items():
        if 'metrics' in eval_data:
            all_metrics[model_name] = {
                'metrics': eval_data.get('metrics', {}),
                'exog_metrics': eval_data.get('exog_metrics', {}),
                'confusion_matrix': eval_data.get('confusion_matrix', {}),
                'classification_report': eval_data.get('classification_report', {}),
            }
    
    payload = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'models': _jsonify(all_metrics),
    }
    
    if config:
        payload['config'] = _jsonify({
            'data_path': config.get('data_path'),
            'target_col': config.get('target_col'),
            'problem_type': config.get('problem_type'),
            'n_steps': config.get('n_steps'),
            'm_steps': config.get('m_steps'),
            'horizon': config.get('horizon'),
        })
    
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    
    return output_path


def save_metrics_to_csv(
    evaluation: Dict[str, Dict],
    output_path: str
) -> str:
    """
    Save metrics for all models to a CSV file.
    
    Parameters:
    -----------
    evaluation : dict
        Dictionary containing evaluation results for all models
    output_path : str
        Path to save the CSV file
    
    Returns:
    --------
    str
        Path to the saved file
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    rows = []
    for model_name, eval_data in evaluation.items():
        if 'metrics' in eval_data:
            row = {'model': model_name}
            row.update(eval_data.get('metrics', {}))
            
            # Add confusion matrix metrics if available
            cm_metrics = eval_data.get('confusion_matrix', {})
            if cm_metrics:
                row['cm_accuracy'] = cm_metrics.get('accuracy')
                row['cm_precision'] = cm_metrics.get('precision')
                row['cm_recall'] = cm_metrics.get('recall')
                row['cm_f1'] = cm_metrics.get('f1')
            
            rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    return output_path


def save_forecast_to_file(
    future_forecasts: Dict[str, Any],
    output_path: str,
    output_format: str,
    config: Dict,
    df_last_date: Optional[datetime] = None,
    confidence_levels: Optional[Dict[str, List]] = None
) -> str:
    """
    Save horizon forecasts to a file (CSV or Parquet).
    
    Parameters:
    -----------
    future_forecasts : dict
        Dictionary containing forecasts from each model
    output_path : str
        Path to save the output file
    output_format : str
        Format of the output file ('csv' or 'parquet')
    config : dict
        Configuration dictionary containing date_col, horizon, etc.
    df_last_date : datetime, optional
        Last date from the dataframe to calculate future dates
    confidence_levels : dict, optional
        Dictionary containing confidence levels for each model's forecasts
    
    Returns:
    --------
    str
        Path to the saved file
    """
    if not future_forecasts:
        raise ValueError("No forecasts to save")
    
    # Get configuration parameters
    horizon = config.get('horizon', 60)
    date_col = config.get('date_col', 'DATE')
    target_col = config.get('target_col', 'LOW')
    target_features = config.get('target_features', [target_col])
    
    # Create date range for forecasts
    if df_last_date is None:
        df_last_date = datetime.now()
    
    future_dates = pd.date_range(
        start=df_last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq='D'
    )
    
    # Build DataFrame with forecasts from all models
    forecast_data = {'date': future_dates}
    
    for model_name, forecast_info in future_forecasts.items():
        if 'future' in forecast_info:
            future_values = forecast_info['future']
            
            # Handle different forecast structures
            if isinstance(future_values, list) and len(future_values) > 0:
                # Handle nested list structure (samples, m_steps, n_targets)
                if isinstance(future_values[0], list):
                    # Extract first timestep for each sample
                    for t_idx, target_feat in enumerate(target_features):
                        values = []
                        for f in future_values[:horizon]:
                            if isinstance(f, list) and len(f) > 0:
                                if isinstance(f[0], list):
                                    values.append(f[0][t_idx] if len(f[0]) > t_idx else f[0][0])
                                else:
                                    values.append(f[t_idx] if len(f) > t_idx else f[0])
                            else:
                                values.append(f)
                        forecast_data[f'{model_name}_{target_feat}_forecast'] = values
                else:
                    # Simple list of values
                    forecast_data[f'{model_name}_{target_features[0]}_forecast'] = future_values[:horizon]
            elif isinstance(future_values, np.ndarray):
                # Handle numpy array
                if future_values.ndim >= 2:
                    for t_idx, target_feat in enumerate(target_features):
                        if future_values.ndim == 3:
                            values = future_values[:horizon, 0, t_idx].tolist()
                        else:
                            values = future_values[:horizon, t_idx].tolist()
                        forecast_data[f'{model_name}_{target_feat}_forecast'] = values
                else:
                    forecast_data[f'{model_name}_{target_features[0]}_forecast'] = future_values[:horizon].tolist()
    
    # Add confidence levels if provided
    if confidence_levels:
        for model_name, conf_values in confidence_levels.items():
            if len(conf_values) >= horizon:
                forecast_data[f'{model_name}_confidence'] = conf_values[:horizon]
    
    # Create DataFrame
    df_forecast = pd.DataFrame(forecast_data)
    
    # Save to file
    output_format = output_format.lower()
    if output_format == 'csv':
        df_forecast.to_csv(output_path, index=False)
    elif output_format == 'parquet':
        df_forecast.to_parquet(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")
    
    return output_path


def load_metrics_from_json(metrics_path: str) -> Dict[str, Any]:
    """
    Load metrics from a JSON file.
    
    Parameters:
    -----------
    metrics_path : str
        Path to the JSON file
    
    Returns:
    --------
    dict
        Loaded metrics
    """
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compare_metrics(
    current_metrics: Dict[str, Any],
    previous_metrics_path: str
) -> Dict[str, Any]:
    """
    Compare current metrics with previously saved metrics.
    
    Parameters:
    -----------
    current_metrics : dict
        Current metrics dictionary
    previous_metrics_path : str
        Path to the previous metrics JSON file
    
    Returns:
    --------
    dict
        Comparison results showing improvements/regressions
    """
    previous_data = load_metrics_from_json(previous_metrics_path)
    previous_metrics = previous_data.get('models', {})
    
    comparison = {}
    
    for model_name, current_data in current_metrics.items():
        if model_name in previous_metrics:
            prev_data = previous_metrics[model_name]
            model_comparison = {}
            
            current_m = current_data.get('metrics', {})
            previous_m = prev_data.get('metrics', {})
            
            for metric_name in ['MAE', 'RMSE', 'MAPE', 'R2']:
                if metric_name in current_m and metric_name in previous_m:
                    curr_val = current_m[metric_name]
                    prev_val = previous_m[metric_name]
                    
                    if isinstance(curr_val, (int, float)) and isinstance(prev_val, (int, float)):
                        diff = curr_val - prev_val
                        pct_change = (diff / prev_val * 100) if prev_val != 0 else 0
                        
                        model_comparison[metric_name] = {
                            'current': curr_val,
                            'previous': prev_val,
                            'difference': diff,
                            'percent_change': pct_change,
                            'improved': (metric_name == 'R2' and diff > 0) or 
                                       (metric_name != 'R2' and diff < 0)
                        }
            
            comparison[model_name] = model_comparison
    
    return comparison
