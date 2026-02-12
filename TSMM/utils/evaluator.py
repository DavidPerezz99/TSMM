"""
Evaluator Module

This module provides comprehensive evaluation functionality for time series forecasting models,
including confusion matrix metrics for trend direction classification and confidence interval
estimation.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from datetime import datetime
from typing import List, Dict, Tuple, Any  # Add this import
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from sklearn.preprocessing import StandardScaler
import holidays 
import joblib
import logging
import tempfile
from pathlib import Path
import seaborn as sns
import time
import matplotlib.dates as mdates
from models.multivariate_models import grid_search_sarimax
from utils.sequence_utils import prepare_sequences_cached as prepare_sequences
import torch

def save_confusion_matrix_plot(
    cm: np.ndarray,
    target_labels: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6)
) -> str:
    """
    Save a confusion matrix plot to a temporary file.
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix
    target_labels : list
        List of target labels
    title : str
        Plot title
    figsize : tuple
        Figure size
    
    Returns:
    --------
    str
        Path to the saved plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_labels, yticklabels=target_labels, ax=ax)
    ax.set_title(title, weight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    
    return tmp.name


def calculate_confusion_matrix_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.0,
    target_labels: List[str] = None
) -> Dict[str, Any]:
    """
    Calculate confusion matrix and classification metrics for trend direction.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted target values
    threshold : float
        Decision threshold for binary classification
    target_labels : list
        Labels for the classes (e.g., ['Downtrend', 'Uptrend'])
    
    Returns:
    --------
    dict
        Dictionary containing confusion matrix and classification metrics
    """
    if target_labels is None:
        target_labels = ['Downtrend', 'Uptrend']
    
    # Binarize predictions and true values based on threshold
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    
    result = {
        'confusion_matrix': cm.tolist(),
        'target_labels': target_labels,
        'threshold': threshold,
    }
    
    # Check if confusion matrix is valid for unpacking
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        result['true_negatives'] = int(tn)
        result['false_positives'] = int(fp)
        result['false_negatives'] = int(fn)
        result['true_positives'] = int(tp)
        
        # Calculate additional metrics
        result['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
        result['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        result['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        result['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Classification report
        result['classification_report'] = classification_report(
            y_true_binary, y_pred_binary, 
            target_names=target_labels, 
            zero_division=0,
            output_dict=True
        )
    else:
        result['note'] = 'Confusion matrix is not 2x2. Some metrics unavailable.'
        result['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
    
    return result


def evaluate_model_with_confusion_matrices(
    model_name: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    target_features: List[str],
    threshold: float = 0.0,
    target_labels: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate the model and generate confusion matrices for each target separately.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    y_test : np.ndarray
        Test target data
    y_pred : np.ndarray
        Predicted target data
    target_features : list
        List of target feature names
    threshold : float
        Decision threshold for binary classification
    target_labels : list
        List of labels (e.g., ['Downtrend', 'Uptrend'])
    
    Returns:
    --------
    dict
        Dictionary containing confusion matrix results and plot paths
    """
    if target_labels is None:
        target_labels = ['Sell', 'Buy']
    
    results = {
        'confusion_matrices': {},
        'figures': [],
        'summary': {}
    }
    
    # Ensure arrays are 2D
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # Evaluate for each target feature (column)
    for i, feature in enumerate(target_features):
        if i >= y_test.shape[1]:
            break
            
        y_true_feat = y_test[:, i]
        y_pred_feat = y_pred[:, i]
        
        # Calculate confusion matrix metrics
        cm_metrics = calculate_confusion_matrix_metrics(
            y_true_feat, y_pred_feat, threshold, target_labels
        )
        
        results['confusion_matrices'][feature] = cm_metrics
        
        # Generate confusion matrix plot
        cm = np.array(cm_metrics['confusion_matrix'])
        plot_path = save_confusion_matrix_plot(
            cm, target_labels, 
            title=f'{model_name} - Confusion Matrix: {feature}'
        )
        results['figures'].append(plot_path)
    
    # Create summary for primary target
    if target_features and target_features[0] in results['confusion_matrices']:
        primary_cm = results['confusion_matrices'][target_features[0]]
        results['summary'] = {
            'accuracy': primary_cm.get('accuracy'),
            'precision': primary_cm.get('precision'),
            'recall': primary_cm.get('recall'),
            'f1': primary_cm.get('f1'),
            'true_positives': primary_cm.get('true_positives'),
            'true_negatives': primary_cm.get('true_negatives'),
            'false_positives': primary_cm.get('false_positives'),
            'false_negatives': primary_cm.get('false_negatives'),
        }
    
    return results


def forecast_sarimax(model, horizon, exog_test=None):
    """Generate forecasts with confidence intervals"""
    forecast_obj = model.get_forecast(steps=horizon, exog=exog_test)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()
    return forecast, conf_int.values


def save_forecast_plot(ds, yhat, yhat_lower, yhat_upper, *,
                       n_future=None,
                       title="Forecast",
                       xlabel="Date", ylabel="Value",
                       color="#0072B2", alpha_band=0.25,
                       figsize=(12, 6), dpi=300):
    """
    Save a forecast plot with confidence intervals.
    """
    ds = pd.to_datetime(ds)
    yhat = np.asarray(yhat)
    yhat_lower = np.asarray(yhat_lower)
    yhat_upper = np.asarray(yhat_upper)

    if n_future is not None:
        ds = ds[:n_future]
        yhat = yhat[:n_future]
        yhat_lower = yhat_lower[:n_future]
        yhat_upper = yhat_upper[:n_future]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(ds, yhat, color=color, lw=2, label="Forecast")
    ax.fill_between(ds, yhat_lower, yhat_upper,
                    color=color, alpha=alpha_band,
                    label="Uncertainty interval")

    ax.set_title(title, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.autofmt_xdate()

    sns.despine(ax=ax, trim=True)
    ax.grid(True, which="major", alpha=0.3)

    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return fig, tmp.name  


def prepare_data(df, target_col='GROSS_ADDS'):
    """Prepare time series data with exogenous features"""
    df = df.asfreq('D')
    
    us_holidays = holidays.UnitedStates()
    df['holiday'] = df.index.map(lambda x: x in us_holidays).astype(float)
    df['dow'] = df.index.dayofweek.astype(float)
    df['month'] = df.index.month.astype(float)
    df['quarter'] = df.index.quarter.astype(float)
    
    dow_dummies = pd.get_dummies(df['dow'], prefix='dow', drop_first=True).astype(float)
    df = pd.concat([df, dow_dummies], axis=1)
    
    df[target_col] = df[target_col].interpolate(method='linear')
    
    for col in df.select_dtypes(include=['int', 'bool']).columns:
        df[col] = df[col].astype(float)
    
    return df


def recursive_forecast(model, scalers, initial_window, steps, model_type, n_steps, n_features, config, max_window, input_features, target_features, m_steps):
    """Recursive forecasting engine for sequence models."""
    predictions = []
    current_window = initial_window.copy()
    
    history_gross = []
    history_diff = []
    
    last_gross = initial_window[-1, input_features.index(config['target_col'])]  
    
    gross_idx = input_features.index(config['target_col'])
    history_gross = list(initial_window[:, gross_idx].flatten())  
    if 'y_diff' in input_features:
        diff_idx = input_features.index('y_diff')
        history_diff = list(initial_window[:, diff_idx].flatten())  
    
    iterations = (steps + m_steps - 1) // m_steps
    
    for _ in range(iterations):
        X_scaled = scalers['X'].transform(current_window)  
        
        if model_type in ['ulr', 'mlr']:
            X_flat = X_scaled.reshape(1, -1)
            y_pred_scaled = model.predict(X_flat)
            y_pred_2d = y_pred_scaled.reshape(-1, len(target_features))
        else:
            X_3d = X_scaled.reshape(1, n_steps, n_features)
            y_pred_scaled = model.predict(X_3d)
            y_pred_2d = y_pred_scaled.reshape(-1, len(target_features))
        
        y_pred = scalers['y'].inverse_transform(y_pred_2d)
        
        predictions.append(y_pred)
        
        new_features, last_gross, history_gross, history_diff = update_features(
            y_pred, last_gross, input_features, target_features, config, max_window,
            history_gross, history_diff
        )
        
        current_window = np.vstack([current_window[new_features.shape[0]:], new_features])
    
    return np.vstack(predictions)[:steps]


def update_features(predictions, last_gross, input_features, target_features,
                    config, max_window, history_gross, history_diff):
    """Update features for recursive forecasting."""
    updated_features = []
    current_gross = last_gross

    for pred in predictions:
        if 'y_diff' in target_features:
            y_diff_idx = target_features.index('y_diff')
            y_diff = pred[y_diff_idx]
            current_gross += y_diff
        elif config['target_col'] in target_features:
            gross_idx = target_features.index(config['target_col'])
            current_gross = pred[gross_idx]
            y_diff = current_gross - last_gross
        else:
            y_diff = 0

        history_gross.append(current_gross)
        history_diff.append(y_diff)
        last_gross = current_gross

        feature_vector = []
        for feat in input_features:
            if feat == 'y_diff':
                feature_vector.append(y_diff)
            elif feat == config['target_col']:
                feature_vector.append(current_gross)
            elif feat.startswith('SMA_') and not feat.endswith('_diff'):
                window = int(feat.split('_')[1])
                if len(history_gross) >= window:
                    sma = np.mean(history_gross[-window:])
                    feature_vector.append(sma)
                else:
                    feature_vector.append(0)
            elif feat.startswith('EMA_') and not feat.endswith('_diff'):
                window = int(feat.split('_')[1])
                if len(history_gross) >= window:
                    ema = pd.Series(history_gross).ewm(span=window, adjust=False).mean().iloc[-1]
                    feature_vector.append(ema)
                else:
                    feature_vector.append(0)
            elif feat.startswith('Volatility_') and not feat.endswith('_diff'):
                window = int(feat.split('_')[1])
                if len(history_gross) >= window:
                    vol = np.std(history_gross[-window:])
                    feature_vector.append(vol)
                else:
                    feature_vector.append(0)
            elif feat.startswith('SMA_') and feat.endswith('_diff'):
                window = int(feat.split('_')[1])
                if len(history_diff) >= window:
                    sma = np.mean(history_diff[-window:])
                    feature_vector.append(sma)
                else:
                    feature_vector.append(0)
            elif feat.startswith('EMA_') and feat.endswith('_diff'):
                window = int(feat.split('_')[1])
                if len(history_diff) >= window:
                    ema = pd.Series(history_diff).ewm(span=window, adjust=False).mean().iloc[-1]
                    feature_vector.append(ema)
                else:
                    feature_vector.append(0)
            elif feat.startswith('Volatility_') and feat.endswith('_diff'):
                window = int(feat.split('_')[1])
                if len(history_diff) >= window:
                    vol = np.std(history_diff[-window:])
                    feature_vector.append(vol)
                else:
                    feature_vector.append(0)
            elif feat in target_features:
                idx = target_features.index(feat)
                feature_vector.append(pred[idx])
            else:
                feature_vector.append(0)

        updated_features.append(feature_vector)

    if len(history_gross) > max_window * 5: 
        history_gross = history_gross[-max_window:]
    if len(history_diff) > max_window * 5:
        history_diff = history_diff[-max_window:]

    return np.array(updated_features), current_gross, history_gross, history_diff


def recursive_forecast_svr(model, scalers, initial_window, steps, n_steps, n_features,
                          config, max_window, input_features, target_features, m_steps):
    """Recursive forecasting for SVR models with multi-step prediction"""
    predictions = []
    current_window = initial_window.copy()

    history_gross = []
    history_diff = []

    last_gross = initial_window[-1, input_features.index(config['target_col'])]

    gross_idx = input_features.index(config['target_col'])
    history_gross = list(initial_window[:, gross_idx].flatten())

    if 'y_diff' in input_features:
        diff_idx = input_features.index('y_diff')
        history_diff = list(initial_window[:, diff_idx].flatten())

    iterations = (steps + m_steps - 1) // m_steps

    for _ in range(iterations):
        current_window_flat = current_window.reshape(1, -1) 
        X_scaled = scalers['X'].transform(current_window_flat)

        y_pred_scaled = model.predict(X_scaled)  
        y_pred = scalers['y'].inverse_transform(y_pred_scaled)
        y_pred_2d = y_pred.reshape(-1, len(target_features))

        predictions.append(y_pred_2d)

        new_features, last_gross, history_gross, history_diff = update_features(
            y_pred_2d, last_gross, input_features, target_features, config, max_window,
            history_gross, history_diff
        )

        current_window = np.vstack([current_window[len(new_features):], new_features])

    return np.vstack(predictions)[:steps]


def recursive_forecast_nbeats(model, scalers, initial_window, steps, n_steps, n_features,
                            config, max_window, input_features, target_features, m_steps, device='cpu'):
    """Recursive forecasting for N-BEATS models."""
    predictions = []
    current_window = initial_window.copy()

    history_gross = []
    history_diff = []

    last_gross = initial_window[-1, input_features.index(config['target_col'])]

    gross_idx = input_features.index(config['target_col'])
    history_gross = list(initial_window[:, gross_idx].flatten())

    if 'y_diff' in input_features:
        diff_idx = input_features.index('y_diff')
        history_diff = list(initial_window[:, diff_idx].flatten())

    iterations = (steps + m_steps - 1) // m_steps
    model = model.to(device)

    for _ in range(iterations):
        current_window_flat = current_window.reshape(1, -1)
        X_scaled = scalers['X'].transform(current_window_flat)

        # Convert to tensor and predict
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        with torch.no_grad():
            y_pred_scaled = model(X_tensor).cpu().numpy()

        # Inverse scaling
        y_pred = scalers['y'].inverse_transform(y_pred_scaled)
        y_pred_2d = y_pred.reshape(-1, len(target_features))

        predictions.append(y_pred_2d)

        new_features, last_gross, history_gross, history_diff = update_features(
            y_pred_2d, last_gross, input_features, target_features, config, max_window,
            history_gross, history_diff
        )

        current_window = np.vstack([current_window[new_features.shape[0]:], new_features])

    return np.vstack(predictions)[:steps]


def save_true_vs_pred_1d(y_true, y_pred,
                         *, title="Validation: True vs Predicted",
                         xlabel="Sample Index", ylabel="Value",
                         figsize=(10, 4), dpi=300,
                         true_color="#1f77b4", pred_color="#d62728"):
    """Save a 1D true vs predicted plot."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")

    x = np.arange(len(y_true))

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y_true, color=true_color, lw=2, label="True")
    ax.plot(x, y_pred, color=pred_color, lw=2, alpha=0.75, label="Predicted")

    ax.set_title(title, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    sns.despine(ax=ax, trim=True)

    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return fig, tmp.name  


def save_scatter_plot(y_true, y_pred,
                      *, title="Predicted vs Actual",
                      xlabel="Actual", ylabel="Predicted",
                      figsize=(10, 6), dpi=300):
    """Save a scatter plot of predicted vs actual values."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    
    ax.set_title(title, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    sns.despine(ax=ax, trim=True)

    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return fig, tmp.name


def save_training_fit_plot(y_train_true, y_train_pred,
                           *, title="Training Fit",
                           xlabel="Sample Index", ylabel="Value",
                           figsize=(10, 4), dpi=300):
    """Save a training fit plot."""
    y_train_true = np.asarray(y_train_true).flatten()
    y_train_pred = np.asarray(y_train_pred).flatten()

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(y_train_true))
    ax.plot(x, y_train_true, color="#1f77b4", lw=2, label="Actual")
    ax.plot(x, y_train_pred, color="#ff7f0e", lw=2, alpha=0.75, label="Predicted")

    ax.set_title(title, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    sns.despine(ax=ax, trim=True)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)

    return fig, tmp.name


def evaluate_models(models, df, config):
    """
    Evaluate all trained models and generate comprehensive metrics including
    confusion matrices for trend direction classification.
    """
    from utils.confidence_level import (
        train_confidence_discriminator, get_forecast_confidence_levels,
        detect_forecast_explosion
    )
    
    evaluation = {}
    future_forecasts = {}
    confidence_discriminators = {}
    
    test_size = config['test_size']
    target = config['target_features'][0]
    date_col = config['date_col']
    train, test = df.iloc[:-test_size], df.iloc[-test_size:]    
   
    n_steps = config['n_steps']
    m_steps = config['m_steps']
    horizon = config['horizon']

    input_features = config['input_features']
    target_features = config['target_features']
    target_col = config['target_col']
    n_features = len(input_features)
    rolling_windows = config.get('rolling_windows', [7, 30, 60, 10])
    max_window = max(rolling_windows) if rolling_windows else 60
    
    # Get last real values for explosion detection
    last_real_values = df[target_col].iloc[-n_steps:].values
    
    for model_name, model_data in models.items():
        if 'error' in model_data or 'model' not in model_data:
            continue 
            
        model_eval = {
            'metrics': {},
            'exog_metrics': {},
            'figures': [],
            'confusion_matrix': {},
            'classification_report': {},
            'confidence_levels': [],
            'explosion_detection': {}
        }
        model = model_data['model']
        
        try:
            # Sequence models (ulr, mlr, rnn, cnn_bilstm)
            if model_name in ['ulr', 'mlr', 'rnn', 'cnn_bilstm']:
                print(f'Starting evaluation for {model_name}')
                scalers = model_data['scalers']

                # Prepare validation window (last n_steps before test set)
                val_start = -(test_size + n_steps)
                val_end = -test_size
                X_val = df[input_features].iloc[val_start:val_end].values

                # Prepare future window (last n_steps in data)
                X_last = df[input_features].iloc[-n_steps:].values

                # Check if recursive forecasting is possible
                exog_in_input = [f for f in input_features
                                 if f not in ['y_diff', config['target_col']]
                                 and not f.startswith(('SMA_', 'EMA_', 'Volatility_'))]
                exog_in_target = [f for f in exog_in_input if f in target_features]
                can_recursive = len(exog_in_input) == len(exog_in_target)

                if test_size > m_steps and not can_recursive:
                    raise ValueError(
                        f"Cannot recursively forecast for {model_name} - "
                        "exogenous features not in target_features"
                    )

                # Get predictions using recursive engine
                y_val_pred = recursive_forecast(
                    model, scalers, X_val, test_size,
                    model_name, n_steps, n_features, config, max_window, input_features, target_features, m_steps
                )

                future_pred = recursive_forecast(
                    model, scalers, X_last, horizon,
                    model_name, n_steps, n_features, config, max_window, input_features, target_features, m_steps
                )

                # Get true values for validation period
                y_val_true = df[target_features].iloc[-test_size:].values

                # Ensure matching lengths
                min_val_length = min(len(y_val_true), len(y_val_pred))

                # Calculate confidence intervals
                residuals = y_val_true[:min_val_length, 0] - y_val_pred[:min_val_length, 0]
                sigma = residuals.std(ddof=1)
                z = 1.96  # 95% confidence
                margin = z * sigma

                # Generate figures and metrics for each target feature
                for i, feat in enumerate(target_features):
                    y_true_feat = y_val_true[:min_val_length, i]
                    y_pred_feat = y_val_pred[:min_val_length, i]

                    metrics = {
                        'MAE': mean_absolute_error(y_true_feat, y_pred_feat),
                        'RMSE': np.sqrt(mean_squared_error(y_true_feat, y_pred_feat)),
                        'R2': r2_score(y_true_feat, y_pred_feat),
                        'MAPE': np.mean(np.abs((y_true_feat - y_pred_feat) /
                                            np.maximum(np.abs(y_true_feat), 1))) * 100
                    }

                    # Classify metrics
                    if feat == 'y_diff' or feat == target_col:
                        model_eval['metrics'] = metrics
                    else:
                        model_eval['exog_metrics'][feat] = metrics

                # Generate 4 required plots for primary target
                if target_features:
                    # 1. Training fit plot (using training data)
                    train_size = int(min_val_length * 0.8)
                    if train_size > 10:
                        fig, train_path = save_training_fit_plot(
                            y_val_true[:train_size, 0],
                            y_val_pred[:train_size, 0],
                            title=f"{model_name.upper()} - Training Fit: {target_features[0]}"
                        )
                        model_eval['figures'].append(train_path)

                    # 2. Validation/Test fit plot
                    fig, val_path = save_true_vs_pred_1d(
                        y_val_true[:min_val_length, 0],
                        y_val_pred[:min_val_length, 0],
                        title=f"{model_name.upper()} - Validation: {target_features[0]}"
                    )
                    model_eval['figures'].append(val_path)

                    # 3. Scatter plot
                    fig, scatter_path = save_scatter_plot(
                        y_val_true[:min_val_length, 0],
                        y_val_pred[:min_val_length, 0],
                        title=f"{model_name.upper()} - Predicted vs Actual: {target_features[0]}"
                    )
                    model_eval['figures'].append(scatter_path)

                    # 4. Future plot with confidence intervals
                    yhat = future_pred[:, 0]
                    yhat_lower = yhat - margin
                    yhat_upper = yhat + margin

                    plt.figure(figsize=(12, 6))
                    plt.plot(yhat, label='Forecast', color='blue', linewidth=2)
                    plt.fill_between(range(len(yhat)), yhat_lower, yhat_upper,
                                   color='blue', alpha=0.2, label='95% Confidence Interval')
                    plt.title(f"{model_name.upper()} - Future Forecast: {target_features[0]}")
                    plt.xlabel('Future Time Steps')
                    plt.ylabel(target_features[0])
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                        future_path = tmpfile.name
                    plt.savefig(future_path, bbox_inches='tight')
                    plt.close()
                    model_eval['figures'].append(future_path)

                # Calculate confusion matrix metrics
                cm_results = evaluate_model_with_confusion_matrices(
                    model_name,
                    y_val_true[:min_val_length],
                    y_val_pred[:min_val_length],
                    target_features,
                    threshold=0.0,
                    target_labels=['Sell', 'Buy']
                )
                model_eval['confusion_matrix'] = cm_results['summary']
                model_eval['classification_report'] = cm_results['confusion_matrices']
                model_eval['figures'].extend(cm_results['figures'])

                # Train confidence discriminator
                discriminator = train_confidence_discriminator(
                    model_data, X_val, y_val_true, y_val_pred, config
                )
                if discriminator:
                    confidence_discriminators[model_name] = discriminator
                    # Get confidence levels for future forecasts
                    future_windows = np.array([X_last] * min(horizon, 10))  # Simplified
                    conf_levels = get_forecast_confidence_levels(discriminator, future_windows)
                    model_eval['confidence_levels'] = conf_levels

                # Detect forecast explosion
                explosion_result = detect_forecast_explosion(
                    future_pred[:, 0], last_real_values, config
                )
                model_eval['explosion_detection'] = explosion_result

                # Store forecasts
                future_forecasts[model_name] = {
                    'validation': y_val_pred.tolist(),
                    'future': future_pred.tolist(),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }

                evaluation[model_name] = model_eval

            elif model_name == 'nbeats':
                print('Starting N-BEATS Evaluation')
                scalers = model_data['scalers']
                device = model_data.get('device', 'cpu')

                # Prepare validation window
                val_start = -(test_size + n_steps)
                val_end = -test_size
                X_val = df[input_features].iloc[val_start:val_end].values
                X_last = df[input_features].iloc[-n_steps:].values

                # Get predictions
                y_val_pred = recursive_forecast_nbeats(
                    model, scalers, X_val, test_size,
                    n_steps, n_features, config, max_window,
                    input_features, target_features, m_steps, device
                )

                future_pred = recursive_forecast_nbeats(
                    model, scalers, X_last, horizon,
                    n_steps, n_features, config, max_window,
                    input_features, target_features, m_steps, device
                )

                y_val_true = df[target_features].iloc[-test_size:].values
                min_val_length = min(len(y_val_true), len(y_val_pred))

                # Calculate confidence intervals
                residuals = y_val_true[:min_val_length, 0] - y_val_pred[:min_val_length, 0]
                sigma = residuals.std(ddof=1)
                z = 1.96
                margin = z * sigma

                # Generate metrics
                for i, feat in enumerate(target_features):
                    y_true_feat = y_val_true[:min_val_length, i]
                    y_pred_feat = y_val_pred[:min_val_length, i]

                    metrics = {
                        'MAE': mean_absolute_error(y_true_feat, y_pred_feat),
                        'RMSE': np.sqrt(mean_squared_error(y_true_feat, y_pred_feat)),
                        'R2': r2_score(y_true_feat, y_pred_feat),
                        'MAPE': np.mean(np.abs((y_true_feat - y_pred_feat) /
                                            np.maximum(np.abs(y_true_feat), 1))) * 100
                    }

                    if feat == 'y_diff' or feat == target_col:
                        model_eval['metrics'] = metrics
                    else:
                        model_eval['exog_metrics'][feat] = metrics

                # Generate 4 required plots
                if target_features:
                    # 1. Training history plot (N-BEATS specific)
                    if 'history' in model_data and model_data['history']:
                        history = model_data['history']
                        plt.figure(figsize=(10, 5))
                        plt.plot(history['train_loss'], label='Train Loss')
                        plt.plot(history['val_loss'], label='Val Loss')
                        plt.title(f'N-BEATS Training History')
                        plt.xlabel('Epoch')
                        plt.ylabel('MSE Loss')
                        plt.legend()
                        plt.grid(True)
                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                            train_path = tmpfile.name
                        plt.savefig(train_path, bbox_inches='tight')
                        plt.close()
                        model_eval['figures'].append(train_path)

                    # 2. Validation plot
                    fig, val_path = save_true_vs_pred_1d(
                        y_val_true[:min_val_length, 0],
                        y_val_pred[:min_val_length, 0],
                        title=f"N-BEATS - Validation: {target_features[0]}"
                    )
                    model_eval['figures'].append(val_path)

                    # 3. Scatter plot
                    fig, scatter_path = save_scatter_plot(
                        y_val_true[:min_val_length, 0],
                        y_val_pred[:min_val_length, 0],
                        title=f"N-BEATS - Predicted vs Actual: {target_features[0]}"
                    )
                    model_eval['figures'].append(scatter_path)

                    # 4. Future plot
                    yhat = future_pred[:, 0]
                    yhat_lower = yhat - margin
                    yhat_upper = yhat + margin

                    plt.figure(figsize=(12, 6))
                    plt.plot(yhat, label='Forecast', color='blue', linewidth=2)
                    plt.fill_between(range(len(yhat)), yhat_lower, yhat_upper,
                                   color='blue', alpha=0.2, label='95% Confidence Interval')
                    plt.title(f"N-BEATS - Future Forecast: {target_features[0]}")
                    plt.xlabel('Future Time Steps')
                    plt.ylabel(target_features[0])
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                        future_path = tmpfile.name
                    plt.savefig(future_path, bbox_inches='tight')
                    plt.close()
                    model_eval['figures'].append(future_path)

                # Confusion matrix
                cm_results = evaluate_model_with_confusion_matrices(
                    model_name,
                    y_val_true[:min_val_length],
                    y_val_pred[:min_val_length],
                    target_features,
                    threshold=0.0,
                    target_labels=['Sell', 'Buy']
                )
                model_eval['confusion_matrix'] = cm_results['summary']
                model_eval['classification_report'] = cm_results['confusion_matrices']
                model_eval['figures'].extend(cm_results['figures'])

                # Explosion detection
                explosion_result = detect_forecast_explosion(
                    future_pred[:, 0], last_real_values, config
                )
                model_eval['explosion_detection'] = explosion_result

                future_forecasts[model_name] = {
                    'validation': y_val_pred.tolist(),
                    'future': future_pred.tolist(),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }

                evaluation[model_name] = model_eval

            elif model_name == 'svr':
                print('Starting SVR Evaluation')
                scalers = model_data['scalers']

                # Prepare validation window
                val_start = -(test_size + n_steps)
                val_end = -test_size
                X_val = df[input_features].iloc[val_start:val_end].values
                X_last = df[input_features].iloc[-n_steps:].values

                # Get predictions
                y_val_pred = recursive_forecast_svr(
                    model, scalers, X_val, test_size,
                    n_steps, n_features, config, max_window,
                    input_features, target_features, m_steps
                )

                future_pred = recursive_forecast_svr(
                    model, scalers, X_last, horizon,
                    n_steps, n_features, config, max_window,
                    input_features, target_features, m_steps
                )

                y_val_true = df[target_features].iloc[-test_size:].values
                min_val_length = min(len(y_val_true), len(y_val_pred))

                # Calculate confidence intervals
                residuals = y_val_true[:min_val_length, 0] - y_val_pred[:min_val_length, 0]
                sigma = residuals.std(ddof=1)
                z = 1.96
                margin = z * sigma

                # Generate metrics
                for i, feat in enumerate(target_features):
                    y_true_feat = y_val_true[:min_val_length, i]
                    y_pred_feat = y_val_pred[:min_val_length, i]

                    metrics = {
                        'MAE': mean_absolute_error(y_true_feat, y_pred_feat),
                        'RMSE': np.sqrt(mean_squared_error(y_true_feat, y_pred_feat)),
                        'R2': r2_score(y_true_feat, y_pred_feat),
                        'MAPE': np.mean(np.abs((y_true_feat - y_pred_feat) /
                                            np.maximum(np.abs(y_true_feat), 1))) * 100
                    }

                    if feat == 'y_diff' or feat == target_col:
                        model_eval['metrics'] = metrics
                    else:
                        model_eval['exog_metrics'][feat] = metrics

                # Generate 4 required plots
                if target_features:
                    # For SVR, we use validation data as proxy for training fit
                    train_size = int(min_val_length * 0.5)
                    if train_size > 10:
                        fig, train_path = save_training_fit_plot(
                            y_val_true[:train_size, 0],
                            y_val_pred[:train_size, 0],
                            title=f"SVR - Training Fit: {target_features[0]}"
                        )
                        model_eval['figures'].append(train_path)

                    # 2. Validation plot
                    fig, val_path = save_true_vs_pred_1d(
                        y_val_true[:min_val_length, 0],
                        y_val_pred[:min_val_length, 0],
                        title=f"SVR - Validation: {target_features[0]}"
                    )
                    model_eval['figures'].append(val_path)

                    # 3. Scatter plot
                    fig, scatter_path = save_scatter_plot(
                        y_val_true[:min_val_length, 0],
                        y_val_pred[:min_val_length, 0],
                        title=f"SVR - Predicted vs Actual: {target_features[0]}"
                    )
                    model_eval['figures'].append(scatter_path)

                    # 4. Future plot
                    yhat = future_pred[:, 0]
                    yhat_lower = yhat - margin
                    yhat_upper = yhat + margin

                    plt.figure(figsize=(12, 6))
                    plt.plot(yhat, label='Forecast', color='blue', linewidth=2)
                    plt.fill_between(range(len(yhat)), yhat_lower, yhat_upper,
                                   color='blue', alpha=0.2, label='95% Confidence Interval')
                    plt.title(f"SVR - Future Forecast: {target_features[0]}")
                    plt.xlabel('Future Time Steps')
                    plt.ylabel(target_features[0])
                    plt.legend()
                    plt.grid(True, alpha=0.3)

                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                        future_path = tmpfile.name
                    plt.savefig(future_path, bbox_inches='tight')
                    plt.close()
                    model_eval['figures'].append(future_path)

                # Confusion matrix
                cm_results = evaluate_model_with_confusion_matrices(
                    model_name,
                    y_val_true[:min_val_length],
                    y_val_pred[:min_val_length],
                    target_features,
                    threshold=0.0,
                    target_labels=['Sell', 'Buy']
                )
                model_eval['confusion_matrix'] = cm_results['summary']
                model_eval['classification_report'] = cm_results['confusion_matrices']
                model_eval['figures'].extend(cm_results['figures'])

                # Explosion detection
                explosion_result = detect_forecast_explosion(
                    future_pred[:, 0], last_real_values, config
                )
                model_eval['explosion_detection'] = explosion_result

                future_forecasts[model_name] = {
                    'validation': y_val_pred.tolist(),
                    'future': future_pred.tolist(),
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }

                evaluation[model_name] = model_eval

        except Exception as e:
            model_eval['error'] = f"Evaluation failed: {str(e)}"
            evaluation[model_name] = model_eval
            logging.error(f"Error evaluating {model_name}: {str(e)}")
    
    return evaluation, future_forecasts


def save_best_model(models, evaluation, model_dir, logger):
    """Save the best performing model based on MAE."""
    os.makedirs(model_dir, exist_ok=True)
    best_score = float('inf')
    best_model_name = None

    for model_name, eval_data in evaluation.items():
        if 'metrics' in eval_data and 'MAE' in eval_data['metrics']:
            mae = eval_data['metrics']['MAE']
            if mae < best_score:
                best_score = mae
                best_model_name = model_name
    
    if not best_model_name:
        logger.error("No valid model found for saving")
        return
    
    try:
        model_data = models[best_model_name]
        model = model_data['model']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"{best_model_name}_{timestamp}.joblib")
        
        if best_model_name == 'sarimax':
            if isinstance(model, SARIMAXResults):
                model.save(f"{model_path}.pkl")
            else:
                joblib.dump(model, model_path)
        else:
            joblib.dump(model, model_path)
        
        artifacts = {}
        if 'scaler_X' in model_data:
            artifacts['scaler_X'] = model_data['scaler_X']
        if 'scaler_y' in model_data:
            artifacts['scaler_y'] = model_data['scaler_y']
        
        if artifacts:
            artifacts_path = os.path.join(model_dir, f"{best_model_name}_artifacts_{timestamp}.joblib")
            joblib.dump(artifacts, artifacts_path)
        
        logger.info(f"Saved best model ({best_model_name}) to {model_path}")
        return model_path
    
    except Exception as e:
        logger.error(f"Error saving best model: {str(e)}")
        return None
