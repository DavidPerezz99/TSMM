"""
Univariate Models Module

This module provides training functions for univariate time series forecasting models,
including ULR, SVR, N-BEATS, XGBoost, Prophet, SARIMAX, and LSTM.
All models generate the 4 required plots: training fit, validation fit, scatter plot, and future forecast.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import holidays 
import logging
import matplotlib.dates as mdates
import tempfile
import os
import time
import seaborn as sns
from datetime import datetime

from models.multivariate_models import (
    grid_search_sarimax, train_cnn_bilstm_model, 
    multiVrecurrent_LR, train_nbeats_model, multiVrecurrent_SVR
)
from utils.sequence_utils import prepare_sequences_cached as prepare_sequences


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


def save_validation_plot(y_true, y_pred,
                         *, title="Validation: True vs Predicted",
                         xlabel="Sample Index", ylabel="Value",
                         figsize=(10, 4), dpi=300):
    """Save a validation plot."""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(y_true))
    ax.plot(x, y_true, color="#1f77b4", lw=2, label="True")
    ax.plot(x, y_pred, color="#d62728", lw=2, alpha=0.75, label="Predicted")

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


def save_future_forecast_plot(forecast_values, 
                              *, title="Future Forecast",
                              xlabel="Future Time Steps", ylabel="Value",
                              confidence_intervals=None,
                              figsize=(12, 6), dpi=300):
    """Save a future forecast plot."""
    forecast_values = np.asarray(forecast_values).flatten()

    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(forecast_values))
    ax.plot(x, forecast_values, color='blue', lw=2, label='Forecast')
    
    # Add confidence intervals if provided
    if confidence_intervals is not None:
        lower = confidence_intervals['lower']
        upper = confidence_intervals['upper']
        ax.fill_between(x, lower, upper, color='blue', alpha=0.2, label='Confidence Interval')
    
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


def train_univariate_models(df, config, logger, input_features, target_features, 
                            exclude_cols, n_steps, m_steps, split_ratio):
    """
    Train univariate forecasting models based on config selection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance
    input_features : list
        List of input feature names
    target_features : list
        List of target feature names
    exclude_cols : list
        Columns to exclude
    n_steps : int
        Lookback window size
    m_steps : int
        Forecast horizon
    split_ratio : float
        Train/test split ratio
    
    Returns:
    --------
    dict
        Dictionary containing trained model results
    """
    models = {}
    
    # Get models to run from config
    models_to_run = config.get('models_to_run', {}).get('univariate', 
        ['ulr', 'svr', 'nbeats', 'xgboost', 'prophet', 'sarimax', 'lstm'])
    
    logger.info(f"Training univariate models: {models_to_run}")
    
    # ULR - Univariate Linear Regression
    if 'ulr' in models_to_run:
        try:
            logger.info("Training ULR model")
            ulr_result = multiVrecurrent_LR(
                df,
                input_features,
                target_features,
                n_steps,
                m_steps,
                split_ratio
            )
            ulr_result['parameters'].update({
                'model_type': 'Univariate Linear Regression',
                'n_steps': n_steps,
                'm_steps': m_steps,
                'input_features': input_features,
                'target_features': target_features,
                'split_ratio': split_ratio
            })
            models['ulr'] = ulr_result
            logger.info("ULR model training completed")
        except Exception as e:
            logger.error(f"ULR training failed: {str(e)}")
            models['ulr'] = {'error': str(e)}
    
    # SVR - Support Vector Regression
    if 'svr' in models_to_run:
        try:
            logger.info("Training SVR model")
            svr_config = config.get('svr', {})
            svr_result = multiVrecurrent_SVR(
                df,
                input_features,
                target_features,
                n_steps,
                m_steps,
                split_ratio,
                svr_params=svr_config
            )
            svr_result['parameters'].update({
                'model_type': 'Support Vector Regression',
                'n_steps': n_steps,
                'm_steps': m_steps,
                'input_features': input_features,
                'target_features': target_features,
                'split_ratio': split_ratio
            })
            models['svr'] = svr_result
            logger.info("SVR model training completed")
        except Exception as e:
            logger.error(f"SVR training failed: {str(e)}")
            models['svr'] = {'error': str(e)}
    
    # N-BEATS
    if 'nbeats' in models_to_run:
        try:
            logger.info("Training N-BEATS model")
            nbeats_config = config.get('nbeats', {})
            nbeats_result = train_nbeats_model(
                df,
                input_features,
                target_features,
                n_steps,
                m_steps,
                split_ratio,
                model_type=nbeats_config.get('model_type', 'interpretable'),
                hidden_size=nbeats_config.get('hidden_size', 512),
                epochs=nbeats_config.get('epochs', 100),
                batch_size=nbeats_config.get('batch_size', 32),
                learning_rate=nbeats_config.get('learning_rate', 0.001),
                patience_es=nbeats_config.get('patience_es', 20),
                patience_rlr=nbeats_config.get('patience_rlr', 10),
                stacks_config=nbeats_config.get('stacks_config'),
                blackbox_config=nbeats_config.get('blackbox_config'),
                device=nbeats_config.get('device', 'cpu')
            )
            models['nbeats'] = nbeats_result
            logger.info("N-BEATS model training completed")
        except Exception as e:
            logger.error(f"N-BEATS training failed: {str(e)}")
            models['nbeats'] = {'error': str(e)}
    
    # XGBoost
    if 'xgboost' in models_to_run:
        try:
            logger.info("Training XGBoost model")
            xgb_result = train_xgboost_model(df, config, input_features, target_features)
            models['xgboost'] = xgb_result
            logger.info("XGBoost model training completed")
        except Exception as e:
            logger.error(f"XGBoost training failed: {str(e)}")
            models['xgboost'] = {'error': str(e)}
    
    # Prophet
    if 'prophet' in models_to_run:
        try:
            logger.info("Training Prophet model")
            prophet_result = train_prophet_model(df, config)
            models['prophet'] = prophet_result
            logger.info("Prophet model training completed")
        except Exception as e:
            logger.error(f"Prophet training failed: {str(e)}")
            models['prophet'] = {'error': str(e)}
    
    # SARIMAX
    if 'sarimax' in models_to_run:
        try:
            logger.info("Training SARIMAX model")
            sarimax_result = train_sarimax_model(df, config)
            models['sarimax'] = sarimax_result
            logger.info("SARIMAX model training completed")
        except Exception as e:
            logger.error(f"SARIMAX training failed: {str(e)}")
            models['sarimax'] = {'error': str(e)}
    
    # LSTM
    if 'lstm' in models_to_run:
        try:
            logger.info("Training LSTM model")
            lstm_config = config.get('lstm', {})
            lstm_result = train_cnn_bilstm_model(
                df,
                input_features,
                target_features,
                n_steps,
                m_steps,
                split_ratio,
                conv_filters=lstm_config.get('conv_filters', 32),
                lstm_units=lstm_config.get('lstm_units', 20),
                epochs=lstm_config.get('epochs', 20),
                batch_size=lstm_config.get('batch_size', 30),
                learning_rate=lstm_config.get('learning_rate', 0.001),
                patience_es=lstm_config.get('patience_es', 30),
                patience_rlr=lstm_config.get('patience_rlr', 5),
                seed=lstm_config.get('seed', 42)
            )
            lstm_result['parameters'].update({
                'model_type': 'LSTM/CNN-BiLSTM',
                'n_steps': n_steps,
                'm_steps': m_steps,
                'input_features': input_features,
                'target_features': target_features,
                'split_ratio': split_ratio
            })
            models['lstm'] = lstm_result
            logger.info("LSTM model training completed")
        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            models['lstm'] = {'error': str(e)}
    
    return models


def train_xgboost_model(df, config, input_features, target_features):
    """Train XGBoost model with all 4 required plots."""
    from utils.evaluator import save_forecast_plot
    
    results = {
        'model': None,
        'metrics': {},
        'figures': [],
        'parameters': {
            'model_type': 'XGBoost',
            'input_features': input_features,
            'target_features': target_features
        },
        'scalers': {'X': None, 'y': None}
    }
    
    try:
        target_col = target_features[0]
        date_col = config['date_col']
        
        # Prepare data
        df_xgb = df.reset_index().copy()
        df_xgb[date_col] = pd.to_datetime(df_xgb[date_col])
        df_xgb = df_xgb.set_index(date_col).sort_index()
        
        # Add time features
        df_xgb['year'] = df_xgb.index.year
        df_xgb['month'] = df_xgb.index.month
        df_xgb['day'] = df_xgb.index.day
        df_xgb['dayofweek'] = df_xgb.index.dayofweek
        df_xgb['weekofyear'] = df_xgb.index.isocalendar().week
        df_xgb['quarter'] = df_xgb.index.quarter
        
        # Add lag features
        lags = config.get('lags', [2, 7, 30, 60])
        for lag in lags:
            df_xgb[f'{target_col}_lag{lag}'] = df_xgb[target_col].shift(lag)
        
        # Add rolling window features
        rolling_windows = config.get('rolling_windows', [2, 7, 30, 60])
        for window in rolling_windows:
            df_xgb[f'{target_col}_roll{window}'] = df_xgb[target_col].rolling(window).mean()
        
        # Drop NaN values
        df_xgb = df_xgb.dropna()
        exclude_cols = config.get('exclude_cols', [])
        # Define features
        feature_cols = [col for col in df_xgb.columns 
                       if col not in exclude_cols + [target_col] + [config.get('target_col', target_col)]]
        feature_cols = [c for c in feature_cols if c in df_xgb.columns]
        
        X = df_xgb[feature_cols]
        y = df_xgb[target_col]
        
        # Split data
        split_idx = int(len(X) * config.get('split_ratio', 0.9))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Train model
        xgb_config = config.get('xgboost', {})
        model = XGBRegressor(
            n_estimators=xgb_config.get('n_estimators', 100),
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.1),
            subsample=xgb_config.get('subsample', 0.8),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
            random_state=42
        )
        model.fit(X_train, y_train)
        results['model'] = model
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        results['metrics'] = {
            'MAE': mean_absolute_error(y_test, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'R2': r2_score(y_test, y_test_pred),
            'MAPE': np.mean(np.abs((y_test - y_test_pred) / np.maximum(np.abs(y_test), 1))) * 100
        }
        
        # Generate 4 required plots
        # 1. Training fit plot
        fig, train_path = save_training_fit_plot(
            y_train.values,
            y_train_pred,
            title=f"XGBoost - Training Fit: {target_col}"
        )
        results['figures'].append(train_path)
        
        # 2. Validation plot
        fig, val_path = save_validation_plot(
            y_test.values,
            y_test_pred,
            title=f"XGBoost - Validation: {target_col}"
        )
        results['figures'].append(val_path)
        
        # 3. Scatter plot
        fig, scatter_path = save_scatter_plot(
            y_test.values,
            y_test_pred,
            title=f"XGBoost - Predicted vs Actual: {target_col}"
        )
        results['figures'].append(scatter_path)
        
        # 4. Feature importance (as training insight)
        fi = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        plt.barh(fi['Feature'], fi['Importance'])
        plt.title('XGBoost - Top 10 Feature Importances')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            fi_path = tmpfile.name
        plt.savefig(fi_path, bbox_inches='tight')
        plt.close()
        results['figures'].append(fi_path)
        
        # Future forecast
        horizon = config.get('horizon', 60)
        future_forecast = generate_xgboost_forecast(model, df_xgb, target_col, horizon, lags, rolling_windows)
        
        # Calculate confidence intervals
        residuals = y_test.values - y_test_pred
        sigma = residuals.std()
        z = 1.96
        margin = z * sigma
        
        fig, future_path = save_future_forecast_plot(
            future_forecast,
            title=f"XGBoost - Future Forecast: {target_col}",
            confidence_intervals={
                'lower': future_forecast - margin,
                'upper': future_forecast + margin
            }
        )
        results['figures'].append(future_path)
        
    except Exception as e:
        logging.error(f"XGBoost training failed: {str(e)}")
        results['error'] = str(e)
    
    return results


def generate_xgboost_forecast(model, df, target_col, horizon, lags, rolling_windows):
    """Generate future forecast for XGBoost model."""
    future_values = []
    df_extended = df.copy()
    
    for i in range(horizon):
        # Create feature row for next prediction
        last_row = df_extended.iloc[-1:].copy()
        
        # Update lag features
        for lag in lags:
            if len(df_extended) >= lag:
                last_row[f'{target_col}_lag{lag}'] = df_extended[target_col].iloc[-lag]
        
        # Update rolling features
        for window in rolling_windows:
            if len(df_extended) >= window:
                last_row[f'{target_col}_roll{window}'] = df_extended[target_col].iloc[-window:].mean()
        
        # Predict
        feature_cols = [c for c in df_extended.columns if c != target_col]
        pred = model.predict(last_row[feature_cols])[0]
        future_values.append(pred)
        
        # Add prediction to extended dataframe
        new_row = last_row.copy()
        new_row[target_col] = pred
        df_extended = pd.concat([df_extended, new_row])
    
    return np.array(future_values)


def train_prophet_model(df, config):
    """Train Prophet model with all 4 required plots."""
    results = {
        'model': None,
        'metrics': {},
        'figures': [],
        'parameters': {
            'model_type': 'Prophet'
        }
    }
    
    try:
        target_col = config['target_features'][0]
        date_col = config['date_col']
        test_size = config.get('test_size', 30)
        
        # Prepare data for Prophet
        prophet_data = df.reset_index()[[date_col, target_col]].copy()
        prophet_data.columns = ['ds', 'y']
        prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])
        prophet_data = prophet_data.sort_values('ds').reset_index(drop=True)
        
        # Split data
        train_data = prophet_data.iloc[:-test_size]
        test_data = prophet_data.iloc[-test_size:]
        
        # Train model
        prophet_config = config.get('prophet', {})
        model = Prophet(
            growth=prophet_config.get('growth', 'linear'),
            seasonality_mode=prophet_config.get('seasonality_mode', 'multiplicative'),
            yearly_seasonality=prophet_config.get('yearly_seasonality', True),
            weekly_seasonality=prophet_config.get('weekly_seasonality', True),
            daily_seasonality=prophet_config.get('daily_seasonality', False),
            changepoint_prior_scale=prophet_config.get('changepoint_prior_scale', 0.05),
            seasonality_prior_scale=prophet_config.get('seasonality_prior_scale', 10.0)
        )
        model.fit(train_data)
        results['model'] = model
        
        # Predictions on training data
        future_train = model.make_future_dataframe(periods=0, include_history=True)
        forecast_train = model.predict(future_train)
        y_train_pred = forecast_train['yhat'].values
        y_train_true = train_data['y'].values
        
        # Predictions on test data
        future_test = model.make_future_dataframe(periods=test_size, include_history=False)
        forecast_test = model.predict(future_test)
        y_test_pred = forecast_test['yhat'].values
        y_test_true = test_data['y'].values
        
        # Calculate metrics
        results['metrics'] = {
            'MAE': mean_absolute_error(y_test_true, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_true, y_test_pred)),
            'R2': r2_score(y_test_true, y_test_pred),
            'MAPE': np.mean(np.abs((y_test_true - y_test_pred) / np.maximum(np.abs(y_test_true), 1))) * 100
        }
        
        # Generate 4 required plots
        # 1. Training fit plot
        fig, train_path = save_training_fit_plot(
            y_train_true,
            y_train_pred,
            title=f"Prophet - Training Fit: {target_col}"
        )
        results['figures'].append(train_path)
        
        # 2. Validation plot
        fig, val_path = save_validation_plot(
            y_test_true,
            y_test_pred,
            title=f"Prophet - Validation: {target_col}"
        )
        results['figures'].append(val_path)
        
        # 3. Scatter plot
        fig, scatter_path = save_scatter_plot(
            y_test_true,
            y_test_pred,
            title=f"Prophet - Predicted vs Actual: {target_col}"
        )
        results['figures'].append(scatter_path)
        
        # 4. Future forecast with Prophet components
        horizon = config.get('horizon', 60)
        future = model.make_future_dataframe(periods=horizon)
        forecast = model.predict(future)
        
        # Plot forecast components
        fig = model.plot_components(forecast)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            components_path = tmpfile.name
        fig.savefig(components_path, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(components_path)
        
        # Future forecast plot
        future_forecast = forecast['yhat'].iloc[-horizon:].values
        future_lower = forecast['yhat_lower'].iloc[-horizon:].values
        future_upper = forecast['yhat_upper'].iloc[-horizon:].values
        
        fig, future_path = save_future_forecast_plot(
            future_forecast,
            title=f"Prophet - Future Forecast: {target_col}",
            confidence_intervals={
                'lower': future_lower,
                'upper': future_upper
            }
        )
        results['figures'].append(future_path)
        
    except Exception as e:
        logging.error(f"Prophet training failed: {str(e)}")
        results['error'] = str(e)
    
    return results


def train_sarimax_model(df, config):
    """Train SARIMAX model with all 4 required plots."""
    results = {
        'model': None,
        'metrics': {},
        'figures': [],
        'parameters': {
            'model_type': 'SARIMAX'
        }
    }
    
    try:
        target_col = config['target_features'][0]
        test_size = config.get('test_size', 30)
        seasonal_period = config.get('seasonal_period', 7)
        
        # Prepare data
        y = df[target_col]
        train_data = y.iloc[:-test_size]
        test_data = y.iloc[-test_size:]
        
        # Grid search for best parameters
        sarimax_result = grid_search_sarimax(train_data, seasonal_period=seasonal_period)
        model = sarimax_result['model']
        results['model'] = model
        results['parameters'].update(sarimax_result['parameters'])
        
        # Predictions on training data (fitted values)
        y_train_pred = model.fittedvalues
        y_train_true = train_data.values
        
        # Predictions on test data
        forecast = model.get_forecast(steps=test_size)
        y_test_pred = forecast.predicted_mean.values
        y_test_true = test_data.values
        conf_int = forecast.conf_int().values
        
        # Calculate metrics
        results['metrics'] = {
            'MAE': mean_absolute_error(y_test_true, y_test_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test_true, y_test_pred)),
            'R2': r2_score(y_test_true, y_test_pred),
            'MAPE': np.mean(np.abs((y_test_true - y_test_pred) / np.maximum(np.abs(y_test_true), 1))) * 100
        }
        
        # Generate 4 required plots
        # 1. Training fit plot
        fig, train_path = save_training_fit_plot(
            y_train_true[-100:],  # Last 100 points for clarity
            y_train_pred[-100:],
            title=f"SARIMAX - Training Fit: {target_col}"
        )
        results['figures'].append(train_path)
        
        # 2. Validation plot
        fig, val_path = save_validation_plot(
            y_test_true,
            y_test_pred,
            title=f"SARIMAX - Validation: {target_col}"
        )
        results['figures'].append(val_path)
        
        # 3. Scatter plot
        fig, scatter_path = save_scatter_plot(
            y_test_true,
            y_test_pred,
            title=f"SARIMAX - Predicted vs Actual: {target_col}"
        )
        results['figures'].append(scatter_path)
        
        # 4. Diagnostic plots
        fig = model.plot_diagnostics(figsize=(12, 8))
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            diag_path = tmpfile.name
        fig.savefig(diag_path, bbox_inches='tight')
        plt.close(fig)
        results['figures'].append(diag_path)
        
        # Future forecast
        horizon = config.get('horizon', 60)
        future_forecast_obj = model.get_forecast(steps=horizon)
        future_forecast = future_forecast_obj.predicted_mean.values
        future_conf_int = future_forecast_obj.conf_int().values
        
        fig, future_path = save_future_forecast_plot(
            future_forecast,
            title=f"SARIMAX - Future Forecast: {target_col}",
            confidence_intervals={
                'lower': future_conf_int[:, 0],
                'upper': future_conf_int[:, 1]
            }
        )
        results['figures'].append(future_path)
        
    except Exception as e:
        logging.error(f"SARIMAX training failed: {str(e)}")
        results['error'] = str(e)
    
    return results
