"""
Multivariate Models Module

This module provides training functions for multivariate time series forecasting models,
including MLR, SARIMAX, and LSTM. All models generate the 4 required plots.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, LeakyReLU, Bidirectional, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import tensorflow as tf
import itertools
import logging
import holidays
import tempfile
from pathlib import Path
import seaborn as sns
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

from utils.sequence_utils import prepare_sequences_cached as prepare_sequences


# N-BEATS Model Classes (from original code)
class GenericBlock(nn.Module):
    def __init__(self, input_size, theta_size, hidden_size=512, num_layers=4):
        super(GenericBlock, self).__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.hidden_size = hidden_size

        layers = []
        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_size
            out_features = hidden_size
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())

        self.fc_stack = nn.Sequential(*layers)
        self.theta_fc = nn.Linear(hidden_size, theta_size)
        self.backcast_fc = nn.Linear(theta_size, input_size)
        self.forecast_fc = nn.Linear(theta_size, input_size)

    def forward(self, x):
        hidden = self.fc_stack(x)
        theta = self.theta_fc(hidden)
        backcast = self.backcast_fc(theta)
        forecast = self.forecast_fc(theta)
        return backcast, forecast


class TrendBlock(nn.Module):
    def __init__(self, input_size, forecast_size, degree=3, hidden_size=512):
        super(TrendBlock, self).__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.degree = degree

        self.polynomial_basis = nn.Parameter(
            torch.tensor([[(i/input_size)**p for p in range(degree+1)]
                         for i in range(input_size)], dtype=torch.float32),
            requires_grad=False
        )

        self.forecast_basis = nn.Parameter(
            torch.tensor([[(i/forecast_size)**p for p in range(degree+1)]
                         for i in range(forecast_size)], dtype=torch.float32),
            requires_grad=False
        )

        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, degree + 1)
        )

    def forward(self, x):
        theta = self.fc_stack(x)
        backcast = torch.matmul(self.polynomial_basis, theta.unsqueeze(-1)).squeeze(-1)
        forecast = torch.matmul(self.forecast_basis, theta.unsqueeze(-1)).squeeze(-1)
        return backcast, forecast


class SeasonalityBlock(nn.Module):
    def __init__(self, input_size, forecast_size, num_harmonics=4, hidden_size=512):
        super(SeasonalityBlock, self).__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.num_harmonics = num_harmonics

        backcast_basis = []
        forecast_basis = []

        for i in range(input_size):
            basis = []
            for h in range(1, num_harmonics + 1):
                basis.append(np.cos(2 * np.pi * h * i / input_size))
                basis.append(np.sin(2 * np.pi * h * i / input_size))
            backcast_basis.append(basis)

        for i in range(forecast_size):
            basis = []
            for h in range(1, num_harmonics + 1):
                basis.append(np.cos(2 * np.pi * h * i / input_size))
                basis.append(np.sin(2 * np.pi * h * i / input_size))
            forecast_basis.append(basis)

        self.backcast_basis = nn.Parameter(
            torch.tensor(backcast_basis, dtype=torch.float32),
            requires_grad=False
        )
        self.forecast_basis = nn.Parameter(
            torch.tensor(forecast_basis, dtype=torch.float32),
            requires_grad=False
        )

        theta_size = 2 * num_harmonics
        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, theta_size)
        )

    def forward(self, x):
        theta = self.fc_stack(x)
        backcast = torch.matmul(self.backcast_basis, theta.unsqueeze(-1)).squeeze(-1)
        forecast = torch.matmul(self.forecast_basis, theta.unsqueeze(-1)).squeeze(-1)
        return backcast, forecast


class InterpretableTrendBlock(nn.Module):
    def __init__(self, input_size, forecast_size, degree=3, hidden_size=256):
        super(InterpretableTrendBlock, self).__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.degree = degree

        self.backcast_basis = nn.Parameter(
            self._create_polynomial_basis(input_size, degree),
            requires_grad=False
        )
        self.forecast_basis = nn.Parameter(
            self._create_polynomial_basis(forecast_size, degree),
            requires_grad=False
        )

        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, degree + 1)
        )

    def _create_polynomial_basis(self, size, degree):
        t = torch.linspace(0, 1, size).reshape(-1, 1)
        basis = [t ** p for p in range(degree + 1)]
        return torch.cat(basis, dim=1).float()

    def forward(self, x):
        coefficients = self.fc_stack(x)
        backcast = torch.matmul(self.backcast_basis, coefficients.unsqueeze(-1)).squeeze(-1)
        forecast = torch.matmul(self.forecast_basis, coefficients.unsqueeze(-1)).squeeze(-1)
        return backcast, forecast, coefficients


class InterpretableSeasonalityBlock(nn.Module):
    def __init__(self, input_size, forecast_size, num_harmonics=6, hidden_size=256):
        super(InterpretableSeasonalityBlock, self).__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.num_harmonics = num_harmonics

        self.backcast_basis = nn.Parameter(
            self._create_fourier_basis(input_size, num_harmonics),
            requires_grad=False
        )
        self.forecast_basis = nn.Parameter(
            self._create_fourier_basis(forecast_size, num_harmonics),
            requires_grad=False
        )

        self.fc_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * num_harmonics)
        )

    def _create_fourier_basis(self, size, num_harmonics):
        t = torch.linspace(0, 1, size).reshape(-1, 1)
        basis = []
        for h in range(1, num_harmonics + 1):
            basis.append(torch.sin(2 * np.pi * h * t))
            basis.append(torch.cos(2 * np.pi * h * t))
        return torch.cat(basis, dim=1).float()

    def forward(self, x):
        coefficients = self.fc_stack(x)
        backcast = torch.matmul(self.backcast_basis, coefficients.unsqueeze(-1)).squeeze(-1)
        forecast = torch.matmul(self.forecast_basis, coefficients.unsqueeze(-1)).squeeze(-1)
        return backcast, forecast, coefficients


class NBeatsStack(nn.Module):
    def __init__(self, input_size, forecast_size, hidden_size=512,
                 num_blocks=3, block_type='generic', **block_kwargs):
        super(NBeatsStack, self).__init__()
        self.blocks = nn.ModuleList()

        filtered_kwargs = {}
        if block_type == 'trend':
            filtered_kwargs = {k: v for k, v in block_kwargs.items() if k in ['degree']}
        elif block_type == 'seasonality':
            filtered_kwargs = {k: v for k, v in block_kwargs.items() if k in ['num_harmonics']}
        elif block_type == 'generic':
            filtered_kwargs = {k: v for k, v in block_kwargs.items() if k in ['num_layers']}

        for _ in range(num_blocks):
            if block_type == 'trend':
                block = TrendBlock(input_size, forecast_size, hidden_size=hidden_size, **filtered_kwargs)
            elif block_type == 'seasonality':
                block = SeasonalityBlock(input_size, forecast_size, hidden_size=hidden_size, **filtered_kwargs)
            else:
                theta_size = input_size
                block = GenericBlock(input_size, theta_size, hidden_size=hidden_size, **filtered_kwargs)
            self.blocks.append(block)

    def forward(self, x):
        stack_forecast = torch.zeros(x.shape[0], self.blocks[0].forecast_size).to(x.device)
        for block in self.blocks:
            backcast, forecast = block(x)
            x = x - backcast
            stack_forecast = stack_forecast + forecast
        return stack_forecast


class InterpretableNBeatsStack(nn.Module):
    def __init__(self, input_size, forecast_size, hidden_size=256,
                 num_blocks=3, block_type='trend', **block_kwargs):
        super(InterpretableNBeatsStack, self).__init__()
        self.block_type = block_type
        self.blocks = nn.ModuleList()

        filtered_kwargs = {}
        if block_type == 'trend':
            filtered_kwargs = {k: v for k, v in block_kwargs.items() if k in ['degree']}
        elif block_type == 'seasonality':
            filtered_kwargs = {k: v for k, v in block_kwargs.items() if k in ['num_harmonics']}

        for i in range(num_blocks):
            if block_type == 'trend':
                block = InterpretableTrendBlock(
                    input_size, forecast_size, hidden_size=hidden_size, **filtered_kwargs)
            elif block_type == 'seasonality':
                block = InterpretableSeasonalityBlock(
                    input_size, forecast_size, hidden_size=hidden_size, **filtered_kwargs)
            else:
                raise ValueError("Only 'trend' and 'seasonality' blocks are interpretable")
            self.blocks.append(block)

    def forward(self, x, return_components=False):
        stack_forecast = torch.zeros(x.shape[0], self.blocks[0].forecast_size).to(x.device)
        block_forecasts = []
        block_backcasts = []
        all_coefficients = []
        residual = x

        for block in self.blocks:
            if hasattr(block, 'fc_stack') and self.block_type in ['trend', 'seasonality']:
                backcast, forecast, coefficients = block(residual)
                block_forecasts.append(forecast)
                block_backcasts.append(backcast)
                all_coefficients.append(coefficients)
            else:
                backcast, forecast = block(residual)
                block_forecasts.append(forecast)
                block_backcasts.append(backcast)
                all_coefficients.append(None)

            residual = residual - backcast
            stack_forecast = stack_forecast + forecast

        if return_components:
            return stack_forecast, residual, block_forecasts, block_backcasts, all_coefficients
        else:
            return stack_forecast


class NBeats(nn.Module):
    def __init__(self, input_size=64, forecast_size=16, hidden_size=512,
                 stacks_config=None):
        super(NBeats, self).__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size

        if stacks_config is None:
            stacks_config = [
                {'type': 'generic', 'num_blocks': 3, 'hidden_size': hidden_size},
                {'type': 'generic', 'num_blocks': 3, 'hidden_size': hidden_size}
            ]

        self.stacks = nn.ModuleList()
        for config in stacks_config:
            block_kwargs = {k: v for k, v in config.items() if k not in ['type', 'num_blocks', 'hidden_size']}

            stack = NBeatsStack(
                input_size=input_size,
                forecast_size=forecast_size,
                hidden_size=config['hidden_size'],
                num_blocks=config['num_blocks'],
                block_type=config['type'],
                **block_kwargs
            )
            self.stacks.append(stack)

    def forward(self, x):
        forecast = torch.zeros(x.shape[0], self.forecast_size).to(x.device)
        for stack in self.stacks:
            stack_forecast = stack(x)
            forecast = forecast + stack_forecast
        return forecast


class InterpretableNBeats(nn.Module):
    def __init__(self, input_size=64, forecast_size=16, hidden_size=256,
                 stacks_config=None):
        super(InterpretableNBeats, self).__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size

        if stacks_config is None:
            stacks_config = [
                {'type': 'trend', 'num_blocks': 2, 'hidden_size': hidden_size, 'degree': 4},
                {'type': 'seasonality', 'num_blocks': 2, 'hidden_size': hidden_size, 'num_harmonics': 8}
            ]

        self.stacks = nn.ModuleList()
        for config in stacks_config:
            block_kwargs = {k: v for k, v in config.items() if k not in ['type', 'num_blocks', 'hidden_size']}

            stack = InterpretableNBeatsStack(
                input_size=input_size,
                forecast_size=forecast_size,
                hidden_size=config['hidden_size'],
                num_blocks=config['num_blocks'],
                block_type=config['type'],
                **block_kwargs
            )
            self.stacks.append(stack)

    def forward(self, x, return_components=False):
        if return_components:
            return self.forward_interpretable(x)
        else:
            total_forecast = torch.zeros(x.shape[0], self.forecast_size).to(x.device)
            for stack in self.stacks:
                stack_forecast = stack(x)
                total_forecast = total_forecast + stack_forecast
            return total_forecast

    def forward_interpretable(self, x):
        total_forecast = torch.zeros(x.shape[0], self.forecast_size).to(x.device)
        residuals = x.clone()

        components = {
            'trend_forecasts': [],
            'seasonality_forecasts': [],
            'trend_backcasts': [],
            'seasonality_backcasts': [],
            'trend_coefficients': [],
            'seasonality_coefficients': [],
            'final_residual': None
        }

        for stack in self.stacks:
            (stack_forecast, residual,
             block_forecasts, block_backcasts, coefficients) = stack(residuals, return_components=True)

            if stack.block_type == 'trend':
                components['trend_forecasts'].extend(block_forecasts)
                components['trend_backcasts'].extend(block_backcasts)
                components['trend_coefficients'].extend(coefficients)
            else:
                components['seasonality_forecasts'].extend(block_forecasts)
                components['seasonality_backcasts'].extend(block_backcasts)
                components['seasonality_coefficients'].extend(coefficients)

            total_forecast = total_forecast + stack_forecast
            residuals = residual

        components['final_residual'] = residuals
        components['total_forecast'] = total_forecast

        return total_forecast, components


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


def prepare_data(df, target_col='GROSS_ADDS'):
    """Prepare time series data with exogenous features."""
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


def prepare_sequences_nbeats(df, input_features, target_features, n_steps, m_steps):
    """Prepare sequences for N-BEATS training."""
    X, y = [], []
    for i in range(len(df) - n_steps - m_steps + 1):
        X.append(df[input_features].iloc[i:i + n_steps].values)
        y.append(df[target_features].iloc[i + n_steps:i + n_steps + m_steps].values)
    return np.array(X), np.array(y)


def train_nbeats_model(
        df,
        input_features,
        target_features,
        n_steps,
        m_steps,
        split_ratio=0.8,
        model_type="interpretable",
        hidden_size=512,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        patience_es=20,
        patience_rlr=10,
        stacks_config=None,
        blackbox_config=None,
        device="cpu"):
    """
    N-BEATS model training with all 4 required plots.
    """
    results = {
        'model': None,
        'metrics': {},
        'exog_metrics': {},
        'figures': [],
        'parameters': {
            'model_type': f'N-BEATS ({model_type})',
            'n_steps': n_steps,
            'm_steps': m_steps,
            'input_features': input_features,
            'target_features': target_features,
            'split_ratio': split_ratio,
            'hidden_size': hidden_size,
            'epochs': epochs
        },
        'scalers': {'X': None, 'y': None},
        'history': None,
        'device': device
    }

    try:
        df = df.dropna().reset_index(drop=True)
        X, y = prepare_sequences_nbeats(df, input_features, target_features, n_steps, m_steps)

        if len(X) == 0:
            raise ValueError("No valid sequences – check n_steps / m_steps.")

        n_input_features = len(input_features)
        n_target_features = len(target_features)

        X_reshaped = X.reshape(X.shape[0], -1) 
        y_reshaped = y.reshape(y.shape[0], -1)  

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_scaled = scaler_X.fit_transform(X_reshaped)
        y_scaled = scaler_y.fit_transform(y_reshaped)

        results['scalers']['X'] = scaler_X
        results['scalers']['y'] = scaler_y

        split_idx = int(len(X_scaled) * split_ratio)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)

        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        device = torch.device(device if torch.cuda.is_available() else "cpu")

        input_size = n_steps * n_input_features
        forecast_size = m_steps * n_target_features

        if model_type == "interpretable":
            model = InterpretableNBeats(
                input_size=input_size,
                forecast_size=forecast_size,
                hidden_size=hidden_size,
                stacks_config=stacks_config
            )
        else:
            if blackbox_config is None:
                blackbox_config = {'num_blocks': 4, 'num_layers': 4}

            stacks_config = [
                {'type': 'generic', 'num_blocks': blackbox_config['num_blocks'],
                 'hidden_size': hidden_size, 'num_layers': blackbox_config['num_layers']},
                {'type': 'generic', 'num_blocks': blackbox_config['num_blocks'],
                 'hidden_size': hidden_size, 'num_layers': blackbox_config['num_layers']}
            ]
            model = NBeats(
                input_size=input_size,
                forecast_size=forecast_size,
                hidden_size=hidden_size,
                stacks_config=stacks_config
            )

        model = model.to(device)
        results['model'] = model

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience_rlr, factor=0.5)

        train_losses = []
        val_losses = []

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                val_predictions = model(X_test_tensor.to(device))
                val_loss = criterion(val_predictions, y_test_tensor.to(device))
                val_losses.append(val_loss.item())
            model.train()

            scheduler.step(val_loss)

            if epoch % 10 == 0:
                logging.info(f'Epoch {epoch}, Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss.item():.6f}')

            if epoch > patience_es and min(val_losses[-patience_es:]) >= min(val_losses):
                logging.info(f'Early stopping at epoch {epoch}')
                break

        results['history'] = {'train_loss': train_losses, 'val_loss': val_losses}

        # 1. Training history plot
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('N-BEATS Training History')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            history_path = tmpfile.name
        plt.savefig(history_path, bbox_inches='tight')
        plt.close()
        results['figures'].append(history_path)

        # Get predictions
        model.eval()
        with torch.no_grad():
            y_pred_scaled = model(X_test_tensor.to(device)).cpu().numpy()
            y_train_pred_scaled = model(X_train_tensor.to(device)).cpu().numpy()

        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_original = scaler_y.inverse_transform(y_test)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
        y_train_original = scaler_y.inverse_transform(y_train)

        y_pred_3d = y_pred.reshape(-1, m_steps, n_target_features)
        y_test_3d = y_test_original.reshape(-1, m_steps, n_target_features)
        y_train_pred_3d = y_train_pred.reshape(-1, m_steps, n_target_features)
        y_train_3d = y_train_original.reshape(-1, m_steps, n_target_features)

        # Calculate metrics and generate plots for each target feature
        for i, feature in enumerate(target_features):
            y_pred_feat = y_pred_3d[:, 0, i]
            y_test_feat = y_test_3d[:, 0, i]
            y_train_pred_feat = y_train_pred_3d[:, 0, i]
            y_train_feat = y_train_3d[:, 0, i]

            metrics = {
                'MSE': mean_squared_error(y_test_feat, y_pred_feat),
                'MAE': mean_absolute_error(y_test_feat, y_pred_feat),
                'R2': r2_score(y_test_feat, y_pred_feat),
                'MAPE': np.mean(np.abs((y_test_feat - y_pred_feat) / np.maximum(np.abs(y_test_feat), 1))) * 100
            }

            if feature == target_features[0]:
                results['metrics'] = metrics
                
                # 2. Training fit plot (using first 100 samples)
                train_samples = min(100, len(y_train_feat))
                fig, train_path = save_training_fit_plot(
                    y_train_feat[:train_samples],
                    y_train_pred_feat[:train_samples],
                    title=f"N-BEATS - Training Fit: {feature}"
                )
                results['figures'].append(train_path)
                
                # 3. Validation plot
                fig, val_path = save_validation_plot(
                    y_test_feat,
                    y_pred_feat,
                    title=f"N-BEATS - Validation: {feature}"
                )
                results['figures'].append(val_path)
                
                # 4. Scatter plot
                fig, scatter_path = save_scatter_plot(
                    y_test_feat,
                    y_pred_feat,
                    title=f"N-BEATS - Predicted vs Actual: {feature}"
                )
                results['figures'].append(scatter_path)
                
                # 5. Future forecast (using last predictions as proxy)
                residuals = y_test_feat - y_pred_feat
                sigma = residuals.std()
                z = 1.96
                margin = z * sigma
                
                fig, future_path = save_future_forecast_plot(
                    y_pred_feat[-60:] if len(y_pred_feat) >= 60 else y_pred_feat,
                    title=f"N-BEATS - Future Forecast: {feature}",
                    confidence_intervals={
                        'lower': (y_pred_feat[-60:] - margin) if len(y_pred_feat) >= 60 else (y_pred_feat - margin),
                        'upper': (y_pred_feat[-60:] + margin) if len(y_pred_feat) >= 60 else (y_pred_feat + margin)
                    }
                )
                results['figures'].append(future_path)
            else:
                results['exog_metrics'][feature] = metrics

    except Exception as e:
        logging.error(f"N-BEATS training failed: {e}")
        results['error'] = str(e)

    return results


def multiVrecurrent_SVR(df, input_features, target_features, n_steps, m_steps, split_ratio, svr_params=None):
    """Multivariate SVR with all 4 required plots."""
    results = {
        'model': None,
        'metrics': {},
        'exog_metrics': {},
        'figures': [],
        'parameters': {
            'model_type': 'Multivariate SVR',
            'n_steps': n_steps,
            'm_steps': m_steps,
            'input_features': input_features,
            'target_features': target_features,
            'split_ratio': split_ratio,
            'svr_params': svr_params
        },
        'scalers': {'X': None, 'y': None}
    }

    try:
        X, y = prepare_sequences(df, input_features, target_features, n_steps, m_steps)
        logging.info(f"Original shapes - X: {X.shape}, y: {y.shape}")

        y_reshaped = y.reshape(y.shape[0], -1)  
        logging.info(f"Multi-step target shape: {y_reshaped.shape}")

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_flat = X.reshape(X.shape[0], -1)  
        logging.info(f"Flattened X shape: {X_flat.shape}")

        X_scaled = scaler_X.fit_transform(X_flat)
        y_scaled = scaler_y.fit_transform(y_reshaped)

        results['scalers']['X'] = scaler_X
        results['scalers']['y'] = scaler_y

        split_idx = int(len(X_scaled) * split_ratio)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]

        logging.info(f"Final train shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        logging.info(f"Final test shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")

        if svr_params is None:
            svr_params = {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1,
                'gamma': 'scale',
                'max_iter': 10000
            }

        base_svr = SVR(**svr_params)
        model = MultiOutputRegressor(base_svr)
        model.fit(X_train, y_train)

        results['model'] = model

        y_pred_scaled = model.predict(X_test)
        y_train_pred_scaled = model.predict(X_train)
        logging.info(f"Multi-step prediction shape: {y_pred_scaled.shape}")

        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_inv = scaler_y.inverse_transform(y_test)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled)
        y_train_inv = scaler_y.inverse_transform(y_train)

        y_pred_3d = y_pred.reshape(-1, m_steps, len(target_features))
        y_test_3d = y_test_inv.reshape(-1, m_steps, len(target_features))
        y_train_pred_3d = y_train_pred.reshape(-1, m_steps, len(target_features))
        y_train_3d = y_train_inv.reshape(-1, m_steps, len(target_features))

        logging.info(f"Reshaped predictions - y_pred_3d: {y_pred_3d.shape}, y_test_3d: {y_test_3d.shape}")

        for i, feature in enumerate(target_features):
            y_pred_feat = y_pred_3d[:, 0, i]
            y_test_feat = y_test_3d[:, 0, i]
            y_train_pred_feat = y_train_pred_3d[:, 0, i]
            y_train_feat = y_train_3d[:, 0, i]

            metrics = {
                'MSE': mean_squared_error(y_test_feat, y_pred_feat),
                'MAE': mean_absolute_error(y_test_feat, y_pred_feat),
                'R2': r2_score(y_test_feat, y_pred_feat),
                'MAPE': np.mean(np.abs((y_test_feat - y_pred_feat) / np.maximum(np.abs(y_test_feat), 1))) * 100
            }

            logging.info(f"Metrics for {feature}: {metrics}")

            if feature == 'y_diff' or feature == target_features[0]:
                results['metrics'] = metrics
                
                # Generate 4 required plots for primary target
                train_samples = min(100, len(y_train_feat))
                
                # 1. Training fit plot
                fig, train_path = save_training_fit_plot(
                    y_train_feat[:train_samples],
                    y_train_pred_feat[:train_samples],
                    title=f"SVR - Training Fit: {feature}"
                )
                results['figures'].append(train_path)
                
                # 2. Validation plot
                fig, val_path = save_validation_plot(
                    y_test_feat,
                    y_pred_feat,
                    title=f"SVR - Validation: {feature}"
                )
                results['figures'].append(val_path)
                
                # 3. Scatter plot
                fig, scatter_path = save_scatter_plot(
                    y_test_feat,
                    y_pred_feat,
                    title=f"SVR - Predicted vs Actual: {feature}"
                )
                results['figures'].append(scatter_path)
                
                # 4. Future forecast
                residuals = y_test_feat - y_pred_feat
                sigma = residuals.std()
                z = 1.96
                margin = z * sigma
                
                fig, future_path = save_future_forecast_plot(
                    y_pred_feat[-60:] if len(y_pred_feat) >= 60 else y_pred_feat,
                    title=f"SVR - Future Forecast: {feature}",
                    confidence_intervals={
                        'lower': (y_pred_feat[-60:] - margin) if len(y_pred_feat) >= 60 else (y_pred_feat - margin),
                        'upper': (y_pred_feat[-60:] + margin) if len(y_pred_feat) >= 60 else (y_pred_feat + margin)
                    }
                )
                results['figures'].append(future_path)
            else:
                results['exog_metrics'][feature] = metrics

    except Exception as e:
        logging.error(f"Multivariate SVR failed: {str(e)}", exc_info=True)
        results['error'] = str(e)

    return results


def multiVrecurrent_LR(df, input_features, target_features, n_steps, m_steps, split_ratio):
    """Multivariate Linear Regression with all 4 required plots."""
    results = {
        'model': None,
        'metrics': {},
        'exog_metrics': {},
        'figures': [],
        'parameters': {
            'model_type': 'Multivariate Linear Regression',
            'n_steps': n_steps,
            'm_steps': m_steps,
            'input_features': input_features,
            'target_features': target_features,
            'split_ratio': split_ratio
        },
        'scalers': {'X': None, 'y': None}
    }

    try:
        X, y = prepare_sequences(df, input_features, target_features, n_steps, m_steps)
        logging.info(f"Original shapes - X: {X.shape}, y: {y.shape}")
        
        X_shape = X.shape
        y_shape = y.shape
     
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_flat = X.reshape(-1, len(input_features))
        y_flat = y.reshape(-1, len(target_features))
        
        X_scaled_flat = scaler_X.fit_transform(X_flat)
        y_scaled_flat = scaler_y.fit_transform(y_flat)
        
        X_scaled = X_scaled_flat.reshape(X_shape)
        y_scaled = y_scaled_flat.reshape(y_shape)
        
        results['scalers']['X'] = scaler_X
        results['scalers']['y'] = scaler_y
        
        split_idx = int(len(X_scaled) * split_ratio)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y_scaled[:split_idx], y_scaled[split_idx:]
        logging.info(f"Train/test shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_test_2d = X_test.reshape(X_test.shape[0], -1)
        y_train_2d = y_train.reshape(y_train.shape[0], -1)
        y_test_2d = y_test.reshape(y_test.shape[0], -1)
        logging.info(f"2D shapes - X_train_2d: {X_train_2d.shape}, y_train_2d: {y_train_2d.shape}")
        
        model = LinearRegression()
        model.fit(X_train_2d, y_train_2d)
        results['model'] = model
        
        y_pred = model.predict(X_test_2d)
        y_train_pred = model.predict(X_train_2d)
        logging.info(f"Prediction shape: {y_pred.shape}")
        
        y_pred_2d = y_pred.reshape(-1, len(target_features))
        y_pred_inv = scaler_y.inverse_transform(y_pred_2d)
        
        y_test_2d_flat = y_test_2d.reshape(-1, len(target_features))
        y_test_inv = scaler_y.inverse_transform(y_test_2d_flat)
        
        y_train_pred_2d = y_train_pred.reshape(-1, len(target_features))
        y_train_pred_inv = scaler_y.inverse_transform(y_train_pred_2d)
        y_train_2d_flat = y_train_2d.reshape(-1, len(target_features))
        y_train_inv = scaler_y.inverse_transform(y_train_2d_flat)
        
        y_pred_inv_3d = y_pred_inv.reshape(len(X_test), m_steps, len(target_features))
        y_test_inv_3d = y_test_inv.reshape(len(X_test), m_steps, len(target_features))
        y_train_pred_inv_3d = y_train_pred_inv.reshape(len(X_train), m_steps, len(target_features))
        y_train_inv_3d = y_train_inv.reshape(len(X_train), m_steps, len(target_features))
        
        for i, feature in enumerate(target_features):
            y_pred_feat = y_pred_inv_3d[:, 0, i]
            y_test_feat = y_test_inv_3d[:, 0, i]
            y_train_pred_feat = y_train_pred_inv_3d[:, 0, i]
            y_train_feat = y_train_inv_3d[:, 0, i]
            
            metrics = {
                'MSE': mean_squared_error(y_test_feat, y_pred_feat),
                'MAE': mean_absolute_error(y_test_feat, y_pred_feat),
                'R2': r2_score(y_test_feat, y_pred_feat)
            }
            
            if feature == 'y_diff' or feature == target_features[0]:
                results['metrics'] = metrics
                
                # Generate 4 required plots for primary target
                train_samples = min(100, len(y_train_feat))
                
                # 1. Training fit plot
                fig, train_path = save_training_fit_plot(
                    y_train_feat[:train_samples],
                    y_train_pred_feat[:train_samples],
                    title=f"ULR - Training Fit: {feature}"
                )
                results['figures'].append(train_path)
                
                # 2. Validation plot
                fig, val_path = save_validation_plot(
                    y_test_feat,
                    y_pred_feat,
                    title=f"ULR - Validation: {feature}"
                )
                results['figures'].append(val_path)
                
                # 3. Scatter plot
                fig, scatter_path = save_scatter_plot(
                    y_test_feat,
                    y_pred_feat,
                    title=f"ULR - Predicted vs Actual: {feature}"
                )
                results['figures'].append(scatter_path)
                
                # 4. Future forecast
                residuals = y_test_feat - y_pred_feat
                sigma = residuals.std()
                z = 1.96
                margin = z * sigma
                
                fig, future_path = save_future_forecast_plot(
                    y_pred_feat[-60:] if len(y_pred_feat) >= 60 else y_pred_feat,
                    title=f"ULR - Future Forecast: {feature}",
                    confidence_intervals={
                        'lower': (y_pred_feat[-60:] - margin) if len(y_pred_feat) >= 60 else (y_pred_feat - margin),
                        'upper': (y_pred_feat[-60:] + margin) if len(y_pred_feat) >= 60 else (y_pred_feat + margin)
                    }
                )
                results['figures'].append(future_path)
            else:
                results['exog_metrics'][feature] = metrics

    except Exception as e:
        logging.error(f"Multivariate LR failed: {str(e)}", exc_info=True)
        results['error'] = str(e)
        
    return results


def grid_search_sarimax(y_train, exog_train=None, seasonal_period=7):
    """Grid search for SARIMAX parameters."""
    results = {
        'model': None,
        'metrics': {},
        'figures': [],
        'parameters': {
            'model_type': 'SARIMAX',
            'seasonal_period': seasonal_period
        }
    }
    
    try:
        best_aic = float('inf')
        best_model = None
        best_order = None
        best_seasonal_order = None
        
        p_params = [0, 1]
        d_params = [1]
        q_params = [0, 1]
        P_params = [0, 1]
        D_params = [1]
        Q_params = [0, 1]
        
        for order in itertools.product(p_params, d_params, q_params):
            for seasonal_order in itertools.product(P_params, D_params, Q_params):
                s_order = (seasonal_order[0], seasonal_order[1], seasonal_order[2], seasonal_period)
                
                try:
                    model = SARIMAX(
                        endog=y_train,
                        exog=exog_train,
                        order=order,
                        seasonal_order=s_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    model_fit = model.fit(disp=False)
                    
                    if model_fit.aic < best_aic:
                        best_aic = model_fit.aic
                        best_model = model_fit
                        best_order = order
                        best_seasonal_order = s_order
                except:
                    continue
        
        if best_model is None:
            raise RuntimeError("No valid SARIMAX models were trained")
        
        results['model'] = best_model
        results['parameters']['order'] = best_order
        results['parameters']['seasonal_order'] = best_seasonal_order
        
        plt.figure(figsize=(12, 8))
        best_model.plot_diagnostics(figsize=(12, 8))
        plt.tight_layout()
        plt.close()
        
    except Exception as e:
        logging.error(f"SARIMAX failed: {str(e)}")
        results['error'] = str(e)
    
    return results


def train_cnn_bilstm_model(
        df,
        input_features,
        target_features,
        n_steps,
        m_steps,
        split_ratio=0.8,
        conv_filters=32,
        lstm_units=20,
        epochs=20,
        batch_size=30,
        learning_rate=0.001,
        patience_es=30,
        patience_rlr=5,
        seed=42):
    """CNN-BiLSTM model with all 4 required plots."""
    results = {
        'model': None,
        'metrics': {},
        'figures': [],
        'parameters': {
            'model_type': 'CNN_BiLSTM',
            'n_steps': n_steps,
            'm_steps': m_steps,
            'input_features': input_features,
            'target_features': target_features,
            'split_ratio': split_ratio,
            'conv_filters': conv_filters,
            'lstm_units': lstm_units,
            'epochs': epochs,
            'batch_size': batch_size
        },
        'scalers': {'X': None, 'y': None},
        'history': None
    }

    try:
        df = df.dropna().reset_index(drop=True)
        X, y = prepare_sequences(df, input_features, target_features,
                                 n_steps, m_steps)

        if len(X) == 0:
            raise ValueError("No valid sequences – check n_steps / m_steps.")

        n_input_features = len(input_features)
        n_target_features = len(target_features)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_shape, y_shape = X.shape, y.shape        

        X = scaler_X.fit_transform(X.reshape(-1, n_input_features)
                                   ).reshape(X_shape)
        y = scaler_y.fit_transform(y.reshape(-1, n_target_features)
                                   ).reshape(y_shape)

        results['scalers']['X'] = scaler_X
        results['scalers']['y'] = scaler_y

        split_idx = int(len(X) * split_ratio)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        tf.keras.utils.set_random_seed(seed)

        input_shape = (n_steps, n_input_features)
        output_size = n_target_features

        model = Sequential([
            Conv1D(filters=conv_filters, kernel_size=1,
                   kernel_regularizer=l2(0.001),
                   input_shape=input_shape),
            LeakyReLU(alpha=0.01),

            Bidirectional(LSTM(lstm_units,
                               return_sequences=True,
                               kernel_regularizer=l2(0.001))),
            Dropout(0.2),

            Bidirectional(LSTM(lstm_units,
                               return_sequences=False,
                               kernel_regularizer=l2(0.001))),
            Dropout(0.2),

            Dense(m_steps * output_size, kernel_regularizer=l2(0.001)),
            layers.Reshape((m_steps, output_size))
        ])

        optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
        model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

        class ValidationMetrics(Callback):
            def __init__(self, validation_data, scaler_y):
                super().__init__()
                self.X_val, self.y_val = validation_data
                self.scaler_y = scaler_y

            def on_epoch_end(self, epoch, logs=None):
                y_pred = self.model.predict(self.X_val, verbose=0)
                y_pred = self.scaler_y.inverse_transform(
                            y_pred.reshape(-1, y_pred.shape[-1]))
                y_true = self.scaler_y.inverse_transform(
                            self.y_val.reshape(-1, self.y_val.shape[-1]))

                val_r2 = r2_score(y_true, y_pred)
                val_mae = mean_absolute_error(y_true, y_pred)
                print(f"Epoch {epoch+1:03d} – val R²: {val_r2: .4f}  "
                      f"val MAE: {val_mae: .4f}")

        early_stop = EarlyStopping(monitor='val_mae',
                                   patience=patience_es,
                                   restore_best_weights=True,
                                   verbose=1)

        reduce_lr = ReduceLROnPlateau(monitor='val_mae',
                                       factor=0.5,
                                       patience=patience_rlr,
                                       min_lr=1e-6,
                                       verbose=1)

        val_cbk = ValidationMetrics(validation_data=(X_test, y_test),
                                       scaler_y=scaler_y)

        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stop, reduce_lr, val_cbk],
                            verbose=1)

        results['model'] = model
        results['history'] = history.history

        # 1. Training history plot
        plt.figure(figsize=(10,5))
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='validation')
        plt.title('CNN-BiLSTM Training History')
        plt.xlabel('epoch')
        plt.ylabel('MAE Loss')
        plt.legend()
        plt.grid(True)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            history_path = tmpfile.name
        plt.savefig(history_path, bbox_inches='tight')
        plt.close()
        results['figures'].append(history_path)

        y_pred = model.predict(X_test, verbose=0)      
        y_train_pred = model.predict(X_train, verbose=0)

        y_pred_step1 = y_pred[:, 0, :]                
        y_test_step1 = y_test[:, 0, :]                
        y_train_pred_step1 = y_train_pred[:, 0, :]
        y_train_step1 = y_train[:, 0, :]

        y_pred_inv = scaler_y.inverse_transform(y_pred_step1)
        y_test_inv = scaler_y.inverse_transform(y_test_step1)
        y_train_pred_inv = scaler_y.inverse_transform(y_train_pred_step1)
        y_train_inv = scaler_y.inverse_transform(y_train_step1)

        for i, feat in enumerate(target_features):
            if feat == target_features[0]:
                # Generate 4 required plots for primary target
                train_samples = min(100, len(y_train_inv))
                
                # 1. Training fit plot
                fig, train_path = save_training_fit_plot(
                    y_train_inv[:train_samples, i],
                    y_train_pred_inv[:train_samples, i],
                    title=f"LSTM - Training Fit: {feat}"
                )
                results['figures'].append(train_path)
                
                # 2. Validation plot
                fig, val_path = save_validation_plot(
                    y_test_inv[:, i],
                    y_pred_inv[:, i],
                    title=f"LSTM - Validation: {feat}"
                )
                results['figures'].append(val_path)
                
                # 3. Scatter plot
                fig, scatter_path = save_scatter_plot(
                    y_test_inv[:, i],
                    y_pred_inv[:, i],
                    title=f"LSTM - Predicted vs Actual: {feat}"
                )
                results['figures'].append(scatter_path)
                
                # 4. Future forecast
                residuals = y_test_inv[:, i] - y_pred_inv[:, i]
                sigma = residuals.std()
                z = 1.96
                margin = z * sigma
                
                fig, future_path = save_future_forecast_plot(
                    y_pred_inv[-60:, i] if len(y_pred_inv) >= 60 else y_pred_inv[:, i],
                    title=f"LSTM - Future Forecast: {feat}",
                    confidence_intervals={
                        'lower': (y_pred_inv[-60:, i] - margin) if len(y_pred_inv) >= 60 else (y_pred_inv[:, i] - margin),
                        'upper': (y_pred_inv[-60:, i] + margin) if len(y_pred_inv) >= 60 else (y_pred_inv[:, i] + margin)
                    }
                )
                results['figures'].append(future_path)

            # metrics
            mse = mean_squared_error(y_test_inv[:, i], y_pred_inv[:, i])
            mae = mean_absolute_error(y_test_inv[:, i], y_pred_inv[:, i])
            r2 = r2_score(y_test_inv[:, i], y_pred_inv[:, i])
            results['metrics'] = {'MSE': mse, 'MAE': mae, 'R2': r2}

    except Exception as e:
        logging.error(f"CNN-BiLSTM training failed: {e}")
        results['error'] = str(e)

    return results


def train_multivariate_models(df, config, logger):
    """
    Train multivariate forecasting models based on config selection.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    config : dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance
    
    Returns:
    --------
    dict
        Dictionary containing trained model results
    """
    models = {}
    
    # Get models to run from config
    models_to_run = config.get('models_to_run', {}).get('multivariate', 
        ['mlr', 'sarimax', 'lstm'])
    
    base_params = {
        'n_steps': config['n_steps'],
        'm_steps': config['m_steps'],
        'input_features': config['input_features'],
        'target_features': config['target_features'],
        'split_ratio': config['split_ratio']
    }
    
    logger.info(f"Training multivariate models: {models_to_run}")
    
    # MLR - Multivariate Linear Regression
    if 'mlr' in models_to_run:
        try:
            logger.info("Training Multivariate Linear Regression model")
            mlr_result = multiVrecurrent_LR(
                df,
                config['input_features'],
                config['target_features'],
                config['n_steps'], 
                config['m_steps'], 
                config['split_ratio']
            )
            mlr_result['parameters'].update(base_params)
            models['mlr'] = mlr_result
        except Exception as e:
            logger.error(f"Multivariate LR failed: {str(e)}")
            models['mlr'] = {'error': str(e)}
    
    # SARIMAX
    if 'sarimax' in models_to_run:
        try:
            logger.info("Training SARIMAX model")
            df_prepared = prepare_data(df)
            horizon = config['m_steps']
            train_size = len(df_prepared) - horizon
            train_df = df_prepared.iloc[:train_size]
            y_train = train_df[config['target_col']]
            exog_cols = [col for col in df_prepared.columns if col.startswith('dow_') or col in ['holiday', 'month', 'quarter']]
            exog_train = train_df[exog_cols] if exog_cols else None
            
            sarimax_result = grid_search_sarimax(y_train, exog_train, config.get('seasonal_period', 7))
            sarimax_result['parameters'].update(base_params)
            models['sarimax'] = sarimax_result
        except Exception as e:
            logger.error(f"SARIMAX failed: {str(e)}")
            models['sarimax'] = {'error': str(e)}
    
    # LSTM
    if 'lstm' in models_to_run:
        try:
            logger.info("Training LSTM model")
            lstm_config = config.get('lstm', {})
            lstm_result = train_cnn_bilstm_model(
                df,
                config['input_features'], 
                config['target_features'], 
                config['n_steps'],
                config['m_steps'],
                config['split_ratio'],
                epochs=lstm_config.get('epochs', 20),
                batch_size=lstm_config.get('batch_size', 30)
            )
            lstm_result['parameters'].update(base_params)
            models['lstm'] = lstm_result
        except Exception as e:
            logger.error(f"LSTM failed: {str(e)}")
            models['lstm'] = {'error': str(e)}
    
    return models
