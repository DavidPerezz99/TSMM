"""
Models Package

This package provides forecasting models for time series analysis.
"""

from .univariate_models import (
    train_univariate_models,
    train_xgboost_model,
    train_prophet_model,
    train_sarimax_model,
    save_training_fit_plot,
    save_validation_plot,
    save_scatter_plot,
    save_future_forecast_plot
)
from .multivariate_models import (
    train_multivariate_models,
    train_nbeats_model,
    train_cnn_bilstm_model,
    multiVrecurrent_LR,
    multiVrecurrent_SVR,
    grid_search_sarimax,
    NBeats,
    InterpretableNBeats,
    NBeatsStack,
    InterpretableNBeatsStack,
    GenericBlock,
    TrendBlock,
    SeasonalityBlock,
    InterpretableTrendBlock,
    InterpretableSeasonalityBlock
)

__all__ = [
    'train_univariate_models',
    'train_xgboost_model',
    'train_prophet_model',
    'train_sarimax_model',
    'save_training_fit_plot',
    'save_validation_plot',
    'save_scatter_plot',
    'save_future_forecast_plot',
    'train_multivariate_models',
    'train_nbeats_model',
    'train_cnn_bilstm_model',
    'multiVrecurrent_LR',
    'multiVrecurrent_SVR',
    'grid_search_sarimax',
    'NBeats',
    'InterpretableNBeats',
    'NBeatsStack',
    'InterpretableNBeatsStack',
    'GenericBlock',
    'TrendBlock',
    'SeasonalityBlock',
    'InterpretableTrendBlock',
    'InterpretableSeasonalityBlock'
]
