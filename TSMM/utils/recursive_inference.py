"""Shared recursive forecasting helpers for live inference."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd


def _next_feature_rows(
    predictions: np.ndarray,
    last_target: float,
    input_features: Sequence[str],
    target_features: Sequence[str],
    target_col: str,
    max_window: int,
    target_history: list[float],
    diff_history: list[float],
) -> tuple[np.ndarray, float, list[float], list[float]]:
    rows: list[list[float]] = []
    current_target = float(last_target)

    for prediction in predictions:
        if "y_diff" in target_features:
            y_diff = float(prediction[target_features.index("y_diff")])
            current_target += y_diff
        elif target_col in target_features:
            next_target = float(prediction[target_features.index(target_col)])
            y_diff = next_target - current_target
            current_target = next_target
        else:
            y_diff = 0.0

        target_history.append(current_target)
        diff_history.append(y_diff)

        feature_row: list[float] = []
        for feature in input_features:
            if feature == target_col:
                value = current_target
            elif feature == "y_diff":
                value = y_diff
            elif feature in target_features:
                value = float(prediction[target_features.index(feature)])
            elif feature.startswith(("SMA_", "EMA_", "Volatility_")):
                is_diff = feature.endswith("_diff")
                base_name = feature[:-5] if is_diff else feature
                try:
                    window = int(base_name.split("_")[1])
                except (IndexError, ValueError):
                    value = 0.0
                else:
                    history = diff_history if is_diff else target_history
                    values = history[-window:]
                    if len(values) < window:
                        value = 0.0
                    elif base_name.startswith("SMA_"):
                        value = float(np.mean(values))
                    elif base_name.startswith("EMA_"):
                        value = float(pd.Series(values).ewm(span=window, adjust=False).mean().iloc[-1])
                    else:
                        value = float(np.std(values))
            else:
                value = 0.0
            feature_row.append(value)
        rows.append(feature_row)

    history_limit = max(int(max_window), 1) * 5
    if len(target_history) > history_limit:
        target_history = target_history[-max(int(max_window), 1):]
    if len(diff_history) > history_limit:
        diff_history = diff_history[-max(int(max_window), 1):]
    return np.asarray(rows, dtype=np.float64), current_target, target_history, diff_history


def recursive_forecast_matrix(
    predict_window: Callable[[np.ndarray], np.ndarray],
    initial_window: np.ndarray,
    steps: int,
    m_steps: int,
    input_features: Sequence[str],
    target_features: Sequence[str],
    target_col: str,
    max_window: int,
) -> np.ndarray:
    """Generate a full future path by feeding each prediction into the next window."""
    current_window = np.asarray(initial_window, dtype=np.float64).copy()
    input_features = [str(value) for value in input_features]
    target_features = [str(value) for value in target_features]
    steps = max(int(steps), 1)
    m_steps = max(int(m_steps), 1)

    if current_window.ndim != 2 or current_window.shape[1] != len(input_features):
        raise ValueError("initial_window_shape_does_not_match_input_features")
    if not target_features:
        raise ValueError("target_features_required")
    if target_col not in input_features:
        raise ValueError(f"target_col_missing_from_inputs:{target_col}")

    target_index = input_features.index(target_col)
    last_target = float(current_window[-1, target_index])
    target_history = current_window[:, target_index].astype(float).tolist()
    diff_history = (
        current_window[:, input_features.index("y_diff")].astype(float).tolist()
        if "y_diff" in input_features
        else []
    )
    predictions: list[np.ndarray] = []

    iterations = (steps + m_steps - 1) // m_steps
    for _ in range(iterations):
        predicted = np.asarray(predict_window(current_window), dtype=np.float64)
        if predicted.size % len(target_features) != 0:
            raise ValueError(
                f"prediction_shape_does_not_match_targets:{tuple(predicted.shape)}:{len(target_features)}"
            )
        predicted = predicted.reshape(-1, len(target_features))
        if predicted.shape[0] == 0:
            raise ValueError("empty_prediction")
        predictions.append(predicted)

        new_rows, last_target, target_history, diff_history = _next_feature_rows(
            predicted,
            last_target,
            input_features,
            target_features,
            target_col,
            max_window,
            target_history,
            diff_history,
        )
        current_window = np.vstack([current_window[len(new_rows):], new_rows])

    return np.vstack(predictions)[:steps]
