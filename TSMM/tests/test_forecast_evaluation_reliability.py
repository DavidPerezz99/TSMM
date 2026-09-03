import unittest

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from models.multivariate_models import multiVrecurrent_LR
from utils.evaluator import (
    _forecast_metrics,
    _recursive_exogenous_defaults,
    _rolling_origin_predictions,
)


class ForecastEvaluationReliabilityTests(unittest.TestCase):
    def test_recursive_cross_asset_assumptions_are_causal_and_feature_aware(self):
        window = np.asarray(
            [
                [100.0, 1.0, 5000.0, 2.5],
                [101.0, 1.0, 5010.0, -1.5],
            ]
        )
        defaults = _recursive_exogenous_defaults(
            window,
            ["HIGH", "y_diff", "US500_CLOSE", "US500_Price_return"],
            ["y_diff"],
            {"target_col": "HIGH"},
        )

        self.assertEqual(defaults["US500_CLOSE"], 5010.0)
        self.assertEqual(defaults["US500_Price_return"], 0.0)

    def test_rolling_origin_uses_actual_preceding_windows(self):
        frame = pd.DataFrame(
            {
                "value": np.arange(20, dtype=float),
                "y_diff": np.ones(20, dtype=float),
            }
        )
        n_steps = 3
        test_size = 5
        windows = []
        targets = []
        for index in range(n_steps, len(frame) - test_size):
            windows.append(frame[["value", "y_diff"]].iloc[index - n_steps:index].to_numpy())
            targets.append([frame.iloc[index]["y_diff"]])
        windows = np.asarray(windows)
        targets = np.asarray(targets)
        scaler_x = StandardScaler().fit(windows.reshape(-1, 2))
        scaler_y = StandardScaler().fit(targets)
        model = LinearRegression().fit(
            scaler_x.transform(windows.reshape(-1, 2)).reshape(len(windows), -1),
            scaler_y.transform(targets),
        )

        predictions = _rolling_origin_predictions(
            model,
            {"X": scaler_x, "y": scaler_y},
            frame,
            ["value", "y_diff"],
            ["y_diff"],
            n_steps,
            1,
            test_size,
            "ulr",
        )

        self.assertEqual(predictions.shape, (test_size, 1))
        np.testing.assert_allclose(predictions[:, 0], 1.0, atol=1e-8)

    def test_metrics_record_protocol_samples_and_zero_baseline_skill(self):
        metrics = _forecast_metrics([1.0, -1.0, 2.0], [0.8, -0.7, 1.8])
        self.assertEqual(metrics["sample_count"], 3)
        self.assertEqual(metrics["evaluation_protocol"], "rolling_origin_one_step")
        self.assertEqual(metrics["direction_accuracy"], 1.0)
        self.assertGreater(metrics["MAE_skill_vs_zero_change"], 0.0)

    def test_ulr_scalers_are_fit_without_holdout_rows(self):
        values = np.arange(80, dtype=float)
        frame = pd.DataFrame({"x": values, "y": values * 2.0})
        result = multiVrecurrent_LR(
            frame, ["x"], ["y"], n_steps=2, m_steps=1,
            split_ratio=0.8, test_size=20,
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["parameters"]["test_size"], 20)
        # Training input sequences end well before the extreme holdout values;
        # a full-data fit would have a materially larger mean.
        self.assertLess(float(result["scalers"]["X"].mean_[0]), 32.0)


if __name__ == "__main__":
    unittest.main()
