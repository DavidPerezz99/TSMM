import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.investing_agent import _latest_inference_window
from utils.recursive_inference import recursive_forecast_matrix


class InferenceWindowFreshnessTests(unittest.TestCase):
    def test_sequence_uses_latest_n_steps_without_horizon_offset(self):
        rows = [{"value": value} for value in range(1, 7)]

        selected = _latest_inference_window(rows, n_steps=3)

        self.assertEqual([row["value"] for row in selected], [4, 5, 6])

    def test_dataframe_window_includes_latest_updated_bucket(self):
        frame = pd.DataFrame(
            {
                "DATE": pd.date_range("2026-07-12 21:00:00", periods=4, freq="10min"),
                "CLOSE": [100.0, 101.0, 102.0, 109.0],
            }
        )

        selected = _latest_inference_window(frame, n_steps=2)

        self.assertEqual(selected["CLOSE"].tolist(), [102.0, 109.0])
        self.assertEqual(selected.iloc[-1]["DATE"], frame.iloc[-1]["DATE"])

    def test_recursive_forecast_returns_every_configured_step(self):
        calls = []

        def predict_window(window):
            calls.append(window.copy())
            return np.array([[1.0, window[-1, 0] / 100.0]])

        result = recursive_forecast_matrix(
            predict_window=predict_window,
            initial_window=np.array([[100.0, 0.0], [101.0, 1.0]]),
            steps=6,
            m_steps=1,
            input_features=["HIGH", "y_diff"],
            target_features=["y_diff", "Price_return"],
            target_col="HIGH",
            max_window=2,
        )

        self.assertEqual(result.shape, (6, 2))
        self.assertEqual(result[:, 0].tolist(), [1.0] * 6)
        self.assertEqual(len(calls), 6)
        self.assertEqual(calls[-1][-1, 0], 106.0)


if __name__ == "__main__":
    unittest.main()
