import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.inference_performance import InferencePerformanceStore
from utils.market_db import upsert_ohlc_1m


class InferencePerformanceStoreTests(unittest.TestCase):
    def test_matures_forecasts_and_scores_latest_forecast_per_origin_bucket(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            market_db = root / "market.sqlite"
            metrics_db = root / "metrics.sqlite"
            candles = pd.DataFrame(
                [
                    {"DATE": "2026-07-13 00:00:00", "OPEN": 99.0, "HIGH": 100.0, "LOW": 98.0, "CLOSE": 99.5, "VOLUME": 1.0},
                    {"DATE": "2026-07-13 00:05:00", "OPEN": 101.0, "HIGH": 102.0, "LOW": 100.0, "CLOSE": 101.5, "VOLUME": 1.0},
                    {"DATE": "2026-07-13 00:10:00", "OPEN": 104.0, "HIGH": 105.0, "LOW": 103.0, "CLOSE": 104.5, "VOLUME": 1.0},
                    {"DATE": "2026-07-13 00:15:00", "OPEN": 108.0, "HIGH": 109.0, "LOW": 107.0, "CLOSE": 108.5, "VOLUME": 1.0},
                ]
            )
            upsert_ohlc_1m(str(market_db), candles, symbol="XAUUSD")
            store = InferencePerformanceStore(metrics_db)

            common = {
                "timeframe": "5m",
                "timeframe_minutes": 5,
                "family": "high",
                "model": "test",
                "model_path": "model_v1.joblib",
                "model_updated_at_utc": "2026-07-12 00:00:00",
                "target_feature": "y_diff",
                "step": 1,
                "inference_strength": 0.7,
                "r2_train": 0.4,
            }
            rows = [
                {**common, "generated_at_utc": "2026-07-13 00:00:10", "origin_bucket_utc": "2026-07-13 00:00:00", "predicted_value": 99.0, "input_fingerprint": "old"},
                {**common, "generated_at_utc": "2026-07-13 00:04:50", "origin_bucket_utc": "2026-07-13 00:00:00", "predicted_value": 2.0, "input_fingerprint": "latest"},
                {**common, "generated_at_utc": "2026-07-13 00:09:50", "origin_bucket_utc": "2026-07-13 00:05:00", "predicted_value": 3.0, "input_fingerprint": "second"},
                {**common, "generated_at_utc": "2026-07-13 00:14:50", "origin_bucket_utc": "2026-07-13 00:10:00", "predicted_value": 4.0, "input_fingerprint": "third"},
            ]

            self.assertEqual(store.record_forecasts(rows), 4)
            self.assertEqual(store.mature_pending(str(market_db), "2026-07-13 00:20:00"), 4)
            metrics = store.rolling_metrics("5m", "high", "test", window_samples=10, min_samples=3)

            self.assertEqual(metrics["r2_live_samples"], 3)
            self.assertEqual(metrics["r2_live_rolling"], 1.0)

            current = store.rolling_metrics(
                "5m", "high", "test", model_path="model_v1.joblib", window_samples=10, min_samples=3
            )
            self.assertEqual(current["r2_live_rolling"], 1.0)

    def test_metric_snapshots_expose_r2_delta(self):
        with TemporaryDirectory() as tmpdir:
            store = InferencePerformanceStore(Path(tmpdir) / "metrics.sqlite")
            first = store.record_metric_snapshot(
                "2026-07-13 00:00:00", "10m", "high", "nbeats", "model.joblib",
                {"r2_live_rolling": 0.8, "r2_live_samples": 10, "r2_live_window_samples": 100},
            )
            second = store.record_metric_snapshot(
                "2026-07-13 00:10:00", "10m", "high", "nbeats", "model.joblib",
                {"r2_live_rolling": 0.55, "r2_live_samples": 11, "r2_live_window_samples": 100},
            )

            self.assertIsNone(first["r2_live_previous"])
            self.assertEqual(second["r2_live_previous"], 0.8)
            self.assertEqual(second["r2_live_delta"], -0.25)


if __name__ == "__main__":
    unittest.main()
