import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.momentum import compute_momentum_overlay
from utils.vol_target import compute_vol_target_overlay
from utils.regime import classify_market_regime


class OverlayTests(unittest.TestCase):
    def _make_df(self, n=400, drift=0.0008, sigma=0.01):
        rng = np.random.default_rng(42)
        rets = drift + sigma * rng.normal(size=n)
        price = 100 * np.cumprod(1 + rets)
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
        return pd.DataFrame({"HIGH": price, "y_diff": pd.Series(price).diff().fillna(0.0).values}, index=idx)

    def test_momentum_output_schema(self):
        df = self._make_df()
        config = {
            "target_col": "HIGH",
            "momentum": {"enabled": True, "windows": [20, 60, 120], "weights": [0.5, 0.3, 0.2]},
        }
        out = compute_momentum_overlay(df, config)
        self.assertTrue(out.get("enabled"))
        self.assertIn("momentum_score", out)
        self.assertGreaterEqual(out["momentum_score"], -1.0)
        self.assertLessEqual(out["momentum_score"], 1.0)
        self.assertIn(out.get("trend_state"), {"up", "down", "flat"})
        self.assertIn(out.get("confidence_bucket"), {"low", "medium", "high"})

    def test_vol_target_output_schema(self):
        df = self._make_df()
        config = {
            "target_col": "HIGH",
            "vol_target": {
                "enabled": True,
                "window": 30,
                "target_vol": 0.15,
                "bars_per_year": 252,
                "caps": {"min_scale": 0.25, "max_scale": 1.5, "max_leverage": 2.0, "min_exposure": 0.0},
            },
        }
        mom = {"enabled": True, "trend_state": "up"}
        out = compute_vol_target_overlay(df, config, mom)
        self.assertTrue(out.get("enabled"))
        self.assertIn("realized_vol", out)
        self.assertIn("position_scale", out)
        self.assertGreaterEqual(out["position_scale"], 0.25)
        self.assertLessEqual(out["position_scale"], 1.5)

    def test_regime_risk_off_for_negative_shift(self):
        # Build a clear negative/high-vol shift in the tail.
        n = 500
        rng = np.random.default_rng(123)
        first = 0.0002 + 0.006 * rng.normal(size=n // 2)
        second = -0.003 + 0.03 * rng.normal(size=n // 2)
        rets = np.concatenate([first, second])
        price = 100 * np.cumprod(1 + rets)
        idx = pd.date_range("2021-01-01", periods=n, freq="D")
        df = pd.DataFrame({"HIGH": price, "y_diff": pd.Series(price).diff().fillna(0.0).values}, index=idx)

        cfg = {
            "target_col": "HIGH",
            "regime": {
                "enabled": True,
                "growth_window": 20,
                "vol_window": 30,
                "vol_quantile": 0.75,
                "policy_map": {
                    "risk_on": {"models": ["nbeats", "ulr", "svr"], "risk_scale": 1.0},
                    "risk_off": {"models": ["ulr", "svr"], "risk_scale": 0.5},
                },
            },
        }
        mom = {"momentum_score": -0.6}
        vol = {"position_scale": 0.7}

        out = classify_market_regime(df, cfg, mom, vol)
        self.assertTrue(out.get("enabled"))
        self.assertEqual(out.get("state"), "risk_off")
        self.assertGreaterEqual(out.get("confidence", 0), 0.0)
        self.assertLessEqual(out.get("confidence", 1), 1.0)


if __name__ == "__main__":
    unittest.main()
