import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.rupture_forecaster import forecast_market_rupture
from utils.momentum import compute_momentum_overlay
from utils.regime import classify_market_regime


class DisruptionDetectionTests(unittest.TestCase):
    def _shifted_market_df(self, n=700):
        rng = np.random.default_rng(7)
        # Stable -> disrupted regime.
        part1 = 0.0005 + 0.005 * rng.normal(size=n // 2)
        part2 = -0.002 + 0.03 * rng.normal(size=n // 2)

        # Inject explicit shocks in disrupted phase.
        shock_idx = [30, 85, 140, 180, 230]
        for i in shock_idx:
            if i < len(part2):
                part2[i] += (-0.08 if i % 2 == 0 else 0.06)

        rets = np.concatenate([part1, part2])
        price = 100 * np.cumprod(1 + rets)
        idx = pd.date_range("2022-01-01", periods=n, freq="D")

        return pd.DataFrame(
            {
                "HIGH": price,
                "Price_return": rets,
                "Open_return": rets * 0.9,
                "y_diff": pd.Series(price).diff().fillna(0.0).values,
            },
            index=idx,
        )

    def test_rupture_forecaster_produces_metrics(self):
        df = self._shifted_market_df()
        cfg = {
            "target_col": "HIGH",
            "input_features": ["HIGH", "y_diff", "Price_return", "Open_return"],
            "n_steps": 30,
            "rupture_forecast": {
                "enabled": True,
                "input_features": ["HIGH", "y_diff", "Price_return", "Open_return"],
                "n_steps": 30,
                "split_ratio": 0.8,
                "n_estimators": 120,
                "max_depth": 8,
                "quantile": 0.85,
            },
        }

        out = forecast_market_rupture(df, cfg)
        self.assertTrue(out.get("enabled"))
        self.assertIn("metrics", out)
        self.assertIn("binary", out["metrics"])

        binary = out["metrics"]["binary"]
        self.assertIn("accuracy", binary)
        self.assertIn("f1", binary)
        self.assertIn("threshold_abs_move", binary)
        self.assertGreater(binary.get("threshold_abs_move", 0), 0)
        self.assertGreaterEqual(binary.get("accuracy", 0), 0.0)
        self.assertLessEqual(binary.get("accuracy", 1), 1.0)
        self.assertGreaterEqual(binary.get("f1", 0), 0.0)
        self.assertLessEqual(binary.get("f1", 1), 1.0)
        self.assertGreaterEqual(out["next_step"].get("rupture_probability", 0), 0.0)
        self.assertLessEqual(out["next_step"].get("rupture_probability", 1), 1.0)

    def test_momentum_and_regime_detect_change(self):
        df = self._shifted_market_df()

        mcfg = {"target_col": "HIGH", "momentum": {"enabled": True, "windows": [20, 60, 120]}}
        mres = compute_momentum_overlay(df, mcfg)
        self.assertTrue(mres.get("enabled"))

        rcfg = {
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

        rres = classify_market_regime(df, rcfg, mres, {"position_scale": 0.8})
        self.assertTrue(rres.get("enabled"))
        self.assertEqual(rres.get("state"), "risk_off")


if __name__ == "__main__":
    unittest.main()
