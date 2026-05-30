import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.investing_agent import _apply_agent_a_enrichment_to_plan, _build_mode_a_plan, _collect_all_model_assessment_signals, _extract_endpoint_signal


class AgentAEnrichmentTests(unittest.TestCase):
    def test_build_mode_a_plan_can_trade_contrarian(self):
        df = pd.DataFrame({"CLOSE": [100.0, 101.0, 102.0]})
        app_cfg = {"target_col": "CLOSE", "target_features": ["y_diff"]}
        trading_cfg = {
            "agent": {"signal_interpretation": "contrarian"},
            "risk": {"stop_loss_pct": 1.0, "take_profit_pct": 2.0, "min_confidence_to_trade": 0.5, "min_cm_accuracy_to_trade": 0.5, "max_input_fooling_risk": 0.45},
            "mode_a": {"allow_long": True, "allow_short": True},
            "execution": {"default_volume": 0.01},
        }
        evaluation = {"model_a": {"metrics": {"MAE": 0.1}, "confusion_matrix": {"accuracy": 0.8}, "confidence_levels": [0.7, 0.7, 0.7], "input_fooling_risk": {"probability_wrong_sign": 0.1}}}
        future_forecasts = {
            "model_a": {
                "future": [[5.0]],
                "future_by_feature": {"y_diff": [5.0]},
            }
        }

        plan = _build_mode_a_plan(df, app_cfg, trading_cfg, evaluation, future_forecasts)

        self.assertEqual(plan["decision"], "sell")
        self.assertGreater(plan["stop_loss"], plan["entry"])
        self.assertLess(plan["take_profit"], plan["entry"])
        self.assertEqual(plan["signal_interpretation"], "contrarian")

    def test_extract_endpoint_signal_can_trade_contrarian(self):
        out = _extract_endpoint_signal({"forecast_sign": 1.0, "confidence": 0.8, "_trading_cfg": {"agent": {"signal_interpretation": "contrarian"}}})

        self.assertEqual(out["signal"], -1)
        self.assertEqual(out["confidence"], 0.8)

    def test_opposed_consensus_flip_recomputes_buy_risk_levels(self):
        plan = {
            "decision": "sell",
            "entry": 4718.39,
            "stop_loss": 4756.13712,
            "take_profit": 4642.89576,
            "confidence": 0.514,
            "signal_score": -1.0,
            "risk_notes": [],
            "rationale": "base",
        }
        enrichment = {
            "enabled": True,
            "consensus": "buy",
            "consensus_score": 0.372,
            "signals": {
                "x": {"confidence": 0.95}
            },
            "n_signals": 20,
        }

        out = _apply_agent_a_enrichment_to_plan(plan, enrichment)

        self.assertEqual(out["decision"], "buy")
        self.assertLess(out["stop_loss"], out["entry"])
        self.assertGreater(out["take_profit"], out["entry"])
        self.assertIn(
            "Primary signal side overridden by stronger pretrained multi-timeframe consensus.",
            out["risk_notes"],
        )

    @patch("utils.investing_agent._collect_agent_a_enrichment_signals")
    def test_all_model_assessment_prefers_enrichment_signals(self, enrichment_mock):
        enrichment_mock.return_value = {
            "enabled": True,
            "consensus": "buy",
            "consensus_score": 0.33,
            "signals": {"high:10m": {"signal": 1}},
            "n_signals": 20,
            "avg_confidence": 0.77,
        }

        out = _collect_all_model_assessment_signals({"model_endpoints": {"10m": "http://127.0.0.1:8000/predict/10m"}})

        self.assertEqual(out["assessment_scope"], "all_models")
        self.assertEqual(out["source"], "agent_a_enrichment")
        self.assertEqual(out["n_signals"], 20)
        self.assertEqual(out["consensus"], "buy")


if __name__ == "__main__":
    unittest.main()