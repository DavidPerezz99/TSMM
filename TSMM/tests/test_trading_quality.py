import unittest

from utils.trading_quality import apply_hybrid_trade_gate, model_quality_weight


class TradingQualityTests(unittest.TestCase):
    def test_negative_refreshed_r2_removes_vote_even_when_selected_score_is_high(self):
        result = model_quality_weight(0.74, -0.90, minimum_r2=0.0)
        self.assertFalse(result["qualified"])
        self.assertEqual(result["source"], "refreshed_r2")
        self.assertEqual(result["weight"], 0.0)

    def test_legacy_static_r2_is_discounted_until_revalidated(self):
        legacy = model_quality_weight(0.80, None, minimum_r2=0.0, legacy_static_discount=0.25)
        refreshed = model_quality_weight(0.80, 0.80, minimum_r2=0.0, legacy_static_discount=0.25)
        self.assertTrue(legacy["qualified"])
        self.assertAlmostEqual(legacy["weight"], 0.20)
        self.assertEqual(legacy["reliability_factor"], 0.25)
        self.assertAlmostEqual(refreshed["weight"], 0.80)
        self.assertEqual(refreshed["reliability_factor"], 1.0)

    def test_hybrid_gate_abstains_on_negative_after_cost_expected_value(self):
        plan = {
            "decision": "buy", "entry": 100.0, "stop_loss": 99.0,
            "take_profit": 100.5, "confidence": 0.5,
            "enrichment": {"n_qualified_signals": 3, "consensus_score": 0.5},
            "risk_notes": [],
        }
        config = {
            "hybrid_strategy": {"enabled": True, "min_qualified_models": 2, "min_abs_consensus": 0.1},
            "execution": {"spread_bps": 5.0, "slippage_bps": 5.0},
        }
        result = apply_hybrid_trade_gate(plan, config)
        self.assertEqual(result["decision"], "hold")
        self.assertFalse(result["hybrid_gate"]["passed"])

    def test_hybrid_gate_allows_positive_expected_value_with_quality_breadth(self):
        plan = {
            "decision": "buy", "entry": 100.0, "stop_loss": 99.5,
            "take_profit": 102.0, "confidence": 0.7,
            "enrichment": {"n_qualified_signals": 3, "consensus_score": 0.5},
            "risk_notes": [],
        }
        result = apply_hybrid_trade_gate(plan, {"hybrid_strategy": {"enabled": True}, "execution": {}})
        self.assertEqual(result["decision"], "buy")
        self.assertTrue(result["hybrid_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
