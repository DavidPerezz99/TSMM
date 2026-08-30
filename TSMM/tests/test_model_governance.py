import tempfile
import unittest
from pathlib import Path

from utils.model_governance import assess_challenger, load_registry, promote, rollback


GOOD = {
    "holdout_r2": 0.62,
    "walk_forward_r2": [0.31, 0.42, 0.28],
    "directional_accuracy": 0.58,
    "profit_factor": 1.3,
    "expectancy": 0.2,
    "max_drawdown_pct": 8.0,
    "trades": 50,
}


class ModelGovernanceTests(unittest.TestCase):
    def test_missing_walk_forward_and_trading_metrics_fail_closed(self):
        result = assess_challenger({"holdout_r2": 0.8})
        self.assertFalse(result["approved"])
        self.assertIn("missing_walk_forward_r2", result["failures"])

    def test_challenger_must_beat_incumbent(self):
        result = assess_challenger(GOOD, {"holdout_r2": 0.615})
        self.assertFalse(result["approved"])
        self.assertIn("does_not_beat_champion_r2", result["failures"])

    def test_atomic_promotion_and_rollback_preserve_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "registry.json"
            first = dict(GOOD)
            assessment = assess_challenger(first)
            promote(path, "10m_high", "bundle-a", first, assessment)
            second = {**GOOD, "holdout_r2": 0.75}
            promote(path, "10m_high", "bundle-b", second, assess_challenger(second, first))
            restored = rollback(path, "10m_high")
            self.assertTrue(restored["bundle"].endswith("bundle-a"))
            self.assertTrue(load_registry(path)["endpoints"]["10m_high"]["champion"]["bundle"].endswith("bundle-a"))

    def test_promotion_records_serving_deployment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "registry.json"
            deployment = {"endpoint": "10m_high", "deployment_id": "qualified-model-v1"}
            champion = promote(
                path,
                "10m_high",
                "bundle-a",
                dict(GOOD),
                assess_challenger(GOOD),
                deployment=deployment,
            )
            self.assertEqual(champion["deployment"]["deployment_id"], "qualified-model-v1")


if __name__ == "__main__":
    unittest.main()
