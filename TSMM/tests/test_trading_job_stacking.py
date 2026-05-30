import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.trading_job import _intentional_same_side_stack_policy


class TradingJobStackingTests(unittest.TestCase):
    def test_explicit_stack_intent_allows_multiple_orders_with_cap(self):
        trading_cfg = {
            "trading_job": {
                "intentional_same_side_stacking": {
                    "enabled": True,
                    "max_same_side_orders": 3,
                    "auto_enable_on_high_confidence": True,
                    "auto_stack_confidence_threshold": 0.7,
                    "auto_stack_target_orders": 2,
                }
            }
        }
        plan = {
            "decision": "buy",
            "confidence": 0.61,
            "execution_intent": {"same_side_order_count": 5},
        }

        policy = _intentional_same_side_stack_policy(plan, trading_cfg)

        self.assertTrue(policy.get("enabled"))
        self.assertEqual(policy.get("target_orders"), 3)
        self.assertEqual(policy.get("allowed_existing_similar_orders"), 2)
        self.assertEqual(policy.get("reason"), "explicit_stack_intent")

    def test_auto_high_confidence_stack_applies_when_enabled(self):
        trading_cfg = {
            "trading_job": {
                "intentional_same_side_stacking": {
                    "enabled": True,
                    "max_same_side_orders": 3,
                    "auto_enable_on_high_confidence": True,
                    "auto_stack_confidence_threshold": 0.54,
                    "auto_stack_target_orders": 2,
                }
            }
        }
        plan = {
            "decision": "buy",
            "confidence": 0.56,
        }

        policy = _intentional_same_side_stack_policy(plan, trading_cfg)

        self.assertEqual(policy.get("target_orders"), 2)
        self.assertEqual(policy.get("allowed_existing_similar_orders"), 1)
        self.assertEqual(policy.get("reason"), "high_confidence_auto_stack")

    def test_default_duplicate_guard_behavior_when_stacking_disabled(self):
        trading_cfg = {
            "trading_job": {
                "intentional_same_side_stacking": {
                    "enabled": False,
                    "max_same_side_orders": 3,
                    "auto_enable_on_high_confidence": True,
                    "auto_stack_confidence_threshold": 0.3,
                    "auto_stack_target_orders": 3,
                }
            }
        }
        plan = {
            "decision": "buy",
            "confidence": 0.99,
            "same_side_order_count": 3,
        }

        policy = _intentional_same_side_stack_policy(plan, trading_cfg)

        self.assertFalse(policy.get("enabled"))
        self.assertEqual(policy.get("target_orders"), 1)
        self.assertEqual(policy.get("allowed_existing_similar_orders"), 0)
        self.assertEqual(policy.get("reason"), "duplicate_guard_default")


if __name__ == "__main__":
    unittest.main()
