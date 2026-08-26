import unittest
from unittest.mock import patch

from utils.trading_job import _execute_approved_order


class ShadowModeTests(unittest.TestCase):
    @patch("utils.trading_job._save_job_state")
    @patch("utils.trading_job.MT5Adapter.connect")
    def test_shadow_mode_records_plan_without_connecting_to_mt5(self, connect, save_state):
        state = {
            "job_id": "shadow-test",
            "plan": {"decision": "buy", "entry": 100.0, "stop_loss": 99.0, "take_profit": 102.0},
            "order_submission_mode": "market",
        }
        result = _execute_approved_order(
            app_config={}, trading_cfg={"shadow_mode": {"enabled": True}},
            output_dir="reports/shadow", logger=None, state=state, state_file="shadow.json",
        )
        connect.assert_not_called()
        save_state.assert_called_once()
        self.assertEqual(result["closed_reason"], "shadow_mode_no_broker_submission")
        self.assertTrue(result["shadow_evaluation"]["enabled"])


if __name__ == "__main__":
    unittest.main()
