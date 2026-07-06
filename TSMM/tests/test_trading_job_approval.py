import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.trading_job import _active_agent_b_position_count, _agent_a_approval_decision


class TradingJobApprovalTests(unittest.TestCase):
    def test_mandatory_session_programmed_bypasses_approval(self):
        cfg = {
            "approval_policy": {
                "auto_approve_mandatory_session_programmed": True,
                "auto_approve_below_agent_b_count": 3,
            },
            "risk": {"max_open_positions": 3},
        }

        with patch("utils.trading_job._active_agent_b_position_count", return_value=3):
            required, reason, count, threshold = _agent_a_approval_decision(
                "reports",
                cfg,
                auto_created=False,
                autonomous_trigger="mandatory_session",
                submission_mode="programmed",
            )

        self.assertFalse(required)
        self.assertEqual(reason, "mandatory_session_programmed")
        self.assertEqual(count, 3)
        self.assertEqual(threshold, 3)

    def test_below_agent_b_threshold_bypasses_approval(self):
        cfg = {
            "approval_policy": {"auto_approve_below_agent_b_count": 3},
            "risk": {"max_open_positions": 3},
            "agent": {"followup_agent_a_requires_approval": False},
        }

        with patch("utils.trading_job._active_agent_b_position_count", return_value=2):
            required, reason, count, threshold = _agent_a_approval_decision(
                "reports",
                cfg,
                auto_created=False,
                autonomous_trigger="",
                submission_mode="programmed",
            )

        self.assertFalse(required)
        self.assertEqual(reason, "below_agent_b_threshold")
        self.assertEqual(count, 2)
        self.assertEqual(threshold, 3)

    def test_threshold_reached_requires_manual_approval(self):
        cfg = {
            "approval_policy": {"auto_approve_below_agent_b_count": 3},
            "risk": {"max_open_positions": 3},
            "agent": {"followup_agent_a_requires_approval": False},
        }

        with patch("utils.trading_job._active_agent_b_position_count", return_value=3):
            required, reason, count, threshold = _agent_a_approval_decision(
                "reports",
                cfg,
                auto_created=False,
                autonomous_trigger="",
                submission_mode="programmed",
            )

        self.assertTrue(required)
        self.assertEqual(reason, "manual_approval_required")
        self.assertEqual(count, 3)
        self.assertEqual(threshold, 3)

    def test_opposing_countertrade_bypasses_approval_when_enabled(self):
        cfg = {
            "approval_policy": {
                "auto_approve_opposing_countertrade": True,
                "auto_approve_below_agent_b_count": 0,
            },
            "risk": {"max_open_positions": 3},
        }

        with patch("utils.trading_job._active_agent_b_position_count", return_value=3):
            required, reason, count, threshold = _agent_a_approval_decision(
                "reports",
                cfg,
                auto_created=False,
                autonomous_trigger="opposing_countertrade",
                submission_mode="market",
            )

        self.assertFalse(required)
        self.assertEqual(reason, "opposing_countertrade")
        self.assertEqual(count, 3)
        self.assertEqual(threshold, 3)

    def test_followup_requires_approval_even_when_below_threshold(self):
        cfg = {
            "approval_policy": {"auto_approve_below_agent_b_count": 3},
            "risk": {"max_open_positions": 3},
            "agent": {"followup_agent_a_requires_approval": True},
        }

        with patch("utils.trading_job._active_agent_b_position_count", return_value=0):
            required, reason, count, threshold = _agent_a_approval_decision(
                "reports",
                cfg,
                auto_created=True,
                autonomous_trigger="",
                submission_mode="programmed",
            )

        self.assertTrue(required)
        self.assertEqual(reason, "followup_manual_approval_required")
        self.assertEqual(count, 0)
        self.assertEqual(threshold, 3)

    def test_active_agent_b_position_count_uses_mt5_tsmm_positions(self):
        class Adapter:
            def __init__(self, _cfg):
                pass

            def connect(self):
                return True, "connected"

            def list_open_positions(self):
                return {
                    "ok": True,
                    "positions": [
                        {"ticket": 1, "symbol": "XAUUSD", "comment": "TSMM live", "magic": 7070001},
                        {"ticket": 2, "symbol": "XAUUSD", "comment": "manual", "magic": 0},
                        {"ticket": 3, "symbol": "EURUSD", "comment": "TSMM live", "magic": 7070001},
                    ],
                }

            def shutdown(self):
                return None

        cfg = {"broker": {"mt5": {}}, "execution": {"symbol": "XAUUSD"}}

        with patch("utils.trading_job.MT5Adapter", Adapter):
            count = _active_agent_b_position_count("reports", cfg)

        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()