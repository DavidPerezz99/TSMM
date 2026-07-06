import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.trading_job import _should_auto_request_followup_agent_a


class TradingJobFollowupTests(unittest.TestCase):
    def test_mode_b_consensus_close_remains_eligible(self):
        state = {
            "closed_reason": "mode_b_consensus_close(sell->buy)",
            "close_outcome": {"reason_label": "expert"},
        }

        self.assertTrue(_should_auto_request_followup_agent_a(state, {}))

    def test_position_not_found_is_not_eligible_by_default(self):
        state = {
            "closed_reason": "position_not_found_assumed_closed",
            "close_outcome": {"reason_label": "sl", "comment": "[sl 4515.47]"},
        }

        self.assertFalse(_should_auto_request_followup_agent_a(state, {}))

    def test_position_not_found_remains_ineligible_even_with_flag(self):
        state = {
            "closed_reason": "position_not_found_assumed_closed",
            "close_outcome": {"reason_label": "sl"},
        }
        trading_cfg = {"agent": {"followup_agent_a_allow_position_not_found_close": True}}

        self.assertFalse(_should_auto_request_followup_agent_a(state, trading_cfg))

    def test_manual_or_external_marker_blocks_followup(self):
        state = {
            "closed_reason": "hard_deadline_reached",
            "manual_or_external_close_detected": True,
            "close_outcome": {"reason_label": "expert"},
        }

        self.assertFalse(_should_auto_request_followup_agent_a(state, {}))

    def test_manual_outcome_blocks_followup_even_when_opted_in(self):
        state = {
            "closed_reason": "position_not_found_assumed_closed",
            "close_outcome": {"reason_label": "manual", "comment": "manually closed"},
        }
        trading_cfg = {"agent": {"followup_agent_a_allow_position_not_found_close": True}}

        self.assertFalse(_should_auto_request_followup_agent_a(state, trading_cfg))


if __name__ == "__main__":
    unittest.main()