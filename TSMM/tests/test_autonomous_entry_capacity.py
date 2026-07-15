import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.telegram_command_listener import _autonomous_capacity_limit, _mode_b_can_trigger_new_entries


class AutonomousEntryCapacityTests(unittest.TestCase):
    def test_explicit_autonomous_capacity_has_priority(self):
        cfg = {
            "autonomous_trading": {"max_jobs_per_session": 3},
            "mode_a": {"max_operations_per_session": 2},
            "risk": {"max_open_positions": 5},
        }

        self.assertEqual(_autonomous_capacity_limit(cfg), 3)

    def test_mode_a_limit_is_capacity_fallback(self):
        cfg = {
            "mode_a": {"max_operations_per_session": 2},
            "risk": {"max_open_positions": 5},
        }

        self.assertEqual(_autonomous_capacity_limit(cfg), 2)

    def test_mode_b_must_explicitly_allow_new_entries(self):
        base = {"mode_b": {"enabled": True, "manage_existing_positions": True}}
        self.assertFalse(_mode_b_can_trigger_new_entries(base))

        base["mode_b"]["allow_open_new_positions"] = True
        self.assertTrue(_mode_b_can_trigger_new_entries(base))

    def test_disabled_mode_b_cannot_trigger_new_entries(self):
        cfg = {
            "mode_b": {
                "enabled": False,
                "manage_existing_positions": True,
                "allow_open_new_positions": True,
            }
        }

        self.assertFalse(_mode_b_can_trigger_new_entries(cfg))


if __name__ == "__main__":
    unittest.main()
