import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "trading_agent_oracle.yaml"


class OracleProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = PROFILE.read_text(encoding="utf-8")
        cls.cfg = yaml.safe_load(cls.raw)

    def test_credentials_are_environment_references(self):
        mt5 = self.cfg["broker"]["mt5"]

        self.assertEqual(mt5["login"], "env:MT5_UPCOMERS_ORACLE_LOGIN")
        self.assertEqual(mt5["password"], "env:MT5_UPCOMERS_ORACLE_PASSWORD")
        self.assertNotIn("1395855", self.raw)

    def test_oracle_profile_is_isolated(self):
        self.assertEqual(self.cfg["runtime"]["root_dir"], "reports/runtime/oracle")
        self.assertEqual(self.cfg["runtime"]["job_id_prefix"], "ORACLE")
        self.assertFalse(self.cfg["account_mirror"]["enabled"])
        self.assertEqual(self.cfg["telegram_listener"]["command_prefix"], "/oracle")

    def test_conservative_execution_policy(self):
        self.assertEqual(self.cfg["execution"]["default_volume"], 0.01)
        self.assertFalse(self.cfg["execution"]["delayed_stop_loss"]["enabled"])
        self.assertEqual(self.cfg["risk"]["max_open_positions"], 1)
        self.assertTrue(self.cfg["trading_job"]["prevent_new_programmed_when_open_position"])
        self.assertTrue(self.cfg["trading_job"]["prevent_new_programmed_when_pending_exists"])
        self.assertFalse(self.cfg["autonomous_trading"]["enabled"])
        self.assertFalse(self.cfg["autonomous_trading"]["followup_enabled"])
        self.assertFalse(self.cfg["opposing_countertrade"]["enabled"])

    def test_internal_limits_leave_large_breach_buffers(self):
        guard = self.cfg["prop_firm_guard"]

        self.assertLessEqual(guard["internal_trade_loss_cap_usd"], 20.0)
        self.assertLess(guard["internal_trade_loss_cap_usd"], guard["hard_single_trade_loss_usd"])
        self.assertLess(guard["internal_daily_loss_cap_usd"], guard["hard_daily_loss_usd"])
        self.assertTrue(guard["require_hard_stop"])
        self.assertTrue(guard["block_on_any_account_exposure"])

    def test_no_automatic_approval_paths_are_enabled(self):
        approval = self.cfg["approval_policy"]

        self.assertFalse(approval["auto_approve_mandatory_session_programmed"])
        self.assertFalse(approval["auto_approve_opposing_countertrade"])
        self.assertEqual(approval["auto_approve_below_agent_b_count"], 0)
        self.assertTrue(self.cfg["agent"]["followup_agent_a_requires_approval"])


if __name__ == "__main__":
    unittest.main()
