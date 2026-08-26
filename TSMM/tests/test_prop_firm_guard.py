import os
import unittest
from unittest.mock import patch

from utils.trading_job import _prop_firm_guard_preflight


class _GuardAdapter:
    def __init__(self):
        self.positions = []
        self.orders = []
        self.account = {
            "ok": True,
            "login": 1395855,
            "server": "Upcomers-Server",
            "currency": "USD",
            "balance": 5000.0,
            "equity": 5000.0,
            "trade_allowed": True,
            "trade_expert": True,
        }
        self.estimated_loss = 12.0
        self.realized_pnl = 0.0

    def list_open_positions(self):
        return {"ok": True, "positions": list(self.positions)}

    def list_pending_orders(self):
        return {"ok": True, "orders": list(self.orders)}

    def get_account_snapshot(self):
        return dict(self.account)

    def get_symbol_trade_spec(self, symbol):
        return {"ok": True, "symbol": symbol, "volume_min": 0.01, "volume_max": 100.0, "volume_step": 0.01}

    def estimate_trade_loss(self, **_kwargs):
        return {
            "ok": True,
            "estimated_pnl": -float(self.estimated_loss),
            "estimated_loss": float(self.estimated_loss),
        }

    def get_utc_day_realized_pnl(self):
        return {"ok": True, "realized_pnl": float(self.realized_pnl), "closing_deals": 0}


class PropFirmGuardTests(unittest.TestCase):
    def setUp(self):
        self.adapter = _GuardAdapter()
        self.config = {
            "prop_firm_guard": {
                "enabled": True,
                "expected_login_env": "TEST_ORACLE_LOGIN",
                "expected_server": "Upcomers-Server",
                "expected_currency": "USD",
                "initial_balance_usd": 5000.0,
                "hard_dynamic_shield_pct": 5.0,
                "hard_daily_loss_usd": 200.0,
                "hard_single_trade_loss_usd": 100.0,
                "internal_trade_loss_cap_usd": 20.0,
                "internal_daily_loss_cap_usd": 50.0,
                "internal_equity_buffer_usd": 100.0,
                "require_hard_stop": True,
                "block_on_any_account_exposure": True,
            }
        }

    def _run(self, stop_loss=4988.0, volume=0.01):
        with patch.dict(os.environ, {"TEST_ORACLE_LOGIN": "1395855"}, clear=False):
            return _prop_firm_guard_preflight(
                adapter=self.adapter,
                trading_cfg=self.config,
                symbol="XAUUSD",
                side="buy",
                volume=volume,
                entry=5000.0,
                stop_loss=stop_loss,
            )

    def test_all_conservative_checks_pass(self):
        result = self._run()

        self.assertTrue(result["ok"])
        self.assertEqual(result["reason"], "all_prop_firm_checks_passed")

    def test_hard_stop_is_required(self):
        result = self._run(stop_loss=0.0)

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "hard_stop_required")

    def test_any_existing_exposure_blocks_a_new_order(self):
        self.adapter.positions = [{"ticket": 10, "magic": 0, "comment": "manual"}]

        result = self._run()

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "existing_account_exposure")

    def test_internal_trade_loss_cap_blocks_before_hard_rule(self):
        self.adapter.estimated_loss = 20.01

        result = self._run()

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "internal_trade_loss_cap")

    def test_accumulated_daily_loss_reserves_room_for_stop(self):
        self.adapter.realized_pnl = -39.0
        self.adapter.estimated_loss = 12.0

        result = self._run()

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "internal_daily_loss_cap")

    def test_unexpected_account_is_blocked(self):
        self.adapter.account["login"] = 999999

        result = self._run()

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "unexpected_broker_account")

    def test_unexpected_account_currency_is_blocked(self):
        self.adapter.account["currency"] = "EUR"

        result = self._run()

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "unexpected_account_currency")

    def test_small_account_skips_when_minimum_lot_exceeds_risk_budget(self):
        self.config = {
            "risk": {"risk_per_trade_pct": 0.5, "account_sizing": {"enabled": True}},
            "prop_firm_guard": {"enabled": False},
        }
        self.adapter.account["equity"] = 100.0
        self.adapter.estimated_loss = 100.0

        result = self._run(stop_loss=4999.0)

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "minimum_broker_volume_exceeds_risk_budget")
        self.assertLess(result["sizing"]["raw_volume"], 0.01)

    def test_account_sizing_rounds_down_never_up(self):
        self.config = {
            "risk": {"risk_per_trade_pct": 0.5, "account_sizing": {"enabled": True}},
            "prop_firm_guard": {"enabled": False},
        }
        self.adapter.account["equity"] = 100.0
        self.adapter.estimated_loss = 20.0

        result = self._run(stop_loss=4999.0, volume=1.0)

        self.assertTrue(result["ok"])
        self.assertEqual(result["sized_volume"], 0.02)


if __name__ == "__main__":
    unittest.main()
