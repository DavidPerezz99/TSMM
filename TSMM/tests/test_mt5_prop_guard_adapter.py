import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from utils.investing_agent import MT5Adapter


class _FakeMT5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    DEAL_ENTRY_OUT = 1
    DEAL_ENTRY_OUT_BY = 2
    DEAL_ENTRY_INOUT = 3

    def account_info(self):
        return SimpleNamespace(
            login=1395855,
            server="Upcomers-Server",
            company="Upcomers Ltd",
            currency="USD",
            balance=5000.0,
            equity=4998.0,
            profit=-2.0,
            margin=10.0,
            margin_free=4988.0,
            trade_allowed=True,
            trade_expert=True,
        )

    def symbol_info(self, symbol):
        return SimpleNamespace(volume_min=0.01, volume_max=100.0, volume_step=0.01)

    def order_calc_profit(self, order_type, symbol, volume, entry, stop_loss):
        direction = 1.0 if order_type == self.ORDER_TYPE_BUY else -1.0
        return direction * (stop_loss - entry) * 100.0 * volume

    def last_error(self):
        return (1, "success")

    def history_deals_get(self, start, end):
        self.history_range = (start, end)
        return [
            SimpleNamespace(entry=self.DEAL_ENTRY_OUT, profit=-10.0, commission=-0.5, swap=-0.25, fee=0.0),
            SimpleNamespace(entry=0, profit=0.0, commission=-0.5, swap=0.0, fee=0.0),
            SimpleNamespace(entry=self.DEAL_ENTRY_OUT_BY, profit=5.0, commission=-0.5, swap=0.0, fee=-0.1),
        ]


class MT5PropGuardAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = MT5Adapter({})
        self.adapter._mt5 = _FakeMT5()

    def test_account_snapshot_exposes_risk_fields(self):
        result = self.adapter.get_account_snapshot()

        self.assertTrue(result["ok"])
        self.assertEqual(result["server"], "Upcomers-Server")
        self.assertEqual(result["balance"], 5000.0)
        self.assertEqual(result["equity"], 4998.0)
        self.assertTrue(result["trade_expert"])

    def test_estimate_trade_loss_uses_broker_calculation(self):
        result = self.adapter.estimate_trade_loss("XAUUSD", "buy", 0.01, 5000.0, 4988.0)

        self.assertTrue(result["ok"])
        self.assertEqual(result["estimated_pnl"], -12.0)
        self.assertEqual(result["estimated_loss"], 12.0)

    def test_symbol_trade_spec_exposes_volume_constraints(self):
        result = self.adapter.get_symbol_trade_spec("XAUUSD")

        self.assertTrue(result["ok"])
        self.assertEqual(result["volume_min"], 0.01)
        self.assertEqual(result["volume_step"], 0.01)

    def test_utc_day_pnl_counts_closing_deals_and_costs(self):
        now = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)

        result = self.adapter.get_utc_day_realized_pnl(now)

        self.assertTrue(result["ok"])
        self.assertEqual(result["closing_deals"], 2)
        self.assertAlmostEqual(result["realized_pnl"], -6.35)
        self.assertEqual(self.adapter._mt5.history_range[0].hour, 0)


if __name__ == "__main__":
    unittest.main()
