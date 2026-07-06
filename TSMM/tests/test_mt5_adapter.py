import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.investing_agent import MT5Adapter


class _FakeMT5:
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    TRADE_ACTION_REMOVE = 8
    TRADE_RETCODE_DONE = 10009

    def __init__(self):
        self.requests = []
        self.retcode_sequence = []

    def orders_get(self, ticket=None):
        if ticket == 123:
            return [SimpleNamespace(ticket=123)]
        return []

    def order_send(self, request):
        self.requests.append(request)
        if self.retcode_sequence:
            return SimpleNamespace(retcode=self.retcode_sequence.pop(0))
        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE)

    def symbol_info(self, symbol):
        return SimpleNamespace(filling_mode=self.ORDER_FILLING_RETURN)

    def last_error(self):
        return (1, "Success")


class _FakeMT5MarketStops:
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_RETCODE_DONE = 10009

    def __init__(self):
        self.requests = []
        self.retcode_sequence = [10016, self.TRADE_RETCODE_DONE]
        self._last_success_order = 0

    def symbol_select(self, symbol, _enable):
        return True

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(ask=4400.0, bid=4399.8)

    def symbol_info(self, symbol):
        return SimpleNamespace(
            filling_mode=self.ORDER_FILLING_RETURN,
            digits=2,
            point=0.01,
            trade_stops_level=50,
            trade_freeze_level=0,
        )

    def order_send(self, request):
        self.requests.append(dict(request))
        retcode = self.retcode_sequence.pop(0) if self.retcode_sequence else self.TRADE_RETCODE_DONE
        if retcode == self.TRADE_RETCODE_DONE:
            self._last_success_order = 9001
            return SimpleNamespace(retcode=retcode, order=9001, deal=8001)
        return SimpleNamespace(retcode=retcode, order=0, deal=0)

    def positions_get(self, ticket=None, symbol=None):
        if ticket == self._last_success_order:
            req = self.requests[-1]
            return [
                SimpleNamespace(
                    ticket=ticket,
                    symbol="XAUUSD",
                    volume=req.get("volume", 0.01),
                    price_open=req.get("price", 4400.0),
                    price_current=req.get("price", 4400.0),
                    sl=req.get("sl", 0.0),
                    tp=req.get("tp", 0.0),
                    profit=0.0,
                    type=self.POSITION_TYPE_BUY,
                    time=0,
                    comment="TSMM market order",
                    magic=7070001,
                )
            ]
        return []

    def last_error(self):
        return (1, "Success")


class _FakeMT5MarketStopsFallback:
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_RETCODE_DONE = 10009

    def __init__(self):
        self.requests = []
        self._last_success_order = 0
        self._positions = {}

    def symbol_select(self, symbol, _enable):
        return True

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(ask=4400.0, bid=4399.8)

    def symbol_info(self, symbol):
        return SimpleNamespace(
            filling_mode=self.ORDER_FILLING_RETURN,
            digits=2,
            point=0.01,
            trade_stops_level=0,
            trade_freeze_level=0,
        )

    def order_send(self, request):
        self.requests.append(dict(request))
        action = int(request.get("action", -1) or -1)
        if action == self.TRADE_ACTION_DEAL:
            sl = float(request.get("sl", 0.0) or 0.0)
            tp = float(request.get("tp", 0.0) or 0.0)
            if sl > 0.0 or tp > 0.0:
                return SimpleNamespace(retcode=10016, order=0, deal=0)
            self._last_success_order = 9101
            self._positions[self._last_success_order] = {
                "ticket": self._last_success_order,
                "sl": 0.0,
                "tp": 0.0,
                "price": float(request.get("price", 4400.0) or 4400.0),
                "volume": float(request.get("volume", 0.01) or 0.01),
            }
            return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=self._last_success_order, deal=8101)

        if action == self.TRADE_ACTION_SLTP:
            ticket = int(request.get("position", 0) or 0)
            if ticket in self._positions:
                self._positions[ticket]["sl"] = float(request.get("sl", 0.0) or 0.0)
                self._positions[ticket]["tp"] = float(request.get("tp", 0.0) or 0.0)
            return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=ticket, deal=0)

        return SimpleNamespace(retcode=self.TRADE_RETCODE_DONE, order=0, deal=0)

    def positions_get(self, ticket=None, symbol=None):
        if ticket is None:
            return []
        data = self._positions.get(int(ticket))
        if not data:
            return []
        return [
            SimpleNamespace(
                ticket=int(data["ticket"]),
                symbol="XAUUSD",
                volume=float(data["volume"]),
                price_open=float(data["price"]),
                price_current=float(data["price"]),
                sl=float(data["sl"]),
                tp=float(data["tp"]),
                profit=0.0,
                type=self.POSITION_TYPE_BUY,
                time=0,
                comment="TSMM market order",
                magic=7070001,
            )
        ]

    def last_error(self):
        return (1, "Success")


class MT5AdapterTests(unittest.TestCase):
    def test_cancel_pending_order_uses_minimal_remove_request(self):
        adapter = MT5Adapter({})
        adapter._mt5 = _FakeMT5()

        out = adapter.cancel_pending_order(123)

        self.assertTrue(out["ok"])
        self.assertEqual(adapter._mt5.requests, [{"action": 8, "order": 123}])

    def test_send_order_with_filling_fallback_retries_after_invalid_request(self):
        adapter = MT5Adapter({})
        fake_mt5 = _FakeMT5()
        fake_mt5.retcode_sequence = [10013, fake_mt5.TRADE_RETCODE_DONE]
        adapter._mt5 = fake_mt5

        out = adapter._send_order_with_filling_fallback("XAUUSD", {"action": 1, "symbol": "XAUUSD"})

        self.assertTrue(out["ok"])
        self.assertEqual(out["type_filling"], fake_mt5.ORDER_FILLING_IOC)
        self.assertEqual(out["attempted_filling_modes"], [fake_mt5.ORDER_FILLING_RETURN, fake_mt5.ORDER_FILLING_IOC])
        self.assertEqual(fake_mt5.requests[0]["type_filling"], fake_mt5.ORDER_FILLING_RETURN)
        self.assertEqual(fake_mt5.requests[1]["type_filling"], fake_mt5.ORDER_FILLING_IOC)

    def test_place_market_order_retries_after_invalid_stops(self):
        adapter = MT5Adapter({})
        fake_mt5 = _FakeMT5MarketStops()
        adapter._mt5 = fake_mt5

        out = adapter.place_market_order(
            symbol="XAUUSD",
            side="buy",
            volume=0.01,
            stop_loss=4401.0,
            take_profit=4399.0,
        )

        self.assertTrue(out["ok"])
        self.assertEqual(len(fake_mt5.requests), 2)
        first_request = fake_mt5.requests[0]
        second_request = fake_mt5.requests[1]

        self.assertLess(first_request["sl"], first_request["price"])
        self.assertGreater(first_request["tp"], first_request["price"])
        self.assertLess(second_request["sl"], second_request["price"])
        self.assertGreater(second_request["tp"], second_request["price"])

        min_distance = 50 * 0.01
        self.assertGreaterEqual(first_request["price"] - first_request["sl"], min_distance)
        self.assertGreaterEqual(first_request["tp"] - first_request["price"], min_distance)
        self.assertGreaterEqual(second_request["price"] - second_request["sl"], min_distance)
        self.assertGreaterEqual(second_request["tp"] - second_request["price"], min_distance)

    def test_place_market_order_falls_back_to_post_open_risk_update(self):
        adapter = MT5Adapter({})
        fake_mt5 = _FakeMT5MarketStopsFallback()
        adapter._mt5 = fake_mt5

        out = adapter.place_market_order(
            symbol="XAUUSD",
            side="buy",
            volume=0.01,
            stop_loss=4380.0,
            take_profit=4460.0,
        )

        self.assertTrue(out["ok"])
        deal_requests = [r for r in fake_mt5.requests if int(r.get("action", -1) or -1) == fake_mt5.TRADE_ACTION_DEAL]
        self.assertEqual(len(deal_requests), 3)
        self.assertGreater(deal_requests[0]["sl"], 0.0)
        self.assertGreater(deal_requests[0]["tp"], 0.0)
        self.assertGreater(deal_requests[1]["sl"], 0.0)
        self.assertGreater(deal_requests[1]["tp"], 0.0)
        self.assertEqual(float(deal_requests[2]["sl"]), 0.0)
        self.assertEqual(float(deal_requests[2]["tp"]), 0.0)

        sltp_requests = [r for r in fake_mt5.requests if int(r.get("action", -1) or -1) == fake_mt5.TRADE_ACTION_SLTP]
        self.assertGreaterEqual(len(sltp_requests), 1)
        self.assertGreater(float(sltp_requests[-1].get("sl", 0.0) or 0.0), 0.0)
        self.assertGreater(float(sltp_requests[-1].get("tp", 0.0) or 0.0), 0.0)
        self.assertTrue(bool((out.get("post_open_risk_update") or {}).get("ok", False)))


if __name__ == "__main__":
    unittest.main()