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


if __name__ == "__main__":
    unittest.main()