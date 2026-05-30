import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.trading_job import (
    _apply_forced_agent_a_plan_override,
    _build_opposing_countertrade_plan,
    _enforce_opposing_countertrade_mirror_parity,
    _is_countertrade_target_reached,
)


class OpposingCountertradeTests(unittest.TestCase):
    def test_builds_buy_countertrade_from_high_confidence_sell(self):
        trading_cfg = {
            "execution": {"default_volume": 0.01},
            "risk": {"stop_loss_pct": 0.8},
            "opposing_countertrade": {
                "enabled": True,
                "min_source_confidence": 0.54,
                "stop_distance_multiplier": 1.0,
            },
        }
        source_plan = {
            "decision": "sell",
            "entry": 4499.38,
            "stop_loss": 4535.37504,
            "take_profit": 4427.38992,
            "volume": 0.01,
            "confidence": 0.5481,
            "model": "nbeats",
        }

        out = _build_opposing_countertrade_plan(source_plan, trading_cfg)

        self.assertTrue(out.get("ok"))
        counter = out.get("plan") or {}
        self.assertEqual(counter.get("decision"), "buy")
        self.assertAlmostEqual(float(counter.get("entry", 0.0)), 4427.38992, places=5)
        self.assertAlmostEqual(float(counter.get("take_profit", 0.0)), 4499.38, places=5)
        self.assertLess(float(counter.get("stop_loss", 0.0)), float(counter.get("entry", 0.0)))

    def test_rejects_countertrade_when_source_confidence_is_low(self):
        trading_cfg = {
            "execution": {"default_volume": 0.01},
            "risk": {"stop_loss_pct": 0.8},
            "opposing_countertrade": {
                "enabled": True,
                "min_source_confidence": 0.54,
            },
        }
        source_plan = {
            "decision": "sell",
            "entry": 4499.38,
            "stop_loss": 4535.37504,
            "take_profit": 4427.38992,
            "volume": 0.01,
            "confidence": 0.52,
            "model": "nbeats",
        }

        out = _build_opposing_countertrade_plan(source_plan, trading_cfg)

        self.assertFalse(out.get("ok"))
        self.assertIn("source_confidence_below_threshold", str(out.get("reason") or ""))

    def test_sell_target_reached_when_price_is_near_or_below_estimated_low(self):
        self.assertTrue(_is_countertrade_target_reached("sell", 4428.0, 4427.0, 2.0))
        self.assertTrue(_is_countertrade_target_reached("sell", 4425.0, 4427.0, 2.0))
        self.assertFalse(_is_countertrade_target_reached("sell", 4431.5, 4427.0, 2.0))

    def test_apply_forced_plan_override_replaces_trade_fields(self):
        base_plan = {
            "decision": "sell",
            "entry": 4499.38,
            "stop_loss": 4535.37,
            "take_profit": 4427.38,
            "volume": 0.01,
            "risk_notes": ["original"],
        }
        forced_plan = {
            "decision": "buy",
            "entry": 4427.38,
            "stop_loss": 4391.39,
            "take_profit": 4499.38,
            "volume": 0.01,
            "confidence": 0.91,
        }

        out = _apply_forced_agent_a_plan_override(base_plan, forced_plan, {"execution": {"default_volume": 0.01}})

        self.assertEqual(out.get("decision"), "buy")
        self.assertAlmostEqual(float(out.get("entry", 0.0)), 4427.38, places=5)
        self.assertAlmostEqual(float(out.get("take_profit", 0.0)), 4499.38, places=5)
        self.assertTrue(bool(out.get("forced_plan_override")))

    def test_parity_guard_skips_when_trigger_is_not_countertrade(self):
        state = {
            "request_context": {"autonomous_trigger": "followup"},
            "order": {"order_ticket": 111, "symbol": "XAUUSD", "price_open": 4400.0, "volume": 0.01, "side": "buy"},
            "plan": {"decision": "buy", "entry": 4400.0, "volume": 0.01},
        }

        out = _enforce_opposing_countertrade_mirror_parity(
            adapter=object(),
            app_config={"symbol": "XAUUSD"},
            trading_cfg={"execution": {"symbol": "XAUUSD"}},
            output_dir="reports",
            state=state,
        )

        self.assertFalse(bool(out.get("attempted", False)))
        self.assertEqual(str(out.get("reason") or ""), "not_opposing_countertrade_trigger")

    def test_parity_guard_passes_when_peer_exposure_exists(self):
        class LocalAdapter:
            def cancel_pending_order(self, _order_ticket):
                return {"ok": True}

            def find_position_by_order(self, _order_ticket):
                return {"ok": True, "position": None}

            def close_position_by_ticket(self, _ticket):
                return {"ok": True}

        class PeerAdapter:
            def __init__(self, _cfg):
                pass

            def connect(self):
                return True, "connected"

            def shutdown(self):
                return None

        state = {
            "request_context": {"autonomous_trigger": "opposing_countertrade"},
            "order": {"order_ticket": 222, "symbol": "XAUUSD", "price_open": 4400.0, "volume": 0.01, "side": "buy"},
            "plan": {"decision": "buy", "entry": 4400.0, "volume": 0.01},
            "mirror": {
                "peer_trading_config_path": "config/trading_agent_ftmo.yaml",
                "peer_job_id": "job_FTMO_1",
                "peer_profile": "FTMO",
            },
        }
        trading_cfg = {
            "execution": {"symbol": "XAUUSD", "default_volume": 0.01},
            "account_mirror": {"enabled": True, "peer_trading_config_path": "config/trading_agent_ftmo.yaml", "peer_profile": "FTMO"},
            "opposing_countertrade": {
                "enabled": True,
                "enforce_mirror_parity": True,
                "mirror_parity_wait_seconds": 1,
                "mirror_parity_poll_seconds": 1,
            },
        }

        with patch("utils.trading_job.load_trading_config", return_value={"broker": {"mt5": {}}, "runtime": {"profile_label": "FTMO"}}), \
             patch("utils.trading_job.MT5Adapter", PeerAdapter), \
             patch("utils.trading_job._find_similar_mt5_exposure", return_value={"ok": True, "pending_orders": [{"order_ticket": 999}], "open_positions": []}):
            out = _enforce_opposing_countertrade_mirror_parity(
                adapter=LocalAdapter(),
                app_config={"symbol": "XAUUSD"},
                trading_cfg=trading_cfg,
                output_dir="reports",
                state=state,
            )

        self.assertTrue(bool(out.get("attempted", False)))
        self.assertTrue(bool(out.get("ok", False)))
        self.assertFalse(bool(out.get("reverted", False)))
        self.assertEqual(str(out.get("reason") or ""), "peer_exposure_detected")

    def test_parity_guard_reverts_and_kills_peer_when_exposure_is_missing(self):
        class LocalAdapter:
            def cancel_pending_order(self, order_ticket):
                return {"ok": True, "order_ticket": int(order_ticket)}

            def find_position_by_order(self, _order_ticket):
                return {"ok": True, "position": None}

            def close_position_by_ticket(self, _ticket):
                return {"ok": True}

        class PeerAdapter:
            def __init__(self, _cfg):
                pass

            def connect(self):
                return True, "connected"

            def shutdown(self):
                return None

        state = {
            "request_context": {"autonomous_trigger": "opposing_countertrade"},
            "order": {"order_ticket": 333, "symbol": "XAUUSD", "price_open": 4400.0, "volume": 0.01, "side": "buy"},
            "plan": {"decision": "buy", "entry": 4400.0, "volume": 0.01},
            "mirror": {
                "peer_trading_config_path": "config/trading_agent_ftmo.yaml",
                "peer_job_id": "job_FTMO_2",
                "peer_profile": "FTMO",
            },
        }
        trading_cfg = {
            "execution": {"symbol": "XAUUSD", "default_volume": 0.01},
            "account_mirror": {"enabled": True, "peer_trading_config_path": "config/trading_agent_ftmo.yaml", "peer_profile": "FTMO"},
            "opposing_countertrade": {
                "enabled": True,
                "enforce_mirror_parity": True,
                "mirror_parity_wait_seconds": 1,
                "mirror_parity_poll_seconds": 1,
            },
        }

        with patch("utils.trading_job.load_trading_config", return_value={"broker": {"mt5": {}}, "runtime": {"profile_label": "FTMO"}}), \
             patch("utils.trading_job.MT5Adapter", PeerAdapter), \
             patch("utils.trading_job._find_similar_mt5_exposure", return_value={"ok": True, "pending_orders": [], "open_positions": []}), \
             patch("utils.trading_job._propagate_mirror_job_action", return_value={"ok": True, "action": "kill"}):
            out = _enforce_opposing_countertrade_mirror_parity(
                adapter=LocalAdapter(),
                app_config={"symbol": "XAUUSD"},
                trading_cfg=trading_cfg,
                output_dir="reports",
                state=state,
            )

        self.assertTrue(bool(out.get("attempted", False)))
        self.assertTrue(bool(out.get("reverted", False)))
        self.assertEqual(str(out.get("reason") or ""), "peer_exposure_not_detected_before_timeout")
        self.assertTrue(bool((out.get("local_revert") or {}).get("ok", False)))
        self.assertTrue(bool((out.get("peer_kill") or {}).get("ok", False)))


if __name__ == "__main__":
    unittest.main()
