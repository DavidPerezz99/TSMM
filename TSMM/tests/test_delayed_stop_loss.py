import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.trading_job import _agent_b_risk_adjustment, _order_risk_levels_for_plan, _order_volume_for_plan


class DelayedStopLossTests(unittest.TestCase):
    def test_no_sl_plan_falls_back_to_standard_stop_when_feature_disabled(self):
        plan = {
            "decision": "buy",
            "entry": 100.0,
            "stop_loss": None,
            "take_profit": 103.0,
            "conviction": {"risk_mode": "no_sl", "conviction": 0.9},
        }
        trading_cfg = {
            "risk": {"stop_loss_pct": 1.0},
            "execution": {"delayed_stop_loss": {"enabled": False}},
        }

        stop_loss, take_profit, meta = _order_risk_levels_for_plan(plan, trading_cfg)

        self.assertEqual(stop_loss, 99.0)
        self.assertEqual(take_profit, 103.0)
        self.assertEqual(meta["mode"], "standard")
        self.assertEqual(meta["reason"], "fallback_stop_loss_applied")

    def test_no_sl_plan_sends_zero_initial_stop_when_delayed_protection_enabled(self):
        plan = {
            "decision": "sell",
            "entry": 100.0,
            "stop_loss": None,
            "take_profit": 96.0,
            "conviction": {"risk_mode": "no_sl", "conviction": 0.9},
        }
        trading_cfg = {
            "risk": {"stop_loss_pct": 1.0},
            "execution": {
                "delayed_stop_loss": {
                    "enabled": True,
                    "allowed_conviction_modes": ["no_sl"],
                }
            },
        }

        stop_loss, take_profit, meta = _order_risk_levels_for_plan(plan, trading_cfg)

        self.assertEqual(stop_loss, 0.0)
        self.assertEqual(take_profit, 96.0)
        self.assertEqual(meta["mode"], "delayed_protection")
        self.assertEqual(meta["planned_stop_loss"], 101.0)

    def test_delayed_stop_requires_minimum_conviction_and_caps_volume(self):
        trading_cfg = {
            "execution": {
                "default_volume": 0.02,
                "delayed_stop_loss": {
                    "enabled": True,
                    "allowed_conviction_modes": ["no_sl"],
                    "min_conviction": 0.8,
                    "max_volume_multiplier": 1.0,
                },
            },
            "risk": {"stop_loss_pct": 1.0},
        }
        low_plan = {
            "decision": "buy",
            "entry": 100.0,
            "stop_loss": None,
            "volume": 0.03,
            "conviction": {"risk_mode": "no_sl", "conviction": 0.79},
        }
        high_plan = {**low_plan, "conviction": {"risk_mode": "no_sl", "conviction": 0.9}}

        low_stop, _, low_meta = _order_risk_levels_for_plan(low_plan, trading_cfg)
        high_stop, _, high_meta = _order_risk_levels_for_plan(high_plan, trading_cfg)

        self.assertEqual(low_stop, 99.0)
        self.assertEqual(low_meta["mode"], "standard")
        self.assertEqual(high_stop, 0.0)
        self.assertEqual(high_meta["mode"], "delayed_protection")
        self.assertEqual(_order_volume_for_plan(high_plan, trading_cfg), 0.02)

    @patch("utils.trading_job._now_utc")
    def test_agent_b_attaches_delayed_stop_after_adverse_move(self, now_mock):
        from datetime import datetime

        now_mock.return_value = datetime(2026, 7, 10, 12, 10, 0)
        state = {
            "started_at": "2026-07-10 12:00:00",
            "plan": {
                "decision": "buy",
                "entry": 100.0,
                "stop_loss": None,
                "take_profit": 104.0,
                "conviction": {"risk_mode": "no_sl", "conviction": 0.9},
                "stop_loss_protection": {
                    "mode": "delayed_protection",
                    "planned_stop_loss": 99.0,
                },
            },
        }
        position = {
            "price_open": 100.0,
            "price_current": 99.7,
            "sl": 0.0,
            "tp": 104.0,
        }
        current_plan = {
            "position_side": "buy",
            "recommendation": "monitor_closely",
            "consensus_score": 0.1,
            "close_threshold": 0.25,
        }
        trading_cfg = {
            "execution": {
                "delayed_stop_loss": {
                    "enabled": True,
                    "max_unprotected_seconds": 900,
                    "max_unprotected_adverse_pct": 0.25,
                }
            },
            "risk": {
                "stop_loss_pct": 1.0,
                "take_profit_pct": 2.0,
                "trailing": {"enabled": True, "trail_pct_base": 0.5},
            },
            "mode_b": {"manage_existing_positions": True},
        }

        out = _agent_b_risk_adjustment(state, position, current_plan, trading_cfg)

        self.assertIsNotNone(out)
        self.assertEqual(out["action"], "attach_delayed_stop_loss")
        self.assertEqual(out["stop_loss"], 99.0)
        self.assertEqual(out["take_profit"], 104.0)


if __name__ == "__main__":
    unittest.main()
