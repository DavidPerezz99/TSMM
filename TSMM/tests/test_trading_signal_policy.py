from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from utils.trading_job import _try_programmed_market_fallback
from utils.strategy_backtest import WalkForwardModelSignalProvider
from utils.trading_signal_policy import (
    all_training_cutoffs_precede,
    apply_volatility_protection,
    evaluate_joint_ohlc_policy,
)


def _policy_config(required_confirmations: int = 2):
    return {
        "signal_policy": {
            "enabled": True,
            "direction_timeframe": "7h",
            "select_best_direction_timeframe": False,
            "confirmation_timeframes": ["10m", "30m", "1h"],
            "required_confirmations": required_confirmations,
            "min_direction_score": 0.10,
            "min_confirmation_score": 0.05,
            "minimum_coverage": 0.70,
            "family_weights": {"high": 0.30, "low": 0.30, "close": 0.25, "open": 0.15},
            "entry_range_fraction": 0.20,
            "max_entry_offset_pct": 2.0,
            "volatility": {
                "enabled": True,
                "timeframe": "10m",
                "atr_period": 3,
                "stop_atr_multiplier": 1.0,
                "target_atr_multiplier": 1.5,
                "reward_risk_ratio": 1.5,
                "min_stop_pct": 0.05,
                "max_stop_pct": 2.0,
                "max_target_pct": 3.0,
            },
        }
    }


def _bundle(primary_signals=(1, 1, 1, 1), confirmation_signals=(1, 1, -1)):
    signals = {}
    for family, signal in zip(("high", "low", "close", "open"), primary_signals):
        signals[f"{family}:7h"] = {
            "signal": signal,
            "confidence": 0.8,
            "forecast_price": 102.0 if family == "high" else (98.0 if family == "low" else None),
        }
    for timeframe, signal in zip(("10m", "30m", "1h"), confirmation_signals):
        for family in ("high", "low", "close", "open"):
            signals[f"{family}:{timeframe}"] = {"signal": signal, "confidence": 0.8}
    return {"signals": signals, "coverage": 1.0, "n_usable": len(signals)}


class TradingSignalPolicyTests(unittest.TestCase):
    def test_joint_ohlc_uses_high_and_low_range_and_short_confirmations(self):
        result = evaluate_joint_ohlc_policy(_bundle(), _policy_config(), market_price=100.0)

        self.assertEqual(result["decision"], "buy")
        self.assertEqual(result["confirmations"], 2)
        self.assertAlmostEqual(result["projected_range"]["projected_low"], 98.0)
        self.assertAlmostEqual(result["projected_range"]["projected_high"], 102.0)
        self.assertAlmostEqual(result["entry"], 98.8)

    def test_high_cannot_control_direction_when_other_ohlc_families_disagree(self):
        result = evaluate_joint_ohlc_policy(
            _bundle(primary_signals=(1, -1, -1, -1), confirmation_signals=(-1, -1, 1)),
            _policy_config(),
            market_price=100.0,
        )

        self.assertEqual(result["decision"], "sell")
        self.assertEqual(result["confirmations"], 2)

    def test_policy_holds_when_short_timeframes_do_not_confirm(self):
        result = evaluate_joint_ohlc_policy(
            _bundle(confirmation_signals=(1, -1, -1)),
            _policy_config(required_confirmations=2),
            market_price=100.0,
        )

        self.assertEqual(result["decision"], "hold")
        self.assertEqual(result["reason"], "short_timeframe_confirmation_failed")

    def test_unqualified_models_have_no_ohlc_vote(self):
        bundle = _bundle()
        for family in ("high", "low", "close"):
            bundle["signals"][f"{family}:7h"]["quality"] = {
                "qualified": False,
                "weight": 0.0,
            }

        result = evaluate_joint_ohlc_policy(bundle, _policy_config(), market_price=100.0)

        self.assertEqual(result["decision"], "hold")
        self.assertEqual(result["reason"], "joint_ohlc_direction_not_strong_enough")
        self.assertEqual(result["direction_vote"]["available_families"], 1)

    def test_best_qualified_high_low_timeframe_becomes_direction_anchor(self):
        bundle = _bundle()
        bundle["signals"]["high:7h"]["quality"] = {"qualified": True, "weight": 0.2}
        bundle["signals"]["low:7h"]["quality"] = {"qualified": True, "weight": 0.2}
        for family in ("high", "low", "close", "open"):
            bundle["signals"][f"{family}:10m"].update(
                {
                    "quality": {"qualified": True, "weight": 0.9},
                    "forecast_price": 101.0 if family == "high" else (99.0 if family == "low" else None),
                }
            )
        config = _policy_config(required_confirmations=1)
        config["signal_policy"].update(
            {
                "direction_timeframes": ["10m", "7h"],
                "select_best_direction_timeframe": True,
                "confirmation_timeframes": ["30m", "1h", "7h"],
            }
        )

        result = evaluate_joint_ohlc_policy(bundle, config, market_price=100.0)

        self.assertEqual(result["selected_direction_timeframe"], "10m")
        self.assertTrue(result["projected_range"]["available"])

    def test_unqualified_high_low_cannot_set_entry_range(self):
        bundle = _bundle()
        bundle["signals"]["high:7h"]["quality"] = {"qualified": False, "weight": 0.0}
        bundle["signals"]["low:7h"]["quality"] = {"qualified": False, "weight": 0.0}
        result = evaluate_joint_ohlc_policy(
            bundle,
            _policy_config(required_confirmations=1),
            market_price=100.0,
        )
        self.assertEqual(result["decision"], "hold")
        self.assertFalse(result["projected_range"]["available"])
        self.assertEqual(result["reason"], "qualified_high_low_range_unavailable")

    def test_crossed_high_low_forecasts_are_not_silently_swapped(self):
        bundle = _bundle()
        bundle["signals"]["high:7h"]["forecast_price"] = 97.0
        bundle["signals"]["low:7h"]["forecast_price"] = 103.0
        result = evaluate_joint_ohlc_policy(
            bundle,
            _policy_config(required_confirmations=1),
            market_price=100.0,
        )
        self.assertEqual(result["decision"], "hold")
        self.assertEqual(
            result["projected_range"]["reason"],
            "independent_high_low_forecasts_crossed_or_collapsed",
        )

    def test_volatility_protection_uses_atr_and_preserves_side(self):
        frame = pd.DataFrame(
            {
                "HIGH": [101.0, 102.0, 103.0, 104.0],
                "LOW": [99.0, 100.0, 101.0, 102.0],
                "CLOSE": [100.0, 101.0, 102.0, 103.0],
            }
        )
        plan = apply_volatility_protection(
            {"decision": "buy", "entry": 103.0, "stop_loss": 100.0, "take_profit": 108.0},
            _policy_config(),
            frame,
        )

        self.assertLess(plan["stop_loss"], 103.0)
        self.assertGreater(plan["take_profit"], 103.0)
        self.assertTrue(plan["volatility_protection"]["enabled"])

    def test_point_in_time_requires_explicit_training_cutoffs(self):
        self.assertFalse(
            all_training_cutoffs_precede(
                [{"model_modified_at_utc": "2026-01-01 00:00:00"}],
                "2026-07-01",
            )
        )

    def test_walk_forward_selects_latest_package_with_pre_tick_cutoff(self):
        common = {
            "family": "high",
            "timeframe": "7h",
            "model": "nbeats",
            "n_steps": 1,
            "input_features": ["HIGH"],
            "target_features": ["y_diff"],
        }
        provider = WalkForwardModelSignalProvider(
            [
                {**common, "deployment_id": "old", "training_data_last_index": "2026-06-01"},
                {**common, "deployment_id": "new", "training_data_last_index": "2026-07-01"},
            ]
        )

        selected = provider.spec_for(provider.base_specs[0], pd.Timestamp("2026-07-15"))

        self.assertEqual(selected["deployment_id"], "new")
        self.assertTrue(provider.point_in_time_safe(pd.Timestamp("2026-06-15")))
        self.assertTrue(
            all_training_cutoffs_precede(
                [{"training_data_last_index": "2026-06-30 23:00:00"}],
                "2026-07-01",
            )
        )

    @patch("utils.trading_job._collect_all_model_assessment_signals")
    def test_live_market_fallback_rechecks_signal_cancels_then_submits(self, assessment_mock):
        assessment_mock.return_value = {**_bundle(), "assessment_scope": "all_models"}

        class Adapter:
            def __init__(self):
                self._mt5 = SimpleNamespace(symbol_info_tick=lambda _symbol: SimpleNamespace(bid=99.9, ask=100.1))
                self.cancelled = False

            def cancel_pending_order(self, ticket):
                self.cancelled = ticket == 7
                return {"ok": True, "order_ticket": ticket}

            def place_market_order(self, **kwargs):
                return {"ok": True, "position": {"ticket": 11}, "request": kwargs}

        config = _policy_config()
        config.update(
            {
                "trading_job": {
                    "market_fallback": {
                        "enabled": True,
                        "allowed_triggers": ["mandatory_session"],
                        "min_direction_score": 0.10,
                    }
                },
                "execution": {"symbol": "XAUUSD", "default_volume": 0.01},
                "risk": {"stop_loss_pct": 0.25},
            }
        )
        adapter = Adapter()
        result = _try_programmed_market_fallback(
            adapter=adapter,
            app_config={"symbol": "XAUUSD"},
            trading_cfg=config,
            state={
                "request_context": {"autonomous_trigger": "mandatory_session"},
                "plan": {
                    "decision": "buy",
                    "entry": 99.0,
                    "stop_loss": 98.0,
                    "take_profit": 101.0,
                    "volume": 0.01,
                },
            },
            order_ticket=7,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(adapter.cancelled)
        self.assertEqual(result["reason"], "market_fallback_submitted")


if __name__ == "__main__":
    unittest.main()
