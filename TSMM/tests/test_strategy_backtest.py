from __future__ import annotations

from datetime import datetime, timezone
import io
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from utils.strategy_backtest import (
    AsOfMarketTape,
    ConsoleProgressBar,
    _intrabar_exit,
    previous_calendar_month,
    run_historical_strategy_backtest,
)


class _FakeProvider:
    def predict(self, spec, rows):
        return {
            "raw_signal": 1,
            "confidence": 0.90,
            "forecast_sign": 1.0,
            "feature_horizons": {"y_diff": [1.0], "Price_return": [1.0]},
            "input_window_start": rows[0]["DATE"],
            "input_window_end": rows[-1]["DATE"],
        }

    def manifest(self):
        return [
            {
                "key": "fake",
                "model_modified_at_utc": "2026-06-01 00:00:00",
                "model_path": "fake.joblib",
            }
        ]


def _minute_frame(start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="1min")
    close = [2000.0 + index * 0.01 for index in range(periods)]
    return pd.DataFrame(
        {
            "DATE": dates,
            "OPEN": close,
            "HIGH": [value + 0.02 for value in close],
            "LOW": [value - 0.02 for value in close],
            "CLOSE": close,
            "VOLUME": 1.0,
        }
    )


class StrategyBacktestTests(unittest.TestCase):
    def test_console_progress_bar_reports_percentage_ticks_and_eta(self):
        stream = io.StringIO()
        progress = ConsoleProgressBar(stream=stream, width=10, min_interval_seconds=0.0)

        progress(0, 20, "2026-07-01 00:00:00")
        progress(10, 20, "2026-07-01 00:50:00")
        progress(20, 20, "2026-07-01 01:40:00")

        output = stream.getvalue()
        self.assertIn("50.00%", output)
        self.assertIn("100.00%", output)
        self.assertIn("20/20 ticks", output)
        self.assertIn("ETA", output)
        self.assertIn("simulated 2026-07-01 01:40:00", output)

    def test_previous_calendar_month_uses_full_local_month(self):
        start, end = previous_calendar_month(
            reference=datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc),
            timezone_name="UTC",
        )
        self.assertEqual(start, pd.Timestamp("2026-07-01 00:00:00"))
        self.assertEqual(end, pd.Timestamp("2026-07-31 23:59:00"))

    def test_partial_timeframe_candle_does_not_see_future_minutes(self):
        frame = _minute_frame("2026-07-01 00:00:00", 20)
        frame.loc[15, "HIGH"] = 9999.0
        tape = AsOfMarketTape(frame, ["10m"])

        as_of = tape.timeframe_as_of("10m", pd.Timestamp("2026-07-01 00:12:00"))

        self.assertEqual(len(as_of), 2)
        self.assertLess(float(as_of.iloc[-1]["HIGH"]), 9999.0)
        self.assertEqual(as_of.iloc[-1]["DATE"], pd.Timestamp("2026-07-01 00:10:00"))

    def test_intrabar_policy_is_conservative_when_both_barriers_hit(self):
        position = {"side": "buy", "stop_loss": 99.0, "take_profit": 101.0}
        bar = pd.Series({"LOW": 98.0, "HIGH": 102.0})

        self.assertEqual(_intrabar_exit(position, bar), ("stop_loss", 99.0))

    def test_fake_model_replay_writes_trade_lifecycle_and_report(self):
        frame = _minute_frame("2026-06-28 00:00:00", 3 * 24 * 60 + 20)
        specs = []
        for timeframe in ("7h", "3h"):
            specs.append(
                {
                    "family": "high",
                    "timeframe": timeframe,
                    "model": "fake",
                    "r2": 0.5,
                    "config_path": "fake.yaml",
                    "n_steps": 1,
                    "m_steps": 1,
                    "horizon": 1,
                    "input_features": ["HIGH", "y_diff", "Price_return"],
                    "target_features": ["y_diff", "Price_return"],
                    "target_col": "HIGH",
                    "rolling_windows": [1],
                }
            )
        trading_cfg = {
            "agent": {"signal_interpretation": "momentum", "timezone": "UTC"},
            "mode_a": {"allow_long": True, "allow_short": True, "max_operations_per_session": 3},
            "mode_b": {"poll_seconds": 300, "close_consensus_threshold": 0.25},
            "conviction": {
                "min_conviction_for_no_sl": 0.8,
                "min_conviction_for_wide": 0.65,
                "min_conviction_for_standard": 0.45,
                "min_conviction_for_tight": 0.3,
            },
            "risk": {
                "stop_loss_pct": 0.8,
                "take_profit_pct": 1.6,
                "max_open_positions": 5,
                "daily_max_loss_pct": 2.0,
                "max_drawdown_pct": 15.0,
                "min_cm_accuracy_to_trade": 0.5,
                "max_input_fooling_risk": 0.45,
                "trailing": {"enabled": False},
            },
            "execution": {
                "symbol": "XAUUSD",
                "default_volume": 0.01,
                "spread_bps": 0.0,
                "slippage_bps": 0.0,
                "commission_per_trade": 0.0,
                "delayed_stop_loss": {"enabled": False},
            },
            "trading_job": {"session_hours": 7, "programmed_order_expiration_minutes": 420},
            "autonomous_trading": {
                "timezone": "UTC",
                "max_jobs_per_session": 3,
                "followup_enabled": False,
                "session_windows": [{"name": "test", "start": "00:00", "end": "23:59"}],
                "pending_order_maintenance": {"cancel_opposed_consensus_threshold": 0.25},
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_historical_strategy_backtest(
                market_source="unused.csv",
                trading_cfg=trading_cfg,
                output_dir=temp_dir,
                start_date="2026-07-01 00:00:00",
                end_date="2026-07-01 00:15:00",
                specs=specs,
                provider=_FakeProvider(),
                market_frame=frame,
                max_ticks=3,
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["summary"]["overall"]["n_trades"], 1)
            operation = result["summary"]["operations"][0]
            self.assertEqual(operation["trigger"], "mandatory_session")
            self.assertEqual(operation["exit_reason"], "forced_period_end")
            self.assertTrue(Path(result["report_path"]).exists())
            self.assertTrue(Path(result["operations_path"]).exists())
            self.assertTrue(result["summary"]["validity"]["point_in_time_market_data"])


if __name__ == "__main__":
    unittest.main()
