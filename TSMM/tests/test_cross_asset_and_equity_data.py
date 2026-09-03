from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from utils.data_loader import load_data, merge_cross_asset_frame
from utils.equity_data import refresh_us500_proxy
from utils.market_db import init_market_db, master_table_name, upsert_ohlc_1m


def _frame(start: str, periods: int, base: float) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="1min")
    close = pd.Series([base + index * 0.01 for index in range(periods)])
    return pd.DataFrame(
        {
            "DATE": dates,
            "OPEN": close,
            "HIGH": close + 0.02,
            "LOW": close - 0.02,
            "CLOSE": close,
            "VOLUME": 1.0,
        }
    )


class CrossAssetAndEquityDataTests(unittest.TestCase):
    def test_cross_asset_merge_is_backward_only(self):
        target = _frame("2026-01-01 00:00:00", 3, 100.0)
        source = _frame("2026-01-01 00:01:00", 3, 10.0)

        merged, coverage = merge_cross_asset_frame(
            target,
            source,
            {
                "symbol": "US500",
                "prefix": "US500",
                "max_staleness_minutes": 1,
                "minimum_coverage": 0.0,
                "require_match": False,
            },
            1,
        )

        self.assertTrue(pd.isna(merged.iloc[0]["US500_CLOSE"]))
        self.assertAlmostEqual(float(merged.iloc[1]["US500_CLOSE"]), 10.0)
        self.assertAlmostEqual(coverage, 2 / 3)

    def test_stale_cross_asset_levels_are_held_but_returns_are_zero(self):
        target = _frame("2026-01-01 00:00:00", 3, 100.0)
        source = _frame("2026-01-01 00:00:00", 2, 10.0)

        merged, coverage = merge_cross_asset_frame(
            target,
            source,
            {
                "symbol": "US500",
                "prefix": "US500",
                "max_staleness_minutes": 5,
                "minimum_coverage": 1.0,
                "require_match": True,
            },
            1,
        )

        self.assertEqual(coverage, 1.0)
        self.assertAlmostEqual(merged.iloc[-1]["US500_CLOSE"], 10.01)
        self.assertEqual(merged.iloc[-1]["US500_Price_return"], 0.0)
        self.assertEqual(merged.iloc[-1]["US500_AGE_MINUTES"], 1.0)

    def test_baseline_recipe_does_not_load_unused_cross_asset_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "xau.csv"
            _frame("2026-01-01", 20, 100.0).to_csv(csv_path, index=False)
            loaded = load_data(
                str(csv_path),
                "DATE",
                "HIGH",
                {
                    "use_sql_master": False,
                    "data_timeframe_minutes": 1,
                    "input_features": ["HIGH", "y_diff"],
                    "target_features": ["y_diff"],
                    "rolling_windows": [2],
                    "cross_asset_features": {
                        "enabled": True,
                        "db_path": str(Path(temp_dir) / "missing.sqlite"),
                        "symbol": "US500",
                        "prefix": "US500",
                    },
                },
            )
            self.assertFalse(loaded.empty)
            self.assertNotIn("US500_CLOSE", loaded.columns)

    @patch("utils.equity_data.fetch_tiingo_iex_minutes")
    def test_us500_proxy_refresh_keeps_raw_source_and_records_provenance(self, fetch):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "us500.sqlite")
            native = _frame("2026-01-01 00:00:00", 200, 1000.0)
            # A stable 10x overlap plus five new source minutes.
            source = native.copy()
            for column in ["OPEN", "HIGH", "LOW", "CLOSE"]:
                source[column] = source[column] / 10.0
            future = _frame("2026-01-01 03:20:00", 5, float(source.iloc[-1]["CLOSE"]) + 0.01)
            fetched = pd.concat([source, future], ignore_index=True)
            fetch.return_value = {
                "ok": True,
                "data": fetched,
                "used_token_env": "TIINGO_API_TOKEN",
                "rotated": False,
            }
            init_market_db(db_path, symbol="US500")
            upsert_ohlc_1m(db_path, native, symbol="US500")

            result = refresh_us500_proxy(
                db_path=db_path,
                minimum_calibration_samples=100,
                calibration_lookback_days=5,
                maximum_relative_mad=0.02,
                maximum_seam_jump_pct=2.0,
                end_date="2026-01-02",
            )

            self.assertTrue(result["ok"])
            self.assertEqual(result["target_rows_written"], 5)
            connection = sqlite3.connect(db_path)
            try:
                us500_rows = connection.execute(
                    f"SELECT COUNT(*) FROM {master_table_name('US500')}"
                ).fetchone()[0]
                spy_rows = connection.execute(
                    f"SELECT COUNT(*) FROM {master_table_name('SPY')}"
                ).fetchone()[0]
                provenance_rows = connection.execute(
                    "SELECT COUNT(*) FROM market_data_provenance"
                ).fetchone()[0]
            finally:
                connection.close()
            self.assertEqual(us500_rows, 205)
            self.assertEqual(spy_rows, 205)
            self.assertEqual(provenance_rows, 1)


if __name__ == "__main__":
    unittest.main()
