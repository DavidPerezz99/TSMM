import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.market_db import (
    create_timeframe_cache_table,
    create_timeframe_view,
    materialize_timeframe_cache_tables,
    query_ohlc,
    upsert_ohlc_1m,
)


class MarketDbSourceModeTests(unittest.TestCase):
    def test_view_and_cache_modes_return_grouped_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "market.sqlite")
            dates = pd.date_range("2026-08-01", periods=30, freq="min")
            frame = pd.DataFrame(
                {
                    "DATE": dates,
                    "OPEN": range(30),
                    "HIGH": [value + 2 for value in range(30)],
                    "LOW": [value - 1 for value in range(30)],
                    "CLOSE": [value + 1 for value in range(30)],
                    "VOLUME": [1.0] * 30,
                }
            )
            upsert_ohlc_1m(db_path, frame)
            create_timeframe_view(db_path, 10)
            create_timeframe_cache_table(db_path, 10)

            from_view = query_ohlc(db_path, 10, latest_records=3, source_mode="view")
            from_cache = query_ohlc(db_path, 10, latest_records=3, source_mode="cache")
            self.assertEqual(len(from_view), 3)
            self.assertEqual(len(from_cache), 3)
            self.assertEqual(from_view["DATE"].tolist(), from_cache["DATE"].tolist())
            self.assertEqual(from_view["HIGH"].tolist(), from_cache["HIGH"].tolist())

    def test_invalid_source_mode_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "market.sqlite")
            with self.assertRaises(ValueError):
                query_ohlc(db_path, 10, source_mode="mystery")

    def test_multi_timeframe_materializer_builds_cache_tables(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "market.sqlite")
            dates = pd.date_range("2026-08-01", periods=60, freq="min")
            frame = pd.DataFrame(
                {
                    "DATE": dates,
                    "OPEN": range(60),
                    "HIGH": [value + 2 for value in range(60)],
                    "LOW": [value - 1 for value in range(60)],
                    "CLOSE": [value + 1 for value in range(60)],
                    "VOLUME": [1.0] * 60,
                }
            )
            upsert_ohlc_1m(db_path, frame)
            created = materialize_timeframe_cache_tables(db_path, [10, 30])
            self.assertEqual(created, ["ohlc_10m_cache", "ohlc_30m_cache"])
            self.assertEqual(len(query_ohlc(db_path, 10, 10, source_mode="cache")), 6)
            self.assertEqual(len(query_ohlc(db_path, 30, 10, source_mode="cache")), 2)


if __name__ == "__main__":
    unittest.main()
