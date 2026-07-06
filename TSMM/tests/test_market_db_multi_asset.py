import sqlite3
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_db import (
    create_timeframe_views,
    get_row_count,
    master_table_name,
    query_ohlc,
    timeframe_view_name,
    upsert_ohlc_1m,
)


class MarketDbMultiAssetTests(unittest.TestCase):
    def _sample_df(self, close_values):
        rows = []
        for idx, close in enumerate(close_values):
            rows.append(
                {
                    "DATE": f"2026-05-29 00:{idx:02}:00",
                    "OPEN": float(close) - 0.5,
                    "HIGH": float(close) + 0.5,
                    "LOW": float(close) - 0.8,
                    "CLOSE": float(close),
                    "VOLUME": float(10 + idx),
                }
            )
        return pd.DataFrame(rows)

    def test_symbol_namespaces_are_isolated_in_single_db(self):
        with TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "market_data.sqlite")

            xau_df = self._sample_df([2400.0, 2400.2, 2400.4])
            us500_df = self._sample_df([5300.0, 5300.3, 5300.6])

            upsert_ohlc_1m(db_path, xau_df, symbol="XAUUSD")
            upsert_ohlc_1m(db_path, us500_df, symbol="US500")

            self.assertEqual(get_row_count(db_path, symbol="XAUUSD"), 3)
            self.assertEqual(get_row_count(db_path, symbol="US500"), 3)

            xau_q = query_ohlc(db_path, timeframe_minutes=1, latest_records=3, symbol="XAUUSD")
            us500_q = query_ohlc(db_path, timeframe_minutes=1, latest_records=3, symbol="US500")

            self.assertEqual(len(xau_q), 3)
            self.assertEqual(len(us500_q), 3)
            self.assertAlmostEqual(float(xau_q.iloc[-1]["CLOSE"]), 2400.4, places=6)
            self.assertAlmostEqual(float(us500_q.iloc[-1]["CLOSE"]), 5300.6, places=6)

            self.assertEqual(master_table_name("XAUUSD"), "ohlc_1m")
            self.assertEqual(master_table_name("US500"), "ohlc_1m_us500")

    def test_symbol_scoped_views_are_created(self):
        with TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "market_data.sqlite")
            xau_df = self._sample_df([2400.0, 2400.2, 2400.4, 2400.1, 2400.3])
            us500_df = self._sample_df([5300.0, 5300.3, 5300.6, 5300.2, 5300.1])

            upsert_ohlc_1m(db_path, xau_df, symbol="XAUUSD")
            upsert_ohlc_1m(db_path, us500_df, symbol="US500")

            create_timeframe_views(db_path, [10], symbol="XAUUSD")
            create_timeframe_views(db_path, [10], symbol="US500")

            xau_view = timeframe_view_name(10, symbol="XAUUSD")
            us500_view = timeframe_view_name(10, symbol="US500")

            conn = sqlite3.connect(db_path)
            try:
                names = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='view'"
                    ).fetchall()
                }
            finally:
                conn.close()

            self.assertIn(xau_view, names)
            self.assertIn(us500_view, names)


if __name__ == "__main__":
    unittest.main()
