import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.migrate_market_data_to_sqlite import migrate_master_directory


class MigrateMarketDataMultiAssetTests(unittest.TestCase):
    def test_import_histdata_directory_without_headers(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "DataBuild" / "HISTDATA_COM_MT_SPXUSD_M12026"
            source_dir.mkdir(parents=True, exist_ok=True)

            csv_path = source_dir / "DAT_MT_SPXUSD_M1_2026.csv"
            csv_path.write_text(
                "2026.01.02,00:00,5000.0,5000.5,4999.8,5000.2,0\n"
                "2026.01.02,00:01,5000.2,5000.6,5000.0,5000.4,0\n"
                "2026.01.02,00:02,5000.4,5000.7,5000.2,5000.5,0\n",
                encoding="utf-8",
            )

            db_path = root / "market_data_us500.sqlite"
            report = migrate_master_directory(
                master_dir=str(root / "DataBuild"),
                db_path=str(db_path),
                chunksize=1000,
                symbol="US500",
                csv_glob="**/*.csv",
            )

            self.assertEqual(report.get("symbol"), "US500")
            self.assertEqual(int(report.get("files_processed", 0)), 1)
            self.assertEqual(int(report.get("db_row_count", 0)), 3)
            self.assertTrue(str(report.get("db_latest_date") or "").startswith("2026-01-02 00:02:00"))


if __name__ == "__main__":
    unittest.main()
