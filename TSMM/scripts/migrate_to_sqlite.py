"""
Migrate market OHLC CSV into SQLite market DB.

Usage:
  python -m scripts.migrate_to_sqlite --csv data/xauusd/master_table.csv --db data/market_data.sqlite
"""

from __future__ import annotations

import argparse

from utils.market_db import import_csv_to_db, init_market_db, get_latest_date


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to source CSV")
    p.add_argument("--db", default="data/market_data.sqlite", help="Path to SQLite DB")
    return p.parse_args()


def main():
    args = parse_args()
    init_market_db(args.db)
    n = import_csv_to_db(args.csv, args.db)
    latest = get_latest_date(args.db)
    print({"rows_upserted": int(n), "db": args.db, "latest_date": latest})


if __name__ == "__main__":
    main()
