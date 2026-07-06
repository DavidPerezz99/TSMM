"""
Migrate market OHLC CSV into SQLite market DB.

Usage:
    python -m scripts.migrate_to_sqlite --csv data/xauusd/master_table.csv --db data/market_data.sqlite --symbol XAUUSD
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_db import import_csv_to_db, init_market_db, get_latest_date, normalize_market_symbol


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to source CSV")
    p.add_argument("--db", default="data/market_data.sqlite", help="Path to SQLite DB")
    p.add_argument("--symbol", default="XAUUSD", help="Asset symbol namespace")
    return p.parse_args()


def main():
    args = parse_args()
    symbol = normalize_market_symbol(args.symbol)
    init_market_db(args.db, symbol=symbol)
    n = import_csv_to_db(args.csv, args.db, symbol=symbol)
    latest = get_latest_date(args.db, symbol=symbol)
    print({"rows_upserted": int(n), "db": args.db, "symbol": symbol, "latest_date": latest})


if __name__ == "__main__":
    main()
