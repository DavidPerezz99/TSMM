"""
Migrate large market CSV(s) into SQLite OHLC store.

Usage:
    python -m scripts.migrate_market_data_to_sqlite --master-csv data/xauusd/master_table.csv --db-path data/market_data.sqlite --chunksize 250000 --update-trading-config
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd

from utils.market_db import (
    init_market_db,
    upsert_ohlc_1m,
    get_row_count,
    get_latest_date,
    create_timeframe_views,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--master-csv", required=True, help="Path to 1m master CSV")
    p.add_argument("--db-path", default="data/market_data.sqlite", help="SQLite output path")
    p.add_argument("--chunksize", type=int, default=250000, help="CSV chunk size")
    p.add_argument(
        "--views",
        default="10,30",
        help="Comma-separated timeframe minutes to expose as SQL views (e.g. 10,30,60)",
    )
    p.add_argument(
        "--update-trading-config",
        action="store_true",
        help="Set dashboard.master_table_path in config/trading_agent.yaml to DB path",
    )
    p.add_argument(
        "--create-cache-tables",
        action="store_true",
        help="Also materialize timeframe cache tables after creating views",
    )
    p.add_argument(
        "--trading-config",
        default="config/trading_agent.yaml",
        help="Trading config path to update when --update-trading-config is set",
    )
    return p.parse_args()


def migrate_master_csv(master_csv: str, db_path: str, chunksize: int = 250000) -> dict:
    csv_path = Path(master_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")

    init_market_db(db_path)
    total_in = 0
    total_upsert = 0

    for chunk in pd.read_csv(csv_path, chunksize=max(int(chunksize), 50000)):
        n_in = int(len(chunk))
        n_up = int(upsert_ohlc_1m(db_path, chunk))
        total_in += n_in
        total_upsert += n_up
        print(f"chunk_in={n_in} upserted={n_up} total_in={total_in}")

    return {
        "total_csv_rows_read": int(total_in),
        "total_rows_upserted": int(total_upsert),
        "db_row_count": int(get_row_count(db_path)),
        "db_latest_date": get_latest_date(db_path),
        "db_path": db_path,
    }


def update_trading_dashboard_db_path(trading_config_path: str, db_path: str) -> str:
    cfg_path = Path(trading_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Trading config not found: {trading_config_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("dashboard", {})
    cfg["dashboard"]["master_table_path"] = db_path

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return str(cfg_path)


def main():
    args = parse_args()
    result = migrate_master_csv(args.master_csv, args.db_path, args.chunksize)

    views = []
    if str(args.views).strip():
        vals = [v.strip() for v in str(args.views).split(",") if v.strip()]
        tfs = [int(v) for v in vals]
        views = create_timeframe_views(
            args.db_path,
            tfs,
            include_cache_tables=bool(args.create_cache_tables),
        )

    result["created_views"] = views
    print("Migration summary:")
    print(result)

    if bool(args.update_trading_config):
        path = update_trading_dashboard_db_path(args.trading_config, args.db_path)
        print(f"Updated dashboard master_table_path in {path}")


if __name__ == "__main__":
    main()
