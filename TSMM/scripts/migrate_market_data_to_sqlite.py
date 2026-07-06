"""
Migrate large market CSV(s) into SQLite OHLC store.

Usage:
    python -m scripts.migrate_market_data_to_sqlite --master-csv data/xauusd/master_table.csv --db-path data/market_data.sqlite --symbol XAUUSD --chunksize 250000 --update-trading-config
    python -m scripts.migrate_market_data_to_sqlite --master-dir "C:/Users/USUARIO/Documents/DataBuild" --db-path data/market_data_us500.sqlite --symbol US500 --views 10,30,60,180,420
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import pandas as pd
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.market_db import (
    init_market_db,
    upsert_ohlc_1m,
    get_row_count,
    get_latest_date,
    create_timeframe_views,
    normalize_market_symbol,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--master-csv", default="", help="Path to 1m master CSV")
    p.add_argument("--master-dir", default="", help="Path to a folder containing CSV files to ingest recursively")
    p.add_argument("--master-glob", default="**/*.csv", help="Glob expression used with --master-dir")
    p.add_argument("--db-path", default="data/market_data.sqlite", help="SQLite output path")
    p.add_argument("--symbol", default="XAUUSD", help="Asset symbol namespace (e.g. XAUUSD, US500)")
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


def _csv_has_header(csv_path: Path) -> bool:
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
    except Exception:
        return True
    sample = str(first_line or "").strip()
    if not sample:
        return True
    return any(ch.isalpha() for ch in sample)


def _iter_csv_chunks(csv_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    chunk_size = max(int(chunksize), 50000)
    has_header = _csv_has_header(csv_path)

    if has_header:
        yield from pd.read_csv(csv_path, chunksize=chunk_size)
        return

    # HISTDATA-like format: DATE,TIME,OPEN,HIGH,LOW,CLOSE,VOLUME with no header.
    names = ["DATE_PART", "TIME_PART", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    yield from pd.read_csv(csv_path, header=None, names=names, chunksize=chunk_size)


def _prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    if chunk is None or chunk.empty:
        return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

    out = chunk.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]

    if "DATE_PART" in out.columns and "TIME_PART" in out.columns:
        out["DATE"] = out["DATE_PART"].astype(str).str.strip() + " " + out["TIME_PART"].astype(str).str.strip()
    elif "DATE" in out.columns and "TIME" in out.columns:
        out["DATE"] = out["DATE"].astype(str).str.strip() + " " + out["TIME"].astype(str).str.strip()

    if "VOL" in out.columns and "VOLUME" not in out.columns:
        out["VOLUME"] = out["VOL"]
    if "TICKVOL" in out.columns and "VOLUME" not in out.columns:
        out["VOLUME"] = out["TICKVOL"]
    if "VOLUME" not in out.columns:
        out["VOLUME"] = 0.0

    required = ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]
    for col in required:
        if col not in out.columns:
            out[col] = None
    return out[required]


def migrate_master_csv(
    master_csv: str,
    db_path: str,
    chunksize: int = 250000,
    symbol: str = "XAUUSD",
) -> dict:
    csv_path = Path(master_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Master CSV not found: {master_csv}")

    resolved_symbol = normalize_market_symbol(symbol)
    init_market_db(db_path, symbol=resolved_symbol)
    total_in = 0
    total_upsert = 0

    for chunk in _iter_csv_chunks(csv_path, chunksize=chunksize):
        prepared = _prepare_chunk(chunk)
        if prepared.empty:
            continue
        n_in = int(len(chunk))
        n_up = int(upsert_ohlc_1m(db_path, prepared, symbol=resolved_symbol))
        total_in += n_in
        total_upsert += n_up
        print(f"file={csv_path.name} chunk_in={n_in} upserted={n_up} total_in={total_in}")

    return {
        "source_file": str(csv_path),
        "symbol": resolved_symbol,
        "total_csv_rows_read": int(total_in),
        "total_rows_upserted": int(total_upsert),
        "db_row_count": int(get_row_count(db_path, symbol=resolved_symbol)),
        "db_latest_date": get_latest_date(db_path, symbol=resolved_symbol),
        "db_path": db_path,
    }


def migrate_master_directory(
    master_dir: str,
    db_path: str,
    chunksize: int = 250000,
    symbol: str = "XAUUSD",
    csv_glob: str = "**/*.csv",
) -> Dict[str, Any]:
    root = Path(master_dir)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Master directory not found: {master_dir}")

    files = sorted([p for p in root.glob(csv_glob) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No CSV files found under {master_dir} with glob {csv_glob}")

    resolved_symbol = normalize_market_symbol(symbol)
    file_reports: List[Dict[str, Any]] = []
    total_rows_read = 0
    total_rows_upserted = 0

    for file_path in files:
        report = migrate_master_csv(
            master_csv=str(file_path),
            db_path=db_path,
            chunksize=chunksize,
            symbol=resolved_symbol,
        )
        total_rows_read += int(report.get("total_csv_rows_read", 0) or 0)
        total_rows_upserted += int(report.get("total_rows_upserted", 0) or 0)
        file_reports.append(
            {
                "source_file": str(file_path),
                "rows_read": int(report.get("total_csv_rows_read", 0) or 0),
                "rows_upserted": int(report.get("total_rows_upserted", 0) or 0),
            }
        )

    return {
        "symbol": resolved_symbol,
        "source_dir": str(root),
        "files_processed": int(len(files)),
        "total_csv_rows_read": int(total_rows_read),
        "total_rows_upserted": int(total_rows_upserted),
        "db_row_count": int(get_row_count(db_path, symbol=resolved_symbol)),
        "db_latest_date": get_latest_date(db_path, symbol=resolved_symbol),
        "db_path": db_path,
        "file_reports": file_reports,
    }


def update_trading_dashboard_db_path(trading_config_path: str, db_path: str, symbol: str) -> str:
    cfg_path = Path(trading_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Trading config not found: {trading_config_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault("dashboard", {})
    cfg["dashboard"]["master_table_path"] = db_path
    cfg["dashboard"]["sql_symbol"] = normalize_market_symbol(symbol)

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    return str(cfg_path)


def main():
    args = parse_args()
    resolved_symbol = normalize_market_symbol(args.symbol)

    if not str(args.master_csv or "").strip() and not str(args.master_dir or "").strip():
        raise ValueError("Provide --master-csv or --master-dir")

    reports: List[Dict[str, Any]] = []
    if str(args.master_csv or "").strip():
        reports.append(
            migrate_master_csv(
                master_csv=str(args.master_csv),
                db_path=str(args.db_path),
                chunksize=int(args.chunksize),
                symbol=resolved_symbol,
            )
        )

    if str(args.master_dir or "").strip():
        reports.append(
            migrate_master_directory(
                master_dir=str(args.master_dir),
                db_path=str(args.db_path),
                chunksize=int(args.chunksize),
                symbol=resolved_symbol,
                csv_glob=str(args.master_glob or "**/*.csv"),
            )
        )

    result: Dict[str, Any] = {
        "symbol": resolved_symbol,
        "db_path": str(args.db_path),
        "reports": reports,
        "db_row_count": int(get_row_count(str(args.db_path), symbol=resolved_symbol)),
        "db_latest_date": get_latest_date(str(args.db_path), symbol=resolved_symbol),
    }

    views = []
    if str(args.views).strip():
        vals = [v.strip() for v in str(args.views).split(",") if v.strip()]
        tfs = [int(v) for v in vals]
        views = create_timeframe_views(
            args.db_path,
            tfs,
            include_cache_tables=bool(args.create_cache_tables),
            symbol=resolved_symbol,
        )

    result["created_views"] = views
    print("Migration summary:")
    print(result)

    if bool(args.update_trading_config):
        path = update_trading_dashboard_db_path(args.trading_config, args.db_path, symbol=resolved_symbol)
        print(f"Updated dashboard master_table_path in {path}")


if __name__ == "__main__":
    main()
