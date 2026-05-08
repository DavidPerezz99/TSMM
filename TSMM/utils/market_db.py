"""
SQLite market data store for 1m master candles and grouped timeframe queries.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional

import pandas as pd


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_market_db(db_path: str) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ohlc_1m (
                DATE TEXT PRIMARY KEY,
                OPEN REAL,
                HIGH REAL,
                LOW REAL,
                CLOSE REAL,
                VOLUME REAL DEFAULT 0.0
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlc_1m_date ON ohlc_1m(DATE)")
        conn.commit()
    finally:
        conn.close()


def normalize_ohlc_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

    out = df.copy()
    out.columns = [str(c).strip().upper() for c in out.columns]

    for col in ["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        if col not in out.columns:
            out[col] = None

    # Recover mixed legacy rows where actual row order is DATE,OPEN,HIGH,LOW,CLOSE
    # but file header may still be OPEN,HIGH,LOW,CLOSE,VOLUME,DATE.
    parsed_open = pd.to_datetime(out["OPEN"], errors="coerce")
    parsed_date = pd.to_datetime(out["DATE"], errors="coerce")
    shifted_mask = parsed_open.notna() & parsed_date.isna()
    if bool(shifted_mask.any()):
        out.loc[shifted_mask, "DATE"] = out.loc[shifted_mask, "OPEN"]
        out.loc[shifted_mask, "OPEN"] = out.loc[shifted_mask, "HIGH"]
        out.loc[shifted_mask, "HIGH"] = out.loc[shifted_mask, "LOW"]
        out.loc[shifted_mask, "LOW"] = out.loc[shifted_mask, "CLOSE"]
        out.loc[shifted_mask, "CLOSE"] = out.loc[shifted_mask, "VOLUME"]
        out.loc[shifted_mask, "VOLUME"] = 0.0

    # Handle mixed legacy layout where DATE may be first value despite header mismatch.
    if "DATE" not in out.columns and len(out.columns) >= 5:
        first_col = out.columns[0]
        parsed_first = pd.to_datetime(out[first_col], errors="coerce")
        if int(parsed_first.notna().sum()) > 0:
            # Assume DATE,OPEN,HIGH,LOW,CLOSE order.
            renamed = {}
            cols = list(out.columns)
            renamed[cols[0]] = "DATE"
            if len(cols) > 1:
                renamed[cols[1]] = "OPEN"
            if len(cols) > 2:
                renamed[cols[2]] = "HIGH"
            if len(cols) > 3:
                renamed[cols[3]] = "LOW"
            if len(cols) > 4:
                renamed[cols[4]] = "CLOSE"
            out = out.rename(columns=renamed)

    if "DATE" not in out.columns:
        # Fallback detection by date-like content.
        best_col = None
        best_valid = -1
        for col in out.columns:
            s = out[col].astype(str).str.strip()
            mask = s.str.contains(r"[-/:T]", regex=True, na=False)
            if int(mask.sum()) <= 0:
                continue
            parsed = pd.to_datetime(s.where(mask, None), errors="coerce")
            valid = int(parsed.notna().sum())
            if valid > best_valid:
                best_valid = valid
                best_col = col
        if best_col is not None:
            out = out.rename(columns={best_col: "DATE"})

    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce").dt.tz_localize(None)
    out = out.dropna(subset=["DATE"])

    for col in ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
    out = out[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]].copy()
    out["DATE"] = out["DATE"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out = out.drop_duplicates(subset=["DATE"]).sort_values("DATE")
    return out


def upsert_ohlc_1m(db_path: str, df: pd.DataFrame) -> int:
    clean = normalize_ohlc_df(df)
    if clean.empty:
        return 0

    init_market_db(db_path)
    conn = _connect(db_path)
    try:
        rows = list(clean.itertuples(index=False, name=None))
        conn.executemany(
            """
            INSERT INTO ohlc_1m (DATE, OPEN, HIGH, LOW, CLOSE, VOLUME)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(DATE) DO UPDATE SET
                OPEN=excluded.OPEN,
                HIGH=excluded.HIGH,
                LOW=excluded.LOW,
                CLOSE=excluded.CLOSE,
                VOLUME=excluded.VOLUME
            """,
            rows,
        )
        conn.commit()
        return int(len(rows))
    finally:
        conn.close()


def import_csv_to_db(csv_path: str, db_path: str) -> int:
    df = pd.read_csv(csv_path)
    return upsert_ohlc_1m(db_path, df)


def get_latest_date(db_path: str) -> Optional[str]:
    if not os.path.exists(db_path):
        return None
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT MAX(DATE) FROM ohlc_1m").fetchone()
        return str(row[0]) if row and row[0] else None
    finally:
        conn.close()


def get_row_count(db_path: str, table: str = "ohlc_1m") -> int:
    if not os.path.exists(db_path):
        return 0
    conn = _connect(db_path)
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def create_timeframe_view(db_path: str, timeframe_minutes: int) -> str:
    """Create or replace a grouped OHLC SQL view from ohlc_1m."""
    tf = int(max(timeframe_minutes, 1))
    view_name = f"ohlc_{tf}m"
    init_market_db(db_path)
    conn = _connect(db_path)
    try:
        conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        conn.execute(
            f"""
            CREATE VIEW {view_name} AS
            WITH base AS (
                SELECT
                    DATE,
                    OPEN,
                    HIGH,
                    LOW,
                    CLOSE,
                    VOLUME,
                    (CAST(strftime('%s', DATE) AS INTEGER) / ({tf} * 60)) AS bucket_id,
                    CAST(strftime('%s', DATE) AS INTEGER) AS ts
                FROM ohlc_1m
            ),
            agg AS (
                SELECT
                    datetime(bucket_id * {tf} * 60, 'unixepoch') AS DATE,
                    MAX(HIGH) AS HIGH,
                    MIN(LOW) AS LOW,
                    SUM(COALESCE(VOLUME, 0.0)) AS VOLUME,
                    MIN(ts) AS first_ts,
                    MAX(ts) AS last_ts
                FROM base
                GROUP BY bucket_id
            )
            SELECT
                agg.DATE AS DATE,
                b_open.OPEN AS OPEN,
                agg.HIGH AS HIGH,
                agg.LOW AS LOW,
                b_close.CLOSE AS CLOSE,
                agg.VOLUME AS VOLUME
            FROM agg
            LEFT JOIN base b_open ON b_open.ts = agg.first_ts
            LEFT JOIN base b_close ON b_close.ts = agg.last_ts
            ORDER BY agg.DATE
            """
        )
        conn.commit()
        return view_name
    finally:
        conn.close()


def create_timeframe_views(db_path: str, timeframes_minutes: list[int]) -> list[str]:
    created = []
    for tf in timeframes_minutes:
        created.append(create_timeframe_view(db_path, int(tf)))
    return created


def query_ohlc(
    db_path: str,
    timeframe_minutes: int = 1,
    latest_records: int = 50000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    init_market_db(db_path)
    conn = _connect(db_path)
    try:
        where = ["1=1"]
        params = []
        if start_date:
            where.append("DATE >= ?")
            params.append(start_date)
        if end_date:
            where.append("DATE <= ?")
            params.append(end_date)

        where_sql = " AND ".join(where)

        if int(timeframe_minutes) <= 1:
            sql = f"""
                SELECT DATE, OPEN, HIGH, LOW, CLOSE, VOLUME
                FROM ohlc_1m
                WHERE {where_sql}
                ORDER BY DATE DESC
                LIMIT ?
            """
            df = pd.read_sql_query(sql, conn, params=params + [int(max(latest_records, 1))])
            if df.empty:
                return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
            df = df.sort_values("DATE").reset_index(drop=True)
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
            return df

        tf = int(max(timeframe_minutes, 1))
        sql = f"""
            WITH base AS (
                SELECT
                    DATE,
                    OPEN,
                    HIGH,
                    LOW,
                    CLOSE,
                    VOLUME,
                    (CAST(strftime('%s', DATE) AS INTEGER) / ({tf} * 60)) AS bucket_id,
                    CAST(strftime('%s', DATE) AS INTEGER) AS ts
                FROM ohlc_1m
                WHERE {where_sql}
            ),
            agg AS (
                SELECT
                    datetime(bucket_id * {tf} * 60, 'unixepoch') AS DATE,
                    MAX(HIGH) AS HIGH,
                    MIN(LOW) AS LOW,
                    SUM(COALESCE(VOLUME, 0.0)) AS VOLUME,
                    MIN(ts) AS first_ts,
                    MAX(ts) AS last_ts
                FROM base
                GROUP BY bucket_id
            )
            SELECT
                agg.DATE,
                b_open.OPEN AS OPEN,
                agg.HIGH,
                agg.LOW,
                b_close.CLOSE AS CLOSE,
                agg.VOLUME
            FROM agg
            LEFT JOIN base b_open ON b_open.ts = agg.first_ts
            LEFT JOIN base b_close ON b_close.ts = agg.last_ts
            ORDER BY agg.DATE DESC
            LIMIT ?
        """
        df = pd.read_sql_query(sql, conn, params=params + [int(max(latest_records, 1))])
        if df.empty:
            return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
        df = df.sort_values("DATE").reset_index(drop=True)
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        return df
    finally:
        conn.close()
