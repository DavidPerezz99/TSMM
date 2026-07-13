"""
SQLite market data store for 1m master candles and grouped timeframe queries.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Optional
import logging

import pandas as pd


DEFAULT_MARKET_SYMBOL = "XAUUSD"
SYMBOL_ALIASES = {
    "SPXUSD": "US500",
    "US500USD": "US500",
    "SP500": "US500",
    "US500": "US500",
}


def normalize_market_symbol(symbol: Optional[str]) -> str:
    raw = str(symbol or DEFAULT_MARKET_SYMBOL).strip().upper().replace("=", "")
    compact = "".join(ch for ch in raw if ch.isalnum() or ch == "_")
    if not compact:
        compact = DEFAULT_MARKET_SYMBOL
    return str(SYMBOL_ALIASES.get(compact, compact))


def _symbol_suffix(symbol: Optional[str]) -> str:
    normalized = normalize_market_symbol(symbol)
    if normalized == DEFAULT_MARKET_SYMBOL:
        return ""
    return f"_{normalized.lower()}"


def master_table_name(symbol: Optional[str] = DEFAULT_MARKET_SYMBOL) -> str:
    return f"ohlc_1m{_symbol_suffix(symbol)}"


def timeframe_view_name(timeframe_minutes: int, symbol: Optional[str] = DEFAULT_MARKET_SYMBOL) -> str:
    tf = int(max(timeframe_minutes, 1))
    return f"ohlc_{tf}m{_symbol_suffix(symbol)}"


def timeframe_cache_table_name(timeframe_minutes: int, symbol: Optional[str] = DEFAULT_MARKET_SYMBOL) -> str:
    tf = int(max(timeframe_minutes, 1))
    return f"ohlc_{tf}m_cache{_symbol_suffix(symbol)}"


def _aggregate_recent_minutes(df: pd.DataFrame, timeframe_minutes: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

    tf = int(max(timeframe_minutes, 1))
    out = df.copy()
    out["DATE"] = pd.to_datetime(out["DATE"], errors="coerce")
    out = out.dropna(subset=["DATE"]).sort_values("DATE")
    if out.empty:
        return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])

    out = out.set_index("DATE")
    grouped = out.resample(f"{tf}min", origin="epoch").agg(
        {
            "OPEN": "first",
            "HIGH": "max",
            "LOW": "min",
            "CLOSE": "last",
            "VOLUME": "sum",
        }
    )
    grouped = grouped.dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"]).reset_index()
    return grouped[["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]


def _merge_cached_history_with_live_tail(
    conn: sqlite3.Connection,
    source_table: str,
    cache_table: str,
    timeframe_minutes: int,
    latest_records: int,
    start_date: Optional[str],
    end_date: Optional[str],
) -> pd.DataFrame:
    """Use the materialized cache for history and rebuild its mutable tail."""
    cache_max_row = conn.execute(f"SELECT MAX(DATE) FROM {cache_table}").fetchone()
    cache_boundary = str((cache_max_row or [None])[0] or "").strip()
    if not cache_boundary:
        return pd.DataFrame()

    cache_where = ["DATE < ?"]
    cache_params: list[object] = [cache_boundary]
    if start_date:
        cache_where.append("DATE >= ?")
        cache_params.append(start_date)
    if end_date:
        cache_where.append("DATE <= ?")
        cache_params.append(end_date)
    cached = pd.read_sql_query(
        f"""
        SELECT DATE, OPEN, HIGH, LOW, CLOSE, VOLUME
        FROM {cache_table}
        WHERE {' AND '.join(cache_where)}
        ORDER BY DATE DESC
        LIMIT ?
        """,
        conn,
        params=cache_params + [int(max(latest_records, 1))],
    )

    live_where = ["DATE >= ?"]
    live_params: list[object] = [cache_boundary]
    if end_date:
        live_where.append("DATE <= ?")
        live_params.append(end_date)
    live_minutes = pd.read_sql_query(
        f"""
        SELECT DATE, OPEN, HIGH, LOW, CLOSE, VOLUME
        FROM {source_table}
        WHERE {' AND '.join(live_where)}
        ORDER BY DATE
        """,
        conn,
        params=live_params,
    )
    live = _aggregate_recent_minutes(live_minutes, timeframe_minutes)

    combined = pd.concat([cached, live], ignore_index=True)
    if combined.empty:
        return combined
    combined["DATE"] = pd.to_datetime(combined["DATE"], errors="coerce")
    combined = combined.dropna(subset=["DATE"])
    if start_date:
        combined = combined[combined["DATE"] >= pd.to_datetime(start_date)]
    if end_date:
        combined = combined[combined["DATE"] <= pd.to_datetime(end_date)]
    return (
        combined.sort_values("DATE")
        .drop_duplicates(subset=["DATE"], keep="last")
        .tail(int(max(latest_records, 1)))
        .reset_index(drop=True)
    )


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_market_db(db_path: str, symbol: str = DEFAULT_MARKET_SYMBOL) -> None:
    table_name = master_table_name(symbol)
    idx_name = f"idx_{table_name}_date"
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_symbols (
                symbol TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                DATE TEXT PRIMARY KEY,
                OPEN REAL,
                HIGH REAL,
                LOW REAL,
                CLOSE REAL,
                VOLUME REAL DEFAULT 0.0
            )
            """
        )
        conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name}(DATE)")
        conn.execute(
            """
            INSERT INTO market_symbols(symbol, table_name)
            VALUES (?, ?)
            ON CONFLICT(symbol) DO UPDATE SET table_name=excluded.table_name
            """,
            (normalize_market_symbol(symbol), table_name),
        )
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


def upsert_ohlc_1m(db_path: str, df: pd.DataFrame, symbol: str = DEFAULT_MARKET_SYMBOL) -> int:
    clean = normalize_ohlc_df(df)
    if clean.empty:
        return 0

    table_name = master_table_name(symbol)
    init_market_db(db_path, symbol=symbol)
    conn = _connect(db_path)
    try:
        rows = list(clean.itertuples(index=False, name=None))
        conn.executemany(
            f"""
            INSERT INTO {table_name} (DATE, OPEN, HIGH, LOW, CLOSE, VOLUME)
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


def import_csv_to_db(csv_path: str, db_path: str, symbol: str = DEFAULT_MARKET_SYMBOL) -> int:
    df = pd.read_csv(csv_path)
    return upsert_ohlc_1m(db_path, df, symbol=symbol)


def get_latest_date(db_path: str, symbol: str = DEFAULT_MARKET_SYMBOL) -> Optional[str]:
    if not os.path.exists(db_path):
        return None
    table_name = master_table_name(symbol)
    init_market_db(db_path, symbol=symbol)
    conn = _connect(db_path)
    try:
        row = conn.execute(f"SELECT MAX(DATE) FROM {table_name}").fetchone()
        return str(row[0]) if row and row[0] else None
    finally:
        conn.close()


def get_row_count(
    db_path: str,
    table: Optional[str] = None,
    symbol: str = DEFAULT_MARKET_SYMBOL,
) -> int:
    if not os.path.exists(db_path):
        return 0
    table_name = str(table or "").strip() or master_table_name(symbol)
    init_market_db(db_path, symbol=symbol)
    conn = _connect(db_path)
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


def create_timeframe_view(
    db_path: str,
    timeframe_minutes: int,
    symbol: str = DEFAULT_MARKET_SYMBOL,
) -> str:
    """Create or replace a grouped OHLC SQL view from symbol-scoped minute candles."""
    tf = int(max(timeframe_minutes, 1))
    if tf <= 1:
        init_market_db(db_path, symbol=symbol)
        return master_table_name(symbol)
    view_name = timeframe_view_name(tf, symbol=symbol)
    table_1m = master_table_name(symbol)
    init_market_db(db_path, symbol=symbol)
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
                FROM {table_1m}
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


def create_timeframe_cache_table(
    db_path: str,
    timeframe_minutes: int,
    symbol: str = DEFAULT_MARKET_SYMBOL,
) -> str:
    """Create or refresh a materialized grouped OHLC table from symbol-scoped minute candles."""
    tf = int(max(timeframe_minutes, 1))
    table_name = timeframe_cache_table_name(tf, symbol=symbol)
    source_table = master_table_name(symbol)
    init_market_db(db_path, symbol=symbol)
    conn = _connect(db_path)
    try:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(
            f"""
            CREATE TABLE {table_name} AS
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
                FROM {source_table}
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
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(DATE)")
        conn.commit()
        return table_name
    finally:
        conn.close()


def create_timeframe_views(
    db_path: str,
    timeframes_minutes: list[int],
    include_cache_tables: bool = False,
    symbol: str = DEFAULT_MARKET_SYMBOL,
) -> list[str]:
    created = []
    for tf in timeframes_minutes:
        created.append(create_timeframe_view(db_path, int(tf), symbol=symbol))
        if include_cache_tables:
            create_timeframe_cache_table(db_path, int(tf), symbol=symbol)
    return created


def ensure_timeframe_view(
    db_path: str,
    timeframe_minutes: int,
    symbol: str = DEFAULT_MARKET_SYMBOL,
) -> str:
    tf = int(max(timeframe_minutes, 1))
    if tf <= 1:
        init_market_db(db_path, symbol=symbol)
        return master_table_name(symbol)
    view_name = timeframe_view_name(tf, symbol=symbol)
    source_table = master_table_name(symbol)
    init_market_db(db_path, symbol=symbol)
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='view' AND name=?",
            (view_name,),
        ).fetchone()
        if row:
            return view_name

        conn.execute(
            f"""
            CREATE VIEW IF NOT EXISTS {view_name} AS
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
                FROM {source_table}
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


def ensure_timeframe_cache_table(
    db_path: str,
    timeframe_minutes: int,
    symbol: str = DEFAULT_MARKET_SYMBOL,
) -> str:
    tf = int(max(timeframe_minutes, 1))
    table_name = timeframe_cache_table_name(tf, symbol=symbol)
    source_table = master_table_name(symbol)
    init_market_db(db_path, symbol=symbol)
    conn = _connect(db_path)
    try:
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
        if row:
            conn.commit()
            return table_name

        conn.execute(
            f"""
            CREATE TABLE {table_name} AS
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
                FROM {source_table}
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
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(DATE)")
        conn.commit()
        return table_name
    finally:
        conn.close()


def ensure_timeframe_artifacts(
    db_path: str,
    timeframes_minutes: list[int],
    symbol: str = DEFAULT_MARKET_SYMBOL,
) -> list[str]:
    created = []
    for tf in timeframes_minutes:
        ensure_timeframe_view(db_path, int(tf), symbol=symbol)
        created.append(ensure_timeframe_cache_table(db_path, int(tf), symbol=symbol))
    return created


def query_ohlc(
    db_path: str,
    timeframe_minutes: int = 1,
    latest_records: int = 50000,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    symbol: str = DEFAULT_MARKET_SYMBOL,
) -> pd.DataFrame:
    source_table = master_table_name(symbol)
    init_market_db(db_path, symbol=symbol)
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
                FROM {source_table}
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
        table_name = timeframe_cache_table_name(tf, symbol=symbol)

        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()

        if row:
            df = _merge_cached_history_with_live_tail(
                conn=conn,
                source_table=source_table,
                cache_table=table_name,
                timeframe_minutes=tf,
                latest_records=latest_records,
                start_date=start_date,
                end_date=end_date,
            )
        else:
            # Avoid write-time cache builds and expensive grouped SQL over the
            # full minute table. Read only the recent minute slice via the DATE
            # index, then aggregate in pandas.
            minute_limit = max(int(max(latest_records, 1)) * tf + tf, tf * 4)
            sql = f"""
                SELECT DATE, OPEN, HIGH, LOW, CLOSE, VOLUME
                FROM {source_table}
                WHERE {where_sql}
                ORDER BY DATE DESC
                LIMIT ?
            """
            recent_df = pd.read_sql_query(sql, conn, params=params + [int(minute_limit)])
            df = _aggregate_recent_minutes(recent_df, tf).tail(int(max(latest_records, 1))).reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=["DATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
        df = df.sort_values("DATE").reset_index(drop=True)
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        return df
    finally:
        conn.close()
