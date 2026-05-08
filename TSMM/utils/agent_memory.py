"""Persistent agent memory and lightweight knowledge-base retrieval.

This module provides:
- SQLite-backed memory records (signals, sentiment, assistance, KB docs)
- Lightweight hashed-vector embeddings for semantic-ish retrieval
- Optional document ingestion for KB context
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return [t for t in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(t) > 1]


def _hash_token(tok: str, dim: int) -> int:
    h = hashlib.md5(tok.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % dim


def _embed_text(text: str, dim: int = 256) -> List[float]:
    v = [0.0] * dim
    toks = _tokenize(text)
    if not toks:
        return v
    for t in toks:
        i = _hash_token(t, dim)
        v[i] += 1.0
    norm = math.sqrt(sum(x * x for x in v))
    if norm <= 1e-12:
        return v
    return [x / norm for x in v]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


class AgentMemoryStore:
    def __init__(self, db_path: str, embedding_dim: int = 256):
        self.db_path = str(db_path)
        self.embedding_dim = int(max(embedding_dim, 64))
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    timeframe TEXT,
                    symbol TEXT,
                    title TEXT,
                    text_payload TEXT NOT NULL,
                    metadata_json TEXT,
                    embedding_json TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_memory_kind ON agent_memory(kind)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_memory_timeframe ON agent_memory(timeframe)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_memory_symbol ON agent_memory(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_memory_created_at ON agent_memory(created_at)")
            conn.commit()
        finally:
            conn.close()

    def add_memory(
        self,
        kind: str,
        text_payload: str,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        txt = str(text_payload or "").strip()
        if not txt:
            return 0

        emb = _embed_text(txt, dim=self.embedding_dim)
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                INSERT INTO agent_memory(created_at, kind, timeframe, symbol, title, text_payload, metadata_json, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_iso(),
                    str(kind or "note"),
                    str(timeframe) if timeframe is not None else None,
                    str(symbol) if symbol is not None else None,
                    str(title) if title is not None else None,
                    txt,
                    json.dumps(metadata or {}, default=str),
                    json.dumps(emb),
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0)
        finally:
            conn.close()

    def ingest_documents(self, paths: List[str], kind: str = "kb_document", symbol: Optional[str] = None) -> Dict[str, Any]:
        out = {"ingested": 0, "skipped": 0, "errors": []}
        for p in paths or []:
            try:
                path = str(p).strip()
                if not path or not os.path.exists(path) or not os.path.isfile(path):
                    out["skipped"] += 1
                    continue
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                title = os.path.basename(path)
                if len(txt) > 120000:
                    txt = txt[:120000] + "\n...<trimmed>"
                rid = self.add_memory(
                    kind=kind,
                    text_payload=txt,
                    symbol=symbol,
                    title=title,
                    metadata={"source_path": path},
                )
                if rid > 0:
                    out["ingested"] += 1
                else:
                    out["skipped"] += 1
            except Exception as e:
                out["errors"].append(f"{p}: {e}")
        return out

    def retrieve(
        self,
        query: str,
        limit: int = 8,
        kinds: Optional[List[str]] = None,
        timeframe: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        q_emb = _embed_text(query, dim=self.embedding_dim)
        conn = self._connect()
        try:
            where = ["1=1"]
            params: List[Any] = []

            if kinds:
                placeholders = ",".join(["?"] * len(kinds))
                where.append(f"kind IN ({placeholders})")
                params.extend([str(k) for k in kinds])
            if timeframe:
                where.append("(timeframe = ? OR timeframe IS NULL)")
                params.append(str(timeframe))
            if symbol:
                where.append("(symbol = ? OR symbol IS NULL)")
                params.append(str(symbol))

            sql = f"""
                SELECT id, created_at, kind, timeframe, symbol, title, text_payload, metadata_json, embedding_json
                FROM agent_memory
                WHERE {' AND '.join(where)}
                ORDER BY id DESC
                LIMIT ?
            """
            params.append(int(max(limit * 20, 50)))
            rows = conn.execute(sql, params).fetchall()

            scored: List[Dict[str, Any]] = []
            for r in rows:
                rid, created_at, kind, tf, sym, title, text_payload, metadata_json, embedding_json = r
                try:
                    emb = json.loads(embedding_json) if embedding_json else []
                except Exception:
                    emb = []
                sim = _cosine(q_emb, emb)
                rec = {
                    "id": int(rid),
                    "created_at": str(created_at),
                    "kind": str(kind),
                    "timeframe": tf,
                    "symbol": sym,
                    "title": title,
                    "text_payload": str(text_payload or ""),
                    "metadata": json.loads(metadata_json) if metadata_json else {},
                    "score": float(sim),
                }
                scored.append(rec)

            scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            return scored[: max(int(limit), 1)]
        finally:
            conn.close()

    def build_context_block(self, query: str, limit: int = 6, **kwargs) -> str:
        hits = self.retrieve(query=query, limit=limit, **kwargs)
        if not hits:
            return ""
        lines: List[str] = []
        for h in hits:
            title = h.get("title") or f"{h.get('kind')}#{h.get('id')}"
            txt = str(h.get("text_payload") or "").strip().replace("\n", " ")
            if len(txt) > 420:
                txt = txt[:420] + "..."
            lines.append(
                f"- [{h.get('kind')}] {title} | score={h.get('score', 0.0):.3f} | at={h.get('created_at')}\n  {txt}"
            )
        return "\n".join(lines)
