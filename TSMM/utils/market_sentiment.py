"""Market sentiment aggregation utilities for Agent A signal analysis.

Sources supported:
- Yahoo Finance RSS (no key required)
- Reddit RSS search (no key required)
- TradingView technicals page (no key required)
- X/Twitter recent search (requires bearer token)
"""

from __future__ import annotations

from typing import Any, Dict, List
import re

import requests


_POS_WORDS = {
    "beat", "beats", "bull", "bullish", "buy", "rally", "surge", "gain", "gains",
    "up", "strong", "optimism", "positive", "record", "growth", "improve", "improves",
}

_NEG_WORDS = {
    "miss", "misses", "bear", "bearish", "sell", "drop", "loss", "losses", "down",
    "weak", "fear", "negative", "recession", "crash", "risk", "decline", "declines",
}


def _score_text(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    toks = [t.strip(".,:;!?()[]{}\"'`).").lower() for t in text.split()]
    pos = sum(1 for t in toks if t in _POS_WORDS)
    neg = sum(1 for t in toks if t in _NEG_WORDS)
    total = max(pos + neg, 1)
    return float((pos - neg) / total)


def _bucket(score: float) -> str:
    if score > 0.2:
        return "positive"
    if score < -0.2:
        return "negative"
    return "neutral"


def _safe_get(url: str, timeout: int = 12, headers: Dict[str, str] | None = None) -> requests.Response | None:
    try:
        return requests.get(url, timeout=timeout, headers=headers)
    except Exception:
        return None


def _extract_rss_titles(xml_text: str, max_items: int) -> List[str]:
    if not xml_text:
        return []
    titles: List[str] = []
    start = 0
    while len(titles) < max_items:
        a = xml_text.find("<title>", start)
        if a < 0:
            break
        b = xml_text.find("</title>", a + 7)
        if b < 0:
            break
        title = xml_text[a + 7:b].strip()
        start = b + 8
        # Ignore channel title and blank values
        if title and "rss" not in title.lower() and "feed" not in title.lower():
            titles.append(title)
    return titles


def fetch_yahoo_finance_sentiment(symbol: str, max_items: int = 12) -> Dict[str, Any]:
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    r = _safe_get(url)
    if r is None or r.status_code >= 400:
        return {"ok": False, "source": "yahoo_finance", "error": "request_failed", "items": []}

    titles = _extract_rss_titles(r.text, max_items=max_items)
    scored = [{"title": t, "score": _score_text(t)} for t in titles]
    avg = float(sum(i["score"] for i in scored) / max(len(scored), 1))
    return {
        "ok": True,
        "source": "yahoo_finance",
        "items": scored,
        "count": len(scored),
        "avg_score": avg,
        "sentiment": _bucket(avg),
    }


def fetch_reddit_sentiment(query: str, max_items: int = 12) -> Dict[str, Any]:
    # Reddit RSS search endpoint; may be rate limited for anonymous traffic.
    q = (query or "gold").strip().replace(" ", "+")
    url = f"https://www.reddit.com/search.rss?q={q}&sort=new"
    r = _safe_get(url, headers={"User-Agent": "tsmm-agent/1.0"})
    if r is None or r.status_code >= 400:
        return {"ok": False, "source": "reddit", "error": "request_failed", "items": []}

    titles = _extract_rss_titles(r.text, max_items=max_items)
    scored = [{"title": t, "score": _score_text(t)} for t in titles]
    avg = float(sum(i["score"] for i in scored) / max(len(scored), 1))
    return {
        "ok": True,
        "source": "reddit",
        "items": scored,
        "count": len(scored),
        "avg_score": avg,
        "sentiment": _bucket(avg),
    }


def fetch_tradingview_sentiment(symbol: str, max_items: int = 12) -> Dict[str, Any]:
    sym = str(symbol or "XAUUSD").strip().upper().replace("=", "")
    url = f"https://www.tradingview.com/symbols/{sym}/technicals/"
    r = _safe_get(url, headers={"User-Agent": "tsmm-agent/1.0"})
    if r is None or r.status_code >= 400:
        return {"ok": False, "source": "tradingview", "error": "request_failed", "items": []}

    text = r.text or ""
    # Keep a bounded window to avoid scoring huge HTML blobs.
    sample = text[:50000]
    sample = re.sub(r"<[^>]+>", " ", sample)
    sample = re.sub(r"\s+", " ", sample).strip()

    score = _score_text(sample)
    title = f"TradingView technicals snapshot for {sym}"
    items = [{"title": title, "score": score}]

    return {
        "ok": True,
        "source": "tradingview",
        "items": items[:max(max_items, 1)],
        "count": 1,
        "avg_score": float(score),
        "sentiment": _bucket(score),
    }


def fetch_x_sentiment(query: str, bearer_token: str, max_items: int = 12) -> Dict[str, Any]:
    if not bearer_token:
        return {"ok": False, "source": "x", "error": "missing_bearer_token", "items": []}

    url = "https://api.x.com/2/tweets/search/recent"
    params = {
        "query": query or "gold OR xauusd OR fed",
        "max_results": str(min(max(max_items, 10), 100)),
        "tweet.fields": "created_at,lang",
    }
    headers = {"Authorization": f"Bearer {bearer_token}"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=12)
        if r.status_code >= 400:
            return {"ok": False, "source": "x", "error": f"status_{r.status_code}", "items": []}
        payload = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        tweets = payload.get("data") or []
        texts = [str(t.get("text", "")) for t in tweets if isinstance(t, dict)]
        scored = [{"title": t[:220], "score": _score_text(t)} for t in texts[:max_items]]
        avg = float(sum(i["score"] for i in scored) / max(len(scored), 1))
        return {
            "ok": True,
            "source": "x",
            "items": scored,
            "count": len(scored),
            "avg_score": avg,
            "sentiment": _bucket(avg),
        }
    except Exception as e:
        return {"ok": False, "source": "x", "error": str(e), "items": []}


def aggregate_market_sentiment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = cfg or {}
    symbol = str(cfg.get("symbol", "GC=F")).strip()  # Yahoo ticker for gold by default
    tradingview_symbol = str(cfg.get("tradingview_symbol", "XAUUSD")).strip()
    reddit_query = str(cfg.get("reddit_query", "xauusd gold market")).strip()
    x_query = str(cfg.get("x_query", "xauusd OR gold OR fed OR inflation")).strip()
    max_items = int(cfg.get("max_items_per_source", 12) or 12)

    sources: List[Dict[str, Any]] = []

    if bool(cfg.get("enable_yahoo", True)):
        sources.append(fetch_yahoo_finance_sentiment(symbol=symbol, max_items=max_items))

    if bool(cfg.get("enable_reddit", True)):
        sources.append(fetch_reddit_sentiment(query=reddit_query, max_items=max_items))

    if bool(cfg.get("enable_tradingview", False)):
        sources.append(fetch_tradingview_sentiment(symbol=tradingview_symbol, max_items=max_items))

    if bool(cfg.get("enable_x", False)):
        env_key = str(cfg.get("x_bearer_env", "X_BEARER_TOKEN")).strip()
        import os

        bearer = os.environ.get(env_key, "")
        sources.append(fetch_x_sentiment(query=x_query, bearer_token=bearer, max_items=max_items))

    ok_scores = [float(s.get("avg_score", 0.0)) for s in sources if bool(s.get("ok", False))]
    total_score = float(sum(ok_scores) / max(len(ok_scores), 1)) if ok_scores else 0.0

    return {
        "enabled": True,
        "sources": sources,
        "aggregate": {
            "score": total_score,
            "sentiment": _bucket(total_score),
            "sources_ok": int(sum(1 for s in sources if bool(s.get("ok", False)))),
            "sources_total": int(len(sources)),
        },
    }
