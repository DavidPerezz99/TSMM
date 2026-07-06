"""TradeMemory — persistent record of completed trades for TSMM.

Records each trade's signal composition and outcome, then surfaces
win-rate statistics for similar setups so the conviction system can
learn from past experience.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
MEMORY_PATH = ROOT / "reports" / "runtime" / "trade_memory.jsonl"


# ── Schema ──────────────────────────────────────────────────────────────────

TRADE_SCHEMA = {
    "job_id": str,
    "decision": str,              # "buy" | "sell"
    "entry": float,
    "exit": Optional[float],
    "outcome": Optional[str],     # "win" | "loss" | "breakeven" | None
    "pnl": Optional[float],
    "pnl_pct": Optional[float],
    "confidence": float,
    "cm_accuracy": float,
    "success_probability": float,
    "input_fooling_risk": float,
    "signal_interpretation": str,
    "enrichment_consensus": Optional[str],
    "enrichment_score": Optional[float],
    "enrichment_alignment": Optional[str],
    "conviction": Optional[float],
    "risk_mode": Optional[str],
    "avg_model_confidence": Optional[float],
    "top_r2": Optional[float],
    "signal_composition": Optional[Dict[str, int]],
    "held_hours": Optional[float],
    "max_favorable": Optional[float],
    "max_adverse": Optional[float],
    "closed_reason": Optional[str],
    "started_at": Optional[str],
    "ended_at": Optional[str],
}


# ── Core Class ──────────────────────────────────────────────────────────────

class TradeMemory:
    """Load, query and persist trade records from a JSONL file."""

    def __init__(self, path: Path = MEMORY_PATH):
        self.path = path
        self._trades: List[Dict[str, Any]] = []
        self._load()

    # ── persistence ──

    def _load(self) -> None:
        if not self.path.exists():
            self._trades = []
            return
        with self.path.open("r", encoding="utf-8") as f:
            self._trades = [json.loads(line) for line in f if line.strip()]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for t in self._trades:
                f.write(json.dumps(t, default=str) + "\n")

    # ── recording ──

    def record(self, trade: Dict[str, Any]) -> None:
        """Append a trade dict and persist."""
        self._trades.append(trade)
        # Keep only last 200 to bound file size
        if len(self._trades) > 200:
            self._trades = self._trades[-200:]
        self._save()

    def latest_job_id(self) -> Optional[str]:
        """Return the job_id of the most recently recorded trade, or None."""
        if not self._trades:
            return None
        return self._trades[-1].get("job_id")

    # ── query ──

    def all(self) -> List[Dict[str, Any]]:
        return list(self._trades)

    def last_n(self, n: int = 15) -> List[Dict[str, Any]]:
        return self._trades[-n:] if self._trades else []

    def wins(self) -> List[Dict[str, Any]]:
        return [t for t in self._trades if t.get("outcome") == "win"]

    def losses(self) -> List[Dict[str, Any]]:
        return [t for t in self._trades if t.get("outcome") == "loss"]

    def _similarity_score(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        """Naive similarity: compare confidence, enrichment score, signal_interpretation."""
        score = 0.0
        # Confidence closeness
        conf_a = float(a.get("confidence", 0.5) or 0.5)
        conf_b = float(b.get("confidence", 0.5) or 0.5)
        score += 0.3 * max(0, 1.0 - abs(conf_a - conf_b) / 0.2)

        # Enrichment score closeness
        es_a = float(a.get("enrichment_score", 0.0) or 0.0)
        es_b = float(b.get("enrichment_score", 0.0) or 0.0)
        score += 0.3 * max(0, 1.0 - abs(es_a - es_b) / 0.5)

        # Same interpretation
        if a.get("signal_interpretation") == b.get("signal_interpretation"):
            score += 0.2

        # Same alignment
        if a.get("enrichment_alignment") == b.get("enrichment_alignment"):
            score += 0.2

        return score

    def similar_setups(self, trade: Dict[str, Any], top_n: int = 10) -> List[Dict[str, Any]]:
        """Return historical trades most similar to *trade* by signal profile."""
        scored = []
        for t in self._trades:
            if t.get("job_id") == trade.get("job_id"):
                continue
            sim = self._similarity_score(trade, t)
            scored.append((sim, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:top_n] if _ > 0.3]

    def win_rate_for_similar(self, trade: Dict[str, Any], min_samples: int = 3) -> Tuple[Optional[float], int]:
        """Win-rate among historical trades similar to *trade*."""
        similar = self.similar_setups(trade, top_n=20)
        outcomes = [t.get("outcome") for t in similar if t.get("outcome") in ("win", "loss")]
        if len(outcomes) < min_samples:
            return None, len(outcomes)  # not enough data
        win_rate = sum(1 for o in outcomes if o == "win") / len(outcomes)
        return win_rate, len(outcomes)

    # ── analysis ──

    def recent_trend_analysis(self, n: int = 15) -> Dict[str, Any]:
        """Summary statistics over the last *n* trades that have outcomes."""
        recent = [t for t in self._trades[-n:] if t.get("outcome")]
        if not recent:
            return {"n_trades": 0, "message": "No completed trades yet."}

        wins = [t for t in recent if t["outcome"] == "win"]
        losses = [t for t in recent if t["outcome"] == "loss"]
        total_pnl = sum(t.get("pnl", 0.0) or 0.0 for t in recent)

        return {
            "n_trades": len(recent),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(recent), 3) if recent else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_pnl_per_trade": round(total_pnl / len(recent), 2) if recent else 0,
            "avg_confidence": round(sum(t.get("confidence", 0.5) or 0.5 for t in recent) / len(recent), 3),
            "avg_enrichment_score": round(
                sum(float(t.get("enrichment_score", 0.0) or 0.0) for t in recent) / len(recent), 3
            ) if recent else 0,
            "best_trade": max(wins, key=lambda t: t.get("pnl", 0.0) or 0.0).get("pnl") if wins else None,
            "worst_trade": min(losses, key=lambda t: t.get("pnl", 0.0) or 0.0).get("pnl") if losses else None,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def weekly_analysis(self) -> Dict[str, Any]:
        """Analysis for last 7 days (Friday-style report)."""
        week_ago = datetime.now() - timedelta(days=7)
        week_trades = [
            t for t in self._trades
            if t.get("started_at") and datetime.strptime(t["started_at"][:19], "%Y-%m-%d %H:%M:%S") >= week_ago
        ]
        if not week_trades and len(self._trades) > 0:
            week_trades = self._trades[-20:]  # fallback to last 20

        wins = [t for t in week_trades if t.get("outcome") == "win"]
        losses = [t for t in week_trades if t.get("outcome") == "loss"]

        best_setups = sorted(
            [t for t in wins if t.get("pnl")],
            key=lambda x: x["pnl"], reverse=True
        )[:3]
        worst_setups = sorted(
            [t for t in losses if t.get("pnl")],
            key=lambda x: x["pnl"]
        )[:3]

        return {
            "period": "last_7_days",
            "total": len(week_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(week_trades), 3) if week_trades else 0,
            "total_pnl": round(sum(t.get("pnl", 0.0) or 0.0 for t in week_trades), 2),
            "avg_hold_hours": round(
                sum(float(t.get("held_hours", 0.0) or 0.0) for t in week_trades if t.get("held_hours")) / max(len([t for t in week_trades if t.get("held_hours")]), 1), 1
            ),
            "best_setups": [
                {"decision": t.get("decision"), "pnl": t.get("pnl"), "confidence": t.get("confidence"), "reason": t.get("closed_reason")}
                for t in best_setups
            ],
            "worst_setups": [
                {"decision": t.get("decision"), "pnl": t.get("pnl"), "confidence": t.get("confidence"), "reason": t.get("closed_reason")}
                for t in worst_setups
            ],
        }

    # ── builder ──

    @staticmethod
    def from_registry_job(job: Dict[str, Any]) -> Dict[str, Any]:
        """Build a trade record dict from a trading_job_registry entry."""
        plan = job.get("plan", {}) or {}
        pos = job.get("position", {}) or {}
        outcome = job.get("close_outcome", {}) or {}

        pnl = outcome.get("profit")
        pnl = float(pnl) if pnl is not None else None
        entry = plan.get("entry") or pos.get("price_open") or 0.0
        exit_price = outcome.get("price") or outcome.get("close_price")

        held = None
        if job.get("started_at") and job.get("ended_at"):
            try:
                s = datetime.strptime(job["started_at"][:19], "%Y-%m-%d %H:%M:%S")
                e = datetime.strptime(job["ended_at"][:19], "%Y-%m-%d %H:%M:%S")
                held = (e - s).total_seconds() / 3600.0
            except Exception:
                pass

        if pnl is not None and entry and float(entry) > 0:
            pnl_pct = round(pnl / float(entry) * 100, 4)
        else:
            pnl_pct = None

        if pnl is not None:
            outcome_label = "win" if pnl > 0 else ("loss" if pnl < 0 else "breakeven")
        else:
            outcome_label = None

        enrichment = plan.get("enrichment", {}) or {}

        return {
            "job_id": job.get("job_id"),
            "decision": plan.get("decision"),
            "entry": float(entry) if entry else 0.0,
            "exit": float(exit_price) if exit_price else None,
            "outcome": outcome_label,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "confidence": float(plan.get("confidence", 0.5) or 0.5),
            "cm_accuracy": float(plan.get("cm_accuracy", 0.5) or 0.5),
            "success_probability": float(plan.get("success_probability", 0.5) or 0.5),
            "input_fooling_risk": float(plan.get("input_fooling_risk", 0.5) or 0.5),
            "signal_interpretation": str(plan.get("signal_interpretation") or ""),
            "enrichment_consensus": enrichment.get("consensus"),
            "enrichment_score": enrichment.get("consensus_score"),
            "enrichment_alignment": enrichment.get("alignment"),
            "conviction": plan.get("conviction", {}).get("conviction") if isinstance(plan.get("conviction"), dict) else None,
            "risk_mode": plan.get("conviction", {}).get("risk_mode") if isinstance(plan.get("conviction"), dict) else None,
            "avg_model_confidence": enrichment.get("avg_confidence"),
            "top_r2": None,  # populated separately when recording
            "signal_composition": None,  # derived from enrichment signals
            "held_hours": held,
            "max_favorable": None,
            "max_adverse": None,
            "closed_reason": job.get("closed_reason"),
            "started_at": job.get("started_at"),
            "ended_at": job.get("ended_at"),
        }


# ── Convenience helpers ─────────────────────────────────────────────────────

_default_memory: Optional[TradeMemory] = None


def get_memory() -> TradeMemory:
    global _default_memory
    if _default_memory is None:
        _default_memory = TradeMemory()
    return _default_memory


def record_trade(trade_data: Dict[str, Any]) -> None:
    get_memory().record(trade_data)


def recent_analysis(n: int = 15) -> Dict[str, Any]:
    return get_memory().recent_trend_analysis(n)


def weekly_analysis() -> Dict[str, Any]:
    return get_memory().weekly_analysis()
