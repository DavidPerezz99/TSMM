"""Summarize TSMM operation feedback logs into operational learning reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


TERMINAL_STATUSES = {"closed", "completed", "failed", "killed", "stopped"}


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def _stats(values: Iterable[Any]) -> Dict[str, Any]:
    nums = [v for v in (_safe_float(value) for value in values) if v is not None]
    if not nums:
        return {"n": 0}
    return {
        "n": len(nums),
        "avg": round(statistics.mean(nums), 6),
        "median": round(statistics.median(nums), 6),
        "min": round(min(nums), 6),
        "max": round(max(nums), 6),
    }


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8-sig")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            yield {
                "_parse_error": True,
                "path": str(path),
                "line": lineno,
                "error": str(exc),
            }
            continue
        if isinstance(payload, dict):
            yield payload


def load_events(feedback_root: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    daily_root = feedback_root / "daily"
    search_root = daily_root if daily_root.exists() else feedback_root
    for path in sorted(search_root.glob("**/*.jsonl")):
        events.extend(_read_jsonl(path))
    return events


def _latest_by_job(events: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        job_id = str(event.get("job_id") or "").strip()
        if job_id:
            grouped[job_id].append(event)
    latest: Dict[str, Dict[str, Any]] = {}
    for job_id, rows in grouped.items():
        latest[job_id] = sorted(rows, key=lambda item: str(item.get("timestamp_utc") or ""))[-1]
    return latest


def _latest_terminal_by_job(events: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return _latest_by_job(
        event
        for event in events
        if str(event.get("status") or "").strip().lower() in TERMINAL_STATUSES
    )


def build_job_rows(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest = _latest_by_job(event for event in events if not event.get("_parse_error"))
    rows: List[Dict[str, Any]] = []
    for job_id, event in sorted(latest.items()):
        perf = event.get("performance_snapshot") or {}
        exec_snap = event.get("execution_snapshot") or {}
        mode_b = event.get("mode_b_snapshot") or {}
        rows.append(
            {
                "job_id": job_id,
                "profile": event.get("profile"),
                "last_timestamp_utc": event.get("timestamp_utc"),
                "status": event.get("status"),
                "stage": event.get("stage"),
                "outcome_label": event.get("outcome_label"),
                "close_reason_family": event.get("close_reason_family"),
                "model": perf.get("model_name"),
                "decision": perf.get("decision"),
                "confidence": perf.get("confidence"),
                "cm_accuracy": perf.get("cm_accuracy"),
                "success_probability": perf.get("success_probability"),
                "input_fooling_risk": perf.get("input_fooling_risk"),
                "backtest_win_rate": perf.get("backtest_win_rate"),
                "backtest_n_trades": perf.get("backtest_n_trades"),
                "order_submission_mode": exec_snap.get("order_submission_mode"),
                "order_ticket": exec_snap.get("order_ticket"),
                "position_ticket": exec_snap.get("position_ticket"),
                "position_profit": exec_snap.get("position_profit"),
                "close_outcome_profit": exec_snap.get("close_outcome_profit"),
                "agent_b_recommendation": mode_b.get("recommendation"),
                "agent_b_consensus": mode_b.get("consensus"),
                "agent_b_consensus_score": mode_b.get("consensus_score"),
                "agent_b_should_close": mode_b.get("should_close"),
                "agent_b_risk_action": mode_b.get("risk_action"),
            }
        )
    return rows


def summarize_events(events: List[Dict[str, Any]], job_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_events = [event for event in events if not event.get("_parse_error")]
    parse_errors = [event for event in events if event.get("_parse_error")]
    agent_a = [event for event in valid_events if event.get("source") == "agent_a_sample"]
    agent_b = [event for event in valid_events if event.get("source") == "agent_b_sample"]
    terminal_events = _latest_terminal_by_job(valid_events)
    terminal = build_job_rows(list(terminal_events.values()))

    return {
        "event_count": len(valid_events),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:20],
        "unique_jobs": len(job_rows),
        "source_counts": Counter(event.get("source") for event in valid_events).most_common(),
        "event_kind_counts": Counter(event.get("event_kind") for event in valid_events).most_common(),
        "latest_status_counts": Counter(row.get("status") for row in job_rows).most_common(),
        "latest_outcome_counts": Counter(row.get("outcome_label") for row in job_rows).most_common(),
        "terminal_operation_count": len(terminal),
        "terminal_outcome_counts": Counter(row.get("outcome_label") for row in terminal).most_common(),
        "latest_close_reason_counts": Counter(row.get("close_reason_family") for row in job_rows if row.get("close_reason_family")).most_common(),
        "agent_a_samples": len(agent_a),
        "agent_b_samples": len(agent_b),
        "agent_a_metrics": {
            "confidence": _stats((event.get("performance_snapshot") or {}).get("confidence") for event in agent_a),
            "cm_accuracy": _stats((event.get("performance_snapshot") or {}).get("cm_accuracy") for event in agent_a),
            "success_probability": _stats((event.get("performance_snapshot") or {}).get("success_probability") for event in agent_a),
            "input_fooling_risk": _stats((event.get("performance_snapshot") or {}).get("input_fooling_risk") for event in agent_a),
            "backtest_win_rate": _stats((event.get("performance_snapshot") or {}).get("backtest_win_rate") for event in agent_a),
        },
        "agent_b_recommendation_counts": Counter((event.get("mode_b_snapshot") or {}).get("recommendation") for event in agent_b).most_common(),
        "agent_b_risk_action_counts": Counter((event.get("mode_b_snapshot") or {}).get("risk_action") for event in agent_b).most_common(),
        "terminal_metrics": {
            "good_confidence": _stats(row.get("confidence") for row in terminal if row.get("outcome_label") == "good"),
            "bad_confidence": _stats(row.get("confidence") for row in terminal if row.get("outcome_label") == "bad"),
            "good_input_fooling_risk": _stats(row.get("input_fooling_risk") for row in terminal if row.get("outcome_label") == "good"),
            "bad_input_fooling_risk": _stats(row.get("input_fooling_risk") for row in terminal if row.get("outcome_label") == "bad"),
        },
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: Dict[str, Any], csv_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def top(name: str, limit: int = 12) -> str:
        rows = summary.get(name) or []
        if not rows:
            return "- none"
        return "\n".join(f"- {key}: {value}" for key, value in rows[:limit])

    lines = [
        "# TSMM Operation Feedback Summary",
        "",
        f"- Events parsed: {summary.get('event_count', 0)}",
        f"- Parse errors: {summary.get('parse_error_count', 0)}",
        f"- Unique jobs: {summary.get('unique_jobs', 0)}",
        f"- Job CSV: `{csv_path}`",
        "",
        "## Terminal Outcomes",
        top("terminal_outcome_counts"),
        "",
        "## Latest Job States",
        top("latest_outcome_counts"),
        "",
        "## Close Reasons",
        top("latest_close_reason_counts"),
        "",
        "## Agent A Metrics",
    ]
    for key, value in (summary.get("agent_a_metrics") or {}).items():
        lines.append(f"- {key}: `{json.dumps(value, sort_keys=True)}`")
    lines.extend(
        [
            "",
            "## Agent B Recommendations",
            top("agent_b_recommendation_counts"),
            "",
            "## Agent B Risk Actions",
            top("agent_b_risk_action_counts"),
            "",
            "## Event Kinds",
            top("event_kind_counts", limit=20),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback-root", default="reports/runtime/operation_feedback")
    parser.add_argument("--out-dir", default="reports/analysis/operation_feedback")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    feedback_root = Path(args.feedback_root)
    if not feedback_root.exists():
        raise SystemExit(f"feedback root does not exist: {feedback_root}")

    prefix = f"{args.prefix}_" if args.prefix else ""
    out_dir = Path(args.out_dir)
    events = load_events(feedback_root)
    job_rows = build_job_rows(events)
    summary = summarize_events(events, job_rows)

    csv_path = out_dir / f"{prefix}jobs.csv"
    json_path = out_dir / f"{prefix}summary.json"
    md_path = out_dir / f"{prefix}summary.md"
    write_csv(csv_path, job_rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    write_markdown(md_path, summary, csv_path)
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
