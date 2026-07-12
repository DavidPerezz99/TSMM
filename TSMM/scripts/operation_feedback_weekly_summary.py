"""Build a weekly summary from operation feedback logs.

Scans daily feedback JSONL files, aggregates operation outcomes, highlights
frequent good/bad execution patterns, and reports model drift indicators.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.operation_feedback_store import resolve_operation_feedback_root
from utils.runtime_scope import resolve_runtime_dir


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _parse_utc(raw_value: Any) -> datetime | None:
    raw = str(raw_value or "").strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except Exception:
            continue
    return None


def _safe_float(raw_value: Any) -> float | None:
    try:
        if raw_value is None:
            return None
        return float(raw_value)
    except Exception:
        return None


def _load_trading_cfg(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _window_range(args: argparse.Namespace) -> Tuple[datetime, datetime]:
    if args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            raise ValueError("--start-date and --end-date must be supplied together")
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        if end_date < start_date:
            raise ValueError("--end-date must be >= --start-date")
        start_dt = start_date.replace(hour=0, minute=0, second=0)
        end_dt = end_date.replace(hour=23, minute=59, second=59)
        return start_dt, end_dt

    days = max(int(args.days or 7), 1)
    end_dt = datetime.utcnow().replace(microsecond=0)
    start_dt = (end_dt - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0)
    return start_dt, end_dt


def _iter_feedback_files(feedback_root: Path) -> Iterable[Path]:
    daily_root = feedback_root / "daily"
    if not daily_root.exists():
        return []
    return sorted(daily_root.glob("**/operations_*.jsonl"))


def _load_events(feedback_root: Path, start_dt: datetime, end_dt: datetime) -> Tuple[List[Dict[str, Any]], List[str]]:
    events: List[Dict[str, Any]] = []
    files_scanned: List[str] = []

    for file_path in _iter_feedback_files(feedback_root):
        files_scanned.append(str(file_path))
        try:
            with file_path.open("r", encoding="utf-8", errors="ignore") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except Exception:
                        continue
                    ts = _parse_utc(payload.get("timestamp_utc"))
                    if ts is None:
                        continue
                    if ts < start_dt or ts > end_dt:
                        continue
                    events.append(payload)
        except Exception:
            continue

    events.sort(key=lambda e: str(e.get("timestamp_utc") or ""))
    return events, files_scanned


def _event_outcome_score(event: Dict[str, Any]) -> int:
    outcome = str(event.get("outcome_label") or "").strip().lower()
    if outcome == "good":
        return 1
    if outcome == "bad":
        return -1
    return 0


def _close_reason_family(event: Dict[str, Any]) -> str:
    raw = str(event.get("close_reason_family") or "").strip().lower()
    if raw:
        return raw
    exec_snapshot = (event.get("execution_snapshot") or {}) if isinstance(event, dict) else {}
    maybe = str(exec_snapshot.get("close_outcome_reason") or "").strip().lower()
    return maybe


def _terminal_operation_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_job: Dict[str, Dict[str, Any]] = {}
    terminal_statuses = {"closed", "completed", "failed", "stopped"}
    for event in events:
        job_id = str(event.get("job_id") or "").strip()
        status = str(event.get("status") or "").strip().lower()
        if not job_id or status not in terminal_statuses:
            continue
        current = by_job.get(job_id)
        if current is None or str(event.get("timestamp_utc") or "") > str(current.get("timestamp_utc") or ""):
            by_job[job_id] = event

    return list(by_job.values())


def _pattern_key(event: Dict[str, Any]) -> Tuple[str, str, str, bool, str]:
    perf = (event.get("performance_snapshot") or {}) if isinstance(event, dict) else {}
    exec_snapshot = (event.get("execution_snapshot") or {}) if isinstance(event, dict) else {}
    metadata = (event.get("metadata") or {}) if isinstance(event, dict) else {}

    decision = str(perf.get("decision") or "").strip().lower() or "n/a"
    model = str(perf.get("model_name") or "").strip().lower() or "n/a"
    submission_mode = str(exec_snapshot.get("order_submission_mode") or "").strip().lower() or "n/a"
    fallback_used = bool(metadata.get("fallback_attempts") or 0)
    close_reason = _close_reason_family(event) or "n/a"
    return decision, model, submission_mode, fallback_used, close_reason


def _patterns_summary(events: List[Dict[str, Any]], min_count: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str, bool, str], Dict[str, Any]] = {}

    for event in events:
        key = _pattern_key(event)
        group = groups.setdefault(
            key,
            {
                "decision": key[0],
                "model": key[1],
                "submission_mode": key[2],
                "fallback_used": key[3],
                "close_reason_family": key[4],
                "count": 0,
                "good": 0,
                "bad": 0,
                "neutral": 0,
                "pending": 0,
                "profit_sum": 0.0,
                "profit_count": 0,
            },
        )
        group["count"] += 1

        outcome = str(event.get("outcome_label") or "pending").strip().lower()
        if outcome in {"good", "bad", "neutral", "pending"}:
            group[outcome] += 1
        else:
            group["pending"] += 1

        exec_snapshot = (event.get("execution_snapshot") or {}) if isinstance(event, dict) else {}
        profit = _safe_float(exec_snapshot.get("close_outcome_profit"))
        if profit is None:
            profit = _safe_float(exec_snapshot.get("position_profit"))
        if profit is not None:
            group["profit_sum"] += float(profit)
            group["profit_count"] += 1

    rows: List[Dict[str, Any]] = []
    for group in groups.values():
        if int(group.get("count", 0)) < max(min_count, 1):
            continue
        count = float(group["count"])
        row = {
            **group,
            "good_rate": round(float(group["good"]) / count, 4),
            "bad_rate": round(float(group["bad"]) / count, 4),
            "avg_profit": round(float(group["profit_sum"]) / float(group["profit_count"]), 6) if int(group["profit_count"]) > 0 else None,
        }
        rows.append(row)

    good_patterns = sorted(rows, key=lambda x: (x["good_rate"], x["count"]), reverse=True)[:15]
    bad_patterns = sorted(rows, key=lambda x: (x["bad_rate"], x["count"]), reverse=True)[:15]
    return good_patterns, bad_patterns


def _avg(values: List[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _metric_avg(rows: List[Dict[str, Any]], key: str) -> float | None:
    values: List[float] = []
    for row in rows:
        metric = _safe_float(row.get(key))
        if metric is not None:
            values.append(metric)
    return _avg(values)


def _drift_summary(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for event in events:
        perf = (event.get("performance_snapshot") or {}) if isinstance(event, dict) else {}
        model_name = str(perf.get("model_name") or "").strip()
        if not model_name:
            continue
        row = {
            "timestamp_utc": str(event.get("timestamp_utc") or ""),
            "confidence": perf.get("confidence"),
            "cm_accuracy": perf.get("cm_accuracy"),
            "success_probability": perf.get("success_probability"),
            "input_fooling_risk": perf.get("input_fooling_risk"),
            "outcome_score": _event_outcome_score(event),
        }
        by_model[model_name].append(row)

    drift_rows: List[Dict[str, Any]] = []
    for model_name, rows in by_model.items():
        rows = sorted(rows, key=lambda x: x["timestamp_utc"])
        if len(rows) < 4:
            continue

        split_idx = max(len(rows) // 2, 1)
        first = rows[:split_idx]
        second = rows[split_idx:]

        first_metrics = {
            "confidence": _metric_avg(first, "confidence"),
            "cm_accuracy": _metric_avg(first, "cm_accuracy"),
            "success_probability": _metric_avg(first, "success_probability"),
            "input_fooling_risk": _metric_avg(first, "input_fooling_risk"),
            "outcome_score": _metric_avg(first, "outcome_score"),
        }
        second_metrics = {
            "confidence": _metric_avg(second, "confidence"),
            "cm_accuracy": _metric_avg(second, "cm_accuracy"),
            "success_probability": _metric_avg(second, "success_probability"),
            "input_fooling_risk": _metric_avg(second, "input_fooling_risk"),
            "outcome_score": _metric_avg(second, "outcome_score"),
        }

        deltas: Dict[str, float | None] = {}
        for key in first_metrics.keys():
            first_val = first_metrics.get(key)
            second_val = second_metrics.get(key)
            if first_val is None or second_val is None:
                deltas[key] = None
            else:
                deltas[key] = round(float(second_val) - float(first_val), 6)

        reasons: List[str] = []
        if deltas.get("confidence") is not None and float(deltas["confidence"] or 0.0) <= -0.05:
            reasons.append("confidence_down")
        if deltas.get("cm_accuracy") is not None and float(deltas["cm_accuracy"] or 0.0) <= -0.05:
            reasons.append("cm_accuracy_down")
        if deltas.get("success_probability") is not None and float(deltas["success_probability"] or 0.0) <= -0.05:
            reasons.append("success_probability_down")
        if deltas.get("input_fooling_risk") is not None and float(deltas["input_fooling_risk"] or 0.0) >= 0.05:
            reasons.append("input_fooling_risk_up")
        if deltas.get("outcome_score") is not None and float(deltas["outcome_score"] or 0.0) <= -0.2:
            reasons.append("realized_outcome_score_down")

        drift_rows.append(
            {
                "model_name": model_name,
                "sample_count": len(rows),
                "first_half": first_metrics,
                "second_half": second_metrics,
                "deltas": deltas,
                "drift_alert": bool(reasons),
                "drift_reasons": reasons,
            }
        )

    return sorted(drift_rows, key=lambda x: (bool(x.get("drift_alert", False)), int(x.get("sample_count", 0))), reverse=True)


def _markdown_summary(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Operation Feedback Weekly Summary")
    lines.append("")
    lines.append(f"- Window start (UTC): {summary.get('window_start_utc')}")
    lines.append(f"- Window end (UTC): {summary.get('window_end_utc')}")
    lines.append(f"- Total feedback events: {summary.get('total_events')}")
    lines.append(f"- Total terminal operations: {summary.get('total_terminal_operations')}")
    lines.append("")

    outcome_counts = summary.get("outcome_counts") or {}
    lines.append("## Outcome Counts")
    lines.append(f"- good: {outcome_counts.get('good', 0)}")
    lines.append(f"- bad: {outcome_counts.get('bad', 0)}")
    lines.append(f"- neutral: {outcome_counts.get('neutral', 0)}")
    lines.append(f"- pending: {outcome_counts.get('pending', 0)}")
    lines.append("")

    lines.append("## Top Good Patterns")
    for row in (summary.get("good_patterns") or [])[:10]:
        lines.append(
            "- decision={decision}, model={model}, mode={submission_mode}, fallback={fallback_used}, close_reason={close_reason_family}, count={count}, good_rate={good_rate}, avg_profit={avg_profit}".format(
                **row
            )
        )
    if not summary.get("good_patterns"):
        lines.append("- none")
    lines.append("")

    lines.append("## Top Bad Patterns")
    for row in (summary.get("bad_patterns") or [])[:10]:
        lines.append(
            "- decision={decision}, model={model}, mode={submission_mode}, fallback={fallback_used}, close_reason={close_reason_family}, count={count}, bad_rate={bad_rate}, avg_profit={avg_profit}".format(
                **row
            )
        )
    if not summary.get("bad_patterns"):
        lines.append("- none")
    lines.append("")

    lines.append("## Model Drift Signals")
    for row in (summary.get("model_drift") or [])[:20]:
        lines.append(
            f"- model={row.get('model_name')}, samples={row.get('sample_count')}, drift_alert={row.get('drift_alert')}, reasons={','.join(row.get('drift_reasons') or []) or '-'}"
        )
    if not summary.get("model_drift"):
        lines.append("- none")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize weekly operation feedback logs")
    parser.add_argument("--trading-config", default="config/trading_agent.yaml", help="Trading config path used to resolve runtime scope")
    parser.add_argument("--days", type=int, default=7, help="Number of days to include when no explicit date range is provided")
    parser.add_argument("--start-date", default="", help="Start date (UTC) in YYYY-MM-DD")
    parser.add_argument("--end-date", default="", help="End date (UTC) in YYYY-MM-DD")
    parser.add_argument("--min-pattern-count", type=int, default=2, help="Minimum occurrences for a pattern to be listed")
    parser.add_argument("--output", default="", help="Output JSON path. Defaults under operation feedback weekly folder.")
    parser.add_argument("--markdown-output", default="", help="Optional output markdown path. Defaults to JSON path with .md suffix.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        start_dt, end_dt = _window_range(args)
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2

    trading_cfg_path = ROOT / str(args.trading_config)
    trading_cfg = _load_trading_cfg(trading_cfg_path)
    runtime_dir = resolve_runtime_dir(output_dir=str(ROOT / "reports"), trading_cfg=trading_cfg, base_dir=ROOT)
    feedback_root = resolve_operation_feedback_root(output_dir=str(ROOT / "reports"), trading_cfg=trading_cfg)

    events, files_scanned = _load_events(feedback_root, start_dt, end_dt)
    terminal_events = _terminal_operation_events(events)

    outcome_counts = {"good": 0, "bad": 0, "neutral": 0, "pending": 0}
    for event in terminal_events:
        outcome = str(event.get("outcome_label") or "pending").strip().lower()
        if outcome not in outcome_counts:
            outcome = "pending"
        outcome_counts[outcome] += 1

    good_patterns, bad_patterns = _patterns_summary(terminal_events, min_count=max(int(args.min_pattern_count or 1), 1))
    model_drift = _drift_summary(events)

    summary = {
        "ok": True,
        "generated_at_utc": _iso(datetime.utcnow()),
        "window_start_utc": _iso(start_dt),
        "window_end_utc": _iso(end_dt),
        "runtime_dir": str(runtime_dir),
        "feedback_root": str(feedback_root),
        "files_scanned": files_scanned,
        "total_events": len(events),
        "total_terminal_operations": len(terminal_events),
        "outcome_counts": outcome_counts,
        "good_patterns": good_patterns,
        "bad_patterns": bad_patterns,
        "model_drift": model_drift,
    }

    if str(args.output or "").strip():
        output_json = Path(str(args.output)).resolve()
    else:
        output_json = feedback_root / "weekly" / f"operation_feedback_weekly_{end_dt.strftime('%Y%m%d')}.json"

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if str(args.markdown_output or "").strip():
        output_md = Path(str(args.markdown_output)).resolve()
    else:
        output_md = output_json.with_suffix(".md")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(_markdown_summary(summary), encoding="utf-8")

    print(json.dumps({"ok": True, "output_json": str(output_json), "output_markdown": str(output_md), "total_events": len(events)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
