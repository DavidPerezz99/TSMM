"""Timed runtime watchdog for TSMM trading runner integrity.

Checks that:
- Telegram listener remains singleton (count == 1)
- Local endpoint service is running (count >= 1)
- Broker parity enforcer is running (count >= 1)
- No duplicate `app.py trading-job resume --job-id ...` processes exist

Writes a JSON summary report and returns non-zero when anomalies are detected.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

import psutil


def _is_python_process(proc: psutil.Process) -> bool:
    try:
        name = str(proc.info.get("name") or "").lower()
    except Exception:
        return False
    return name in {"python", "python.exe"}


def _scan_runtime_state() -> Dict[str, Any]:
    listener_count = 0
    endpoint_count = 0
    parity_count = 0
    job_runners: Dict[str, List[int]] = {}

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if not _is_python_process(proc):
            continue
        try:
            cmdline_parts = [str(item or "") for item in (proc.info.get("cmdline") or [])]
        except Exception:
            continue
        if not cmdline_parts:
            continue

        cmdline_text = " ".join(cmdline_parts).lower()
        if "telegram_command_listener.py" in cmdline_text:
            listener_count += 1
        if "local_signal_endpoint_service.py" in cmdline_text:
            endpoint_count += 1
        if "enforce_broker_parity.py" in cmdline_text:
            parity_count += 1

        if "app.py" not in cmdline_text or "trading-job" not in cmdline_text or "resume" not in cmdline_text:
            continue
        if "--job-id" not in cmdline_parts:
            continue
        idx = cmdline_parts.index("--job-id")
        if idx + 1 >= len(cmdline_parts):
            continue

        job_id = str(cmdline_parts[idx + 1] or "").strip()
        if not job_id:
            continue
        job_runners.setdefault(job_id, []).append(int(proc.info.get("pid") or 0))

    duplicates = {
        job_id: sorted(pid for pid in pids if pid > 0)
        for job_id, pids in job_runners.items()
        if len([pid for pid in pids if pid > 0]) > 1
    }

    return {
        "listener_count": listener_count,
        "endpoint_count": endpoint_count,
        "parity_count": parity_count,
        "job_counts": {job_id: len(pids) for job_id, pids in sorted(job_runners.items())},
        "duplicate_job_runners": duplicates,
    }


def _run_watchdog(duration_sec: int, interval_sec: int) -> Dict[str, Any]:
    started_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    deadline = time.time() + float(duration_sec)
    anomalies: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []

    while time.time() < deadline:
        now_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        state = _scan_runtime_state()

        sample = {
            "ts_utc": now_utc,
            "listener_count": int(state.get("listener_count") or 0),
            "endpoint_count": int(state.get("endpoint_count") or 0),
            "parity_count": int(state.get("parity_count") or 0),
            "duplicate_job_ids": sorted((state.get("duplicate_job_runners") or {}).keys()),
        }
        samples.append(sample)

        if sample["listener_count"] != 1:
            anomalies.append({"ts_utc": now_utc, "kind": "listener_count", "value": sample["listener_count"]})
        if sample["endpoint_count"] < 1:
            anomalies.append({"ts_utc": now_utc, "kind": "endpoint_missing", "value": sample["endpoint_count"]})
        if sample["parity_count"] < 1:
            anomalies.append({"ts_utc": now_utc, "kind": "parity_missing", "value": sample["parity_count"]})
        if sample["duplicate_job_ids"]:
            anomalies.append(
                {
                    "ts_utc": now_utc,
                    "kind": "duplicate_job_runners",
                    "value": state.get("duplicate_job_runners") or {},
                }
            )

        dupes_text = ",".join(sample["duplicate_job_ids"]) if sample["duplicate_job_ids"] else "-"
        print(
            f"[{now_utc}] listener={sample['listener_count']} endpoint={sample['endpoint_count']} "
            f"parity={sample['parity_count']} dupes={dupes_text}",
            flush=True,
        )
        time.sleep(float(interval_sec))

    final_state = _scan_runtime_state()
    ended_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "watchdog_started_utc": started_utc,
        "watchdog_ended_utc": ended_utc,
        "duration_sec": int(duration_sec),
        "interval_sec": int(interval_sec),
        "sample_count": len(samples),
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "final_listener_count": int(final_state.get("listener_count") or 0),
        "final_endpoint_count": int(final_state.get("endpoint_count") or 0),
        "final_parity_count": int(final_state.get("parity_count") or 0),
        "final_job_counts": final_state.get("job_counts") or {},
        "final_duplicate_job_runners": final_state.get("duplicate_job_runners") or {},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TSMM runtime runner integrity watchdog")
    parser.add_argument("--duration-sec", type=int, default=300, help="Watchdog duration in seconds")
    parser.add_argument("--interval-sec", type=int, default=15, help="Sampling interval in seconds")
    parser.add_argument(
        "--output",
        default="reports/runtime/runner_integrity_watchdog_last.json",
        help="Path to JSON output report",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    duration_sec = max(int(args.duration_sec or 0), 30)
    interval_sec = max(int(args.interval_sec or 0), 5)

    report = _run_watchdog(duration_sec=duration_sec, interval_sec=interval_sec)
    print(json.dumps(report, indent=2))

    output_path = Path(str(args.output)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0 if int(report.get("anomaly_count") or 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
