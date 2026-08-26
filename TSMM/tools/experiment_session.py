#!/usr/bin/env python
"""Manual, resumable, resource-guarded experiment session runner."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, List, Optional

import psutil
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hypersearch import (  # noqa: E402
    build_sweep_plan,
    generate_factorial_experiments,
    generate_smart_experiments,
    unique_experiments,
)
from utils.experiment_planning import (  # noqa: E402
    duration_signature,
    estimate_experiment_duration_seconds,
    estimate_experiment_memory,
    next_local_deadline,
)
from utils.market_db import (  # noqa: E402
    create_timeframe_views,
    materialize_timeframe_cache_tables,
    master_table_name,
    timeframe_cache_table_name,
)


MANIFEST_VERSION = 1
SUCCESS_STATUS = "SUCCESS"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "session")).strip("_.")
    return cleaned or "session"


def _yaml_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return payload


def _canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temp_path, path)


def _deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Merge a session profile into a sweep without mutating either source."""
    out = deepcopy(base)
    for key, value in (patch or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _iter_entry_configs(base_cfg: Dict[str, Any], sweep_cfg: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    if bool(sweep_cfg.get("smart_generation", True)):
        generated = generate_smart_experiments(base_cfg, sweep_cfg, verbose=False)
    else:
        generated = generate_factorial_experiments(base_cfg, sweep_cfg)
    return unique_experiments(generated)


def _session_paths(session_cfg: Dict[str, Any]) -> tuple[str, Path, Path, Path]:
    name = _slug(str(session_cfg.get("session_name") or "xauusd_nightly"))
    output_root = _resolve_path(session_cfg.get("output_root") or "output/hypersearch_sessions")
    session_dir = output_root / name
    return name, output_root, session_dir, session_dir / "session_manifest.json"


def _entry_source(
    entry: Dict[str, Any],
    session_cfg: Optional[Dict[str, Any]] = None,
) -> tuple[Path, Path, Dict[str, Any], Dict[str, Any]]:
    base_path = _resolve_path(entry.get("base_config") or "config_templates/univariate.yaml")
    sweep_path = _resolve_path(entry.get("sweep_config") or entry.get("param_grid") or "")
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep config not found: {sweep_path}")
    base_cfg = _yaml_load(base_path)
    sweep_cfg = _yaml_load(sweep_path)
    profile_name = str(entry.get("target_profile") or "").strip().upper()
    if profile_name:
        profiles = (session_cfg or {}).get("target_profiles") or {}
        profile = profiles.get(profile_name)
        if not isinstance(profile, dict):
            raise ValueError(
                f"Entry '{entry.get('name') or sweep_path.stem}' references missing "
                f"target_profile '{profile_name}'"
            )
        sweep_cfg = _deep_merge(sweep_cfg, profile)
    overrides = entry.get("sweep_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("sweep_overrides must be a mapping")
    sweep_cfg = _deep_merge(sweep_cfg, overrides)
    return base_path, sweep_path, base_cfg, sweep_cfg


def _single_sweep_value(sweep_cfg: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = sweep_cfg.get(key, default)
    if isinstance(value, list):
        return value[0] if len(value) == 1 else None
    return value


def build_session_plan(session_cfg: Dict[str, Any]) -> Dict[str, Any]:
    resources = session_cfg.get("resources") or {}
    ram_limit_gb = float(resources.get("max_process_ram_gb", 20.0) or 20.0)
    cpu_threads = int(resources.get("cpu_threads_per_experiment", 6) or 6)
    max_duration_minutes = float(
        resources.get("max_estimated_experiment_minutes", 90.0) or 90.0
    )
    entries = list(session_cfg.get("experiments") or [])
    if not entries:
        raise ValueError("Session config must contain a non-empty 'experiments' list")

    planned_entries = []
    total = 0
    for order, entry in enumerate(entries, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"Session experiment #{order} must be a mapping")
        base_path, sweep_path, base_cfg, sweep_cfg = _entry_source(entry, session_cfg)
        sweep_plan = build_sweep_plan(
            base_cfg,
            sweep_cfg,
            ram_limit_gb=ram_limit_gb,
            cpu_threads=cpu_threads,
            max_duration_minutes=max_duration_minutes,
        )
        limit = int(entry.get("max_experiments", 0) or 0)
        count = int(sweep_plan["unique_experiments"])
        if limit and count > limit:
            raise ValueError(
                f"Entry '{entry.get('name') or order}' has {count} experiments, above its limit of {limit}"
            )
        worst = sweep_plan.get("worst_memory_estimate") or {}
        if float(worst.get("estimated_peak_gb", 0.0)) > ram_limit_gb:
            raise ValueError(
                f"Entry '{entry.get('name') or order}' is estimated at "
                f"{worst.get('estimated_peak_gb')} GB, above the {ram_limit_gb} GB limit"
            )
        worst_duration = sweep_plan.get("worst_duration_estimate") or {}
        if float(worst_duration.get("estimated_minutes", 0.0)) > max_duration_minutes:
            raise ValueError(
                f"Entry '{entry.get('name') or order}' is estimated at "
                f"{worst_duration.get('estimated_minutes')} minutes, above the "
                f"{max_duration_minutes} minute per-experiment limit"
            )
        total += count
        planned_entries.append(
            {
                "order": order,
                "name": _slug(str(entry.get("name") or sweep_path.stem)),
                "base_config": str(base_path),
                "base_hash": _file_hash(base_path),
                "sweep_config": str(sweep_path),
                "sweep_hash": _file_hash(sweep_path),
                "effective_sweep_hash": _canonical_hash(sweep_cfg),
                "target_profile": str(entry.get("target_profile") or "").strip().upper() or None,
                "target_col": _single_sweep_value(sweep_cfg, "target_col"),
                "data_timeframe_minutes": _single_sweep_value(
                    sweep_cfg, "data_timeframe_minutes"
                ),
                "max_experiments": limit or None,
                "plan": sweep_plan,
            }
        )

    return {
        "session_name": _slug(str(session_cfg.get("session_name") or "xauusd_nightly")),
        "manual_start_only": bool(session_cfg.get("manual_start_only", True)),
        "deadline_local": str(session_cfg.get("deadline_local") or "05:00"),
        "total_experiments": total,
        "resources": {
            "max_process_ram_gb": ram_limit_gb,
            "min_free_ram_before_start_gb": float(
                resources.get("min_free_ram_before_start_gb", 6.0) or 6.0
            ),
            "critical_free_ram_gb": float(
                resources.get("critical_free_ram_gb", 2.0) or 2.0
            ),
            "cpu_threads_per_experiment": cpu_threads,
            "max_estimated_experiment_minutes": max_duration_minutes,
        },
        "entries": planned_entries,
    }


def _materialize_session(
    session_config_path: Path,
    session_cfg: Dict[str, Any],
    plan: Dict[str, Any],
) -> Dict[str, Any]:
    name, _, session_dir, manifest_path = _session_paths(session_cfg)
    config_hash = _canonical_hash(session_cfg)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("session_config_hash") != config_hash:
            raise RuntimeError(
                f"Session '{name}' already exists with a different definition. "
                "Use a new session_name to preserve reproducible resume behavior."
            )
        for entry in manifest.get("entries") or []:
            if _file_hash(Path(entry["base_config"])) != entry.get("base_hash"):
                raise RuntimeError(f"Base config changed after session creation: {entry['base_config']}")
            if _file_hash(Path(entry["sweep_config"])) != entry.get("sweep_hash"):
                raise RuntimeError(f"Sweep config changed after session creation: {entry['sweep_config']}")
        return manifest

    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    source_entries = list(session_cfg.get("experiments") or [])
    for planned, source_entry in zip(plan["entries"], source_entries):
        entry_dir = session_dir / f"{planned['order']:02d}_{planned['name']}"
        configs_dir = entry_dir / "configs"
        summaries_dir = entry_dir / "summaries"
        logs_dir = entry_dir / "logs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        summaries_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)

        _, _, base_cfg, sweep_cfg = _entry_source(source_entry, session_cfg)
        count = 0
        for count, experiment in enumerate(_iter_entry_configs(base_cfg, sweep_cfg), 1):
            cfg_path = configs_dir / f"cfg_{count:05d}.yaml"
            with cfg_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(experiment, stream, sort_keys=False)
        expected = int(planned["plan"]["unique_experiments"])
        if count != expected:
            raise RuntimeError(
                f"Materialized {count} configs for {planned['name']}, expected {expected}"
            )
        manifest_entries.append(
            {
                **planned,
                "configs_dir": str(configs_dir),
                "summaries_dir": str(summaries_dir),
                "logs_dir": str(logs_dir),
                "config_count": count,
            }
        )

    manifest = {
        "version": MANIFEST_VERSION,
        "session_name": name,
        "created_at": _now_iso(),
        "session_config": str(session_config_path),
        "session_config_hash": config_hash,
        "manual_start_only": True,
        "deadline_local": plan["deadline_local"],
        "worthy_r2_threshold": float(session_cfg.get("worthy_r2_threshold", 0.6)),
        "resources": plan["resources"],
        "total_experiments": plan["total_experiments"],
        "entries": manifest_entries,
    }
    _atomic_json(manifest_path, manifest)
    _atomic_json(
        session_dir / "session_state.json",
        {
            "session_name": name,
            "updated_at": _now_iso(),
            "runs": [],
            "durations_seconds": [],
        },
    )
    return manifest


def _data_source_settings(session_cfg: Dict[str, Any]) -> Dict[str, Any]:
    source = session_cfg.get("data_source") or {}
    timeframes = list(source.get("timeframes_minutes") or [])
    if not timeframes:
        for entry in session_cfg.get("experiments") or []:
            try:
                _, _, _, sweep = _entry_source(entry, session_cfg)
                value = _single_sweep_value(sweep, "data_timeframe_minutes")
                if value:
                    timeframes.append(int(value))
            except Exception:
                continue
    return {
        "db_path": _resolve_path(source.get("db_path") or "data/market_data.sqlite"),
        "symbol": str(source.get("symbol") or "XAUUSD").strip().upper(),
        "timeframes_minutes": list(dict.fromkeys(int(value) for value in timeframes)),
    }


def _cache_inventory(session_cfg: Dict[str, Any]) -> Dict[str, Any]:
    source = _data_source_settings(session_cfg)
    db_path = Path(source["db_path"])
    if not db_path.exists():
        raise FileNotFoundError(f"Market database not found: {db_path}")
    uri = f"file:{db_path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5)
    try:
        master = master_table_name(source["symbol"])
        master_row = conn.execute(
            f"SELECT COUNT(*), MIN(DATE), MAX(DATE) FROM {master}"
        ).fetchone()
        timeframes = {}
        for tf in source["timeframes_minutes"]:
            table = timeframe_cache_table_name(tf, symbol=source["symbol"])
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if exists:
                row = conn.execute(
                    f"SELECT COUNT(*), MIN(DATE), MAX(DATE) FROM {table}"
                ).fetchone()
                timeframes[str(tf)] = {
                    "cache_table": table,
                    "exists": True,
                    "records": int(row[0] or 0),
                    "first": row[1],
                    "latest": row[2],
                }
            else:
                timeframes[str(tf)] = {
                    "cache_table": table,
                    "exists": False,
                    "records": 0,
                    "first": None,
                    "latest": None,
                }
    finally:
        conn.close()
    return {
        "db_path": str(db_path),
        "symbol": source["symbol"],
        "master_records": int(master_row[0] or 0),
        "master_first": master_row[1],
        "master_latest": master_row[2],
        "timeframes": timeframes,
    }


def prepare_session_data(session_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh every configured SQL view and materialized cache in one pass."""
    source = _data_source_settings(session_cfg)
    db_path = str(source["db_path"])
    if not Path(db_path).exists():
        raise FileNotFoundError(f"Market database not found: {db_path}")
    timeframes = source["timeframes_minutes"]
    if not timeframes:
        raise ValueError("No data_source.timeframes_minutes configured")
    create_timeframe_views(
        db_path,
        timeframes,
        include_cache_tables=False,
        symbol=source["symbol"],
    )
    materialize_timeframe_cache_tables(db_path, timeframes, symbol=source["symbol"])
    return _cache_inventory(session_cfg)


def build_capacity_report(
    session_cfg: Dict[str, Any],
    plan: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combine hardware estimates with actual cached history availability."""
    resolved_plan = plan or build_session_plan(session_cfg)
    inventory = _cache_inventory(session_cfg)
    entries = []
    for entry in resolved_plan["entries"]:
        tf = int(entry.get("data_timeframe_minutes") or 0)
        cache = inventory["timeframes"].get(str(tf), {})
        available = int(cache.get("records", 0) or 0)
        ram_limit = int(entry["plan"].get("estimated_max_records_for_worst_shape", 0) or 0)
        duration_limit = int(entry["plan"].get("estimated_max_records_for_duration", 0) or 0)
        positive_limits = [value for value in (available, ram_limit, duration_limit) if value > 0]
        effective = min(positive_limits) if len(positive_limits) == 3 else 0
        by_model = {}
        for model_name, limits in (entry["plan"].get("capacity_by_model") or {}).items():
            model_ram = int(limits.get("ram_limited_records", 0) or 0)
            model_duration = int(limits.get("duration_limited_records", 0) or 0)
            model_positive = [
                value for value in (available, model_ram, model_duration) if value > 0
            ]
            by_model[model_name] = {
                "ram_limited_records": model_ram,
                "duration_limited_records": model_duration,
                "effective_safe_records": (
                    min(model_positive) if len(model_positive) == 3 else 0
                ),
            }
        entries.append(
            {
                "order": entry["order"],
                "name": entry["name"],
                "target_col": entry.get("target_col"),
                "timeframe_minutes": tf,
                "available_cached_records": available,
                "ram_limited_records": ram_limit,
                "duration_limited_records": duration_limit,
                "effective_safe_records": effective,
                "configured_max_records": int(
                    entry["plan"].get("worst_memory_estimate", {}).get("records", 0) or 0
                ),
                "by_model": by_model,
            }
        )
    return {"inventory": inventory, "entries": entries}


def _summary_candidates(summary_dir: Path, cfg_stem: str) -> List[Path]:
    candidates = list(summary_dir.glob(f"{cfg_stem}__summary.json"))
    candidates.extend(summary_dir.glob(f"{cfg_stem}__*__summary.json"))
    return list({path.resolve(): path for path in candidates}.values())


def _latest_summary(summary_dir: Path, cfg_stem: str) -> Optional[Dict[str, Any]]:
    candidates = _summary_candidates(summary_dir, cfg_stem)
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    payload["_path"] = str(latest)
    return payload


def _is_complete(summary_dir: Path, cfg_path: Path) -> bool:
    summary = _latest_summary(summary_dir, cfg_path.stem)
    return bool(summary and summary.get("status") == SUCCESS_STATUS)


def _process_tree_rss_bytes(pid: int) -> int:
    try:
        process = psutil.Process(pid)
        processes = [process] + process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    total = 0
    for item in processes:
        try:
            total += int(item.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _terminate_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        return
    for process in children:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        parent.terminate()
    except psutil.NoSuchProcess:
        pass
    _, alive = psutil.wait_procs(children + [parent], timeout=5)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass


def _write_runner_summary(summary_dir: Path, cfg_path: Path, status: str, reason: str) -> None:
    suffix = status.lower()
    _atomic_json(
        summary_dir / f"{cfg_path.stem}__{suffix}__summary.json",
        {
            "config_path": str(cfg_path),
            "status": status,
            "metric": {},
            "reason": reason,
            "wall_time": time.time(),
        },
    )


def _tail(path: Path, limit: int = 3000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return text[-limit:]


def _run_guarded_experiment(
    cfg_path: Path,
    summary_dir: Path,
    log_dir: Path,
    deadline: datetime,
    resources: Dict[str, Any],
    config: Dict[str, Any],
    artifact_dir: Path,
    worthy_r2_threshold: float = 0.6,
) -> Dict[str, Any]:
    max_ram_gb = float(resources.get("max_process_ram_gb", 20.0) or 20.0)
    max_ram_bytes = int(max_ram_gb * 1024 ** 3)
    critical_free_gb = float(resources.get("critical_free_ram_gb", 2.0) or 2.0)
    threads = max(int(resources.get("cpu_threads_per_experiment", 6) or 6), 1)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[variable] = str(threads)

    stdout_path = log_dir / f"{cfg_path.stem}.stdout.log"
    stderr_path = log_dir / f"{cfg_path.stem}.stderr.log"
    command = [
        sys.executable,
        str(ROOT / "tools" / "search_mode.py"),
        "--config",
        str(cfg_path),
        "--summary-dir",
        str(summary_dir),
        "--bulk-search",
        "--worthy-artifact-dir",
        str(artifact_dir),
        "--worthy-r2-threshold",
        str(worthy_r2_threshold),
    ]
    existing_bundles = set(artifact_dir.glob(f"{cfg_path.stem}__*")) if artifact_dir.exists() else set()
    started = time.monotonic()
    peak_rss = 0
    status = "FAILED"
    reason = "process_failed"
    with stdout_path.open("wb") as stdout_stream, stderr_path.open("wb") as stderr_stream:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=stdout_stream,
            stderr=stderr_stream,
        )
        while process.poll() is None:
            rss = _process_tree_rss_bytes(process.pid)
            peak_rss = max(peak_rss, rss)
            if datetime.now().astimezone() >= deadline:
                status = "INTERRUPTED"
                reason = f"hard_deadline_reached:{deadline.isoformat(timespec='minutes')}"
                _terminate_process_tree(process.pid)
                break
            if rss > max_ram_bytes:
                status = "RESOURCE_LIMIT"
                reason = f"process_ram_exceeded:{rss / 1024 ** 3:.2f}GB>{max_ram_gb:.2f}GB"
                _terminate_process_tree(process.pid)
                break
            available_gb = psutil.virtual_memory().available / 1024 ** 3
            if available_gb < critical_free_gb:
                status = "RESOURCE_LIMIT"
                reason = (
                    f"system_free_ram_below_guard:{available_gb:.2f}GB<"
                    f"{critical_free_gb:.2f}GB"
                )
                _terminate_process_tree(process.pid)
                break
            time.sleep(2)
        return_code = process.poll()

    duration = time.monotonic() - started
    if return_code == 0:
        summary = _latest_summary(summary_dir, cfg_path.stem)
        if summary and summary.get("status") == SUCCESS_STATUS:
            status = SUCCESS_STATUS
            reason = "completed"
        else:
            status = str((summary or {}).get("status") or "NO_METRICS")
            reason = "experiment_finished_without_success_metrics"
    elif status not in {"INTERRUPTED", "RESOURCE_LIMIT"}:
        reason = f"process_exit_code:{return_code};stderr_tail:{_tail(stderr_path)}"

    if status != SUCCESS_STATUS:
        _write_runner_summary(summary_dir, cfg_path, status, reason)
    matching_bundles = set(artifact_dir.glob(f"{cfg_path.stem}__*")) if artifact_dir.exists() else set()
    new_bundles = sorted(matching_bundles - existing_bundles, key=lambda path: path.stat().st_mtime)
    return {
        "config": str(cfg_path),
        "duration_signature": duration_signature(config),
        "status": status,
        "reason": reason,
        "duration_seconds": round(duration, 2),
        "peak_process_ram_gb": round(peak_rss / 1024 ** 3, 2),
        "finished_at": _now_iso(),
        "worthy_bundle": str(new_bundles[-1]) if new_bundles else None,
    }


def _load_state(session_dir: Path) -> Dict[str, Any]:
    path = session_dir / "session_state.json"
    if not path.exists():
        return {"runs": [], "durations_seconds": []}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"runs": [], "durations_seconds": []}


def _all_configs(manifest: Dict[str, Any]) -> List[tuple[Dict[str, Any], Path]]:
    ordered = []
    for entry in sorted(manifest.get("entries") or [], key=lambda item: int(item["order"])):
        config_dir = Path(entry["configs_dir"])
        for cfg_path in sorted(config_dir.glob("cfg_*.yaml")):
            ordered.append((entry, cfg_path))
    return ordered


def session_status(manifest: Dict[str, Any]) -> Dict[str, Any]:
    complete = 0
    per_entry = []
    for entry in sorted(manifest.get("entries") or [], key=lambda item: int(item["order"])):
        configs = sorted(Path(entry["configs_dir"]).glob("cfg_*.yaml"))
        entry_complete = sum(
            1 for cfg in configs if _is_complete(Path(entry["summaries_dir"]), cfg)
        )
        complete += entry_complete
        per_entry.append(
            {
                "order": entry["order"],
                "name": entry["name"],
                "completed": entry_complete,
                "total": len(configs),
            }
        )
    total = int(manifest.get("total_experiments", 0) or 0)
    return {
        "session_name": manifest.get("session_name"),
        "completed": complete,
        "pending": max(total - complete, 0),
        "total": total,
        "completion_percent": round((complete / total * 100.0) if total else 100.0, 2),
        "entries": per_entry,
    }


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _print_progress(status: Dict[str, Any], durations: List[float]) -> None:
    total = max(int(status["total"]), 1)
    completed = int(status["completed"])
    fraction = completed / total
    width = 28
    filled = min(int(round(width * fraction)), width)
    bar = "#" * filled + "-" * (width - filled)
    average = sum(durations) / len(durations) if durations else 0.0
    eta = average * int(status["pending"]) if average else 0.0
    eta_text = _format_duration(eta) if eta else "learning"
    print(
        f"[{bar}] {fraction * 100:6.2f}%  {completed}/{status['total']} "
        f"| pending={status['pending']} | ETA={eta_text}"
    )


def run_session(
    session_config_path: Path,
    session_cfg: Dict[str, Any],
    max_experiments_override: int = 0,
    deadline_override: Optional[str] = None,
) -> Dict[str, Any]:
    if not bool(session_cfg.get("manual_start_only", True)):
        raise ValueError("Experiment sessions must set manual_start_only: true")
    plan = build_session_plan(session_cfg)
    if session_cfg.get("data_source"):
        inventory = _cache_inventory(session_cfg)
        missing = [
            details["cache_table"]
            for details in inventory["timeframes"].values()
            if not details["exists"]
        ]
        if missing:
            raise RuntimeError(
                "Required SQL cache tables are missing: "
                f"{', '.join(missing)}. Run 'python tools/experiment_session.py "
                "prepare-data --config config/experiment_sessions/xauusd_nightly.yaml' first."
            )
    manifest = _materialize_session(session_config_path, session_cfg, plan)
    _, _, session_dir, _ = _session_paths(session_cfg)
    state = _load_state(session_dir)
    durations = [float(value) for value in state.get("durations_seconds") or [] if float(value) > 0]
    signature_durations = state.get("signature_durations") or {}
    resources = manifest.get("resources") or {}
    minimum_free = float(resources.get("min_free_ram_before_start_gb", 6.0) or 6.0)
    deadline_text = str(deadline_override or manifest.get("deadline_local") or "05:00")
    deadline = next_local_deadline(datetime.now().astimezone(), deadline_text)
    minimum_start_window = int(session_cfg.get("minimum_start_window_minutes", 30) or 30)
    configured_batch = int(session_cfg.get("max_experiments_per_run", 0) or 0)
    run_limit = int(max_experiments_override or configured_batch or 0)

    print(f"Manual session: {manifest['session_name']}")
    print(f"Hard stop:      {deadline.isoformat(timespec='minutes')}")
    print(f"RAM guard:      {resources.get('max_process_ram_gb')} GB per process")
    print(f"System reserve: {resources.get('critical_free_ram_gb')} GB minimum")
    print(f"CPU threads:    {resources.get('cpu_threads_per_experiment')}")
    _print_progress(session_status(manifest), durations)

    started_count = 0
    satisfied_entries = set(state.get("satisfied_entries") or [])
    stop_reason = "all_experiments_complete"
    for entry, cfg_path in _all_configs(manifest):
        if str(entry["name"]) in satisfied_entries:
            continue
        summary_dir = Path(entry["summaries_dir"])
        if _is_complete(summary_dir, cfg_path):
            continue
        if run_limit and started_count >= run_limit:
            stop_reason = f"manual_batch_limit_reached:{run_limit}"
            break

        now = datetime.now().astimezone()
        remaining_seconds = (deadline - now).total_seconds()
        if remaining_seconds <= minimum_start_window * 60:
            stop_reason = "deadline_start_window_reached"
            break
        available_gb = psutil.virtual_memory().available / 1024 ** 3
        if available_gb < minimum_free:
            stop_reason = f"insufficient_free_ram:{available_gb:.2f}GB<{minimum_free:.2f}GB"
            break

        cfg = _yaml_load(cfg_path)
        estimate = estimate_experiment_memory(cfg)
        if float(estimate["estimated_peak_gb"]) > float(resources["max_process_ram_gb"]):
            stop_reason = (
                f"planned_ram_exceeds_limit:{estimate['estimated_peak_gb']}GB>"
                f"{resources['max_process_ram_gb']}GB:{cfg_path.name}"
            )
            break
        signature = duration_signature(cfg)
        signature_history = [
            float(value) for value in signature_durations.get(signature, []) if float(value) > 0
        ]
        if signature_history:
            expected_duration = sum(signature_history[-10:]) / min(len(signature_history), 10)
        else:
            expected_duration = estimate_experiment_duration_seconds(
                cfg,
                cpu_threads=int(resources.get("cpu_threads_per_experiment", 6) or 6),
            )
        if expected_duration > remaining_seconds:
            stop_reason = (
                f"estimated_next_experiment_would_cross_deadline:"
                f"{expected_duration / 60.0:.1f}minutes"
            )
            break

        print(
            f"Starting {entry['order']:02d}/{entry['name']}/{cfg_path.name} "
            f"(estimated RAM {estimate['estimated_peak_gb']} GB)"
        )
        result = _run_guarded_experiment(
            cfg_path=cfg_path,
            summary_dir=summary_dir,
            log_dir=Path(entry["logs_dir"]),
            deadline=deadline,
            resources=resources,
            config=cfg,
            artifact_dir=Path(entry["configs_dir"]).parent / "worthy_artifacts",
            worthy_r2_threshold=float(manifest.get("worthy_r2_threshold", 0.6)),
        )
        started_count += 1
        state.setdefault("runs", []).append(result)
        if result.get("worthy_bundle") and bool(session_cfg.get("stop_entry_on_worthy_artifact", True)):
            satisfied_entries.add(str(entry["name"]))
            state["satisfied_entries"] = sorted(satisfied_entries)
            print(
                f"R2 target met for {entry['name']}; preserving {result['worthy_bundle']} "
                "and advancing to the next endpoint."
            )
        if result["status"] == SUCCESS_STATUS:
            durations.append(float(result["duration_seconds"]))
            state["durations_seconds"] = durations[-100:]
            signature_durations.setdefault(result["duration_signature"], []).append(
                float(result["duration_seconds"])
            )
            signature_durations[result["duration_signature"]] = signature_durations[
                result["duration_signature"]
            ][-20:]
            state["signature_durations"] = signature_durations
        state["updated_at"] = _now_iso()
        _atomic_json(session_dir / "session_state.json", state)
        print(
            f"Finished {cfg_path.name}: {result['status']} in "
            f"{_format_duration(result['duration_seconds'])}; "
            f"peak RAM={result['peak_process_ram_gb']} GB"
        )
        current_status = session_status(manifest)
        _print_progress(current_status, durations)
        if result["status"] in {"INTERRUPTED", "RESOURCE_LIMIT"}:
            stop_reason = result["reason"]
            break

    final_status = session_status(manifest)
    state["last_stop_reason"] = stop_reason
    state["last_deadline"] = deadline.isoformat()
    state["updated_at"] = _now_iso()
    _atomic_json(session_dir / "session_state.json", state)
    print(f"Session stopped: {stop_reason}")
    _print_progress(final_status, durations)
    return {**final_status, "stop_reason": stop_reason, "started_this_run": started_count}


def _load_session_config(path_value: str) -> tuple[Path, Dict[str, Any]]:
    path = _resolve_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Session config not found: {path}")
    return path, _yaml_load(path)


def _print_plan(plan: Dict[str, Any]) -> None:
    print(f"Session:             {plan['session_name']}")
    print(f"Manual start only:   {plan['manual_start_only']}")
    print(f"Hard stop each run:  {plan['deadline_local']} local time")
    print(f"Total experiments:   {plan['total_experiments']}")
    print(f"Per-process RAM cap: {plan['resources']['max_process_ram_gb']} GB")
    for entry in plan["entries"]:
        sweep = entry["plan"]
        worst = sweep.get("worst_memory_estimate") or {}
        print(
            f"  {entry['order']:02d}. {entry['name']}: "
            f"{sweep['unique_experiments']} experiments; "
            f"worst RAM={worst.get('estimated_peak_gb', 0.0)} GB; "
            f"RAM rows={sweep['estimated_max_records_for_worst_shape']:,}; "
            f"{sweep.get('max_duration_minutes', 90):.0f}m rows="
            f"{sweep.get('estimated_max_records_for_duration', 0):,}"
        )
        for warning in sweep.get("warnings") or []:
            print(f"      WARNING: {warning}")


def _print_inventory(inventory: Dict[str, Any]) -> None:
    print(
        f"SQL source: {inventory['db_path']} ({inventory['symbol']}); "
        f"minute rows={inventory['master_records']:,}; latest={inventory['master_latest']}"
    )
    for tf, details in inventory["timeframes"].items():
        state = (
            f"{details['records']:,} rows; latest={details['latest']}"
            if details["exists"]
            else "MISSING"
        )
        print(f"  {int(tf):>4}m {details['cache_table']}: {state}")


def _print_capacity(report: Dict[str, Any]) -> None:
    _print_inventory(report["inventory"])
    print("Effective ceiling = min(cached history, 20 GB RAM estimate, 90-minute CPU estimate)")
    for entry in report["entries"]:
        print(
            f"  {entry['name']}: available={entry['available_cached_records']:,}; "
            f"RAM={entry['ram_limited_records']:,}; CPU={entry['duration_limited_records']:,}; "
            f"effective={entry['effective_safe_records']:,}; "
            f"configured_max={entry['configured_max_records']:,}"
        )
        model_text = "; ".join(
            f"{name}:RAM={limits['ram_limited_records']:,},"
            f"CPU={limits['duration_limited_records']:,},"
            f"effective={limits['effective_safe_records']:,}"
            for name, limits in entry.get("by_model", {}).items()
        )
        if model_text:
            print(f"      {model_text}")


def main_cli() -> None:
    parser = argparse.ArgumentParser(description="Manual resumable TSMM experiment sessions")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "status", "capacity", "prepare-data"):
        command = sub.add_parser(name)
        command.add_argument("--config", required=True, help="Experiment session YAML")
        command.add_argument("--json", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--config", required=True, help="Experiment session YAML")
    run.add_argument(
        "--max-experiments",
        type=int,
        default=0,
        help="Run only this many pending experiments, then stop (0 uses session default)",
    )
    run.add_argument("--deadline-local", default=None, help="Override the configured HH:MM hard stop")
    args = parser.parse_args()

    config_path, session_cfg = _load_session_config(args.config)
    if args.command == "plan":
        plan = build_session_plan(session_cfg)
        if args.json:
            print(json.dumps(plan, indent=2, default=str))
        else:
            _print_plan(plan)
        return
    if args.command == "prepare-data":
        if not bool(session_cfg.get("manual_start_only", True)):
            raise ValueError("Data preparation must belong to a manual-only session")
        if not args.json:
            print("Refreshing configured SQL timeframe views and cache tables...")
        inventory = prepare_session_data(session_cfg)
        if args.json:
            print(json.dumps(inventory, indent=2, default=str))
        else:
            _print_inventory(inventory)
        return
    if args.command == "capacity":
        report = build_capacity_report(session_cfg)
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            _print_capacity(report)
        return
    if args.command == "run":
        run_session(
            session_config_path=config_path,
            session_cfg=session_cfg,
            max_experiments_override=args.max_experiments,
            deadline_override=args.deadline_local,
        )
        return

    _, _, _, manifest_path = _session_paths(session_cfg)
    if not manifest_path.exists():
        payload = {
            "created": False,
            "message": "Session has not been started; showing its plan.",
            "plan": build_session_plan(session_cfg),
        }
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = {"created": True, **session_status(manifest)}
    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        if not payload["created"]:
            print(payload["message"])
            _print_plan(payload["plan"])
        else:
            print(
                f"Session {payload['session_name']}: {payload['completed']}/{payload['total']} "
                f"complete ({payload['completion_percent']}%)"
            )
            for entry in payload["entries"]:
                print(
                    f"  {entry['order']:02d}. {entry['name']}: "
                    f"{entry['completed']}/{entry['total']}"
                )


if __name__ == "__main__":
    main_cli()
