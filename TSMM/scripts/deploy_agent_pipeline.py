"""Deploy and trigger the TSMM agent pipeline from one command.

Pipeline stages:
1) Refresh master DB and timeframe views
2) Configure LLM provider (Ollama-first with fallback)
3) Optionally retrain requested timeframe top1 models
4) Deploy local signal endpoint service
5) Start trading-job execution (Agent A -> approval -> Agent B)

Usage:
  python scripts/deploy_agent_pipeline.py
  python scripts/deploy_agent_pipeline.py --pipeline-config config/agent_pipeline.yaml
  python scripts/deploy_agent_pipeline.py --retrain 7h:ulr,3h:nbeats --no-start-job
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import psutil
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.migrate_market_data_to_sqlite import migrate_master_csv  # noqa: E402
from utils.market_db import create_timeframe_views  # noqa: E402
from utils.notification_telegram import send_telegram_broadcast  # noqa: E402
from utils.runtime_scope import resolve_runtime_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deploy and run TSMM agent pipeline")
    p.add_argument("--pipeline-config", default="config/agent_pipeline.yaml", help="Pipeline YAML path")
    p.add_argument("--retrain", default="", help="Comma-separated targets like 7h:ulr,3h:nbeats")
    p.add_argument("--stop", action="store_true", help="Request a running deploy pipeline to stop")
    p.add_argument("--refresh", action="store_true", help="Force refresh stage execution")
    p.add_argument("--no-refresh", action="store_true", help="Skip refresh stage execution")
    p.add_argument("--no-start-job", action="store_true", help="Do not start trading-job stage")
    p.add_argument("--dry-run", action="store_true", help="Only print actions")
    return p.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def _as_abs(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p)


def _resolve_secret(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if raw.startswith("env:"):
        return os.environ.get(raw.split(":", 1)[1], "")
    return raw


def _parse_retrain_targets(value: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for token in [t.strip() for t in str(value or "").split(",") if t.strip()]:
        if ":" in token:
            tf, model = token.split(":", 1)
            out.append((tf.strip(), model.strip().lower()))
    return out


def _parse_r2_from_name(path: Path) -> float:
    stem = path.stem.lower()
    m = re.search(r"_(\d{4,6})$", stem)
    if m:
        digits = m.group(1)
        return float(int(digits) / (10 ** (len(digits) - 1)))
    m2 = re.search(r"(\d+\.\d+)", stem)
    if m2:
        return float(m2.group(1))
    return 0.0


def _stage_log_path() -> Path:
    return ROOT / "reports" / "runtime" / "deployment_pipeline_stage_log.jsonl"


def _forecast_runtime_log_dir() -> Path:
    return ROOT / "reports" / "runtime" / "forecast_runs"


def _deploy_stop_flag_path() -> Path:
    return ROOT / "reports" / "runtime" / "deployment_pipeline_stop.flag"


def _log_stage(stage: str, status: str, payload: Dict[str, Any]) -> None:
    path = _stage_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "status": status,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _tail_text(path: Path, max_chars: int = 1200) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    text = text[-max_chars:]
    return text.strip()


def _should_stop_deploy() -> bool:
    return _deploy_stop_flag_path().exists()


def _request_stop_deploy() -> str:
    p = _deploy_stop_flag_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    return str(p)


def _clear_stop_deploy_flag() -> None:
    p = _deploy_stop_flag_path()
    if p.exists():
        p.unlink()


def _discover_high_result_dirs() -> List[Tuple[str, Path]]:
    cfg_root = ROOT / "config"
    out: List[Tuple[str, Path]] = []
    if not cfg_root.exists():
        return out
    for child in cfg_root.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"^high(.+)Results$", child.name, flags=re.IGNORECASE)
        if not m:
            continue
        timeframe = m.group(1)
        out.append((timeframe, child))
    return sorted(out, key=lambda t: t[0])


def _discover_result_dirs_by_family() -> List[Tuple[str, str, Path]]:
    cfg_root = ROOT / "config"
    out: List[Tuple[str, str, Path]] = []
    if not cfg_root.exists():
        return out
    for child in cfg_root.iterdir():
        if not child.is_dir():
            continue
        m = re.match(r"^(high|low|open|close)(.+)Results$", child.name, flags=re.IGNORECASE)
        if not m:
            continue
        family = m.group(1).lower()
        timeframe = m.group(2)
        out.append((family, timeframe, child))
    return sorted(out, key=lambda t: (t[1], t[0]))


def _retarget_payload(value: Any, src_token: str, dst_token: str, target_col: str) -> Any:
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if str(k) == "target_col":
                out[k] = target_col
            else:
                out[k] = _retarget_payload(v, src_token, dst_token, target_col)
        return out
    if isinstance(value, list):
        return [_retarget_payload(v, src_token, dst_token, target_col) for v in value]
    if isinstance(value, str):
        return value.replace(src_token, dst_token)
    return value


def _target_return_feature(target_col: str) -> str:
    tc = str(target_col or "").strip().upper()
    mapping = {
        "HIGH": "High_return",
        "LOW": "Low_return",
        "OPEN": "Open_return",
        "CLOSE": "Price_return",
    }
    return mapping.get(tc, "")


def _normalize_feature_list(features: Any, target_col: str) -> List[str]:
    if not isinstance(features, list):
        return []

    out = [str(x) for x in features]
    if out and out[0].upper() in {"OPEN", "HIGH", "LOW", "CLOSE"}:
        out[0] = str(target_col).upper()

    drop_feat = _target_return_feature(target_col)
    if drop_feat:
        out = [f for f in out if f != drop_feat]
    return out


def _normalize_target_payload(payload: Dict[str, Any], target_col: str) -> Dict[str, Any]:
    out = dict(payload or {})
    tgt = str(target_col or "").upper()
    out["target_col"] = tgt

    if isinstance(out.get("input_features"), list):
        out["input_features"] = _normalize_feature_list(out.get("input_features"), tgt)

    if isinstance(out.get("target_features"), list):
        out["target_features"] = _normalize_feature_list(out.get("target_features"), tgt)

    it_sets = out.get("input_target_sets")
    if isinstance(it_sets, dict):
        if isinstance(it_sets.get("input_features"), list):
            it_sets["input_features"] = _normalize_feature_list(it_sets.get("input_features"), tgt)
        if isinstance(it_sets.get("target_features"), list):
            it_sets["target_features"] = _normalize_feature_list(it_sets.get("target_features"), tgt)
        out["input_target_sets"] = it_sets

    return out


def _ensure_target_result_ecosystem(pipeline_cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    targets_cfg = (pipeline_cfg.get("targets") or {})
    if not bool(targets_cfg.get("enabled", True)):
        return {"enabled": False, "reason": "disabled_in_pipeline_config"}

    mirror_from = str(targets_cfg.get("mirror_from", "high")).strip().lower()
    if mirror_from != "high":
        return {"enabled": False, "reason": f"unsupported_mirror_source:{mirror_from}"}

    families = [str(x).strip().lower() for x in (targets_cfg.get("families") or ["low", "open", "close"]) if str(x).strip()]
    result: Dict[str, Any] = {
        "enabled": True,
        "families": families,
        "created_dirs": 0,
        "mirrored_configs": 0,
        "normalized_configs": 0,
        "normalized_source_configs": 0,
        "timeframes": {},
    }

    for timeframe, high_root in _discover_high_result_dirs():
        src_token = f"high{timeframe}Results"
        tf_report: Dict[str, Any] = {"models": {}}
        for model_dir in [d for d in high_root.iterdir() if d.is_dir()]:
            source_cfgs = sorted(list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml")))

            source_normalized = 0
            for src_cfg in source_cfgs:
                old_src_payload = _load_yaml(src_cfg)
                patched_src_payload = _normalize_target_payload(old_src_payload, target_col="HIGH")
                if old_src_payload != patched_src_payload:
                    source_normalized += 1
                    if not dry_run:
                        _save_yaml(src_cfg, patched_src_payload)
            result["normalized_source_configs"] += source_normalized

            model_report: Dict[str, Any] = {}

            for target in families:
                dst_root = ROOT / "config" / f"{target}{timeframe}Results" / model_dir.name
                existed_before = dst_root.exists()
                if not existed_before and not dry_run:
                    dst_root.mkdir(parents=True, exist_ok=True)
                if not existed_before:
                    result["created_dirs"] += 1

                existing_cfgs = sorted(list(dst_root.glob("*.yaml")) + list(dst_root.glob("*.yml")))
                copied = 0
                if not existing_cfgs and source_cfgs:
                    for src_cfg in source_cfgs:
                        dst_cfg = dst_root / src_cfg.name
                        if dry_run:
                            copied += 1
                            continue
                        payload = _load_yaml(src_cfg)
                        patched = _retarget_payload(
                            payload,
                            src_token=src_token,
                            dst_token=f"{target}{timeframe}Results",
                            target_col=target.upper(),
                        )
                        patched = _normalize_target_payload(patched, target_col=target.upper())
                        _save_yaml(dst_cfg, patched)
                        copied += 1
                    result["mirrored_configs"] += copied

                # Normalize existing generated files too (repair mode) so bad files get corrected.
                dst_cfgs = sorted(list(dst_root.glob("*.yaml")) + list(dst_root.glob("*.yml")))
                normalized = 0
                for cfg_file in dst_cfgs:
                    old_payload = _load_yaml(cfg_file)
                    patched = _retarget_payload(
                        old_payload,
                        src_token=src_token,
                        dst_token=f"{target}{timeframe}Results",
                        target_col=target.upper(),
                    )
                    patched = _normalize_target_payload(patched, target_col=target.upper())
                    if old_payload != patched:
                        normalized += 1
                        if not dry_run:
                            _save_yaml(cfg_file, patched)
                result["normalized_configs"] += normalized

                model_report[target] = {
                    "path": str(dst_root),
                    "source_count": len(source_cfgs),
                    "mirrored_now": copied,
                    "normalized_now": normalized,
                    "already_had_configs": bool(existing_cfgs),
                }
            tf_report["models"][model_dir.name] = model_report
        result["timeframes"][timeframe] = tf_report
    return result


def _resolve_top1_config(timeframe: str, model: str) -> Optional[Path]:
    folder = ROOT / "config" / f"high{timeframe}Results" / model
    if not folder.exists():
        return None
    cands = list(folder.glob("top*.yaml")) + list(folder.glob("top*.yml"))
    if not cands:
        return None
    cands = sorted(cands, key=_parse_r2_from_name, reverse=True)
    return cands[0]


def _discover_retrain_targets_all_families() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    winner_models = {"nbeats", "ulr"}

    for family, timeframe, root_dir in _discover_result_dirs_by_family():
        best_model = ""
        best_cfg: Optional[Path] = None
        best_r2 = -1.0

        for model_dir in [d for d in root_dir.iterdir() if d.is_dir()]:
            model_name = str(model_dir.name).strip().lower()
            if model_name not in winner_models:
                continue

            cands = list(model_dir.glob("top1*.yaml")) + list(model_dir.glob("top1*.yml"))
            if not cands:
                cands = list(model_dir.glob("top*.yaml")) + list(model_dir.glob("top*.yml"))
            if not cands:
                continue

            model_best = sorted(cands, key=_parse_r2_from_name, reverse=True)[0]
            r2 = _parse_r2_from_name(model_best)
            if r2 > best_r2:
                best_r2 = r2
                best_cfg = model_best
                best_model = model_name

        if best_cfg is None:
            continue

        out.append(
            {
                "family": family,
                "timeframe": str(timeframe),
                "model": best_model,
                "config_path": str(best_cfg),
            }
        )

    return out


def _dedup_retrain_jobs(jobs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep first-seen retrain jobs and drop duplicates by config path or tf/family/model key."""
    out: List[Dict[str, str]] = []
    seen_cfg: Set[str] = set()
    seen_key: Set[Tuple[str, str, str]] = set()

    for job in jobs:
        family = str(job.get("family") or "high").strip().lower()
        timeframe = str(job.get("timeframe") or "").strip()
        model = str(job.get("model") or "").strip().lower()
        cfg_path = str(job.get("config_path") or "").strip()

        norm_cfg = ""
        if cfg_path:
            try:
                norm_cfg = str(Path(cfg_path).resolve())
            except Exception:
                norm_cfg = str(Path(cfg_path))

        tf_key = (family, timeframe, model)
        if norm_cfg and norm_cfg in seen_cfg:
            continue
        if timeframe and model and tf_key in seen_key:
            continue

        out.append(job)
        if norm_cfg:
            seen_cfg.add(norm_cfg)
        if timeframe and model:
            seen_key.add(tf_key)

    return out


def _call_ollama_tags(base_url: str) -> bool:
    try:
        r = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _configure_llm(pipeline_cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    llm_cfg = (pipeline_cfg.get("llm") or {})
    providers_path = _as_abs(str(llm_cfg.get("providers_config_path", "config/llm_providers.yaml")))
    trading_cfg_path = _as_abs(str(llm_cfg.get("trading_config_path", "config/trading_agent.yaml")))

    providers = _load_yaml(providers_path)
    trading = _load_yaml(trading_cfg_path)

    prov_map = (providers.get("providers") or {})
    ollama_cfg = (prov_map.get("local_ollama") or {})
    ollama_up = _call_ollama_tags(str(ollama_cfg.get("base_url", "http://127.0.0.1:11434")))

    preferred = str(llm_cfg.get("prefer_provider", "local_ollama"))
    fallback = str(llm_cfg.get("fallback_provider", "local_transformers"))
    chosen = preferred if ollama_up else fallback

    providers["default_provider"] = chosen
    if preferred in prov_map:
        prov_map[preferred]["enabled"] = True
    if fallback in prov_map:
        prov_map[fallback]["enabled"] = True

    trading.setdefault("llm", {})
    trading["llm"]["enabled"] = bool(llm_cfg.get("enable", True))
    trading["llm"]["provider"] = chosen
    trading["llm"]["providers_config_path"] = str(Path("config") / "llm_providers.yaml")

    if not dry_run:
        _save_yaml(providers_path, providers)
        _save_yaml(trading_cfg_path, trading)

    return {
        "chosen_provider": chosen,
        "ollama_available": bool(ollama_up),
        "providers_path": str(providers_path),
        "trading_config_path": str(trading_cfg_path),
    }


def _refresh_master_and_views(pipeline_cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    refresh_cfg = (pipeline_cfg.get("refresh") or {})
    master_csv = _as_abs(str(refresh_cfg.get("master_csv", "data/xauusd/master_table.csv")))
    db_path = _as_abs(str(refresh_cfg.get("db_path", "data/market_data.sqlite")))
    chunksize = int(refresh_cfg.get("chunksize", 250000) or 250000)
    views = [int(v) for v in (refresh_cfg.get("views") or [10, 30, 60, 180, 420, 720, 1440, 10080])]

    if dry_run:
        return {
            "dry_run": True,
            "master_csv": str(master_csv),
            "db_path": str(db_path),
            "views": views,
        }

    mig = migrate_master_csv(str(master_csv), str(db_path), chunksize=chunksize)
    create_cache_tables = bool(refresh_cfg.get("create_cache_tables", False))
    created = create_timeframe_views(str(db_path), views, include_cache_tables=create_cache_tables)

    if bool(refresh_cfg.get("update_trading_config", True)):
        trading_cfg_path = ROOT / "config" / "trading_agent.yaml"
        trading = _load_yaml(trading_cfg_path)
        trading.setdefault("dashboard", {})
        trading["dashboard"]["master_table_path"] = str(Path("data") / "market_data.sqlite")
        _save_yaml(trading_cfg_path, trading)

    return {
        "migration": mig,
        "created_views": created,
        "created_cache_tables": bool(create_cache_tables),
    }


def _model_file_key(path: Path) -> Optional[Tuple[str, str, str, bool]]:
    # Returns (model, target, timeframe, is_artifacts) + timestamp grouping key.
    m_art = re.match(
        r"^(?P<model>[A-Za-z0-9_]+)_artifacts_(?P<target>high|low|open|close)_(?P<timeframe>[A-Za-z0-9]+)_(?P<ts>\d{8}_\d{6})\.joblib$",
        path.name,
        flags=re.IGNORECASE,
    )
    if m_art:
        family_key = f"{m_art.group('model').lower()}__{m_art.group('target').lower()}__{m_art.group('timeframe').lower()}"
        return family_key, m_art.group("ts"), path.parent.as_posix(), True

    m_mod = re.match(
        r"^(?P<model>[A-Za-z0-9_]+)_(?P<target>high|low|open|close)_(?P<timeframe>[A-Za-z0-9]+)_(?P<ts>\d{8}_\d{6})\.joblib$",
        path.name,
        flags=re.IGNORECASE,
    )
    if m_mod:
        family_key = f"{m_mod.group('model').lower()}__{m_mod.group('target').lower()}__{m_mod.group('timeframe').lower()}"
        return family_key, m_mod.group("ts"), path.parent.as_posix(), False

    # Backward compatibility for legacy timestamped files that omitted target/timeframe.
    m_art_legacy = re.match(r"^(?P<family>[A-Za-z0-9_]+)_artifacts_(?P<ts>\d{8}_\d{6})\.joblib$", path.name)
    if m_art_legacy:
        return m_art_legacy.group("family").lower(), m_art_legacy.group("ts"), path.parent.as_posix(), True

    m_mod_legacy = re.match(r"^(?P<family>[A-Za-z0-9_]+)_(?P<ts>\d{8}_\d{6})\.joblib$", path.name)
    if m_mod_legacy:
        return m_mod_legacy.group("family").lower(), m_mod_legacy.group("ts"), path.parent.as_posix(), False
    return None


def _is_process_alive(pid: int) -> bool:
    try:
        return psutil.pid_exists(int(pid))
    except Exception:
        return False


def _file_size_mb(path: Path) -> float:
    if not path.exists() or not path.is_file():
        return 0.0
    return float(path.stat().st_size / (1024.0 * 1024.0))


def _dir_size_mb(path: Path) -> float:
    if not path.exists() or not path.is_dir():
        return 0.0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return float(total / (1024.0 * 1024.0))


def _optimize_storage(pipeline_cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    cfg = (pipeline_cfg.get("storage_optimizer") or {})
    if not bool(cfg.get("enabled", True)):
        return {"enabled": False, "reason": "disabled_in_pipeline_config"}

    model_dirs = [str(x).strip() for x in (cfg.get("model_dirs") or ["model_files"]) if str(x).strip()]
    keep_recent_model_generations = max(int(cfg.get("keep_recent_model_generations", 1) or 1), 1)
    keep_recent_artifact_generations = max(int(cfg.get("keep_recent_artifact_generations", 1) or 1), 1)
    delete_orphan_artifacts = bool(cfg.get("delete_orphan_artifacts", True))

    prune_runtime_flags = bool(cfg.get("prune_runtime_stale_files", True))
    clear_joblib_cache = bool(cfg.get("clear_joblib_cache_on_refresh", False))
    cache_clear_min_size_mb = float(cfg.get("cache_clear_min_size_mb", 4096) or 4096)

    deleted_files: List[str] = []
    kept_files_count = 0
    scanned_files_count = 0

    for rel_dir in model_dirs:
        base = _as_abs(rel_dir)
        if not base.exists() or not base.is_dir():
            continue

        # Group timestamped model/artifact files by folder + model/target/timeframe.
        per_group: Dict[Tuple[str, str], Dict[str, Set[str]]] = {}
        for f in base.rglob("*.joblib"):
            if not f.is_file():
                continue
            scanned_files_count += 1
            mk = _model_file_key(f)
            if mk is None:
                kept_files_count += 1
                continue
            family_key, ts, folder_key, is_art = mk
            group_key = (folder_key, family_key)
            per_group.setdefault(group_key, {"models": set(), "artifacts": set()})
            if is_art:
                per_group[group_key]["artifacts"].add(ts)
            else:
                per_group[group_key]["models"].add(ts)

        for (folder, family_key), stamps in per_group.items():
            model_ts = sorted(stamps["models"], reverse=True)
            art_ts = sorted(stamps["artifacts"], reverse=True)
            keep_model_ts = set(model_ts[:keep_recent_model_generations])
            keep_art_ts = set(art_ts[:keep_recent_artifact_generations])

            # Keep matching artifacts for kept model generations.
            keep_art_ts.update(t for t in art_ts if t in keep_model_ts)

            model_name, target_name, timeframe_name = (family_key.split("__", 2) + ["", ""])[:3]

            for ts in model_ts[keep_recent_model_generations:]:
                if target_name and timeframe_name:
                    fname = f"{model_name}_{target_name}_{timeframe_name}_{ts}.joblib"
                else:
                    fname = f"{family_key}_{ts}.joblib"
                p = Path(folder) / fname
                if p.exists() and p.is_file():
                    if dry_run:
                        deleted_files.append(str(p))
                    else:
                        p.unlink(missing_ok=True)
                        deleted_files.append(str(p))

            for ts in art_ts:
                if ts in keep_art_ts:
                    continue
                if (not delete_orphan_artifacts) and (ts not in keep_model_ts):
                    kept_files_count += 1
                    continue
                if target_name and timeframe_name:
                    fname = f"{model_name}_artifacts_{target_name}_{timeframe_name}_{ts}.joblib"
                else:
                    fname = f"{family_key}_artifacts_{ts}.joblib"
                p = Path(folder) / fname
                if p.exists() and p.is_file():
                    if dry_run:
                        deleted_files.append(str(p))
                    else:
                        p.unlink(missing_ok=True)
                        deleted_files.append(str(p))

    runtime_cleaned: List[str] = []
    if prune_runtime_flags:
        trading_cfg_path = _as_abs(str(((pipeline_cfg.get("trading") or {}).get("trading_config_path", "config/trading_agent.yaml"))))
        trading_cfg = _load_yaml(trading_cfg_path)
        runtime_dir = resolve_runtime_dir(base_dir=ROOT, trading_cfg=trading_cfg)
        runtime_dir.mkdir(parents=True, exist_ok=True)

        pid_specs = [
            (runtime_dir / "trading_job.pid", "pid"),
            (runtime_dir / "local_signal_endpoint_service.pid", "pid"),
        ]
        for p, kind in pid_specs:
            if not p.exists():
                continue
            try:
                pid = int((p.read_text(encoding="utf-8") or "0").strip() or "0")
            except Exception:
                pid = 0
            stale = (pid <= 0) or (not _is_process_alive(pid))
            if stale:
                if not dry_run:
                    p.unlink(missing_ok=True)
                runtime_cleaned.append(str(p))

        stale_flags = [
            runtime_dir / "mode_b_interrupt.flag",
            _deploy_stop_flag_path(),
        ]
        for p in stale_flags:
            if p.exists():
                if not dry_run:
                    p.unlink(missing_ok=True)
                runtime_cleaned.append(str(p))

    cache_result: Dict[str, Any] = {"cleared": False, "path": str(ROOT / "joblib_cache")}
    if clear_joblib_cache:
        cache_path = ROOT / "joblib_cache"
        size_mb = _dir_size_mb(cache_path)
        cache_result["size_mb_before"] = round(size_mb, 2)
        if size_mb >= cache_clear_min_size_mb and cache_path.exists() and cache_path.is_dir():
            if dry_run:
                cache_result["cleared"] = True
            else:
                import shutil

                shutil.rmtree(cache_path, ignore_errors=True)
                cache_result["cleared"] = True
        cache_result["size_mb_after"] = round(_dir_size_mb(cache_path), 2)

    return {
        "enabled": True,
        "dry_run": bool(dry_run),
        "scanned_files": scanned_files_count,
        "deleted_files_count": len(deleted_files),
        "deleted_files": deleted_files,
        "runtime_cleaned": runtime_cleaned,
        "cache": cache_result,
        "notes": [
            "reports/ and logs/ are never touched by storage optimizer",
            "only timestamped model/artifact .joblib files are pruned",
        ],
    }


def _run_forecast(
    cfg_path: Path,
    dry_run: bool = False,
    heartbeat_seconds: int = 60,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "config_path": str(cfg_path)}

    env = os.environ.copy()
    env["CONFIG_PATH"] = str(cfg_path)
    runtime_log_dir = _forecast_runtime_log_dir()
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_cfg = re.sub(r"[^A-Za-z0-9_.-]+", "_", cfg_path.stem)
    run_log_path = runtime_log_dir / f"{safe_cfg}_{ts}.log"

    with run_log_path.open("a", encoding="utf-8") as run_log:
        run_log.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START config={cfg_path}\n")
        run_log.flush()
        proc = subprocess.Popen(
            [sys.executable, "app.py", "forecast"],
            cwd=str(ROOT),
            env=env,
            stdout=run_log,
            stderr=subprocess.STDOUT,
            text=True,
        )

        started = time.time()
        last_heartbeat = started
        hb_interval = max(int(heartbeat_seconds or 60), 10)
        while proc.poll() is None:
            time.sleep(2)
            now = time.time()
            if progress_callback and (now - last_heartbeat) >= hb_interval:
                progress_callback(
                    {
                        "config_path": str(cfg_path),
                        "pid": int(proc.pid),
                        "elapsed_seconds": int(now - started),
                        "run_log_path": str(run_log_path),
                        "log_tail": _tail_text(run_log_path, max_chars=1000),
                    }
                )
                last_heartbeat = now

        proc.wait()
        run_log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] END rc={int(proc.returncode or 0)}\n")
        run_log.flush()

    log_tail = _tail_text(run_log_path, max_chars=1200)
    return {
        "config_path": str(cfg_path),
        "returncode": int(proc.returncode or 0),
        "ok": (proc.returncode or 0) == 0,
        "stdout_tail": log_tail,
        "stderr_tail": "",
        "duration_seconds": int(time.time() - started),
        "run_log_path": str(run_log_path),
        "log_tail": log_tail,
    }


def _ensure_endpoint_service(script_rel: str, host: str, port: int, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "script": script_rel, "host": host, "port": port}

    # Stop stale service processes first.
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if "local_signal_endpoint_service.py" in cmd:
                p.kill()
        except Exception:
            pass

    env = os.environ.copy()
    env["TSMM_SIGNAL_HOST"] = host
    env["TSMM_SIGNAL_PORT"] = str(port)

    script_abs = _as_abs(script_rel)
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]

    p = subprocess.Popen(
        [sys.executable, str(script_abs)],
        cwd=str(ROOT),
        env=env,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    health_url = f"http://{host}:{port}/health"
    ok = False
    payload: Dict[str, Any] = {}
    for _ in range(25):
        time.sleep(1)
        try:
            r = requests.get(health_url, timeout=5)
            if r.status_code == 200:
                payload = r.json()
                ok = True
                break
        except Exception:
            pass

    pid_file = ROOT / "reports" / "runtime" / "local_signal_endpoint_service.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(p.pid), encoding="utf-8")

    return {
        "pid": int(p.pid),
        "ok": ok,
        "health_url": health_url,
        "health": payload,
        "pid_file": str(pid_file),
    }


def _start_trading_job(pipeline_cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    tcfg = (pipeline_cfg.get("trading") or {})
    if not bool(tcfg.get("start_job", True)):
        return {"started": False, "reason": "disabled_in_pipeline_config"}

    plan_model = str(tcfg.get("plan_model", "")).strip()
    exec_cfg = _as_abs(str(tcfg.get("execution_config", "config/high7hResults/ulr/top1_07890.yaml")))
    trading_cfg = _as_abs(str(tcfg.get("trading_config_path", "config/trading_agent.yaml")))
    pipeline_mt5_terminal_path = _resolve_secret(tcfg.get("mt5_terminal_path", ""))
    trading_cfg_payload = _load_yaml(trading_cfg)
    trading_cfg_mt5_terminal_path = _resolve_secret(
        ((trading_cfg_payload.get("broker") or {}).get("mt5") or {}).get("path", "")
    )

    # Keep terminal selection account-scoped by preferring the per-account trading config.
    mt5_terminal_path = trading_cfg_mt5_terminal_path or pipeline_mt5_terminal_path
    if trading_cfg_mt5_terminal_path:
        mt5_terminal_path_source = "trading_config.broker.mt5.path"
    elif pipeline_mt5_terminal_path:
        mt5_terminal_path_source = "pipeline.trading.mt5_terminal_path"
    else:
        mt5_terminal_path_source = "unset"

    mt5_path_conflict = bool(
        trading_cfg_mt5_terminal_path
        and pipeline_mt5_terminal_path
        and (trading_cfg_mt5_terminal_path != pipeline_mt5_terminal_path)
    )

    cmd = [sys.executable, "app.py", "trading-job", "start"]
    if plan_model:
        cmd += ["--plan-model", plan_model]

    if dry_run:
        return {
            "dry_run": True,
            "cmd": cmd,
            "config_path": str(exec_cfg),
            "trading_config_path": str(trading_cfg),
            "mt5_terminal_path": mt5_terminal_path,
            "mt5_terminal_path_source": mt5_terminal_path_source,
            "mt5_terminal_path_from_trading_config": trading_cfg_mt5_terminal_path,
            "mt5_terminal_path_from_pipeline": pipeline_mt5_terminal_path,
            "mt5_terminal_path_conflict": mt5_path_conflict,
        }

    env = os.environ.copy()
    env["CONFIG_PATH"] = str(exec_cfg)
    env["TRADING_CONFIG_PATH"] = str(trading_cfg)
    if mt5_terminal_path:
        env["MT5_TERMINAL_PATH"] = mt5_terminal_path

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]

    p = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    pid_file = ROOT / "reports" / "runtime" / "trading_job.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(p.pid), encoding="utf-8")

    return {
        "started": True,
        "pid": int(p.pid),
        "pid_file": str(pid_file),
        "cmd": cmd,
        "mt5_terminal_path": mt5_terminal_path,
        "mt5_terminal_path_source": mt5_terminal_path_source,
        "mt5_terminal_path_conflict": mt5_path_conflict,
    }


def _send_telegram(summary: Dict[str, Any], trading_cfg_path: Path) -> Dict[str, Any]:
    trading_cfg = _load_yaml(trading_cfg_path)
    telegram_cfg = (trading_cfg.get("telegram_notifications") or {})
    runtime_dir = resolve_runtime_dir(base_dir=ROOT, trading_cfg=trading_cfg)
    account_label = str(((trading_cfg.get("runtime") or {}).get("profile_label") or "TSMM")).strip() or "TSMM"
    msg = (
        f"{account_label} pipeline deployed. "
        f"llm_provider={summary.get('llm', {}).get('chosen_provider')}; "
        f"endpoint_ok={summary.get('endpoint', {}).get('ok')}; "
        f"trading_started={summary.get('trading', {}).get('started', False)}"
    )
    return send_telegram_broadcast(
        telegram_cfg,
        msg,
        subscribers_path=str(runtime_dir / "telegram_subscribers.json"),
    )


def main() -> int:
    args = parse_args()

    if args.stop:
        path = _request_stop_deploy()
        print(json.dumps({"ok": True, "stop_requested": True, "stop_flag_path": path}, indent=2))
        return 0

    if args.refresh and args.no_refresh:
        raise SystemExit("Cannot use --refresh and --no-refresh together")

    _clear_stop_deploy_flag()

    pipeline_cfg_path = _as_abs(args.pipeline_config)
    pipeline_cfg = _load_yaml(pipeline_cfg_path)

    retrain_targets = []
    retrain_jobs: List[Dict[str, str]] = []
    if args.retrain.strip():
        retrain_targets = _parse_retrain_targets(args.retrain)
        for tf, model in retrain_targets:
            cfg = _resolve_top1_config(tf, model)
            if cfg is not None:
                retrain_jobs.append(
                    {
                        "family": "high",
                        "timeframe": tf,
                        "model": model,
                        "config_path": str(cfg),
                    }
                )
    else:
        for raw in (pipeline_cfg.get("models", {}).get("retrain_targets") or []):
            if isinstance(raw, str) and ":" in raw:
                tf, model = raw.split(":", 1)
                retrain_targets.append((tf.strip(), model.strip().lower()))
        for tf, model in retrain_targets:
            cfg = _resolve_top1_config(tf, model)
            if cfg is not None:
                retrain_jobs.append(
                    {
                        "family": "high",
                        "timeframe": tf,
                        "model": model,
                        "config_path": str(cfg),
                    }
                )

    summary: Dict[str, Any] = {
        "pipeline_config": str(pipeline_cfg_path),
        "dry_run": bool(args.dry_run),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    _log_stage("bootstrap", "started", {"pipeline_config": str(pipeline_cfg_path), "dry_run": bool(args.dry_run)})

    if _should_stop_deploy():
        summary["stopped"] = True
        summary["stopped_stage"] = "bootstrap"
        print(json.dumps(summary, indent=2, default=str))
        return 0

    summary["target_scaffold"] = _ensure_target_result_ecosystem(pipeline_cfg, dry_run=args.dry_run)
    _log_stage("target_scaffold", "completed", summary["target_scaffold"])

    if _should_stop_deploy():
        summary["stopped"] = True
        summary["stopped_stage"] = "target_scaffold"
        print(json.dumps(summary, indent=2, default=str))
        return 0

    refresh_cfg = (pipeline_cfg.get("refresh") or {})
    refresh_enabled = bool(refresh_cfg.get("enabled", True))
    if args.refresh:
        refresh_enabled = True
    if args.no_refresh:
        refresh_enabled = False

    if refresh_enabled:
        summary["refresh"] = _refresh_master_and_views(pipeline_cfg, dry_run=args.dry_run)
        _log_stage("refresh", "completed", summary["refresh"])
    else:
        summary["refresh"] = {"skipped": True, "reason": "disabled_by_flag_or_config"}
        _log_stage("refresh", "skipped", summary["refresh"])

    summary["llm"] = _configure_llm(pipeline_cfg, dry_run=args.dry_run)
    _log_stage("llm", "completed", summary["llm"])

    if _should_stop_deploy():
        summary["stopped"] = True
        summary["stopped_stage"] = "llm"
        print(json.dumps(summary, indent=2, default=str))
        return 0

    models_cfg = (pipeline_cfg.get("models") or {})
    if refresh_enabled and bool(models_cfg.get("retrain_all_targets_on_refresh", True)):
        auto_jobs = _discover_retrain_targets_all_families()
        if auto_jobs:
            retrain_jobs = auto_jobs

    retrain_jobs = _dedup_retrain_jobs(retrain_jobs)

    retrain_results = []
    retrain_interval_seconds = int(models_cfg.get("retrain_progress_log_interval_seconds", 60) or 60)
    _log_stage(
        "retrain",
        "started",
        {
            "explicit_targets": retrain_targets,
            "jobs_count": len(retrain_jobs),
            "progress_log_interval_seconds": retrain_interval_seconds,
        },
    )

    for idx, job in enumerate(retrain_jobs, start=1):
        tf = str(job.get("timeframe") or "")
        model = str(job.get("model") or "")
        family = str(job.get("family") or "high")
        cfg_path = str(job.get("config_path") or "")
        cfg = Path(cfg_path) if cfg_path else None

        _log_stage(
            "retrain",
            "job_started",
            {
                "job_index": idx,
                "jobs_total": len(retrain_jobs),
                "family": family,
                "timeframe": tf,
                "model": model,
                "config_path": cfg_path,
            },
        )

        if cfg is None:
            miss = {
                "family": family,
                "timeframe": tf,
                "model": model,
                "ok": False,
                "error": "top_config_not_found",
            }
            retrain_results.append(miss)
            _log_stage(
                "retrain",
                "job_completed",
                {
                    "job_index": idx,
                    "jobs_total": len(retrain_jobs),
                    **miss,
                },
            )
            continue
        out = _run_forecast(
            cfg,
            dry_run=args.dry_run,
            heartbeat_seconds=retrain_interval_seconds,
            progress_callback=lambda payload, i=idx, fam=family, t=tf, m=model: _log_stage(
                "retrain",
                "progress",
                {
                    "job_index": i,
                    "jobs_total": len(retrain_jobs),
                    "family": fam,
                    "timeframe": t,
                    "model": m,
                    **payload,
                },
            ),
        )
        out.update({"family": family, "timeframe": tf, "model": model})
        retrain_results.append(out)
        _log_stage(
            "retrain",
            "job_completed",
            {
                "job_index": idx,
                "jobs_total": len(retrain_jobs),
                "family": family,
                "timeframe": tf,
                "model": model,
                "ok": bool(out.get("ok", False)),
                "returncode": int(out.get("returncode", 0) or 0),
                "duration_seconds": int(out.get("duration_seconds", 0) or 0),
            },
        )
    summary["retrain"] = retrain_results
    _log_stage(
        "retrain",
        "completed",
        {
            "explicit_targets": retrain_targets,
            "jobs_count": len(retrain_jobs),
            "results_count": len(retrain_results),
        },
    )

    if refresh_enabled:
        summary["storage_optimizer"] = _optimize_storage(pipeline_cfg, dry_run=args.dry_run)
        _log_stage("storage_optimizer", "completed", summary["storage_optimizer"])
    else:
        summary["storage_optimizer"] = {"skipped": True, "reason": "refresh_not_enabled"}
        _log_stage("storage_optimizer", "skipped", summary["storage_optimizer"])

    if _should_stop_deploy():
        summary["stopped"] = True
        summary["stopped_stage"] = "storage_optimizer"
        print(json.dumps(summary, indent=2, default=str))
        return 0

    ep_cfg = (pipeline_cfg.get("endpoints") or {})
    if bool(ep_cfg.get("deploy_local_service", False)):
        summary["endpoint"] = _ensure_endpoint_service(
            script_rel=str(ep_cfg.get("service_script", "scripts/local_signal_endpoint_service.py")),
            host=str(ep_cfg.get("host", "127.0.0.1")),
            port=int(ep_cfg.get("port", 8000) or 8000),
            dry_run=args.dry_run,
        )
    else:
        summary["endpoint"] = {"skipped": True, "reason": "deploy_local_service_disabled"}
    _log_stage("endpoint", "completed", summary["endpoint"])

    if _should_stop_deploy():
        summary["stopped"] = True
        summary["stopped_stage"] = "endpoint"
        print(json.dumps(summary, indent=2, default=str))
        return 0

    if args.no_start_job:
        summary["trading"] = {"started": False, "reason": "disabled_by_cli_flag"}
    else:
        summary["trading"] = _start_trading_job(pipeline_cfg, dry_run=args.dry_run)
    _log_stage("trading", "completed", summary["trading"])

    if _should_stop_deploy():
        summary["stopped"] = True
        summary["stopped_stage"] = "trading"
        print(json.dumps(summary, indent=2, default=str))
        return 0

    trading_cfg_path = _as_abs(str((pipeline_cfg.get("trading") or {}).get("trading_config_path", "config/trading_agent.yaml")))
    if not args.dry_run:
        summary["telegram"] = _send_telegram(summary, trading_cfg_path)
    else:
        summary["telegram"] = {"ok": False, "skipped": True, "reason": "dry_run"}
    _log_stage("telegram", "completed", summary["telegram"])

    summary["backtest_policy"] = {
        "llm_enabled": False,
        "mode": "technical_only",
        "note": "Backtests should use technical signal estimation only. LLM commentary is runtime assistance, not backtest signal generation.",
    }

    out_path = ROOT / "reports" / "runtime" / "deployment_pipeline_last.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _log_stage("pipeline", "completed", {"summary_path": str(out_path)})

    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
