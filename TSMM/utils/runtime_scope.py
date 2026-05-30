from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict


def runtime_scope_enabled(trading_cfg: Dict[str, Any] | None = None, env_var: str = "TSMM_RUNTIME_DIR") -> bool:
    if str(os.environ.get(env_var) or "").strip():
        return True
    cfg = dict((trading_cfg or {}).get("runtime") or {})
    return bool(str(cfg.get("root_dir") or "").strip() or str(cfg.get("namespace") or "").strip())


def resolve_runtime_dir(
    *,
    output_dir: str | Path | None = None,
    trading_cfg: Dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
    env_var: str = "TSMM_RUNTIME_DIR",
) -> Path:
    env_value = str(os.environ.get(env_var) or "").strip()
    if env_value:
        return Path(env_value)

    cfg = dict((trading_cfg or {}).get("runtime") or {})
    root_dir = str(cfg.get("root_dir") or "").strip()
    if root_dir:
        candidate = Path(root_dir)
        if candidate.is_absolute():
            return candidate
        anchor = Path(base_dir) if base_dir is not None else Path.cwd()
        return anchor / candidate

    namespace = str(cfg.get("namespace") or "").strip()
    if namespace:
        if output_dir is not None:
            return Path(output_dir) / "runtime" / namespace
        anchor = Path(base_dir) if base_dir is not None else Path.cwd()
        return anchor / "reports" / "runtime" / namespace

    if output_dir is not None:
        return Path(output_dir) / "runtime"

    anchor = Path(base_dir) if base_dir is not None else Path.cwd()
    return anchor / "reports" / "runtime"


def resolve_runtime_file(
    *,
    configured_path: str | Path | None,
    fallback_name: str,
    output_dir: str | Path | None = None,
    trading_cfg: Dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
    env_var: str = "TSMM_RUNTIME_DIR",
) -> Path:
    configured = str(configured_path or "").strip()
    if configured and not runtime_scope_enabled(trading_cfg, env_var=env_var):
        candidate = Path(configured)
        if candidate.is_absolute():
            return candidate
        anchor = Path(base_dir) if base_dir is not None else Path.cwd()
        return anchor / candidate

    root = resolve_runtime_dir(output_dir=output_dir, trading_cfg=trading_cfg, base_dir=base_dir, env_var=env_var)
    filename = Path(configured).name if configured else str(fallback_name or "").strip()
    return root / filename


def _normalize_job_id_prefix(raw_value: Any) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return ""
    clean = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_")
    return clean.upper()


def resolve_job_id_prefix(trading_cfg: Dict[str, Any] | None = None) -> str:
    cfg = dict((trading_cfg or {}).get("runtime") or {})
    for key in ("job_id_prefix", "job_id_tag"):
        resolved = _normalize_job_id_prefix(cfg.get(key))
        if resolved:
            return resolved
    return ""