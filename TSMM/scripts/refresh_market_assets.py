#!/usr/bin/env python
"""Refresh configured TSMM market assets and rebuild their timeframe caches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.equity_data import refresh_us500_proxy
from utils.live_data import update_fx_master_table_db
from utils.market_db import create_timeframe_views, materialize_timeframe_cache_tables


def _resolve(value: str) -> Path:
    path = Path(str(value or ""))
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return payload


def _refresh_one(name: str, cfg: dict, *, rebuild_caches: bool) -> dict:
    provider = str(cfg.get("provider") or "").strip().lower()
    db_path = _resolve(str(cfg.get("db_path") or "data/market_data.sqlite"))
    symbol = str(cfg.get("symbol") or name).strip().upper()
    token_env = str(cfg.get("token_env") or "TIINGO_API_TOKEN")
    token_envs = cfg.get("token_envs")
    rotation_path = str(_resolve(str(cfg.get("token_rotation_state_path")))) if cfg.get("token_rotation_state_path") else None
    if provider == "tiingo_fx":
        result = update_fx_master_table_db(
            db_path=str(db_path),
            rate=str(cfg.get("rate") or "1min"),
            symbol=str(cfg.get("provider_symbol") or symbol).lower(),
            token_env=token_env,
            token_envs=token_envs,
            token_rotation_state_path=rotation_path,
        )
    elif provider == "tiingo_iex_proxy":
        result = refresh_us500_proxy(
            db_path=str(db_path),
            source_ticker=str(cfg.get("source_ticker") or "SPY"),
            target_symbol=symbol,
            rate=str(cfg.get("rate") or "1min"),
            token_env=token_env,
            token_envs=token_envs,
            token_rotation_state_path=rotation_path,
            calibration_lookback_days=int(cfg.get("calibration_lookback_days", 60) or 60),
            minimum_calibration_samples=int(cfg.get("minimum_calibration_samples", 100) or 100),
            maximum_relative_mad=float(cfg.get("maximum_relative_mad", 0.02) or 0.02),
            maximum_seam_jump_pct=float(cfg.get("maximum_seam_jump_pct", 8.0) or 8.0),
        )
    else:
        raise ValueError(f"Unsupported provider for {name}: {provider}")

    if result.get("ok", result.get("updated", False)) and rebuild_caches:
        timeframes = [int(value) for value in (cfg.get("timeframes_minutes") or [])]
        if timeframes:
            create_timeframe_views(str(db_path), timeframes, include_cache_tables=False, symbol=symbol)
            materialize_timeframe_cache_tables(str(db_path), timeframes, symbol=symbol)
            result["caches_rebuilt"] = timeframes
    result["asset"] = name
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/market_assets.yaml")
    parser.add_argument("--asset", action="append", help="Asset key to refresh; repeatable. Default: all enabled assets")
    parser.add_argument("--skip-caches", action="store_true")
    args = parser.parse_args()
    config_path = _resolve(args.config)
    config = _load(config_path)
    assets = dict(config.get("assets") or {})
    selected = [str(value).strip().lower() for value in (args.asset or [])]
    if not selected:
        selected = [name for name, cfg in assets.items() if bool((cfg or {}).get("enabled", True))]
    unknown = [name for name in selected if name not in assets]
    if unknown:
        raise ValueError(f"Unknown asset keys: {unknown}")
    results = []
    exit_code = 0
    for name in selected:
        try:
            result = _refresh_one(name, dict(assets[name] or {}), rebuild_caches=not args.skip_caches)
        except Exception as exc:
            result = {"asset": name, "ok": False, "error": str(exc)}
        if not result.get("ok", result.get("updated", False)):
            exit_code = 1
        results.append(result)
    print(json.dumps({"config": str(config_path), "results": results}, indent=2, default=str))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
