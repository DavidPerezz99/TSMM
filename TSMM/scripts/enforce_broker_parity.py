"""Enforce cross-broker parity from a source trading config to a target trading config.

This utility is meant as an operational safety net when account mirroring drifts.
It compares source vs target MT5 exposure for the configured symbol and applies
the minimum corrective actions on the target side:

- Open positions: create missing side/volume legs and align SL/TP for matched legs.
- Pending orders: create missing pending orders.

By default this mirrors Pepperstone -> FTMO.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.investing_agent import MT5Adapter


def _load_trading_cfg(path_like: str) -> Dict[str, Any]:
    path = Path(path_like)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _profile_label(cfg: Dict[str, Any], fallback: str) -> str:
    runtime_cfg = dict(cfg.get("runtime") or {})
    label = str(runtime_cfg.get("profile_label") or runtime_cfg.get("job_id_prefix") or "").strip()
    return label or fallback


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _abs_close(a: float, b: float, tol: float) -> bool:
    return abs(float(a) - float(b)) <= float(tol)


def _parse_expiration_utc(raw: Any) -> Optional[datetime]:
    text = str(raw or "").strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except Exception:
            continue
    return None


def _connect_adapter(mt5_cfg: Dict[str, Any]) -> MT5Adapter:
    adapter = MT5Adapter(mt5_cfg)
    ok, msg = adapter.connect()
    if not ok:
        raise RuntimeError(f"MT5 connect failed: {msg}")
    return adapter


def _snapshot(adapter: MT5Adapter, symbol: str) -> Dict[str, List[Dict[str, Any]]]:
    positions = (adapter.list_open_positions() or {}).get("positions") or []
    orders = (adapter.list_pending_orders() or {}).get("orders") or []
    symbol_u = str(symbol or "").upper()
    pos = [p for p in positions if str((p or {}).get("symbol", "")).upper() == symbol_u]
    ords = [o for o in orders if str((o or {}).get("symbol", "")).upper() == symbol_u]
    return {"positions": pos, "orders": ords}


def _match_target_position(
    source_pos: Dict[str, Any],
    target_positions: List[Dict[str, Any]],
    used_indexes: set[int],
    volume_tol: float,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    side = str(source_pos.get("side") or "").lower()
    src_vol = _safe_float(source_pos.get("volume"), 0.0)
    src_entry = _safe_float(source_pos.get("price_open"), 0.0)

    candidates: List[Tuple[float, int, Dict[str, Any]]] = []
    for idx, tgt in enumerate(target_positions):
        if idx in used_indexes:
            continue
        if str(tgt.get("side") or "").lower() != side:
            continue
        tgt_vol = _safe_float(tgt.get("volume"), 0.0)
        if not _abs_close(src_vol, tgt_vol, volume_tol):
            continue
        tgt_entry = _safe_float(tgt.get("price_open"), 0.0)
        candidates.append((abs(src_entry - tgt_entry), idx, tgt))

    if not candidates:
        return None

    candidates.sort(key=lambda row: row[0])
    _, idx, tgt = candidates[0]
    return idx, tgt


def _match_target_pending(
    source_order: Dict[str, Any],
    target_orders: List[Dict[str, Any]],
    used_indexes: set[int],
    entry_tol: float,
    volume_tol: float,
) -> Optional[Tuple[int, Dict[str, Any]]]:
    side = str(source_order.get("side") or "").lower()
    src_entry = _safe_float(source_order.get("price_open"), 0.0)
    src_vol = _safe_float(source_order.get("volume"), 0.0)

    for idx, tgt in enumerate(target_orders):
        if idx in used_indexes:
            continue
        if str(tgt.get("side") or "").lower() != side:
            continue
        tgt_entry = _safe_float(tgt.get("price_open"), 0.0)
        tgt_vol = _safe_float(tgt.get("volume"), 0.0)
        if not _abs_close(src_entry, tgt_entry, entry_tol):
            continue
        if not _abs_close(src_vol, tgt_vol, volume_tol):
            continue
        return idx, tgt
    return None


def _sync_once(
    source_cfg: Dict[str, Any],
    target_cfg: Dict[str, Any],
    entry_tol: float,
    volume_tol: float,
    dry_run: bool,
    prune_target_extras: bool,
) -> Dict[str, Any]:
    src_label = _profile_label(source_cfg, "SOURCE")
    tgt_label = _profile_label(target_cfg, "TARGET")
    symbol = str(((source_cfg.get("execution") or {}).get("symbol")) or "XAUUSD")
    target_symbol = str(((target_cfg.get("execution") or {}).get("symbol")) or "XAUUSD")
    if symbol.upper() != target_symbol.upper():
        raise RuntimeError(f"Symbol mismatch: source={symbol} target={target_symbol}")

    src_mt5 = ((source_cfg.get("broker") or {}).get("mt5") or {})
    tgt_mt5 = ((target_cfg.get("broker") or {}).get("mt5") or {})

    src_adapter = _connect_adapter(src_mt5)
    src_snap = _snapshot(src_adapter, symbol)
    src_adapter.shutdown()

    tgt_adapter = _connect_adapter(tgt_mt5)
    tgt_snap = _snapshot(tgt_adapter, symbol)

    actions: List[Dict[str, Any]] = []
    target_positions = list(tgt_snap["positions"])
    used_target_pos: set[int] = set()

    for src_pos in src_snap["positions"]:
        match = _match_target_position(src_pos, target_positions, used_target_pos, volume_tol)
        src_side = str(src_pos.get("side") or "").lower()
        src_sl = _safe_float(src_pos.get("sl"), 0.0)
        src_tp = _safe_float(src_pos.get("tp"), 0.0)
        src_vol = _safe_float(src_pos.get("volume"), 0.0)

        if match is None:
            action = {
                "kind": "create_position",
                "source": src_label,
                "target": tgt_label,
                "side": src_side,
                "volume": src_vol,
                "sl": src_sl,
                "tp": src_tp,
            }
            if not dry_run:
                action["result"] = tgt_adapter.place_market_order(
                    symbol=symbol,
                    side=src_side,
                    volume=src_vol,
                    stop_loss=src_sl,
                    take_profit=src_tp,
                )
                target_positions = _snapshot(tgt_adapter, symbol)["positions"]
            actions.append(action)
            continue

        tgt_idx, tgt_pos = match
        used_target_pos.add(tgt_idx)
        tgt_sl = _safe_float(tgt_pos.get("sl"), 0.0)
        tgt_tp = _safe_float(tgt_pos.get("tp"), 0.0)
        if not _abs_close(src_sl, tgt_sl, 1e-6) or not _abs_close(src_tp, tgt_tp, 1e-6):
            action = {
                "kind": "align_risk",
                "source": src_label,
                "target": tgt_label,
                "target_ticket": int(tgt_pos.get("ticket") or 0),
                "side": src_side,
                "sl": src_sl,
                "tp": src_tp,
            }
            if not dry_run and int(tgt_pos.get("ticket") or 0) > 0:
                action["result"] = tgt_adapter.modify_position_risk(
                    ticket=int(tgt_pos.get("ticket") or 0),
                    stop_loss=src_sl,
                    take_profit=src_tp,
                )
            actions.append(action)

    target_orders = list(tgt_snap["orders"])
    used_target_orders: set[int] = set()
    for src_order in src_snap["orders"]:
        match = _match_target_pending(src_order, target_orders, used_target_orders, entry_tol, volume_tol)
        if match is not None:
            used_target_orders.add(match[0])
            continue

        src_side = str(src_order.get("side") or "").lower()
        src_entry = _safe_float(src_order.get("price_open"), 0.0)
        src_sl = _safe_float(src_order.get("sl"), 0.0)
        src_tp = _safe_float(src_order.get("tp"), 0.0)
        src_vol = _safe_float(src_order.get("volume"), 0.0)
        exp_dt = _parse_expiration_utc(src_order.get("expiration_utc"))
        action = {
            "kind": "create_pending",
            "source": src_label,
            "target": tgt_label,
            "side": src_side,
            "entry": src_entry,
            "sl": src_sl,
            "tp": src_tp,
            "volume": src_vol,
            "expiration_utc": src_order.get("expiration_utc"),
        }
        if not dry_run:
            action["result"] = tgt_adapter.place_programmed_order(
                symbol=symbol,
                side=src_side,
                volume=src_vol,
                entry=src_entry,
                stop_loss=src_sl,
                take_profit=src_tp,
                expiration_utc=exp_dt,
            )
            target_orders = _snapshot(tgt_adapter, symbol)["orders"]
        actions.append(action)

    if prune_target_extras:
        # Recompute matching after potential create actions, then prune unmatched target exposure.
        refreshed_target = _snapshot(tgt_adapter, symbol)
        refreshed_positions = list(refreshed_target["positions"])
        refreshed_orders = list(refreshed_target["orders"])

        matched_pos_indexes: set[int] = set()
        for src_pos in src_snap["positions"]:
            match = _match_target_position(src_pos, refreshed_positions, matched_pos_indexes, volume_tol)
            if match is not None:
                matched_pos_indexes.add(match[0])

        for idx, tgt_pos in enumerate(refreshed_positions):
            if idx in matched_pos_indexes:
                continue
            ticket = int(tgt_pos.get("ticket") or 0)
            if ticket <= 0:
                continue
            action = {
                "kind": "close_extra_position",
                "source": src_label,
                "target": tgt_label,
                "target_ticket": ticket,
                "side": str(tgt_pos.get("side") or "").lower(),
                "volume": _safe_float(tgt_pos.get("volume"), 0.0),
            }
            if not dry_run:
                action["result"] = tgt_adapter.close_position_by_ticket(ticket)
            actions.append(action)

        matched_order_indexes: set[int] = set()
        for src_order in src_snap["orders"]:
            match = _match_target_pending(src_order, refreshed_orders, matched_order_indexes, entry_tol, volume_tol)
            if match is not None:
                matched_order_indexes.add(match[0])

        for idx, tgt_order in enumerate(refreshed_orders):
            if idx in matched_order_indexes:
                continue
            ticket = int(tgt_order.get("order_ticket") or 0)
            if ticket <= 0:
                continue
            action = {
                "kind": "cancel_extra_pending",
                "source": src_label,
                "target": tgt_label,
                "target_ticket": ticket,
                "side": str(tgt_order.get("side") or "").lower(),
                "volume": _safe_float(tgt_order.get("volume"), 0.0),
                "entry": _safe_float(tgt_order.get("price_open"), 0.0),
            }
            if not dry_run:
                action["result"] = tgt_adapter.cancel_pending_order(ticket)
            actions.append(action)

    final_target = _snapshot(tgt_adapter, symbol)
    tgt_adapter.shutdown()

    return {
        "ok": True,
        "symbol": symbol,
        "source_profile": src_label,
        "target_profile": tgt_label,
        "source_counts": {
            "open": len(src_snap["positions"]),
            "pending": len(src_snap["orders"]),
        },
        "target_counts_after": {
            "open": len(final_target["positions"]),
            "pending": len(final_target["orders"]),
        },
        "actions": actions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce broker parity from source config to target config")
    parser.add_argument("--source-config", default="config/trading_agent.yaml", help="Source trading config path")
    parser.add_argument("--target-config", default="config/trading_agent_ftmo.yaml", help="Target trading config path")
    parser.add_argument("--entry-tol", type=float, default=0.15, help="Pending order entry tolerance")
    parser.add_argument("--volume-tol", type=float, default=1e-9, help="Volume tolerance")
    parser.add_argument("--interval", type=int, default=30, help="Polling interval seconds for continuous mode")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--dry-run", action="store_true", help="Report actions without trading")
    parser.add_argument("--keep-target-extras", action="store_true", help="Do not close/cancel target-only exposure")
    return parser.parse_args()


def _print_result(res: Dict[str, Any]) -> None:
    print(
        "PARITY"
        f" symbol={res.get('symbol')}"
        f" source={res.get('source_profile')}"
        f" target={res.get('target_profile')}"
        f" source_open={((res.get('source_counts') or {}).get('open'))}"
        f" source_pending={((res.get('source_counts') or {}).get('pending'))}"
        f" target_open={((res.get('target_counts_after') or {}).get('open'))}"
        f" target_pending={((res.get('target_counts_after') or {}).get('pending'))}"
        f" actions={len(res.get('actions') or [])}"
    , flush=True)
    for action in (res.get("actions") or []):
        kind = str(action.get("kind") or "action")
        detail = []
        for key in ("side", "volume", "entry", "sl", "tp", "target_ticket"):
            if key in action:
                detail.append(f"{key}={action.get(key)}")
        result = action.get("result")
        if isinstance(result, dict):
            detail.append(f"ok={result.get('ok')}")
            if not bool(result.get("ok", False)):
                detail.append(f"error={result.get('message', result.get('error', 'unknown'))}")
        print(f"- {kind}: {'; '.join(detail)}", flush=True)


def main() -> int:
    args = parse_args()
    source_cfg = _load_trading_cfg(args.source_config)
    target_cfg = _load_trading_cfg(args.target_config)

    interval = max(int(args.interval or 30), 5)
    entry_tol = max(float(args.entry_tol or 0.15), 0.0)
    volume_tol = max(float(args.volume_tol or 1e-9), 0.0)

    while True:
        try:
            result = _sync_once(
                source_cfg=source_cfg,
                target_cfg=target_cfg,
                entry_tol=entry_tol,
                volume_tol=volume_tol,
                dry_run=bool(args.dry_run),
                prune_target_extras=not bool(args.keep_target_extras),
            )
            _print_result(result)
        except Exception as exc:
            print(f"PARITY_ERROR {exc}", flush=True)

        if bool(args.once):
            break
        time.sleep(interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
