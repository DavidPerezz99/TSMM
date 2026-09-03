"""
Deterministic investing agent (Mode A / Mode B scaffold, MT5-first).

Mode A: Generate a single-session deterministic trading plan/report.
Mode B: Optional live-management scaffold with MT5 adapter and model endpoint checks.
"""

from __future__ import annotations

import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import yaml

from .backtester import run_backtest_from_validation
from .market_db import query_ohlc
from .model_deployment import resolve_active_deployment
from .data_loader import merge_cross_asset_frame
from .trading_reporter import generate_trading_plan_report
from .trading_quality import apply_hybrid_trade_gate, model_quality_weight
from .iqoption_adapter import IQOptionAdapter
from .runtime_scope import resolve_runtime_file
from .trading_signal_policy import (
    apply_volatility_protection,
    evaluate_joint_ohlc_policy,
    signal_policy_config,
    weighted_timeframe_consensus,
)


_LAST_ENDPOINT_START_TS: float = 0.0


def _resolve_secret(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value.startswith("env:"):
        return os.environ.get(value.split(":", 1)[1], "")
    return value


def _int_or_default(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


class MT5Adapter:
    """MT5 execution adapter (graceful fallback if MetaTrader5 is unavailable)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self._mt5 = None

    def connect(self) -> tuple[bool, str]:
        if "enabled" in self.cfg and not bool(self.cfg.get("enabled")):
            return False, "MT5 broker disabled by configuration"

        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception:
            return False, "MetaTrader5 package not installed"

        self._mt5 = mt5
        path = _resolve_secret(self.cfg.get("path", "")) or None
        ok_init = mt5.initialize(path=path) if path else mt5.initialize()
        if not ok_init:
            return False, f"MT5 initialize failed: {mt5.last_error()}"

        login_raw = _resolve_secret(self.cfg.get("login", ""))
        try:
            login = int(login_raw or 0)
        except Exception:
            login = 0
        password = _resolve_secret(self.cfg.get("password", ""))
        server = _resolve_secret(self.cfg.get("server", ""))
        if login and password:
            if server:
                ok_login = mt5.login(login=login, password=password, server=server)
            else:
                ok_login = mt5.login(login=login, password=password)
            if not ok_login:
                return False, f"MT5 login failed: {mt5.last_error()}"

        return True, "connected"

    def shutdown(self):
        if self._mt5 is not None:
            try:
                self._mt5.shutdown()
            except Exception:
                pass

    def _require_mt5(self):
        if self._mt5 is None:
            return False, "MT5 not connected"
        return True, "ok"

    def _position_side_from_type(self, position_type: Any) -> str:
        mt5 = self._mt5
        ptype = _int_or_default(position_type, -1)
        if mt5 is not None:
            if ptype == int(getattr(mt5, "POSITION_TYPE_BUY", 0)):
                return "buy"
            if ptype == int(getattr(mt5, "POSITION_TYPE_SELL", 1)):
                return "sell"
        if ptype == 0:
            return "buy"
        if ptype in {1, -1}:
            return "sell"
        return "unknown"

    def _order_side_from_type(self, order_type: Any) -> str:
        mt5 = self._mt5
        otype = _int_or_default(order_type, -1)
        if mt5 is not None:
            buy_types = {
                int(getattr(mt5, "ORDER_TYPE_BUY", 0)),
                int(getattr(mt5, "ORDER_TYPE_BUY_LIMIT", 2)),
                int(getattr(mt5, "ORDER_TYPE_BUY_STOP", 4)),
                int(getattr(mt5, "ORDER_TYPE_BUY_STOP_LIMIT", 6)),
            }
            sell_types = {
                int(getattr(mt5, "ORDER_TYPE_SELL", 1)),
                int(getattr(mt5, "ORDER_TYPE_SELL_LIMIT", 3)),
                int(getattr(mt5, "ORDER_TYPE_SELL_STOP", 5)),
                int(getattr(mt5, "ORDER_TYPE_SELL_STOP_LIMIT", 7)),
            }
            if otype in buy_types:
                return "buy"
            if otype in sell_types:
                return "sell"
        if otype in {0, 2, 4, 6}:
            return "buy"
        if otype in {1, 3, 5, 7, -1}:
            return "sell"
        return "unknown"

    def _serialize_position(self, p: Any) -> Dict[str, Any]:
        ptype = _int_or_default(getattr(p, "type", None), -1)
        return {
            "ticket": int(getattr(p, "ticket", 0) or 0),
            "symbol": str(getattr(p, "symbol", "")),
            "volume": float(getattr(p, "volume", 0.0) or 0.0),
            "price_open": float(getattr(p, "price_open", 0.0) or 0.0),
            "price_current": float(getattr(p, "price_current", 0.0) or 0.0),
            "sl": float(getattr(p, "sl", 0.0) or 0.0),
            "tp": float(getattr(p, "tp", 0.0) or 0.0),
            "profit": float(getattr(p, "profit", 0.0) or 0.0),
            "type": ptype,
            "side": self._position_side_from_type(ptype),
            "time": int(getattr(p, "time", 0) or 0),
            "comment": str(getattr(p, "comment", "") or ""),
            "magic": int(getattr(p, "magic", 0) or 0),
        }

    def _serialize_order(self, order: Any) -> Dict[str, Any]:
        otype = _int_or_default(getattr(order, "type", None), -1)
        volume = float(getattr(order, "volume_current", 0.0) or 0.0)
        if volume <= 0.0:
            volume = float(getattr(order, "volume_initial", 0.0) or 0.0)
        expires_at = int(getattr(order, "time_expiration", 0) or 0)
        expiration_utc = datetime.utcfromtimestamp(expires_at).strftime("%Y-%m-%d %H:%M:%S") if expires_at > 0 else None
        return {
            "order_ticket": int(getattr(order, "ticket", 0) or 0),
            "symbol": str(getattr(order, "symbol", "")),
            "volume": volume,
            "price_open": float(getattr(order, "price_open", 0.0) or 0.0),
            "sl": float(getattr(order, "sl", 0.0) or 0.0),
            "tp": float(getattr(order, "tp", 0.0) or 0.0),
            "type": otype,
            "side": self._order_side_from_type(otype),
            "time": int(getattr(order, "time_setup", 0) or getattr(order, "time_setup_msc", 0) or 0),
            "time_expiration": expires_at,
            "expiration_utc": expiration_utc,
            "comment": str(getattr(order, "comment", "") or ""),
            "magic": int(getattr(order, "magic", 0) or 0),
        }

    def _round_symbol_price(self, value: float, digits: int) -> float:
        try:
            return round(float(value), max(int(digits), 0))
        except Exception:
            return float(value)

    def _normalize_market_sltp(
        self,
        symbol: str,
        side: str,
        price: float,
        stop_loss: float,
        take_profit: float,
        distance_multiplier: float = 1.0,
    ) -> Dict[str, Any]:
        mt5 = self._mt5
        symbol_info = mt5.symbol_info(symbol) if mt5 is not None else None

        digits = _int_or_default(getattr(symbol_info, "digits", 2), 2)
        point = float(getattr(symbol_info, "point", 0.0) or 0.0)
        if point <= 0.0:
            point = 10 ** (-max(digits, 0))

        stops_level_points = max(float(getattr(symbol_info, "trade_stops_level", 0) or 0.0), 0.0)
        freeze_level_points = max(float(getattr(symbol_info, "trade_freeze_level", 0) or 0.0), 0.0)
        min_points = max(stops_level_points, freeze_level_points, 1.0)
        distance_multiplier = max(float(distance_multiplier or 1.0), 1.0)
        min_distance = point * min_points * distance_multiplier

        ref_price = float(price or 0.0)
        sl_raw = float(stop_loss or 0.0)
        tp_raw = float(take_profit or 0.0)
        sl = sl_raw
        tp = tp_raw

        side_token = str(side or "").strip().lower()
        if side_token == "buy":
            if sl > 0.0 and sl >= (ref_price - min_distance):
                sl = ref_price - min_distance
            if tp > 0.0 and tp <= (ref_price + min_distance):
                tp = ref_price + min_distance
        elif side_token == "sell":
            if sl > 0.0 and sl <= (ref_price + min_distance):
                sl = ref_price + min_distance
            if tp > 0.0 and tp >= (ref_price - min_distance):
                tp = ref_price - min_distance

        sl = self._round_symbol_price(sl, digits) if sl > 0.0 else 0.0
        tp = self._round_symbol_price(tp, digits) if tp > 0.0 else 0.0
        ref_price = self._round_symbol_price(ref_price, digits)

        # Guard against rounding drift leaving levels invalid relative to current price.
        if side_token == "buy":
            if sl > 0.0 and sl >= ref_price:
                sl = self._round_symbol_price(ref_price - max(min_distance, point), digits)
            if tp > 0.0 and tp <= ref_price:
                tp = self._round_symbol_price(ref_price + max(min_distance, point), digits)
        elif side_token == "sell":
            if sl > 0.0 and sl <= ref_price:
                sl = self._round_symbol_price(ref_price + max(min_distance, point), digits)
            if tp > 0.0 and tp >= ref_price:
                tp = self._round_symbol_price(ref_price - max(min_distance, point), digits)

        return {
            "stop_loss": float(sl),
            "take_profit": float(tp),
            "price": float(ref_price),
            "digits": int(digits),
            "point": float(point),
            "stops_level_points": float(stops_level_points),
            "freeze_level_points": float(freeze_level_points),
            "min_distance": float(min_distance),
        }

    def list_open_positions(self) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        positions = mt5.positions_get() or []
        return {"ok": True, "positions": [self._serialize_position(p) for p in positions]}

    def list_pending_orders(self) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        orders = mt5.orders_get() or []
        return {"ok": True, "orders": [self._serialize_order(order) for order in orders]}

    def get_account_snapshot(self) -> Dict[str, Any]:
        """Return the account fields needed by pre-trade risk guards."""
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        info = self._mt5.account_info()
        if info is None:
            return {"ok": False, "message": "account_info returned None"}

        return {
            "ok": True,
            "login": int(getattr(info, "login", 0) or 0),
            "server": str(getattr(info, "server", "") or ""),
            "company": str(getattr(info, "company", "") or ""),
            "currency": str(getattr(info, "currency", "") or ""),
            "balance": float(getattr(info, "balance", 0.0) or 0.0),
            "equity": float(getattr(info, "equity", 0.0) or 0.0),
            "profit": float(getattr(info, "profit", 0.0) or 0.0),
            "margin": float(getattr(info, "margin", 0.0) or 0.0),
            "margin_free": float(getattr(info, "margin_free", 0.0) or 0.0),
            "trade_allowed": bool(getattr(info, "trade_allowed", False)),
            "trade_expert": bool(getattr(info, "trade_expert", False)),
        }

    def estimate_trade_loss(
        self,
        symbol: str,
        side: str,
        volume: float,
        entry: float,
        stop_loss: float,
    ) -> Dict[str, Any]:
        """Estimate broker-currency loss at the requested hard stop."""
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        side_token = str(side or "").strip().lower()
        if side_token not in {"buy", "sell"}:
            return {"ok": False, "message": f"unsupported side: {side_token}"}

        entry_price = float(entry or 0.0)
        stop_price = float(stop_loss or 0.0)
        requested_volume = float(volume or 0.0)
        if entry_price <= 0.0 or stop_price <= 0.0 or requested_volume <= 0.0:
            return {"ok": False, "message": "entry, stop_loss, and volume must be positive"}

        order_type = mt5.ORDER_TYPE_BUY if side_token == "buy" else mt5.ORDER_TYPE_SELL
        estimate = mt5.order_calc_profit(order_type, str(symbol), requested_volume, entry_price, stop_price)
        if estimate is None:
            return {
                "ok": False,
                "message": "order_calc_profit returned None",
                "last_error": mt5.last_error(),
            }

        estimated_pnl = float(estimate)
        return {
            "ok": True,
            "symbol": str(symbol),
            "side": side_token,
            "volume": requested_volume,
            "entry": entry_price,
            "stop_loss": stop_price,
            "estimated_pnl": estimated_pnl,
            "estimated_loss": max(-estimated_pnl, 0.0),
        }

    def get_symbol_trade_spec(self, symbol: str) -> Dict[str, Any]:
        """Return broker volume constraints used for fail-closed risk sizing."""
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}
        info = self._mt5.symbol_info(str(symbol))
        if info is None:
            return {"ok": False, "message": "symbol_info returned None"}
        return {
            "ok": True,
            "symbol": str(symbol),
            "volume_min": float(getattr(info, "volume_min", 0.0) or 0.0),
            "volume_max": float(getattr(info, "volume_max", 0.0) or 0.0),
            "volume_step": float(getattr(info, "volume_step", 0.0) or 0.0),
        }

    def get_utc_day_realized_pnl(self, now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        """Return today's closed trading P/L, including costs, from 00:00 UTC."""
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        now = now_utc or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        else:
            now = now.astimezone(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        try:
            deals = self._mt5.history_deals_get(start, now) or []
        except Exception as exc:
            return {"ok": False, "message": f"history_deals_get failed: {exc}"}

        mt5 = self._mt5
        closing_entries = {
            int(getattr(mt5, "DEAL_ENTRY_OUT", -2001)),
            int(getattr(mt5, "DEAL_ENTRY_OUT_BY", -2002)),
            int(getattr(mt5, "DEAL_ENTRY_INOUT", -2003)),
        }
        realized = 0.0
        counted = 0
        for deal in deals:
            entry_code = int(getattr(deal, "entry", -1) or -1)
            if entry_code not in closing_entries:
                continue
            realized += float(getattr(deal, "profit", 0.0) or 0.0)
            realized += float(getattr(deal, "commission", 0.0) or 0.0)
            realized += float(getattr(deal, "swap", 0.0) or 0.0)
            realized += float(getattr(deal, "fee", 0.0) or 0.0)
            counted += 1

        return {
            "ok": True,
            "start_utc": start.isoformat(),
            "end_utc": now.isoformat(),
            "realized_pnl": float(realized),
            "closing_deals": int(counted),
        }

    def _candidate_filling_modes(self, symbol: str) -> list[int]:
        mt5 = self._mt5
        if mt5 is None:
            return []

        candidates: list[int] = []
        try:
            symbol_info = mt5.symbol_info(symbol)
        except Exception:
            symbol_info = None

        preferred_mode = getattr(symbol_info, "filling_mode", None) if symbol_info is not None else None
        for mode in [preferred_mode, getattr(mt5, "ORDER_FILLING_IOC", None), getattr(mt5, "ORDER_FILLING_FOK", None), getattr(mt5, "ORDER_FILLING_RETURN", None)]:
            if isinstance(mode, int) and mode not in candidates:
                candidates.append(mode)
        return candidates

    def _send_order_with_filling_fallback(self, symbol: str, request: Dict[str, Any]) -> Dict[str, Any]:
        mt5 = self._mt5
        if mt5 is None:
            return {"ok": False, "message": "mt5_not_connected"}

        attempted_modes: list[int] = []
        last_retcode = -1
        last_result = None
        retry_retcode_set = {10013, 10030}
        for filling_mode in self._candidate_filling_modes(symbol):
            attempted_modes.append(int(filling_mode))
            current_request = dict(request)
            current_request["type_filling"] = int(filling_mode)
            result = mt5.order_send(current_request)
            last_result = result
            if result is None:
                return {"ok": False, "message": "order_send returned None", "attempted_filling_modes": attempted_modes}

            retcode = int(getattr(result, "retcode", -1))
            last_retcode = retcode
            if retcode == mt5.TRADE_RETCODE_DONE:
                return {
                    "ok": True,
                    "result": result,
                    "retcode": retcode,
                    "type_filling": int(filling_mode),
                    "attempted_filling_modes": attempted_modes,
                }
            if retcode not in retry_retcode_set:
                return {
                    "ok": False,
                    "message": f"order_send failed retcode={retcode}",
                    "retcode": retcode,
                    "type_filling": int(filling_mode),
                    "attempted_filling_modes": attempted_modes,
                }

        return {
            "ok": False,
            "message": f"order_send failed retcode={last_retcode}",
            "retcode": last_retcode,
            "result": last_result,
            "attempted_filling_modes": attempted_modes,
        }

    def _deal_reason_label(self, reason_code: int) -> str:
        mt5 = self._mt5
        if mt5 is None:
            return str(int(reason_code))

        reason_map = {
            int(getattr(mt5, "DEAL_REASON_CLIENT", -1001)): "client",
            int(getattr(mt5, "DEAL_REASON_MOBILE", -1002)): "mobile",
            int(getattr(mt5, "DEAL_REASON_WEB", -1003)): "web",
            int(getattr(mt5, "DEAL_REASON_EXPERT", -1004)): "expert",
            int(getattr(mt5, "DEAL_REASON_SL", -1005)): "sl",
            int(getattr(mt5, "DEAL_REASON_TP", -1006)): "tp",
            int(getattr(mt5, "DEAL_REASON_SO", -1007)): "so",
            int(getattr(mt5, "DEAL_REASON_VMARGIN", -1008)): "vmargin",
            int(getattr(mt5, "DEAL_REASON_ROLLOVER", -1009)): "rollover",
        }
        return reason_map.get(int(reason_code), str(int(reason_code)))

    def get_position_close_outcome(self, ticket: int, lookback_hours: int = 168) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        start = datetime.utcnow().timestamp() - max(int(lookback_hours), 1) * 60 * 60
        end = datetime.utcnow().timestamp()
        try:
            deals = mt5.history_deals_get(start, end) or []
        except Exception as exc:
            return {"ok": False, "message": f"history_deals_get failed: {exc}"}

        out_deals = []
        out_entry_codes = {
            int(getattr(mt5, "DEAL_ENTRY_OUT", -2001)),
            int(getattr(mt5, "DEAL_ENTRY_OUT_BY", -2002)),
        }

        for deal in deals:
            position_id = int(getattr(deal, "position_id", 0) or 0)
            if position_id != int(ticket):
                continue
            entry_code = int(getattr(deal, "entry", -1) or -1)
            if entry_code not in out_entry_codes:
                continue
            out_deals.append(deal)

        if not out_deals:
            return {"ok": True, "found": False, "ticket": int(ticket)}

        deal = sorted(out_deals, key=lambda item: int(getattr(item, "time", 0) or 0))[-1]
        reason_code = int(getattr(deal, "reason", -1) or -1)
        return {
            "ok": True,
            "found": True,
            "ticket": int(ticket),
            "deal_ticket": int(getattr(deal, "ticket", 0) or 0),
            "position_id": int(getattr(deal, "position_id", 0) or 0),
            "order": int(getattr(deal, "order", 0) or 0),
            "time": int(getattr(deal, "time", 0) or 0),
            "price": float(getattr(deal, "price", 0.0) or 0.0),
            "profit": float(getattr(deal, "profit", 0.0) or 0.0),
            "volume": float(getattr(deal, "volume", 0.0) or 0.0),
            "symbol": str(getattr(deal, "symbol", "") or ""),
            "reason_code": reason_code,
            "reason_label": self._deal_reason_label(reason_code),
            "comment": str(getattr(deal, "comment", "") or ""),
        }

    def place_programmed_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        entry: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        expiration_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        if not mt5.symbol_select(symbol, True):
            return {"ok": False, "message": f"symbol_select failed for {symbol}"}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "message": f"No market tick for {symbol}"}

        side = str(side or "").lower()
        if side not in {"buy", "sell"}:
            return {"ok": False, "message": f"Unsupported side: {side}"}

        if side == "buy":
            order_type = mt5.ORDER_TYPE_BUY_LIMIT if entry <= tick.ask else mt5.ORDER_TYPE_BUY_STOP
        else:
            order_type = mt5.ORDER_TYPE_SELL_LIMIT if entry >= tick.bid else mt5.ORDER_TYPE_SELL_STOP

        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(entry),
            "sl": float(stop_loss or 0.0),
            "tp": float(take_profit or 0.0),
            "deviation": 20,
            "magic": 7070001,
            "comment": "TSMM programmed order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        if expiration_utc is not None:
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = int(expiration_utc.replace(tzinfo=timezone.utc).timestamp())

        result = mt5.order_send(request)
        if result is None:
            return {"ok": False, "message": "order_send returned None"}

        retcode = int(getattr(result, "retcode", -1))
        if retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "ok": False,
                "message": f"order_send failed retcode={retcode}",
                "retcode": retcode,
            }

        return {
            "ok": True,
            "order_ticket": int(getattr(result, "order", 0) or 0),
            "deal_ticket": int(getattr(result, "deal", 0) or 0),
            "retcode": retcode,
            "expiration_utc": expiration_utc.strftime("%Y-%m-%d %H:%M:%S") if expiration_utc is not None else None,
        }

    def place_market_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        if not mt5.symbol_select(symbol, True):
            return {"ok": False, "message": f"symbol_select failed for {symbol}"}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "message": f"No market tick for {symbol}"}

        side = str(side or "").lower()
        if side not in {"buy", "sell"}:
            return {"ok": False, "message": f"Unsupported side: {side}"}

        if side == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)
            position_type = int(getattr(mt5, "POSITION_TYPE_BUY", 0))
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)
            position_type = int(getattr(mt5, "POSITION_TYPE_SELL", 1))

        normalized = self._normalize_market_sltp(
            symbol=symbol,
            side=side,
            price=price,
            stop_loss=float(stop_loss or 0.0),
            take_profit=float(take_profit or 0.0),
            distance_multiplier=1.0,
        )
        effective_stop_loss = float(normalized.get("stop_loss", 0.0) or 0.0)
        effective_take_profit = float(normalized.get("take_profit", 0.0) or 0.0)
        effective_price = float(normalized.get("price", price) or price)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(effective_price),
            "sl": float(effective_stop_loss),
            "tp": float(effective_take_profit),
            "deviation": 20,
            "magic": 7070001,
            "comment": "TSMM market order",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        send_res = self._send_order_with_filling_fallback(symbol, request)
        if not send_res.get("ok") and int(send_res.get("retcode", -1) or -1) == 10016:
            fresh_tick = mt5.symbol_info_tick(symbol)
            if fresh_tick is not None:
                retry_price = float(fresh_tick.ask) if side == "buy" else float(fresh_tick.bid)
                retry_norm = self._normalize_market_sltp(
                    symbol=symbol,
                    side=side,
                    price=retry_price,
                    stop_loss=float(stop_loss or 0.0),
                    take_profit=float(take_profit or 0.0),
                    distance_multiplier=2.0,
                )
                retry_request = dict(request)
                retry_request["price"] = float(retry_norm.get("price", retry_price) or retry_price)
                retry_request["sl"] = float(retry_norm.get("stop_loss", 0.0) or 0.0)
                retry_request["tp"] = float(retry_norm.get("take_profit", 0.0) or 0.0)
                send_res = self._send_order_with_filling_fallback(symbol, retry_request)
                if send_res.get("ok"):
                    effective_price = float(retry_request["price"])
                    effective_stop_loss = float(retry_request["sl"])
                    effective_take_profit = float(retry_request["tp"])

        post_open_risk_update = None
        if not send_res.get("ok") and int(send_res.get("retcode", -1) or -1) == 10016:
            # Last-resort fallback for brokers enforcing dynamic stops not exposed
            # via symbol_info: open at market without SL/TP, then apply risk levels
            # after fill using a dedicated SLTP modify request.
            bare_request = dict(request)
            bare_request["sl"] = 0.0
            bare_request["tp"] = 0.0
            bare_send_res = self._send_order_with_filling_fallback(symbol, bare_request)
            if bare_send_res.get("ok"):
                send_res = bare_send_res
                effective_price = float(bare_request.get("price", effective_price) or effective_price)
            else:
                send_res = bare_send_res

        if not send_res.get("ok"):
            return send_res
        result = send_res.get("result")
        retcode = int(send_res.get("retcode", -1) or -1)

        matched_position = None
        order_ticket = int(getattr(result, "order", 0) or 0)
        if order_ticket > 0:
            pos = mt5.positions_get(ticket=order_ticket) or []
            if pos:
                matched_position = pos[0]

        if matched_position is None:
            positions = mt5.positions_get(symbol=symbol) or []
            for p in positions:
                if _int_or_default(getattr(p, "type", None), -1) != position_type:
                    continue
                if int(getattr(p, "magic", 0) or 0) != 7070001:
                    continue
                if str(getattr(p, "comment", "") or "") != "TSMM market order":
                    continue
                if abs(float(getattr(p, "volume", 0.0) or 0.0) - float(volume)) > 1e-9:
                    continue
                if effective_stop_loss and abs(float(getattr(p, "sl", 0.0) or 0.0) - float(effective_stop_loss)) > 0.05:
                    continue
                if effective_take_profit and abs(float(getattr(p, "tp", 0.0) or 0.0) - float(effective_take_profit)) > 0.05:
                    continue
                matched_position = p
                break

        # Retry position lookup if order succeeded but position isn't visible yet.
        # Some brokers (e.g. FTMO) have higher latency before the position appears
        # after a successful order send.
        if matched_position is None and send_res.get("ok") and int(send_res.get("retcode", -1) or -1) == mt5.TRADE_RETCODE_DONE:
            for _ in range(5):
                time.sleep(1.0)
                if order_ticket > 0:
                    pos = mt5.positions_get(ticket=order_ticket) or []
                    if pos:
                        matched_position = pos[0]
                        break
                positions = mt5.positions_get(symbol=symbol) or []
                for p in positions:
                    if _int_or_default(getattr(p, "type", None), -1) != position_type:
                        continue
                    if int(getattr(p, "magic", 0) or 0) != 7070001:
                        continue
                    if str(getattr(p, "comment", "") or "") != "TSMM market order":
                        continue
                    if abs(float(getattr(p, "volume", 0.0) or 0.0) - float(volume)) > 1e-9:
                        continue
                    matched_position = p
                    break
                if matched_position:
                    break

        if int(send_res.get("retcode", -1) or -1) == mt5.TRADE_RETCODE_DONE and (
            float(effective_stop_loss or 0.0) > 0.0 or float(effective_take_profit or 0.0) > 0.0
        ):
            has_live_sltp = False
            if matched_position is not None:
                current_sl = float(getattr(matched_position, "sl", 0.0) or 0.0)
                current_tp = float(getattr(matched_position, "tp", 0.0) or 0.0)
                has_live_sltp = current_sl > 0.0 or current_tp > 0.0

            if not has_live_sltp:
                pos_ticket = int(getattr(matched_position, "ticket", 0) or 0)
                if pos_ticket <= 0:
                    pos_ticket = int(order_ticket)

                if pos_ticket > 0:
                    sltp_tick = mt5.symbol_info_tick(symbol)
                    sltp_price = float(sltp_tick.ask) if (sltp_tick is not None and side == "buy") else (
                        float(sltp_tick.bid) if sltp_tick is not None else float(effective_price)
                    )
                    post_norm = self._normalize_market_sltp(
                        symbol=symbol,
                        side=side,
                        price=float(sltp_price),
                        stop_loss=float(stop_loss or 0.0),
                        take_profit=float(take_profit or 0.0),
                        distance_multiplier=4.0,
                    )
                    post_open_risk_update = self.modify_position_risk(
                        pos_ticket,
                        stop_loss=float(post_norm.get("stop_loss", 0.0) or 0.0),
                        take_profit=float(post_norm.get("take_profit", 0.0) or 0.0),
                    )
                    if not bool(post_open_risk_update.get("ok", False)) and int(post_open_risk_update.get("retcode", -1) or -1) == 10016:
                        post_norm_wide = self._normalize_market_sltp(
                            symbol=symbol,
                            side=side,
                            price=float(sltp_price),
                            stop_loss=float(stop_loss or 0.0),
                            take_profit=float(take_profit or 0.0),
                            distance_multiplier=8.0,
                        )
                        post_open_risk_update = self.modify_position_risk(
                            pos_ticket,
                            stop_loss=float(post_norm_wide.get("stop_loss", 0.0) or 0.0),
                            take_profit=float(post_norm_wide.get("take_profit", 0.0) or 0.0),
                        )

                    if bool(post_open_risk_update.get("ok", False)) and isinstance(post_open_risk_update.get("position"), dict):
                        pos_payload = post_open_risk_update.get("position") or {}
                        effective_stop_loss = float(pos_payload.get("sl", post_open_risk_update.get("stop_loss", effective_stop_loss)) or 0.0)
                        effective_take_profit = float(pos_payload.get("tp", post_open_risk_update.get("take_profit", effective_take_profit)) or 0.0)
                    else:
                        effective_stop_loss = float(post_norm.get("stop_loss", effective_stop_loss) or 0.0)
                        effective_take_profit = float(post_norm.get("take_profit", effective_take_profit) or 0.0)

        return {
            "ok": True,
            "order_ticket": order_ticket,
            "deal_ticket": int(getattr(result, "deal", 0) or 0),
            "retcode": retcode,
            "position": self._serialize_position(matched_position) if matched_position is not None else None,
            "execution_price": effective_price,
            "stop_loss": effective_stop_loss,
            "take_profit": effective_take_profit,
            "execution_mode": "market",
            "type_filling": send_res.get("type_filling"),
            "attempted_filling_modes": send_res.get("attempted_filling_modes") or [],
            "post_open_risk_update": post_open_risk_update,
        }

    def find_position_by_order(self, order_ticket: int) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        positions = mt5.positions_get() or []

        for p in positions:
            p_order = int(getattr(p, "ticket", 0) or 0)
            if p_order == int(order_ticket):
                return {"ok": True, "position": self._serialize_position(p)}

        # Some MT5 pending fills do not preserve the order ticket on the resulting
        # position, so fall back to recent order metadata and match by strategy keys.
        try:
            start = datetime.utcnow().timestamp() - 7 * 24 * 60 * 60
            end = datetime.utcnow().timestamp()
            history_orders = mt5.history_orders_get(start, end) or []
        except Exception:
            history_orders = []

        matched_order = None
        for order in reversed(list(history_orders)):
            if int(getattr(order, "ticket", 0) or 0) == int(order_ticket):
                matched_order = order
                break

        if matched_order is not None:
            position_id = int(getattr(matched_order, "position_id", 0) or 0)
            if position_id:
                pos = mt5.positions_get(ticket=position_id) or []
                if pos:
                    return {"ok": True, "position": self._serialize_position(pos[0])}

            symbol = str(getattr(matched_order, "symbol", "") or "")
            magic = int(getattr(matched_order, "magic", 0) or 0)
            comment = str(getattr(matched_order, "comment", "") or "")
            volume_initial = float(getattr(matched_order, "volume_initial", 0.0) or 0.0)
            price_open = float(getattr(matched_order, "price_open", 0.0) or 0.0)
            sl = float(getattr(matched_order, "sl", 0.0) or 0.0)
            tp = float(getattr(matched_order, "tp", 0.0) or 0.0)

            for p in positions:
                if symbol and str(getattr(p, "symbol", "") or "") != symbol:
                    continue
                if magic and int(getattr(p, "magic", 0) or 0) != magic:
                    continue
                if comment and str(getattr(p, "comment", "") or "") != comment:
                    continue
                if volume_initial and abs(float(getattr(p, "volume", 0.0) or 0.0) - volume_initial) > 1e-9:
                    continue
                if price_open and abs(float(getattr(p, "price_open", 0.0) or 0.0) - price_open) > 0.05:
                    continue
                if sl and abs(float(getattr(p, "sl", 0.0) or 0.0) - sl) > 0.05:
                    continue
                if tp and abs(float(getattr(p, "tp", 0.0) or 0.0) - tp) > 0.05:
                    continue
                return {"ok": True, "position": self._serialize_position(p)}

        return {"ok": True, "position": None}

    def find_live_position_by_plan(
        self,
        symbol: str,
        volume: float,
        entry: float,
        stop_loss: float,
        take_profit: float,
        price_tolerance: float = 0.05,
        volume_tolerance: float = 1e-9,
    ) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        positions = mt5.positions_get() or []
        target_symbol = str(symbol or "").strip()
        target_volume = float(volume or 0.0)
        target_entry = float(entry or 0.0)
        target_sl = float(stop_loss or 0.0)
        target_tp = float(take_profit or 0.0)
        tol = max(float(price_tolerance or 0.05), 0.0)
        vol_tol = max(float(volume_tolerance or 1e-9), 0.0)

        for p in positions:
            if target_symbol and str(getattr(p, "symbol", "") or "") != target_symbol:
                continue
            if target_volume and abs(float(getattr(p, "volume", 0.0) or 0.0) - target_volume) > vol_tol:
                continue
            if target_entry and abs(float(getattr(p, "price_open", 0.0) or 0.0) - target_entry) > tol:
                continue
            if target_sl and abs(float(getattr(p, "sl", 0.0) or 0.0) - target_sl) > tol:
                continue
            if target_tp and abs(float(getattr(p, "tp", 0.0) or 0.0) - target_tp) > tol:
                continue
            return {"ok": True, "position": self._serialize_position(p)}

        return {"ok": True, "position": None}

    def find_pending_order_by_plan(
        self,
        symbol: str,
        volume: float,
        entry: float,
        stop_loss: float,
        take_profit: float,
        price_tolerance: float = 0.05,
        volume_tolerance: float = 1e-9,
    ) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        orders = mt5.orders_get() or []
        target_symbol = str(symbol or "").strip()
        target_volume = float(volume or 0.0)
        target_entry = float(entry or 0.0)
        target_sl = float(stop_loss or 0.0)
        target_tp = float(take_profit or 0.0)
        tol = max(float(price_tolerance or 0.05), 0.0)
        vol_tol = max(float(volume_tolerance or 1e-9), 0.0)

        for order in orders:
            if target_symbol and str(getattr(order, "symbol", "") or "") != target_symbol:
                continue
            live_volume = float(getattr(order, "volume_current", 0.0) or 0.0)
            if live_volume <= 0.0:
                live_volume = float(getattr(order, "volume_initial", 0.0) or 0.0)
            if target_volume and abs(live_volume - target_volume) > vol_tol:
                continue
            if target_entry and abs(float(getattr(order, "price_open", 0.0) or 0.0) - target_entry) > tol:
                continue
            if target_sl and abs(float(getattr(order, "sl", 0.0) or 0.0) - target_sl) > tol:
                continue
            if target_tp and abs(float(getattr(order, "tp", 0.0) or 0.0) - target_tp) > tol:
                continue
            return {"ok": True, "order": self._serialize_order(order)}

        return {"ok": True, "order": None}

    def get_pending_order_by_ticket(self, order_ticket: int) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        orders = mt5.orders_get(ticket=int(order_ticket)) or []
        if not orders:
            return {"ok": True, "order": None}

        return {"ok": True, "order": self._serialize_order(orders[0])}

    def cancel_pending_order(self, order_ticket: int) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        pending = mt5.orders_get(ticket=int(order_ticket))
        if not pending:
            return {
                "ok": True,
                "order_ticket": int(order_ticket),
                "skipped": True,
                "reason": "order_not_pending",
            }

        request = {
            "action": mt5.TRADE_ACTION_REMOVE,
            "order": int(order_ticket),
        }
        result = mt5.order_send(request)
        if result is None:
            return {
                "ok": False,
                "message": "order_send(cancel) returned None",
                "order_ticket": int(order_ticket),
                "last_error": mt5.last_error(),
                "request": request,
            }

        retcode = int(getattr(result, "retcode", -1))
        if retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "ok": False,
                "message": f"cancel order failed retcode={retcode}",
                "retcode": retcode,
                "order_ticket": int(order_ticket),
            }

        return {
            "ok": True,
            "order_ticket": int(order_ticket),
            "retcode": retcode,
        }

    def get_position_by_ticket(self, ticket: int) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        positions = mt5.positions_get(ticket=int(ticket))
        if not positions:
            return {"ok": True, "position": None}

        p = positions[0]
        return {"ok": True, "position": self._serialize_position(p)}

    def modify_position_risk(
        self,
        ticket: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        positions = mt5.positions_get(ticket=int(ticket))
        if not positions:
            return {"ok": False, "message": f"Position not found for ticket={int(ticket)}"}

        p = positions[0]
        current_sl = float(getattr(p, "sl", 0.0) or 0.0)
        current_tp = float(getattr(p, "tp", 0.0) or 0.0)
        desired_sl = float(stop_loss) if stop_loss is not None else current_sl
        desired_tp = float(take_profit) if take_profit is not None else current_tp

        symbol = str(getattr(p, "symbol", "") or "")
        position_type = int(getattr(p, "type", getattr(mt5, "POSITION_TYPE_BUY", 0)) or 0)
        side = "buy" if position_type == int(getattr(mt5, "POSITION_TYPE_BUY", 0)) else "sell"
        current_price = float(getattr(p, "price_current", 0.0) or 0.0)
        tick = mt5.symbol_info_tick(symbol) if symbol else None
        if side == "buy":
            current_price = float(getattr(tick, "bid", 0.0) or current_price)
        else:
            current_price = float(getattr(tick, "ask", 0.0) or current_price)
        normalized: Dict[str, Any] = {"skipped": True, "reason": "live_price_unavailable", "price": current_price}
        if current_price > 0.0:
            normalized = self._normalize_market_sltp(
                symbol=symbol,
                side=side,
                price=current_price,
                stop_loss=desired_sl,
                take_profit=desired_tp,
            )
            desired_sl = float(normalized.get("stop_loss", desired_sl) or 0.0)
            desired_tp = float(normalized.get("take_profit", desired_tp) or 0.0)

        if abs(desired_sl - current_sl) < 1e-9 and abs(desired_tp - current_tp) < 1e-9:
            return {
                "ok": True,
                "skipped": True,
                "reason": "risk_levels_unchanged",
                "ticket": int(ticket),
                "position": self._serialize_position(p),
            }

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": int(ticket),
            "sl": float(desired_sl),
            "tp": float(desired_tp),
            "magic": 7070002,
            "comment": "TSMM Agent B risk update",
        }

        result = mt5.order_send(request)
        if result is None:
            return {"ok": False, "message": "order_send(sltp) returned None"}

        retcode = int(getattr(result, "retcode", -1))
        if retcode != mt5.TRADE_RETCODE_DONE:
            return {
                "ok": False,
                "message": f"modify risk failed retcode={retcode}",
                "retcode": retcode,
                "ticket": int(ticket),
                "normalization": normalized,
            }

        refreshed = mt5.positions_get(ticket=int(ticket)) or []
        return {
            "ok": True,
            "ticket": int(ticket),
            "retcode": retcode,
            "stop_loss": float(desired_sl),
            "take_profit": float(desired_tp),
            "normalization": normalized,
            "position": self._serialize_position(refreshed[0]) if refreshed else None,
        }

    def close_position_by_ticket(self, ticket: int) -> Dict[str, Any]:
        ok, msg = self._require_mt5()
        if not ok:
            return {"ok": False, "message": msg}

        mt5 = self._mt5
        pos = mt5.positions_get(ticket=int(ticket))
        if not pos:
            return {"ok": True, "message": "Position already closed", "ticket": int(ticket)}

        p = pos[0]
        symbol = str(getattr(p, "symbol", ""))
        volume = float(getattr(p, "volume", 0.0) or 0.0)
        ptype = _int_or_default(getattr(p, "type", None), -1)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "message": f"No market tick for {symbol}"}

        if ptype == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = float(tick.bid)
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = float(tick.ask)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": int(ticket),
            "price": price,
            "deviation": 20,
            "magic": 7070002,
            "comment": "TSMM close by Agent B",
            "type_time": mt5.ORDER_TIME_GTC,
        }

        send_res = self._send_order_with_filling_fallback(symbol, request)
        if not send_res.get("ok"):
            if str(send_res.get("message") or "") == "order_send returned None":
                return {"ok": False, "message": "order_send(close) returned None"}
            return {
                "ok": False,
                "message": str(send_res.get("message") or "close order failed"),
                "retcode": int(send_res.get("retcode", -1) or -1),
                "attempted_filling_modes": send_res.get("attempted_filling_modes") or [],
            }
        result = send_res.get("result")
        retcode = int(send_res.get("retcode", -1) or -1)

        return {
            "ok": True,
            "ticket": int(ticket),
            "retcode": retcode,
            "deal_ticket": int(getattr(result, "deal", 0) or 0),
            "type_filling": send_res.get("type_filling"),
        }


def load_trading_config(path: str = "config/trading_agent.yaml") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"agent": {"enabled": False}, "error": f"Trading config not found: {path}"}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _choose_best_model(evaluation: Dict[str, Any], preferred_model: Optional[str] = None) -> Optional[str]:
    if preferred_model and preferred_model in (evaluation or {}):
        return preferred_model

    best = None
    best_mae = float("inf")
    for model_name, ev in (evaluation or {}).items():
        mae = (ev.get("metrics") or {}).get("MAE")
        if isinstance(mae, (int, float)) and mae < best_mae:
            best_mae = mae
            best = model_name
    return best


def _latest_price(df, target_col: str) -> float:
    return float(df[target_col].iloc[-1])


def _signal_interpretation_mode(trading_cfg: Dict[str, Any]) -> str:
    raw = str(((trading_cfg.get("agent") or {}).get("signal_interpretation") or "momentum")).strip().lower()
    if raw in {"contrarian", "mean_reversion", "mean-reversion", "fade"}:
        return "contrarian"
    return "momentum"


def _apply_signal_interpretation(signal_value: float, trading_cfg: Dict[str, Any]) -> float:
    value = float(signal_value or 0.0)
    if _signal_interpretation_mode(trading_cfg) == "contrarian":
        return -value
    return value


def _build_mode_a_plan(
    df,
    app_config: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    evaluation: Dict[str, Any],
    future_forecasts: Dict[str, Any],
    preferred_model: Optional[str] = None,
) -> Dict[str, Any]:
    risk = trading_cfg.get("risk", {}) or {}
    mode_a = trading_cfg.get("mode_a", {}) or {}
    exec_cfg = trading_cfg.get("execution", {}) or {}

    best_model = _choose_best_model(evaluation, preferred_model=preferred_model)
    if best_model is None or best_model not in future_forecasts:
        return {
            "decision": "hold",
            "rationale": "No valid forecast model available",
            "risk_notes": ["Model selection failed; no operation recommended."],
        }

    forecast_block = future_forecasts[best_model] or {}
    pred = forecast_block.get("future") or []
    if not pred:
        return {
            "decision": "hold",
            "rationale": "Missing future forecast block",
            "risk_notes": ["Forecast output missing; no operation recommended."],
        }

    feature_paths = forecast_block.get("future_by_feature", {}) or {}
    target_features = app_config.get("target_features", []) or []
    tf0 = target_features[0] if target_features else None

    first_pred = pred[0][0] if isinstance(pred[0], list) and pred[0] else (pred[0] if isinstance(pred[0], (int, float)) else 0.0)

    # Feature-level first-step values (if available)
    f_open = feature_paths.get("OPEN", [None])[0] if feature_paths.get("OPEN") else None
    f_high = feature_paths.get("HIGH", [None])[0] if feature_paths.get("HIGH") else None
    f_low = feature_paths.get("LOW", [None])[0] if feature_paths.get("LOW") else None
    f_close = feature_paths.get("CLOSE", [None])[0] if feature_paths.get("CLOSE") else None
    f_pr = feature_paths.get("Price_return", [None])[0] if feature_paths.get("Price_return") else None
    f_or = feature_paths.get("Open_return", [None])[0] if feature_paths.get("Open_return") else None
    f_yd = feature_paths.get("y_diff", [None])[0] if feature_paths.get("y_diff") else None

    # Consensus score across OHLC/return dimensions
    signal_score = 0.0
    score_parts = []

    for name, val, w in [
        ("y_diff", f_yd, 1.0),
        ("Price_return", f_pr, 1.0),
        ("Open_return", f_or, 0.5),
    ]:
        if isinstance(val, (int, float)):
            s = 1.0 if float(val) > 0 else -1.0
            signal_score += w * s
            score_parts.append(f"{name}:{float(val):.4f}")

    # If direct OHLC targets are present, use close-open directional cue
    if isinstance(f_close, (int, float)) and isinstance(f_open, (int, float)):
        delta = float(f_close) - float(f_open)
        signal_score += 1.0 if delta > 0 else -1.0
        score_parts.append(f"CLOSE-OPEN:{delta:.4f}")

    # Fallback to primary prediction sign
    if abs(signal_score) < 1e-9:
        signal_score = 1.0 if float(first_pred) > 0 else -1.0

    raw_signal_score = float(signal_score)
    signal_score = _apply_signal_interpretation(raw_signal_score, trading_cfg)
    direction = "buy" if signal_score > 0 else "sell"

    cm_acc = float(((evaluation.get(best_model, {}) or {}).get("confusion_matrix", {}) or {}).get("accuracy", 0.0) or 0.0)
    conf_levels = (evaluation.get(best_model, {}) or {}).get("confidence_levels", []) or []
    confidence = float(np.mean(conf_levels[:3])) if conf_levels else 0.5

    min_conf = float(risk.get("min_confidence_to_trade", 0.55))
    min_cm = float(risk.get("min_cm_accuracy_to_trade", 0.52))
    max_input_fooling_risk = float(risk.get("max_input_fooling_risk", 0.45))

    fooling_info = (evaluation.get(best_model, {}) or {}).get("input_fooling_risk", {}) or {}
    p_wrong = fooling_info.get("probability_wrong_sign")
    try:
        p_wrong = float(p_wrong) if p_wrong is not None else None
    except Exception:
        p_wrong = None

    allow_long = bool(mode_a.get("allow_long", True))
    allow_short = bool(mode_a.get("allow_short", True))
    block_on_confidence_thresholds = bool(mode_a.get("block_on_confidence_thresholds", False))
    block_on_input_fooling_risk = bool(mode_a.get("block_on_input_fooling_risk", False))

    entry = _latest_price(df, app_config["target_col"])
    sl_pct = float(risk.get("stop_loss_pct", 0.8)) / 100.0
    tp_pct = float(risk.get("take_profit_pct", 1.6)) / 100.0

    if direction == "buy":
        stop_loss = entry * (1 - sl_pct)
        take_profit = entry * (1 + tp_pct)
    else:
        stop_loss = entry * (1 + sl_pct)
        take_profit = entry * (1 - tp_pct)

    decision = direction
    rationale = (
        f"Model={best_model}, score={signal_score:.2f}, "
        f"forecast_sign={float(first_pred):.4f}, raw_score={raw_signal_score:.2f}, cm_accuracy={cm_acc:.3f}, confidence={confidence:.3f}, "
        f"features=[{', '.join(score_parts[:6])}]"
    )
    risk_notes: List[str] = [
        f"Risk per trade={risk.get('risk_per_trade_pct', 0.5)}%",
        f"Daily max loss={risk.get('daily_max_loss_pct', 2.0)}%",
        f"Max open positions={risk.get('max_open_positions', 3)}",
    ]
    if _signal_interpretation_mode(trading_cfg) == "contrarian":
        risk_notes.append("Signal interpretation=contrarian: sell strength and buy weakness.")

    confidence_threshold_breached = confidence < min_conf or cm_acc < min_cm
    if confidence_threshold_breached:
        if block_on_confidence_thresholds:
            decision = "hold"
            risk_notes.append("Signal blocked by confidence/confusion thresholds.")
        else:
            risk_notes.append(
                "Confidence/confusion thresholds were breached, but the 7h base plan remains active by policy."
            )

    input_fooling_risk_breached = p_wrong is not None and p_wrong > max_input_fooling_risk
    if input_fooling_risk_breached:
        if block_on_input_fooling_risk:
            decision = "hold"
            risk_notes.append(
                f"Signal blocked by per-timeframe input fooling risk: p_wrong={p_wrong:.3f} > {max_input_fooling_risk:.3f}"
            )
        else:
            risk_notes.append(
                f"Input fooling risk is elevated (p_wrong={p_wrong:.3f} > {max_input_fooling_risk:.3f}), but the 7h base plan remains active by policy."
            )

    if decision == "buy" and not allow_long:
        decision = "hold"
        risk_notes.append("Long operations disabled by config.")
    if decision == "sell" and not allow_short:
        decision = "hold"
        risk_notes.append("Short operations disabled by config.")

    return {
        "decision": decision,
        "model": best_model,
        "entry": round(entry, 6),
        "stop_loss": round(float(stop_loss), 6),
        "take_profit": round(float(take_profit), 6),
        "volume": float(exec_cfg.get("default_volume", 0.01)),
        "confidence": round(confidence, 4),
        "cm_accuracy": round(cm_acc, 4),
        "signal_score": round(signal_score, 4),
        "raw_signal_score": round(raw_signal_score, 4),
        "signal_interpretation": _signal_interpretation_mode(trading_cfg),
        "input_fooling_risk": (round(p_wrong, 4) if p_wrong is not None else None),
        "confidence_threshold_breached": bool(confidence_threshold_breached),
        "input_fooling_risk_breached": bool(input_fooling_risk_breached),
        "target_anchor": tf0,
        "feature_forecasts_step1": {
            "OPEN": f_open,
            "HIGH": f_high,
            "LOW": f_low,
            "CLOSE": f_close,
            "Price_return": f_pr,
            "Open_return": f_or,
            "y_diff": f_yd,
        },
        "rationale": rationale,
        "risk_notes": risk_notes,
    }


def _discover_agent_a_enrichment_candidates(trading_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = (trading_cfg.get("agent_a_fallback") or {})
    target_families = [
        str(x).strip().lower()
        for x in (cfg.get("target_families") or ["high", "low", "close", "open"])
        if str(x).strip()
    ]
    endpoint_map = dict(trading_cfg.get("model_endpoints") or {})
    project_root = Path(__file__).resolve().parents[1]
    config_root = project_root / "config"
    endpoint_versions: Dict[str, Any] = {}
    versions_path = config_root / "model_endpoint_versions.yaml"
    if versions_path.exists():
        try:
            with versions_path.open("r", encoding="utf-8") as stream:
                endpoint_versions = dict((yaml.safe_load(stream) or {}).get("endpoints") or {})
        except Exception:
            endpoint_versions = {}
    if not config_root.exists() or not endpoint_map:
        return []

    discovered: List[Dict[str, Any]] = []
    deployment_root_value = str(
        ((trading_cfg.get("model_registry") or {}).get("deployment_root") or "")
    ).strip()
    deployment_root = None
    if deployment_root_value:
        deployment_root = Path(deployment_root_value)
        if not deployment_root.is_absolute():
            deployment_root = (project_root / deployment_root).resolve()
    for tf in endpoint_map.keys():
        tf_label = str(tf).strip()
        if not tf_label:
            continue
        for family in target_families:
            active_deployment = resolve_active_deployment(
                f"{tf_label}_{family}", deployment_root=deployment_root
            )
            if active_deployment is not None:
                metrics = active_deployment.get("metrics") or {}
                qualification = active_deployment.get("qualification") or {}
                deployed_r2 = metrics.get("holdout_r2", qualification.get("score", 0.0))
                discovered.append(
                    {
                        "family": family,
                        "timeframe": tf_label,
                        "model": str(active_deployment.get("model") or "").lower(),
                        "config_path": str(active_deployment.get("config_path") or ""),
                        "model_path": str(active_deployment.get("model_path") or ""),
                        "artifacts_path": active_deployment.get("artifacts_path"),
                        "deployment_id": active_deployment.get("deployment_id"),
                        "training_data_first_index": active_deployment.get("training_data_first_index"),
                        "training_data_last_index": active_deployment.get("training_data_last_index"),
                        "r2": float(deployed_r2 or 0.0),
                        "refreshed_r2": float(deployed_r2) if deployed_r2 is not None else None,
                        "validation_status": "activated_bundle",
                    }
                )
                continue
            if deployment_root is not None:
                # An asset-specific registry is a hard namespace boundary.
                # Never substitute legacy XAUUSD Result folders when (for
                # example) a US500 endpoint has not been activated yet.
                continue
            tf_dir = config_root / f"{family}{tf_label}Results"
            if not tf_dir.exists() or not tf_dir.is_dir():
                continue

            best_path: Path | None = None
            best_model = ""
            best_r2 = -1.0
            for model_dir in [d for d in tf_dir.iterdir() if d.is_dir()]:
                cfg_files = list(model_dir.glob("top1*.yaml")) + list(model_dir.glob("top1*.yml"))
                if not cfg_files:
                    cfg_files = list(model_dir.glob("*.yaml")) + list(model_dir.glob("*.yml"))
                for cfg_file in cfg_files:
                    r2 = _parse_r2_from_filename(str(cfg_file))
                    if r2 > best_r2:
                        best_r2 = r2
                        best_path = cfg_file
                        best_model = model_dir.name

            if best_path is None:
                continue

            version_current = dict(
                (endpoint_versions.get(f"{tf_label}_{family}") or {}).get("current") or {}
            )
            discovered.append(
                {
                    "family": family,
                    "timeframe": tf_label,
                    "model": best_model,
                    "config_path": str(best_path),
                    "r2": float(best_r2),
                    "refreshed_r2": version_current.get("refreshed_r2"),
                    "validation_status": version_current.get("validation_status"),
                }
            )

    timeframe_priority = {"7h": 0, "3h": 1, "1h": 2, "30m": 3, "10m": 4, "12h": 5, "24h": 6, "1w": 7}
    return sorted(
        discovered,
        key=lambda x: (int(timeframe_priority.get(str(x.get("timeframe")), 999)), -float(x.get("r2", 0.0))),
    )


def _collect_agent_a_enrichment_signals(
    trading_cfg: Dict[str, Any],
    timeout_sec: float = 3.0,
) -> Dict[str, Any]:
    model_endpoints = dict(trading_cfg.get("model_endpoints") or {})
    if not model_endpoints:
        return {"enabled": False, "reason": "No model endpoints configured", "signals": {}, "consensus": "hold", "consensus_score": 0.0}

    candidates = _discover_agent_a_enrichment_candidates(trading_cfg)
    if not candidates:
        return {"enabled": False, "reason": "No pretrained enrichment candidates discovered", "signals": {}, "consensus": "hold", "consensus_score": 0.0}

    timeframe_weights = {
        "7h": 2.4,
        "3h": 1.8,
        "1h": 1.4,
        "30m": 1.1,
        "10m": 0.9,
        "12h": 1.6,
        "24h": 1.5,
        "1w": 1.7,
    }

    signals: Dict[str, Any] = {}
    weighted = 0.0
    total_w = 0.0
    qualified_count = 0
    quality_cfg = dict(trading_cfg.get("model_quality") or {})
    minimum_r2 = float(quality_cfg.get("minimum_r2_for_vote", 0.0) or 0.0)
    legacy_static_discount = float(quality_cfg.get("legacy_static_score_discount", 0.35) or 0.0)

    for item in candidates:
        tf = str(item.get("timeframe") or "").strip()
        endpoint_cfg = model_endpoints.get(tf)
        if endpoint_cfg is None:
            continue

        try:
            sig_bundle = _collect_mode_b_signals(
                model_endpoints={tf: endpoint_cfg},
                trading_cfg=trading_cfg,
                timeout_sec=timeout_sec,
                config_overrides={tf: str(item.get("config_path") or "")},
            )
            tf_sig = ((sig_bundle.get("timeframes") or {}).get(tf) or {})
            conf = float(tf_sig.get("confidence", 0.5) or 0.5)
            signal = int(tf_sig.get("signal", 0) or 0)
            tf_weight = float(timeframe_weights.get(tf, 1.0))
            quality = model_quality_weight(
                item.get("r2"), item.get("refreshed_r2"), minimum_r2, legacy_static_discount
            )
            vote_weight = max(conf, 0.01) * tf_weight * float(quality["weight"])
            if quality["qualified"] and signal != 0:
                qualified_count += 1

            key = f"{item.get('family')}:{tf}"
            signals[key] = {
                "family": item.get("family"),
                "timeframe": tf,
                "model": item.get("model"),
                "config_path": item.get("config_path"),
                "r2": item.get("r2"),
                "refreshed_r2": item.get("refreshed_r2"),
                "validation_status": item.get("validation_status"),
                "quality": quality,
                "signal": signal,
                "confidence": conf,
                "vote_weight": vote_weight,
                "raw": tf_sig.get("raw"),
                "forecast_delta": tf_sig.get("forecast_delta"),
                "reference_price": tf_sig.get("reference_price"),
                "forecast_price": tf_sig.get("forecast_price"),
                "error": tf_sig.get("error"),
            }

            weighted += vote_weight * signal
            total_w += vote_weight
        except Exception as e:
            key = f"{item.get('family')}:{tf}"
            signals[key] = {
                "family": item.get("family"),
                "timeframe": tf,
                "model": item.get("model"),
                "config_path": item.get("config_path"),
                "r2": item.get("r2"),
                "signal": 0,
                "confidence": 0.5,
                "vote_weight": 0.0,
                "error": str(e),
            }

    consensus_score = weighted / total_w if total_w > 0 else 0.0
    consensus = "buy" if consensus_score > 0.1 else ("sell" if consensus_score < -0.1 else "hold")
    return {
        "enabled": True,
        "signals": signals,
        "consensus": consensus,
        "consensus_score": float(consensus_score),
        "n_signals": len(signals),
        "n_qualified_signals": qualified_count,
        "avg_confidence": float(np.mean([
            float(value.get("confidence", 0.5) or 0.5)
            for value in signals.values() if float(value.get("vote_weight", 0.0) or 0.0) > 0.0
        ])) if qualified_count else 0.0,
    }


def _collect_all_model_assessment_signals(
    trading_cfg: Dict[str, Any],
    timeout_sec: float = 3.0,
) -> Dict[str, Any]:
    enrichment = _collect_agent_a_enrichment_signals(trading_cfg=trading_cfg, timeout_sec=timeout_sec)
    if bool(enrichment.get("enabled")) and int(enrichment.get("n_signals", 0) or 0) > 0:
        mode_b_cfg = dict(trading_cfg.get("mode_b") or {})
        management_weights = dict(
            mode_b_cfg.get("management_timeframe_weights")
            or {"10m": 2.4, "30m": 2.0, "1h": 1.5, "3h": 0.8, "7h": 0.4}
        )
        management = weighted_timeframe_consensus(
            dict(enrichment.get("signals") or {}),
            management_weights,
            family_weights=(trading_cfg.get("signal_policy") or {}).get("family_weights"),
            minimum_families=1,
            decision_threshold=float(
                mode_b_cfg.get("management_consensus_threshold", 0.20) or 0.20
            ),
        )
        return {
            "assessment_scope": "all_models",
            "signals": dict(enrichment.get("signals") or {}),
            "consensus": str(management.get("decision") or "hold"),
            "consensus_score": float(management.get("score", 0.0) or 0.0),
            "management_consensus_detail": management,
            "entry_consensus": str(enrichment.get("consensus") or "hold"),
            "entry_consensus_score": float(enrichment.get("consensus_score", 0.0) or 0.0),
            "n_signals": int(enrichment.get("n_signals", 0) or 0),
            "avg_confidence": float(enrichment.get("avg_confidence", 0.0) or 0.0),
            "source": "agent_a_enrichment",
        }

    mtf = _collect_mode_b_signals(trading_cfg.get("model_endpoints", {}), trading_cfg=trading_cfg, timeout_sec=timeout_sec)
    return {
        "assessment_scope": "timeframe_endpoints",
        "timeframes": dict(mtf.get("timeframes") or {}),
        "consensus": str(mtf.get("consensus") or "hold"),
        "consensus_score": float(mtf.get("consensus_score", 0.0) or 0.0),
        "n_timeframes": int(mtf.get("n_timeframes", 0) or 0),
        "source": "mode_b_fallback",
        "fallback_reason": str(enrichment.get("reason") or "agent_a_enrichment_unavailable"),
    }


def _apply_agent_a_enrichment_to_plan(
    plan: Dict[str, Any],
    enrichment: Dict[str, Any],
    trading_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def _recompute_risk_levels(payload: Dict[str, Any], side: str) -> None:
        entry = payload.get("entry")
        stop_loss = payload.get("stop_loss")
        take_profit = payload.get("take_profit")
        try:
            entry_f = float(entry)
            stop_f = float(stop_loss)
            take_f = float(take_profit)
        except Exception:
            return

        sl_distance = abs(entry_f - stop_f)
        tp_distance = abs(take_f - entry_f)
        if side == "buy":
            payload["stop_loss"] = round(entry_f - sl_distance, 6)
            payload["take_profit"] = round(entry_f + tp_distance, 6)
        elif side == "sell":
            payload["stop_loss"] = round(entry_f + sl_distance, 6)
            payload["take_profit"] = round(entry_f - tp_distance, 6)

    out = dict(plan or {})
    if not bool((enrichment or {}).get("enabled")):
        out["enrichment"] = enrichment or {"enabled": False}
        return out

    policy = evaluate_joint_ohlc_policy(
        enrichment,
        trading_cfg or {},
        market_price=out.get("entry"),
    )
    if bool(policy.get("enabled", False)):
        old_entry = float(out.get("entry", 0.0) or 0.0)
        old_stop = out.get("stop_loss")
        old_take = out.get("take_profit")
        stop_distance = abs(old_entry - float(old_stop)) if old_stop is not None else 0.0
        target_distance = abs(float(old_take) - old_entry) if old_take is not None else 0.0
        out["signal_policy"] = policy
        out["enrichment"] = {
            **dict(enrichment or {}),
            "joint_ohlc_policy": policy,
        }
        decision = str(policy.get("decision") or "hold")
        if decision not in {"buy", "sell"}:
            out["decision"] = "hold"
            out["signal_policy_block_reason"] = str(policy.get("reason") or "signal_policy_blocked")
            out.setdefault("risk_notes", []).append(
                f"Joint OHLC/short-timeframe policy blocked entry: {out['signal_policy_block_reason']}."
            )
            return out
        out["decision"] = decision
        if policy.get("entry") is not None:
            out["entry"] = round(float(policy["entry"]), 6)
        out["confidence"] = round(float(policy.get("confidence", out.get("confidence", 0.5)) or 0.5), 4)
        out["signal_score"] = round(float(policy.get("score", 0.0) or 0.0), 4)
        entry = float(out.get("entry", old_entry) or old_entry)
        if decision == "buy":
            out["stop_loss"] = round(entry - stop_distance, 6) if old_stop is not None else None
            out["take_profit"] = round(entry + target_distance, 6)
        else:
            out["stop_loss"] = round(entry + stop_distance, 6) if old_stop is not None else None
            out["take_profit"] = round(entry - target_distance, 6)
        out.setdefault("risk_notes", []).append(
            "Direction selected from the strongest qualified HIGH/LOW timeframe and confirmed by the configured supporting timeframes."
        )
        out["rationale"] = str(out.get("rationale") or "") + (
            f" | joint_ohlc={decision}, score={float(policy.get('score', 0.0)):.3f}, "
            f"confirmations={policy.get('confirmations')}/{policy.get('required_confirmations')}"
        )
        return out

    consensus = str(enrichment.get("consensus") or "hold").lower()
    consensus_score = float(enrichment.get("consensus_score", 0.0) or 0.0)
    signals = dict(enrichment.get("signals") or {})
    usable = [v for v in signals.values() if not v.get("error")]
    avg_conf = float(np.mean([float(v.get("confidence", 0.5) or 0.5) for v in usable])) if usable else 0.5

    primary_decision = str(out.get("decision") or "hold").lower()
    primary_signal = 0.0
    if primary_decision == "buy":
        primary_signal = 1.0
    elif primary_decision == "sell":
        primary_signal = -1.0
    else:
        raw_score = float(out.get("signal_score", 0.0) or 0.0)
        primary_signal = 1.0 if raw_score > 0 else (-1.0 if raw_score < 0 else 0.0)

    alignment = "neutral"
    if primary_signal > 0 and consensus == "buy":
        alignment = "aligned"
    elif primary_signal < 0 and consensus == "sell":
        alignment = "aligned"
    elif consensus in {"buy", "sell"} and primary_signal != 0:
        alignment = "opposed"

    base_conf = float(out.get("confidence", 0.5) or 0.5)
    out["confidence"] = round(float(np.clip(0.7 * base_conf + 0.3 * avg_conf, 0.0, 1.0)), 4)
    out["enrichment"] = {
        "enabled": True,
        "consensus": consensus,
        "consensus_score": round(consensus_score, 4),
        "alignment": alignment,
        "n_signals": int(enrichment.get("n_signals", 0) or 0),
        "n_qualified_signals": int(enrichment.get("n_qualified_signals", 0) or 0),
        "avg_confidence": round(avg_conf, 4),
        "signals": signals,
    }

    out.setdefault("risk_notes", [])
    out["risk_notes"].append(
        f"Pretrained consensus={consensus} score={consensus_score:.3f} across {int(enrichment.get('n_signals', 0) or 0)} refreshed family/timeframe signals."
    )
    out["rationale"] = str(out.get("rationale") or "") + (
        f" | enriched_consensus={consensus}, enriched_score={consensus_score:.3f}, alignment={alignment}"
    )

    prior_decision = str(out.get("decision") or "hold").lower()
    if str(out.get("decision") or "hold").lower() in {"buy", "sell"} and alignment == "opposed" and abs(consensus_score) >= 0.2:
        if consensus in {"buy", "sell"}:
            out["decision"] = consensus
            out["risk_notes"].append("Primary signal side overridden by stronger pretrained multi-timeframe consensus.")
        else:
            out["decision"] = "hold"
            out["risk_notes"].append("Signal blocked by opposing pretrained multi-timeframe consensus.")

    final_decision = str(out.get("decision") or "hold").lower()
    if final_decision in {"buy", "sell"} and final_decision != prior_decision:
        _recompute_risk_levels(out, final_decision)

    return out


def _apply_conviction_to_plan(
    plan: Dict[str, Any],
    enrichment: Dict[str, Any],
    trading_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Assess conviction from enrichment signals and plan quality, then adjust SL/TP/sizing."""
    out = dict(plan or {})
    conviction_cfg = dict(trading_cfg.get("conviction", {}) or {})
    if not bool(conviction_cfg.get("enabled", True)):
        out["conviction"] = {"enabled": False}
        return out

    risk_cfg = dict(trading_cfg.get("risk", {}) or {})
    base_sl_pct = float(risk_cfg.get("stop_loss_pct", 0.8) or 0.8) / 100.0
    base_tp_pct = float(risk_cfg.get("take_profit_pct", 1.6) or 1.6) / 100.0
    base_volume = float((trading_cfg.get("execution", {}) or {}).get("default_volume", 0.01) or 0.01)

    min_no_sl = float(conviction_cfg.get("min_conviction_for_no_sl", 0.80) or 0.80)
    min_wide = float(conviction_cfg.get("min_conviction_for_wide", 0.65) or 0.65)
    min_standard = float(conviction_cfg.get("min_conviction_for_standard", 0.45) or 0.45)
    min_tight = float(conviction_cfg.get("min_conviction_for_tight", 0.30) or 0.30)
    hist_weight = float(conviction_cfg.get("historical_weight", 0.25) or 0.25)

    # ── factors ──
    confidence = float(out.get("confidence", 0.5) or 0.5)
    cm_acc = float(out.get("cm_accuracy", 0.5) or 0.5)
    success_prob = float(out.get("success_probability", 0.5) or 0.5)
    fooling = float(out.get("input_fooling_risk", 0.5) or 0.5)

    enr = out.get("enrichment", {}) or {}
    consensus_score = abs(float(enr.get("consensus_score", 0.0) or 0.0))
    avg_model_conf = float(enr.get("avg_confidence", 0.5) or 0.5)
    alignment = str(enr.get("alignment") or "neutral").lower()
    n_signals = int(enr.get("n_signals", 0) or 0)

    # 1. Signal quality (0-0.45)
    signal_factor = 0.0
    signal_factor += 0.15 * max(0, min(1, (confidence - 0.40) / 0.30))
    signal_factor += 0.10 * max(0, min(1, (cm_acc - 0.40) / 0.30))
    signal_factor += 0.10 * max(0, min(1, (success_prob - 0.35) / 0.30))
    signal_factor += 0.10 * max(0, min(1, (1.0 - fooling) / 0.30))

    # 2. Enrichment alignment (0-0.30)
    enrich_factor = 0.0
    enrich_factor += 0.12 * consensus_score
    enrich_factor += 0.10 * max(0, min(1, (avg_model_conf - 0.45) / 0.25))
    if alignment == "aligned":
        enrich_factor += 0.08
    elif alignment == "opposed":
        enrich_factor -= 0.05
    enrich_factor = max(0, min(0.30, enrich_factor))

    # 3. Breadth bonus (0-0.10)
    breadth = max(0, min(0.10, n_signals / 240.0))  # 24 signals → 0.10

    # 4. Historical memory (0-0.15)
    hist_factor = 0.0
    try:
        from utils.trade_memory import get_memory
        mem = get_memory()
        hist_result, n_similar = mem.win_rate_for_similar(out, min_samples=2)
        if hist_result is not None:
            hist_factor = hist_weight * hist_result
    except Exception:
        pass

    raw_conviction = min(1.0, signal_factor + enrich_factor + breadth + hist_factor)

    # ── risk mode ──
    if raw_conviction >= min_no_sl:
        risk_mode = "standard"
        sl_mult = 1.0
        tp_mult = 1.0
        vol_mult = 1.0
    elif raw_conviction >= min_wide:
        risk_mode = "standard"
        sl_mult = 1.0
        tp_mult = 1.0
        vol_mult = 0.75
    elif raw_conviction >= min_standard:
        risk_mode = "standard"
        sl_mult = 1.0
        tp_mult = 1.0
        vol_mult = 0.5
    elif raw_conviction >= min_tight:
        risk_mode = "tight"
        sl_mult = 0.5
        tp_mult = 0.6
        vol_mult = 0.25
    else:
        risk_mode = "skip"
        sl_mult = 0.0
        tp_mult = 0.0
        vol_mult = 0.0

    # ── apply ──
    decision = str(out.get("decision") or "hold").lower()
    if decision in {"buy", "sell"} and risk_mode != "skip":
        entry = float(out.get("entry") or 0.0)
        sl_pct = base_sl_pct * sl_mult
        tp_pct = base_tp_pct * tp_mult
        if decision == "buy":
            out["stop_loss"] = round(entry * (1 - sl_pct), 6)
            out["take_profit"] = round(entry * (1 + tp_pct), 6)
        else:
            out["stop_loss"] = round(entry * (1 + sl_pct), 6)
            out["take_profit"] = round(entry * (1 - tp_pct), 6)

        if vol_mult != 1.0:
            volume = round(base_volume * vol_mult, 6)
            out["volume"] = max(volume, 0.0)
            out["risk_notes"].append(f"Volume adjusted by conviction ({vol_mult}x) to {out.get('volume')}.")

        out["conviction"] = {
            "conviction": round(raw_conviction, 4),
            "risk_mode": risk_mode,
            "factors": {
                "signal_quality": round(signal_factor, 3),
                "enrichment": round(enrich_factor, 3),
                "breadth": round(breadth, 3),
                "historical": round(hist_factor, 3),
            },
            "position_multiplier": vol_mult,
        }
    elif risk_mode == "skip":
        out["decision"] = "hold"
        out["conviction"] = {"conviction": round(raw_conviction, 4), "risk_mode": "skip", "reasoning": "conviction below tight threshold"}
        out["risk_notes"].append(f"Trade blocked by low conviction ({raw_conviction:.3f} < {min_tight:.2f}).")
    else:
        out["conviction"] = {"conviction": round(raw_conviction, 4), "risk_mode": risk_mode}

    return out


def _estimate_signal_success_probability(plan: Dict[str, Any], backtest: Dict[str, Any]) -> float:
    """Estimate signal success probability using confidence + CM quality + backtest prior."""
    conf = float(plan.get("confidence", 0.5) or 0.5)
    cm_acc = float(plan.get("cm_accuracy", 0.5) or 0.5)

    win_rate = float(backtest.get("win_rate", 0.5) or 0.5)
    n_trades = int(backtest.get("n_trades", 0) or 0)
    prior_strength = 20.0
    empirical = (win_rate * n_trades + 0.5 * prior_strength) / (n_trades + prior_strength)

    p = 0.4 * conf + 0.3 * cm_acc + 0.3 * empirical
    return float(np.clip(p, 0.01, 0.99))


def _build_probability_heatmaps(
    df,
    app_config: Dict[str, Any],
    future_forecasts: Dict[str, Any],
    model_name: str,
    output_dir: str,
    n_paths: int = 1200,
    n_price_bins: int = 70,
) -> Dict[str, Any]:
    """Create 2D/3D probability concentration maps for future price trajectories."""
    try:
        if model_name not in future_forecasts:
            return {"enabled": False, "error": "Model forecast block missing"}

        forecast_block = future_forecasts.get(model_name, {}) or {}
        future = np.asarray(forecast_block.get("future", []), dtype=float)
        validation = np.asarray(forecast_block.get("validation", []), dtype=float)
        if future.size == 0:
            return {"enabled": False, "error": "Future forecast unavailable"}

        if future.ndim == 2:
            yhat = future[:, 0]
        else:
            yhat = future.reshape(-1)

        target_features = app_config.get("target_features", []) or []
        target_col = app_config.get("target_col")

        resid_std = 1.0
        if validation.size > 0 and target_features:
            n_val = validation.shape[0] if validation.ndim > 1 else validation.shape[0]
            y_true = df[target_features].iloc[-n_val:].values
            y_pred = validation if validation.ndim > 1 else validation.reshape(-1, 1)
            min_len = min(len(y_true), len(y_pred))
            if min_len > 5:
                residuals = y_true[:min_len, 0] - y_pred[:min_len, 0]
                resid_std = float(np.std(residuals, ddof=1)) if np.std(residuals) > 0 else 1.0

        if target_features and target_features[0] == "y_diff" and target_col in df.columns:
            anchor = float(df[target_col].iloc[-1])
            mean_path = anchor + np.cumsum(yhat)
            shocks = np.random.normal(loc=0.0, scale=max(resid_std, 1e-6), size=(n_paths, len(yhat)))
            paths = anchor + np.cumsum((yhat.reshape(1, -1) + shocks), axis=1)
        else:
            mean_path = yhat.copy()
            increments = np.diff(np.r_[mean_path[0], mean_path])
            shocks = np.random.normal(loc=0.0, scale=max(resid_std, 1e-6), size=(n_paths, len(yhat)))
            paths = mean_path.reshape(1, -1) + np.cumsum(shocks + increments.reshape(1, -1) * 0.0, axis=1)

        p_min, p_max = float(np.nanmin(paths)), float(np.nanmax(paths))
        if p_min == p_max:
            p_min -= 1.0
            p_max += 1.0
        bins = np.linspace(p_min, p_max, n_price_bins + 1)

        density = np.zeros((len(yhat), n_price_bins), dtype=float)
        for t in range(len(yhat)):
            hist, _ = np.histogram(paths[:, t], bins=bins, density=True)
            density[t, :] = hist

        centers = 0.5 * (bins[:-1] + bins[1:])
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)

        heat2d_path = os.path.join(output_dir, f"probability_heatmap_2d_{model_name}_{ts}.png")
        fig2d, ax2d = plt.subplots(figsize=(10, 5))
        im = ax2d.imshow(
            density.T,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[1, len(yhat), centers[0], centers[-1]],
        )
        ax2d.plot(np.arange(1, len(yhat) + 1), mean_path, color="white", linewidth=1.5, label="Mean path")
        ax2d.set_title("Time-Price Probability Heatmap (2D)")
        ax2d.set_xlabel("Horizon step")
        ax2d.set_ylabel("Price")
        ax2d.legend(loc="upper right")
        fig2d.colorbar(im, ax=ax2d, label="Density")
        fig2d.tight_layout()
        fig2d.savefig(heat2d_path, dpi=200, bbox_inches="tight")
        plt.close(fig2d)

        heat3d_path = os.path.join(output_dir, f"probability_heatmap_3d_{model_name}_{ts}.png")
        fig3d = plt.figure(figsize=(10, 6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        X, Y = np.meshgrid(np.arange(1, len(yhat) + 1), centers)
        Z = density.T
        ax3d.plot_surface(X, Y, Z, cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
        ax3d.set_title("Time-Price Density Surface (3D)")
        ax3d.set_xlabel("Horizon step")
        ax3d.set_ylabel("Price")
        ax3d.set_zlabel("Density")
        fig3d.tight_layout()
        fig3d.savefig(heat3d_path, dpi=200, bbox_inches="tight")
        plt.close(fig3d)

        return {
            "enabled": True,
            "model": model_name,
            "n_paths": int(n_paths),
            "residual_std": float(resid_std),
            "heatmap_2d_path": heat2d_path,
            "heatmap_3d_path": heat3d_path,
        }
    except Exception as e:
        return {"enabled": False, "error": str(e)}


def _check_endpoints(model_endpoints: Dict[str, str], timeout_sec: float = 2.0) -> Dict[str, Any]:
    return _check_endpoints_with_payloads(model_endpoints, trading_cfg=None, timeout_sec=timeout_sec)


def _tf_to_minutes(label: str) -> int:
    s = str(label or "").strip().lower()
    m = re.match(r"^(\d+)([mhw])$", s)
    if not m:
        return 1
    n = int(m.group(1))
    u = m.group(2)
    if u == "m":
        return n
    if u == "h":
        return n * 60
    if u == "w":
        return n * 7 * 24 * 60
    return 1


def _parse_r2_from_filename(path: str) -> float:
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    m = re.search(r"_(\d{4,6})$", stem)
    if m:
        digits = m.group(1)
        return float(int(digits) / (10 ** (len(digits) - 1)))
    m2 = re.search(r"(\d+\.\d+)", stem)
    if m2:
        return float(m2.group(1))
    return 0.0


def _default_config_root() -> str:
    return str(Path(__file__).resolve().parents[1] / "config")


def _normalize_endpoint_cfg(endpoint_cfg: Any) -> Dict[str, Any]:
    if isinstance(endpoint_cfg, str):
        return {"url": endpoint_cfg, "method": "post"}
    if isinstance(endpoint_cfg, dict):
        out = dict(endpoint_cfg)
        out["url"] = str(out.get("url") or out.get("endpoint") or "").strip()
        out["method"] = str(out.get("method") or out.get("http_method") or "post").strip().lower()
        return out
    return {"url": "", "method": "post"}


def _discover_endpoint_specs(
    model_endpoints: Dict[str, Any],
    config_root: Optional[str] = None,
    config_overrides: Optional[Dict[str, str]] = None,
    deployment_root: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    root = str(config_root or _default_config_root())
    out: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(root):
        return out

    overrides = {str(k): str(v) for k, v in (config_overrides or {}).items() if str(k).strip() and str(v).strip()}

    for tf, raw_endpoint_cfg in (model_endpoints or {}).items():
        tf_label = str(tf or "").strip()
        endpoint_cfg = _normalize_endpoint_cfg(raw_endpoint_cfg)
        preferred_model = str(endpoint_cfg.get("preferred_model") or endpoint_cfg.get("model") or "").strip().lower()
        preferred_config = str(overrides.get(tf_label) or endpoint_cfg.get("config_path") or "").strip()
        best: Dict[str, Any] | None = None

        if preferred_config:
            pref_path = preferred_config
            if not os.path.isabs(pref_path):
                pref_path = str((Path(root).parents[0] / pref_path).resolve())
            if os.path.exists(pref_path):
                best = {
                    "timeframe": tf_label,
                    "model": str(Path(pref_path).parents[0].name),
                    "config_path": pref_path,
                    "r2": _parse_r2_from_filename(pref_path),
                }

        # Asset-specific registries may intentionally have no legacy Results
        # folders. Prefer their installed active package before filesystem
        # discovery, using the explicit config family or HIGH as the default.
        preferred_family = str(endpoint_cfg.get("target_col") or "high").strip().lower()
        if best is not None:
            try:
                with open(best["config_path"], "r", encoding="utf-8") as stream:
                    preferred_family = str((yaml.safe_load(stream) or {}).get("target_col") or preferred_family).lower()
            except Exception:
                pass
        active_deployment = resolve_active_deployment(
            f"{tf_label}_{preferred_family}", deployment_root=deployment_root
        )
        if active_deployment is not None:
            best = {
                "timeframe": tf_label,
                "model": str(active_deployment.get("model") or preferred_model or "").lower(),
                "config_path": str(active_deployment.get("config_path") or ""),
                "model_path": str(active_deployment.get("model_path") or ""),
                "artifacts_path": active_deployment.get("artifacts_path"),
                "deployment_id": active_deployment.get("deployment_id"),
                "training_data_first_index": active_deployment.get("training_data_first_index"),
                "training_data_last_index": active_deployment.get("training_data_last_index"),
                "r2": float(
                    ((active_deployment.get("metrics") or {}).get("holdout_r2"))
                    or ((active_deployment.get("qualification") or {}).get("score"))
                    or 0.0
                ),
            }

        if best is None and deployment_root is not None:
            continue

        # An explicit endpoint config is authoritative. Otherwise select from
        # the active top1 files only, so historical runners-up and nested
        # legacy versions cannot silently replace the promoted endpoint model.
        if best is None:
            tf_dir = os.path.join(root, f"high{tf_label}Results")
            if not os.path.isdir(tf_dir):
                continue
            for model_name in os.listdir(tf_dir):
                if preferred_model and str(model_name).strip().lower() != preferred_model:
                    continue
                model_dir = os.path.join(tf_dir, model_name)
                if not os.path.isdir(model_dir):
                    continue
                filenames = [
                    fn for fn in os.listdir(model_dir)
                    if fn.lower().endswith((".yaml", ".yml")) and fn.lower().startswith("top1")
                ]
                if not filenames:
                    filenames = [
                        fn for fn in os.listdir(model_dir)
                        if fn.lower().endswith((".yaml", ".yml"))
                    ]
                for fn in filenames:
                    full = os.path.join(model_dir, fn)
                    rec = {
                        "timeframe": tf_label,
                        "model": str(model_name),
                        "config_path": full,
                        "r2": _parse_r2_from_filename(full),
                    }
                    if best is None or float(rec["r2"]) > float(best["r2"]):
                        best = rec

        if best is None:
            continue

        try:
            with open(best["config_path"], "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            family = str(cfg.get("target_col") or "HIGH").strip().lower()
            active_deployment = resolve_active_deployment(
                f"{tf_label}_{family}", deployment_root=deployment_root
            )
            if active_deployment is not None:
                best.update(
                    {
                        "model": str(active_deployment.get("model") or best.get("model") or "").lower(),
                        "config_path": str(active_deployment.get("config_path") or best["config_path"]),
                        "model_path": str(active_deployment.get("model_path") or ""),
                        "artifacts_path": active_deployment.get("artifacts_path"),
                        "deployment_id": active_deployment.get("deployment_id"),
                        "training_data_first_index": active_deployment.get("training_data_first_index"),
                        "training_data_last_index": active_deployment.get("training_data_last_index"),
                    }
                )
                with open(best["config_path"], "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                metrics = active_deployment.get("metrics") or {}
                qualification = active_deployment.get("qualification") or {}
                best["r2"] = float(metrics.get("holdout_r2", qualification.get("score", best.get("r2", 0.0))) or 0.0)
            best.update(
                {
                    "n_steps": int(cfg.get("n_steps", 1) or 1),
                    "m_steps": int(cfg.get("m_steps", 1) or 1),
                    "horizon": int(cfg.get("horizon", cfg.get("m_steps", 1)) or 1),
                    "input_features": list(cfg.get("input_features") or []),
                    "target_features": list(cfg.get("target_features") or []),
                    "target_col": str(cfg.get("target_col") or "HIGH"),
                    "rolling_windows": list(cfg.get("rolling_windows") or [2, 7, 30, 60]),
                    "cross_asset_features": cfg.get("cross_asset_features"),
                }
            )
            out[tf_label] = best
        except Exception:
            continue

    return out


def _load_endpoint_market_frame(
    master_path: str,
    timeframe_label: str,
    latest_records: int,
    symbol: str = "XAUUSD",
) -> pd.DataFrame:
    p = str(master_path or "").strip()
    if not p:
        return pd.DataFrame()

    tf_minutes = _tf_to_minutes(timeframe_label)
    if p.lower().endswith((".db", ".sqlite")):
        return query_ohlc(
            db_path=p,
            timeframe_minutes=tf_minutes,
            latest_records=max(int(latest_records or 1), 1),
            start_date=None,
            end_date=None,
            symbol=symbol,
        )

    df = pd.read_csv(p)
    if "DATE" not in df.columns:
        return pd.DataFrame()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"]).sort_values("DATE")
    if df.empty:
        return df

    if tf_minutes <= 1:
        return df.tail(max(int(latest_records or 1), 1)).copy()

    w = df.set_index("DATE").copy()
    if "VOLUME" not in w.columns:
        w["VOLUME"] = 0.0
    out = (
        w.resample(f"{int(tf_minutes)}min")
        .agg({"OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last", "VOLUME": "sum"})
        .dropna(subset=["OPEN", "HIGH", "LOW", "CLOSE"])
        .reset_index()
    )
    return out.tail(max(int(latest_records or 1), 1)).copy()


def _enrich_endpoint_features(df: pd.DataFrame, target_col: str, rolling_windows: List[int]) -> pd.DataFrame:
    out = df.copy().sort_values("DATE").reset_index(drop=True)
    if out.empty:
        return out
    if "VOLUME" not in out.columns:
        out["VOLUME"] = 0.0

    out["Price_return"] = pd.to_numeric(out["CLOSE"], errors="coerce").diff()
    out["Open_return"] = pd.to_numeric(out["OPEN"], errors="coerce").diff()
    out["High_return"] = pd.to_numeric(out["HIGH"], errors="coerce").diff()
    out["Low_return"] = pd.to_numeric(out["LOW"], errors="coerce").diff()
    out["daily_return"] = pd.to_numeric(out["CLOSE"], errors="coerce") - pd.to_numeric(out["OPEN"], errors="coerce")

    tgt = pd.to_numeric(out[target_col], errors="coerce") if target_col in out.columns else pd.to_numeric(out["HIGH"], errors="coerce")
    out["y_diff"] = tgt.diff()

    for window in sorted(set(int(w) for w in (rolling_windows or [2, 7, 30, 60]) if int(w) > 0)):
        out[f"SMA_{window}"] = tgt.rolling(window=window).mean()
        out[f"EMA_{window}"] = tgt.ewm(span=window, adjust=False).mean()
        out[f"Volatility_{window}"] = tgt.rolling(window=window).std()
        out[f"SMA_{window}_diff"] = out["y_diff"].rolling(window=window).mean()
        out[f"EMA_{window}_diff"] = out["y_diff"].ewm(span=window, adjust=False).mean()
        out[f"Volatility_{window}_diff"] = out["y_diff"].rolling(window=window).std()

    return out.dropna().reset_index(drop=True)


def _build_endpoint_payloads(
    model_endpoints: Dict[str, Any],
    trading_cfg: Optional[Dict[str, Any]] = None,
    config_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    deployment_root_value = str(
        (((trading_cfg or {}).get("model_registry") or {}).get("deployment_root") or "")
    ).strip()
    deployment_root = None
    if deployment_root_value:
        deployment_root = Path(deployment_root_value)
        if not deployment_root.is_absolute():
            deployment_root = (Path(__file__).resolve().parents[1] / deployment_root).resolve()
    specs = _discover_endpoint_specs(
        model_endpoints,
        config_root=_default_config_root(),
        config_overrides=config_overrides,
        deployment_root=deployment_root,
    )
    dashboard_cfg = ((trading_cfg or {}).get("dashboard") or {})
    master_path = str(dashboard_cfg.get("master_table_path") or os.path.join(Path(__file__).resolve().parents[1], "data", "market_data.sqlite"))
    market_symbol = str(
        dashboard_cfg.get("sql_symbol")
        or dashboard_cfg.get("tiingo_symbol")
        or ((trading_cfg or {}).get("execution") or {}).get("symbol")
        or "XAUUSD"
    ).strip()
    if not os.path.isabs(master_path):
        master_path = str((Path(__file__).resolve().parents[1] / master_path).resolve())

    payloads: Dict[str, Dict[str, Any]] = {}
    for tf, spec in specs.items():
        n_steps = int(spec.get("n_steps", 1) or 1)
        rolling_windows = list(spec.get("rolling_windows") or [2, 7, 30, 60])
        latest_records = max(n_steps + max([int(w) for w in rolling_windows] + [0]) + 5, 200)
        df = _load_endpoint_market_frame(
            master_path=master_path,
            timeframe_label=tf,
            latest_records=latest_records,
            symbol=market_symbol,
        )
        if df.empty:
            payloads[tf] = {"error": f"No market data available for timeframe {tf}"}
            continue

        merged_market = df
        cross_sources = spec.get("cross_asset_features") or []
        if isinstance(cross_sources, dict):
            cross_sources = [cross_sources]
        for source_cfg in cross_sources:
            if not isinstance(source_cfg, dict) or not bool(source_cfg.get("enabled", True)):
                continue
            source_symbol = str(source_cfg.get("symbol") or "").strip().upper()
            source_prefix = re.sub(
                r"[^A-Z0-9_]+",
                "_",
                str(source_cfg.get("prefix") or source_symbol).strip().upper(),
            ).strip("_")
            if not any(
                str(feature).startswith(f"{source_prefix}_")
                for feature in (spec.get("input_features") or [])
            ):
                continue
            cross_path = str(source_cfg.get("db_path") or "").strip()
            if not cross_path:
                payloads[tf] = {"error": f"Missing cross-asset db_path for timeframe {tf}"}
                merged_market = pd.DataFrame()
                break
            if not os.path.isabs(cross_path):
                cross_path = str((Path(__file__).resolve().parents[1] / cross_path).resolve())
            cross_frame = _load_endpoint_market_frame(
                master_path=cross_path,
                timeframe_label=tf,
                latest_records=max(latest_records * 3, 500),
                symbol=source_symbol,
            )
            try:
                merged_market, _ = merge_cross_asset_frame(
                    merged_market,
                    cross_frame,
                    source_cfg,
                    _tf_to_minutes(tf),
                )
            except Exception as exc:
                payloads[tf] = {"error": f"Cross-asset merge failed for timeframe {tf}: {exc}"}
                merged_market = pd.DataFrame()
                break
        if merged_market.empty:
            continue
        enriched = _enrich_endpoint_features(merged_market, target_col=str(spec.get("target_col") or "HIGH"), rolling_windows=rolling_windows)
        if len(enriched) < n_steps:
            payloads[tf] = {"error": f"Insufficient enriched rows for timeframe {tf}: need {n_steps}, got {len(enriched)}"}
            continue

        needed = list(dict.fromkeys(["DATE"] + [str(c) for c in (spec.get("input_features") or [])]))
        missing = [c for c in needed if c != "DATE" and c not in enriched.columns]
        if missing:
            payloads[tf] = {"error": f"Missing engineered columns for timeframe {tf}: {missing}"}
            continue

        rows_df = enriched[needed].tail(n_steps).copy()
        rows: List[Dict[str, Any]] = []
        for _, row in rows_df.iterrows():
            item: Dict[str, Any] = {"DATE": pd.to_datetime(row["DATE"]).strftime("%Y-%m-%d %H:%M:%S")}
            for col in needed:
                if col == "DATE":
                    continue
                val = row[col]
                item[col] = None if pd.isna(val) else float(val)
            rows.append(item)

        payloads[tf] = {
            "rows": rows,
            "timeframe": tf,
            "model": spec.get("model"),
            "config_path": spec.get("config_path"),
            "input_features": list(spec.get("input_features") or []),
            "n_steps": n_steps,
            "target_col": str(spec.get("target_col") or "HIGH"),
            "reference_price": float(enriched.iloc[-1][str(spec.get("target_col") or "HIGH")]),
        }

    return payloads


def _latest_inference_window(rows: Any, n_steps: int) -> Any:
    """Select the newest feature window; forecast horizon does not offset inputs."""
    window_size = max(int(n_steps or 1), 1)
    return rows[-window_size:]


def _is_local_signal_url(url: str, host: str, port: int) -> bool:
    u = str(url or "").strip().lower()
    if not u:
        return False
    return u.startswith(f"http://{host.lower()}:{int(port)}") or u.startswith(f"https://{host.lower()}:{int(port)}")


def _ensure_local_endpoint_on_demand(trading_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    global _LAST_ENDPOINT_START_TS
    cfg = ((trading_cfg or {}).get("endpoint_lifecycle") or {})
    if not bool(cfg.get("on_demand_start", True)):
        return {"ok": False, "skipped": True, "reason": "on_demand_disabled"}

    host = str(cfg.get("host", "127.0.0.1"))
    port = int(cfg.get("port", 8000) or 8000)
    health_url = f"http://{host}:{port}/health"
    startup_wait = int(cfg.get("startup_wait_seconds", 20) or 20)
    service_script = str(cfg.get("service_script", "scripts/local_signal_endpoint_service.py"))

    try:
        r = requests.get(health_url, timeout=2)
        if r.status_code == 200:
            return {"ok": True, "already_running": True}
    except Exception:
        pass

    now = time.time()
    if now - _LAST_ENDPOINT_START_TS < 8:
        return {"ok": False, "skipped": True, "reason": "recent_start_attempt"}
    _LAST_ENDPOINT_START_TS = now

    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["TSMM_SIGNAL_HOST"] = host
    env["TSMM_SIGNAL_PORT"] = str(port)

    exe = sys.executable
    if os.name == "nt":
        exe = sys.executable
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NO_WINDOW  # type: ignore[attr-defined]

    subprocess.Popen(
        [
            exe,
            str((root / service_script).resolve()),
        ],
        cwd=str(root),
        env=env,
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for _ in range(max(startup_wait, 5)):
        time.sleep(1)
        try:
            rr = requests.get(health_url, timeout=2)
            if rr.status_code == 200:
                return {"ok": True, "started": True, "health_url": health_url}
        except Exception:
            continue

    return {"ok": False, "error": "endpoint_start_timeout", "health_url": health_url}


def _call_signal_endpoint(endpoint_cfg: Dict[str, Any], payload: Optional[Dict[str, Any]], timeout_sec: float, trading_cfg: Optional[Dict[str, Any]] = None) -> requests.Response:
    url = str(endpoint_cfg.get("url") or "").strip()
    method = str(endpoint_cfg.get("method") or "post").strip().lower()
    headers = endpoint_cfg.get("headers") or {}

    def _do() -> requests.Response:
        if method == "get":
            return requests.get(url, timeout=timeout_sec, headers=headers)
        return requests.post(url, timeout=timeout_sec, headers=headers, json=payload or {})

    try:
        return _do()
    except Exception:
        cfg = ((trading_cfg or {}).get("endpoint_lifecycle") or {})
        host = str(cfg.get("host", "127.0.0.1"))
        port = int(cfg.get("port", 8000) or 8000)
        if _is_local_signal_url(url, host=host, port=port) and bool(cfg.get("on_demand_start", True)):
            _ensure_local_endpoint_on_demand(trading_cfg)
            return _do()
        raise


def _check_endpoints_with_payloads(
    model_endpoints: Dict[str, Any],
    trading_cfg: Optional[Dict[str, Any]],
    timeout_sec: float = 2.0,
    config_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    payloads = _build_endpoint_payloads(
        model_endpoints,
        trading_cfg=trading_cfg,
        config_overrides=config_overrides,
    )
    out: Dict[str, Any] = {}
    for tf, raw_cfg in (model_endpoints or {}).items():
        endpoint_cfg = _normalize_endpoint_cfg(raw_cfg)
        payload = payloads.get(str(tf), {})
        if payload.get("error"):
            out[tf] = {"ok": False, "error": payload.get("error")}
            continue
        try:
            r = _call_signal_endpoint(endpoint_cfg, payload, timeout_sec=timeout_sec, trading_cfg=trading_cfg)
            out[tf] = {"ok": r.status_code < 500, "status_code": r.status_code}
        except Exception as e:
            out[tf] = {"ok": False, "error": str(e)}
    return out


def _extract_endpoint_signal(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract normalized signal from heterogeneous model endpoint payloads."""
    if not isinstance(payload, dict):
        return {"signal": 0, "confidence": 0.5, "raw": payload}

    if "signal" in payload:
        s = str(payload.get("signal", "hold")).lower()
        signal = 1 if s in ("buy", "long", "up") else (-1 if s in ("sell", "short", "down") else 0)
    elif "forecast_sign" in payload and isinstance(payload.get("forecast_sign"), (int, float)):
        fs = float(payload.get("forecast_sign"))
        signal = 1 if fs > 0 else (-1 if fs < 0 else 0)
    elif "prediction" in payload and isinstance(payload.get("prediction"), (int, float)):
        pv = float(payload.get("prediction"))
        signal = 1 if pv > 0 else (-1 if pv < 0 else 0)
    else:
        signal = 0

    confidence = payload.get("confidence", payload.get("probability", 0.5))
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.5

    forecast_delta = None
    for candidate in (payload.get("forecast_sign"), payload.get("prediction")):
        if isinstance(candidate, (int, float)):
            forecast_delta = float(candidate)
            break

    interpreted_signal = _apply_signal_interpretation(float(signal), payload.get("_trading_cfg") or {})
    signal = 1 if interpreted_signal > 0 else (-1 if interpreted_signal < 0 else 0)

    return {
        "signal": signal,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "forecast_delta": forecast_delta,
        "raw": payload,
    }


def _collect_mode_b_signals(
    model_endpoints: Dict[str, Any],
    trading_cfg: Optional[Dict[str, Any]] = None,
    timeout_sec: float = 2.0,
    config_overrides: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    votes = []
    weighted = 0.0
    total_w = 0.0
    payloads = _build_endpoint_payloads(
        model_endpoints,
        trading_cfg=trading_cfg,
        config_overrides=config_overrides,
    )

    for tf, raw_cfg in (model_endpoints or {}).items():
        endpoint_cfg = _normalize_endpoint_cfg(raw_cfg)
        payload = payloads.get(str(tf), {})
        if payload.get("error"):
            out[tf] = {"signal": 0, "confidence": 0.5, "error": payload.get("error")}
            continue
        try:
            r = _call_signal_endpoint(endpoint_cfg, payload, timeout_sec=timeout_sec, trading_cfg=trading_cfg)
            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            signal_payload = dict(data or {}) if isinstance(data, dict) else {"raw": data}
            signal_payload["_trading_cfg"] = trading_cfg or {}
            sig = _extract_endpoint_signal(signal_payload)
            sig["status_code"] = r.status_code
            sig["request"] = {
                "method": str(endpoint_cfg.get("method") or "post").upper(),
                "url": str(endpoint_cfg.get("url") or ""),
                "payload_rows": len(payload.get("rows") or []),
                "model": payload.get("model"),
                "target_col": payload.get("target_col"),
                "reference_price": payload.get("reference_price"),
            }
            sig["reference_price"] = payload.get("reference_price")
            if sig.get("forecast_delta") is not None and payload.get("reference_price") is not None:
                sig["forecast_price"] = float(payload["reference_price"]) + float(sig["forecast_delta"])
            out[tf] = sig
            votes.append(sig["signal"])
            w = max(sig["confidence"], 0.01)
            weighted += w * sig["signal"]
            total_w += w
        except Exception as e:
            out[tf] = {"signal": 0, "confidence": 0.5, "error": str(e)}

    consensus_score = weighted / total_w if total_w > 0 else 0.0
    consensus = "buy" if consensus_score > 0.1 else ("sell" if consensus_score < -0.1 else "hold")
    return {
        "timeframes": out,
        "consensus": consensus,
        "consensus_score": float(consensus_score),
        "n_timeframes": len(out),
    }


def _write_runtime_state(output_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    state_path = os.path.join(output_dir, "agent_state_latest.json")
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return state_path


def run_investing_agent(
    app_config: Dict[str, Any],
    results: Dict[str, Any],
    trading_cfg: Dict[str, Any],
    output_dir: str,
    selected_model: Optional[str] = None,
    mode_override: Optional[str] = None,
    interrupt_mode_b: bool = False,
) -> Dict[str, Any]:
    agent_cfg = (trading_cfg.get("agent") or {})
    if not agent_cfg.get("enabled", False):
        return {"enabled": False, "reason": "Trading agent disabled"}

    mode = str(mode_override or agent_cfg.get("mode", "mode_a")).lower()
    target_col = app_config.get("target_col", "asset")

    # External interrupt flag (for UI/dashboard stop control)
    mb_cfg = (trading_cfg.get("mode_b") or {})
    interrupt_flag_path = str(
        resolve_runtime_file(
            configured_path=mb_cfg.get("interrupt_flag_path"),
            fallback_name="mode_b_interrupt.flag",
            output_dir=output_dir,
            trading_cfg=trading_cfg,
            base_dir=Path(output_dir).parent,
        )
    )
    interrupt_mode_b = bool(interrupt_mode_b or (interrupt_flag_path and os.path.exists(interrupt_flag_path)))

    plan = _build_mode_a_plan(
        results.get("df"),
        app_config,
        trading_cfg,
        results.get("evaluation", {}),
        results.get("future_forecasts", {}),
        preferred_model=selected_model,
    )

    mode_a_cfg = (trading_cfg.get("mode_a") or {})
    if bool(mode_a_cfg.get("use_pretrained_consensus", True)):
        enrichment = _collect_agent_a_enrichment_signals(trading_cfg=trading_cfg, timeout_sec=3.0)
        plan = _apply_agent_a_enrichment_to_plan(plan, enrichment, trading_cfg)
    else:
        enrichment = {"enabled": False, "reason": "Disabled in mode_a config"}

    # Conviction assessment — modulates SL/TP/sizing based on signal strength
    plan = _apply_conviction_to_plan(plan, enrichment, trading_cfg)
    policy_cfg = signal_policy_config(trading_cfg)
    volatility_cfg = dict(policy_cfg.get("volatility") or {})
    volatility_frame = None
    if bool(volatility_cfg.get("enabled", False)):
        dashboard_cfg = trading_cfg.get("dashboard") or {}
        master_path = str(dashboard_cfg.get("master_table_path") or "data/market_data.sqlite")
        if not os.path.isabs(master_path):
            master_path = str((Path(__file__).resolve().parents[1] / master_path).resolve())
        try:
            volatility_frame = _load_endpoint_market_frame(
                master_path=master_path,
                timeframe_label=str(volatility_cfg.get("timeframe") or "10m"),
                latest_records=max(int(volatility_cfg.get("atr_period", 14) or 14) + 5, 30),
                symbol=str(dashboard_cfg.get("sql_symbol") or ((trading_cfg.get("execution") or {}).get("symbol") or "XAUUSD")),
            )
        except Exception:
            volatility_frame = None
    plan = apply_volatility_protection(plan, trading_cfg, volatility_frame)
    backtest = run_backtest_from_validation(
        results.get("df"),
        app_config,
        results.get("evaluation", {}),
        results.get("future_forecasts", {}),
        trading_cfg,
        preferred_model=plan.get("model"),
    )

    success_probability = _estimate_signal_success_probability(plan, backtest)
    plan["success_probability"] = round(success_probability, 4)
    # The admission gate must consume the calibrated/backtested probability.
    # Previously it ran first and silently substituted raw confidence instead.
    plan = apply_hybrid_trade_gate(plan, trading_cfg)

    pm_cfg = trading_cfg.get("probability_maps", {}) or {}
    if bool(pm_cfg.get("enabled", True)):
        prob_dir = os.path.join(output_dir, "probability_maps")
        heatmaps = _build_probability_heatmaps(
            results.get("df"),
            app_config,
            results.get("future_forecasts", {}),
            plan.get("model"),
            prob_dir,
            n_paths=int(pm_cfg.get("n_paths", 1200)),
            n_price_bins=int(pm_cfg.get("n_price_bins", 70)),
        )
    else:
        heatmaps = {"enabled": False, "reason": "Disabled in trading config"}

    warnings: List[str] = []

    mode_b_status = {"enabled": False}
    if mode == "mode_b" and (trading_cfg.get("mode_b") or {}).get("enabled", False):
        if interrupt_mode_b:
            warnings.append("Mode B execution interrupted by user flag.")
            mode_b_status = {
                "enabled": True,
                "interrupted": True,
                "reason": "interrupt_mode_b flag",
                "interrupt_flag_path": interrupt_flag_path,
            }
        else:
            endpoints = _check_endpoints_with_payloads(trading_cfg.get("model_endpoints", {}), trading_cfg=trading_cfg)
            if bool((trading_cfg.get("mode_b") or {}).get("allow_endpoint_signals", True)):
                mtf_signals = _collect_mode_b_signals(trading_cfg.get("model_endpoints", {}), trading_cfg=trading_cfg)
            else:
                mtf_signals = {"timeframes": {}, "consensus": "hold", "consensus_score": 0.0, "n_timeframes": 0}
            mode_b_status = {
                "enabled": True,
                "endpoints": endpoints,
                "timeframe_signals": mtf_signals,
                "live_execution_requested": not bool(agent_cfg.get("confirm_live_execution", True)),
            }

            broker_block = (trading_cfg.get("broker", {}) or {})
            active_broker = str(broker_block.get("active", "mt5") or "mt5").lower()
            if active_broker == "mt5":
                broker_cfg = (broker_block.get("mt5") or {})
                if broker_cfg.get("enabled", False):
                    adapter = MT5Adapter(broker_cfg)
                    ok, msg = adapter.connect()
                    mode_b_status["mt5_connection"] = {"ok": ok, "message": msg}
                    adapter.shutdown()
                else:
                    warnings.append("Mode B selected but MT5 broker is disabled.")
            elif active_broker == "iqoption":
                broker_cfg = (broker_block.get("iqoption") or {})
                if broker_cfg.get("enabled", False):
                    adapter = IQOptionAdapter(broker_cfg)
                    ok, msg = adapter.connect()
                    mode_b_status["iqoption_connection"] = {"ok": ok, "message": msg}
                    adapter.shutdown()
                else:
                    warnings.append("Mode B selected but IQ Option broker is disabled.")
            else:
                warnings.append(f"Mode B selected but broker '{active_broker}' is not supported for connection checks.")

    report_cfg = trading_cfg.get("reporting", {}) or {}
    rep_dir = report_cfg.get("output_dir") or os.path.join(output_dir, "trading_plans")
    os.makedirs(rep_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rep_prefix = report_cfg.get("filename_prefix", "trading_plan")
    plan_path = os.path.join(rep_dir, f"{rep_prefix}_{target_col}_{ts}.pdf")

    generate_trading_plan_report(
        output_path=plan_path,
        target_col=target_col,
        mode=mode,
        plan=plan,
        backtest=backtest,
        warnings=warnings,
        heatmaps=heatmaps,
    )

    state_dir = os.path.join(output_dir, "runtime")
    state_payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "plan": plan,
        "backtest": {
            "model_name": backtest.get("model_name"),
            "n_trades": backtest.get("n_trades"),
            "win_rate": backtest.get("win_rate"),
            "total_return_pct": backtest.get("total_return_pct"),
            "max_drawdown_pct": backtest.get("max_drawdown_pct"),
        },
        "enrichment": enrichment,
        "mode_b": mode_b_status,
        "open_positions": (backtest.get("trades") or [])[-3:],
        "signal_success_probability": success_probability,
        "heatmaps": heatmaps,
        "report_path": plan_path,
        "warnings": warnings,
    }
    state_path = _write_runtime_state(state_dir, state_payload)

    return {
        "enabled": True,
        "mode": mode,
        "plan": plan,
        "backtest": backtest,
        "enrichment": enrichment,
        "signal_success_probability": success_probability,
        "heatmaps": heatmaps,
        "mode_b": mode_b_status,
        "state_path": state_path,
        "report_path": plan_path,
        "warnings": warnings,
    }
