"""IQ Option adapter for connection checks and practice-mode readiness."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple


def _resolve_secret(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value.startswith("env:"):
        return os.environ.get(value.split(":", 1)[1], "")
    return value


class IQOptionAdapter:
    """Lightweight IQ Option connection adapter.

    This adapter is intentionally scoped to connectivity and account mode checks
    so users can validate demo/practice access without placing real trades.
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg or {}
        self._api = None

    def connect(self) -> Tuple[bool, str]:
        try:
            from iqoptionapi.stable_api import IQ_Option  # type: ignore
        except Exception as e:
            return False, f"iqoptionapi import failed: {e}"

        email = _resolve_secret(self.cfg.get("email", ""))
        password = _resolve_secret(self.cfg.get("password", ""))
        if not email or not password:
            return False, "Missing IQ Option credentials (email/password)"

        api = IQ_Option(email, password)
        try:
            conn_result = api.connect()
        except Exception as e:
            return False, f"IQ Option connect exception: {e}"

        ok = False
        reason = "unknown"
        if isinstance(conn_result, tuple):
            ok = bool(conn_result[0])
            if len(conn_result) > 1:
                reason = str(conn_result[1])
        else:
            ok = bool(conn_result)

        if not ok:
            return False, f"IQ Option connect failed: {reason}"

        # Default to PRACTICE for safe testing unless explicitly set.
        balance_mode = str(self.cfg.get("balance_mode", "PRACTICE") or "PRACTICE").upper()
        if balance_mode in {"PRACTICE", "REAL"}:
            try:
                api.change_balance(balance_mode)
            except Exception as e:
                return False, f"Connected, but could not switch balance to {balance_mode}: {e}"

        self._api = api
        return True, f"connected ({balance_mode})"

    def shutdown(self) -> None:
        api = self._api
        if api is None:
            return
        try:
            api.close_connect()
        except Exception:
            pass
