"""Telegram notification utility for trading-job events and approvals."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import requests


def _read_windows_user_env(var_name: str) -> str:
    """Read a user-scoped environment variable directly from Windows registry."""
    if os.name != "nt":
        return ""
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            value, _ = winreg.QueryValueEx(key, str(var_name))
            return str(value or "").strip()
    except Exception:
        return ""


def _resolve_secret(value: str) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value.startswith("env:"):
        env_name = value.split(":", 1)[1]
        resolved = os.environ.get(env_name, "")
        if resolved:
            return resolved
        return _read_windows_user_env(env_name)
    return value


def _load_subscriber_chat_ids(subscribers_path: str | None, default_chat_id: str = "") -> List[str]:
    chat_ids: List[str] = []
    if default_chat_id:
        chat_ids.append(str(default_chat_id).strip())
    raw_path = str(subscribers_path or "").strip()
    if not raw_path:
        return [chat_id for chat_id in chat_ids if chat_id]

    try:
        payload = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    except Exception:
        return [chat_id for chat_id in chat_ids if chat_id]

    for chat_id in payload.get("chat_ids") or []:
        resolved_chat_id = str(chat_id or "").strip()
        if resolved_chat_id and resolved_chat_id not in chat_ids:
            chat_ids.append(resolved_chat_id)
    return [chat_id for chat_id in chat_ids if chat_id]


def send_telegram_notification(telegram_cfg: Dict[str, Any], message: str) -> Dict[str, Any]:
    cfg = telegram_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return {"ok": False, "skipped": True, "reason": "telegram disabled"}

    token = _resolve_secret(str(cfg.get("bot_token", "")))
    chat_id = _resolve_secret(str(cfg.get("chat_id", "")))
    parse_mode = str(cfg.get("parse_mode", "Markdown")).strip()
    disable_preview = bool(cfg.get("disable_web_page_preview", True))

    if not token or not chat_id:
        return {"ok": False, "error": "missing_telegram_token_or_chat_id"}

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": str(message or "").strip() or "TSMM notification",
        "disable_web_page_preview": disable_preview,
    }
    valid_parse_modes = {"Markdown", "MarkdownV2", "HTML"}
    if parse_mode in valid_parse_modes:
        payload["parse_mode"] = parse_mode

    try:
        r = requests.post(url, json=payload, timeout=20)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
        if r.status_code >= 400 or not bool(data.get("ok", False)):
            # Fallback: retry once without parse mode in case formatting is invalid.
            if "parse_mode" in payload:
                payload_no_parse = dict(payload)
                payload_no_parse.pop("parse_mode", None)
                r2 = requests.post(url, json=payload_no_parse, timeout=20)
                data2 = r2.json() if r2.headers.get("content-type", "").startswith("application/json") else {"raw": r2.text}
                if r2.status_code < 400 and bool(data2.get("ok", False)):
                    return {
                        "ok": True,
                        "status_code": r2.status_code,
                        "chat_id": str(chat_id),
                        "message_id": ((data2.get("result") or {}).get("message_id")),
                        "fallback_no_parse_mode": True,
                    }

            return {
                "ok": False,
                "status_code": r.status_code,
                "error": data.get("description", "telegram_send_failed"),
            }
        return {
            "ok": True,
            "status_code": r.status_code,
            "chat_id": str(chat_id),
            "message_id": ((data.get("result") or {}).get("message_id")),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def send_telegram_broadcast(
    telegram_cfg: Dict[str, Any],
    message: str,
    subscribers_path: str | None = None,
    extra_chat_ids: List[str] | None = None,
) -> Dict[str, Any]:
    cfg = dict(telegram_cfg or {})
    default_chat_id = _resolve_secret(str(cfg.get("chat_id", ""))).strip()
    target_chat_ids = _load_subscriber_chat_ids(subscribers_path, default_chat_id=default_chat_id)
    for chat_id in extra_chat_ids or []:
        resolved_chat_id = str(chat_id or "").strip()
        if resolved_chat_id and resolved_chat_id not in target_chat_ids:
            target_chat_ids.append(resolved_chat_id)

    if not target_chat_ids:
        return send_telegram_notification(cfg, message)

    results = []
    ok = False
    for chat_id in target_chat_ids:
        scoped_cfg = dict(cfg)
        scoped_cfg["chat_id"] = chat_id
        result = send_telegram_notification(scoped_cfg, message)
        results.append(result)
        if bool(result.get("ok", False)):
            ok = True

    return {
        "ok": ok,
        "chat_ids": target_chat_ids,
        "results": results,
        "error": None if ok else "telegram_broadcast_failed",
    }
