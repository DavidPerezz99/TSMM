"""Persistent request/reply bridge between Telegram chat and the active Copilot session."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import secrets
from typing import Any, Dict, List

from .notification_telegram import send_telegram_notification


def _utc_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _bridge_root(root: Path) -> Path:
    path = Path(root) / "reports" / "runtime" / "copilot_bridge"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _requests_dir(root: Path) -> Path:
    path = _bridge_root(root) / "requests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _request_path(root: Path, request_id: str) -> Path:
    return _requests_dir(root) / f"{str(request_id or '').strip()}.json"


def _new_request_id() -> str:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"copilot_{stamp}_{secrets.token_hex(4)}"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def queue_copilot_request(
    root: Path,
    trading_cfg: Dict[str, Any],
    prompt: str,
    chat_id: str,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    clean_prompt = str(prompt or "").strip()
    clean_chat_id = str(chat_id or "").strip()
    if not clean_prompt:
        return {"ok": False, "error": "missing_prompt"}
    if not clean_chat_id:
        return {"ok": False, "error": "missing_chat_id"}

    request_id = _new_request_id()
    payload = {
        "request_id": request_id,
        "status": "pending",
        "source": "telegram",
        "chat_id": clean_chat_id,
        "prompt": clean_prompt,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "response_text": "",
        "response_sent": False,
        "response_sent_at_utc": None,
        "telegram_send_result": None,
        "metadata": metadata or {},
    }
    _write_json(_request_path(root, request_id), payload)
    return {"ok": True, "request": payload, "request_path": str(_request_path(root, request_id))}


def read_copilot_request(root: Path, request_id: str) -> Dict[str, Any]:
    clean_request_id = str(request_id or "").strip()
    if not clean_request_id:
        return {}
    return _read_json(_request_path(root, clean_request_id))


def list_copilot_requests(
    root: Path,
    status: str | None = None,
    chat_id: str | None = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    clean_status = str(status or "").strip().lower()
    clean_chat_id = str(chat_id or "").strip()
    requests: List[Dict[str, Any]] = []
    for path in sorted(_requests_dir(root).glob("copilot_*.json"), reverse=True):
        payload = _read_json(path)
        if not payload:
            continue
        if clean_status and str(payload.get("status") or "").strip().lower() != clean_status:
            continue
        if clean_chat_id and str(payload.get("chat_id") or "").strip() != clean_chat_id:
            continue
        requests.append(payload)
        if len(requests) >= max(int(limit or 20), 1):
            break
    return requests


def record_copilot_response(
    root: Path,
    trading_cfg: Dict[str, Any],
    request_id: str,
    response_text: str,
    send_to_telegram: bool = True,
) -> Dict[str, Any]:
    payload = read_copilot_request(root, request_id)
    if not payload:
        return {"ok": False, "error": "request_not_found", "request_id": str(request_id or "").strip()}

    clean_response = str(response_text or "").strip()
    if not clean_response:
        return {"ok": False, "error": "missing_response_text", "request": payload}

    payload["status"] = "answered"
    payload["response_text"] = clean_response
    payload["updated_at_utc"] = _utc_now()
    payload["answered_at_utc"] = _utc_now()

    telegram_send_result: Dict[str, Any] | None = None
    if send_to_telegram:
        tcfg = dict(trading_cfg.get("telegram_notifications") or {})
        tcfg["chat_id"] = str(payload.get("chat_id") or "").strip()
        telegram_send_result = send_telegram_notification(tcfg, clean_response[:3500])
        payload["telegram_send_result"] = telegram_send_result
        payload["response_sent"] = bool(telegram_send_result.get("ok", False))
        payload["response_sent_at_utc"] = _utc_now() if payload["response_sent"] else None

    _write_json(_request_path(root, str(payload.get("request_id") or "")), payload)
    return {"ok": True, "request": payload, "telegram_send": telegram_send_result}