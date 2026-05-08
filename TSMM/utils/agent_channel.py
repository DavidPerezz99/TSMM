"""Runtime agent communication channel with optional popup and gating controls."""

from __future__ import annotations

import ctypes
import json
import os
from datetime import datetime
from typing import Any, Dict, List


def _iso_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def _communication_cfg(trading_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return (trading_cfg.get("communication") or {})


def get_channel_path(output_dir: str, trading_cfg: Dict[str, Any]) -> str:
    cc = _communication_cfg(trading_cfg)
    return str(cc.get("channel_file_path") or os.path.join(output_dir, "runtime", "agent_channel.jsonl"))


def get_channel_flag_path(output_dir: str, trading_cfg: Dict[str, Any]) -> str:
    cc = _communication_cfg(trading_cfg)
    return str(cc.get("control_flag_path") or os.path.join(output_dir, "runtime", "agent_channel_enabled.flag"))


def is_channel_enabled(output_dir: str, trading_cfg: Dict[str, Any]) -> bool:
    cc = _communication_cfg(trading_cfg)
    if bool(cc.get("enabled_default", False)):
        return True
    return os.path.exists(get_channel_flag_path(output_dir, trading_cfg))


def set_channel_enabled(output_dir: str, trading_cfg: Dict[str, Any], enabled: bool) -> str:
    flag = get_channel_flag_path(output_dir, trading_cfg)
    os.makedirs(os.path.dirname(flag), exist_ok=True)
    if enabled:
        with open(flag, "w", encoding="utf-8") as f:
            f.write(_iso_now())
    else:
        if os.path.exists(flag):
            os.remove(flag)
    return flag


def _popup_message(title: str, message: str) -> None:
    try:
        user32 = ctypes.windll.user32
        MB_OK = 0x00000000
        MB_ICONINFORMATION = 0x00000040
        user32.MessageBoxW(None, str(message), str(title), MB_OK | MB_ICONINFORMATION)
    except Exception:
        return


def publish_channel_message(
    output_dir: str,
    trading_cfg: Dict[str, Any],
    channel: str,
    message: str,
    kind: str = "info",
    requires_approval: bool = False,
    emergency: bool = False,
    approval_deadline_utc: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    enabled = is_channel_enabled(output_dir, trading_cfg)
    if not enabled and not emergency:
        return {"ok": False, "skipped": True, "reason": "channel_disabled"}

    entry = {
        "timestamp_utc": _iso_now(),
        "channel": str(channel or "agent"),
        "kind": str(kind or "info"),
        "message": str(message or ""),
        "requires_approval": bool(requires_approval),
        "emergency": bool(emergency),
        "approval_deadline_utc": approval_deadline_utc,
        "metadata": metadata or {},
    }

    ch_path = get_channel_path(output_dir, trading_cfg)
    os.makedirs(os.path.dirname(ch_path), exist_ok=True)
    with open(ch_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")

    cc = _communication_cfg(trading_cfg)
    popup_on_any = bool(cc.get("popup_on_channel_message", True))
    popup_on_emergency = bool(cc.get("popup_on_emergency", True))
    if (popup_on_any and enabled) or (popup_on_emergency and emergency):
        _popup_message(
            title=f"TSMM Agent Channel: {entry['channel']}",
            message=f"[{entry['kind']}] {entry['message']}",
        )

    return {"ok": True, "entry": entry, "channel_path": ch_path}


def read_channel_messages(output_dir: str, trading_cfg: Dict[str, Any], max_lines: int = 200) -> List[Dict[str, Any]]:
    ch_path = get_channel_path(output_dir, trading_cfg)
    if not os.path.exists(ch_path):
        return []
    out: List[Dict[str, Any]] = []
    with open(ch_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-max(int(max_lines), 1):]
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            out.append(json.loads(ln))
        except Exception:
            continue
    return out
