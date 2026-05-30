"""Inspect and answer Copilot bridge requests queued from Telegram."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.copilot_bridge import list_copilot_requests, read_copilot_request, record_copilot_response  # noqa: E402


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage Copilot bridge requests created from Telegram.")
    parser.add_argument("--trading-config", default="config/trading_agent.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List recent bridge requests")
    list_parser.add_argument("--status", default="pending")
    list_parser.add_argument("--chat-id", default="")
    list_parser.add_argument("--limit", type=int, default=10)

    show_parser = subparsers.add_parser("show", help="Show one bridge request")
    show_parser.add_argument("--request-id", required=True)

    reply_parser = subparsers.add_parser("reply", help="Store a Copilot response and optionally send it to Telegram")
    reply_parser.add_argument("--request-id", required=True)
    reply_parser.add_argument("--message", default="")
    reply_parser.add_argument("--message-file", default="")
    reply_parser.add_argument("--no-send", action="store_true")

    args = parser.parse_args()
    trading_cfg_path = Path(args.trading_config)
    if not trading_cfg_path.is_absolute():
        trading_cfg_path = ROOT / trading_cfg_path
    trading_cfg = _load_yaml(trading_cfg_path)

    if args.command == "list":
        items = list_copilot_requests(ROOT, status=args.status or None, chat_id=args.chat_id or None, limit=args.limit)
        print(json.dumps({"requests": items}, indent=2, default=str))
        return 0

    if args.command == "show":
        payload = read_copilot_request(ROOT, args.request_id)
        if not payload:
            print(json.dumps({"ok": False, "error": "request_not_found", "request_id": args.request_id}, indent=2))
            return 1
        print(json.dumps(payload, indent=2, default=str))
        return 0

    message = str(args.message or "").strip()
    if not message and args.message_file:
        message = Path(args.message_file).read_text(encoding="utf-8").strip()
    if not message:
        print(json.dumps({"ok": False, "error": "missing_message"}, indent=2))
        return 1

    result = record_copilot_response(ROOT, trading_cfg, args.request_id, message, send_to_telegram=not bool(args.no_send))
    print(json.dumps(result, indent=2, default=str))
    return 0 if bool(result.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())