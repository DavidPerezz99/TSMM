import sys
import tempfile
import unittest
from types import ModuleType
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "requests" not in sys.modules:
    requests_stub = ModuleType("requests")
    requests_stub.post = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    sys.modules["requests"] = requests_stub

from utils.notification_telegram import send_telegram_broadcast


class TelegramNotificationTests(unittest.TestCase):
    @patch("utils.notification_telegram.send_telegram_notification")
    def test_broadcast_tracks_delivery_per_chat(self, send_mock):
        send_mock.side_effect = [
            {"ok": True, "message_id": 1},
            {"ok": False, "error": "boom"},
            {"ok": False, "error": "boom"},
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subscribers = root / "telegram_subscribers.json"
            subscribers.write_text('{"chat_ids":["111","222"]}', encoding="utf-8")

            out = send_telegram_broadcast(
                {"enabled": True, "bot_token": "test", "chat_id": "999"},
                "hello world",
                subscribers_path=str(subscribers),
            )

        self.assertTrue(out["ok"])
        self.assertEqual(out["chat_ids"], ["999", "111", "222"])
        self.assertEqual(out["delivered_chat_ids"], ["999"])
        self.assertEqual(out["failed_chat_ids"], ["111", "222"])
        self.assertEqual(len(out["results"]), 3)
        self.assertEqual(send_mock.call_count, 3)


if __name__ == "__main__":
    unittest.main()
