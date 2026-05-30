import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.copilot_bridge import list_copilot_requests, queue_copilot_request, record_copilot_response


class CopilotBridgeTests(unittest.TestCase):
    def test_queue_request_persists_pending_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            out = queue_copilot_request(root, {}, prompt="Search the web for gold macro headlines", chat_id="8362291523")

            self.assertTrue(out.get("ok"))
            request = dict(out.get("request") or {})
            self.assertEqual(request.get("status"), "pending")
            self.assertEqual(request.get("chat_id"), "8362291523")
            self.assertEqual(request.get("prompt"), "Search the web for gold macro headlines")

            pending = list_copilot_requests(root, status="pending", limit=5)
            self.assertEqual(len(pending), 1)
            self.assertEqual(pending[0].get("request_id"), request.get("request_id"))

    @patch("utils.copilot_bridge.send_telegram_notification")
    def test_record_response_updates_request_and_sends_reply(self, send_mock):
        send_mock.return_value = {"ok": True, "message_id": 1234}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queued = queue_copilot_request(root, {}, prompt="Run the latest audit", chat_id="8362291523")
            request_id = str((queued.get("request") or {}).get("request_id") or "")

            out = record_copilot_response(
                root,
                {"telegram_notifications": {"enabled": True, "bot_token": "test", "chat_id": "unused"}},
                request_id,
                "I checked the audit and the listener is healthy.",
                send_to_telegram=True,
            )

            self.assertTrue(out.get("ok"))
            request = dict(out.get("request") or {})
            self.assertEqual(request.get("status"), "answered")
            self.assertTrue(bool(request.get("response_sent")))
            self.assertEqual(request.get("response_text"), "I checked the audit and the listener is healthy.")
            send_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()