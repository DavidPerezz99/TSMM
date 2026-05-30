import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.telegram_command_listener import _handle_command, _infer_natural_language_tail


class TelegramListenerRoutingTests(unittest.TestCase):
    def test_deployment_question_does_not_route_to_deploy(self):
        tail, chat_msg = _infer_natural_language_tail("Are you a real llm deployment ?")

        self.assertIsNone(tail)
        self.assertTrue(bool(chat_msg))

    def test_deploy_stop_phrase_routes_to_stop_action(self):
        tail, chat_msg = _infer_natural_language_tail("Deploy stop")

        self.assertEqual(tail, "deploy stop")
        self.assertIsNone(chat_msg)

    @patch("scripts.telegram_command_listener._set_chat_mode")
    def test_transfer_with_agent_switches_into_agent_mode(self, set_chat_mode_mock):
        out = _handle_command("Transfer with agent", {"telegram_listener": {}}, source_chat_id="8362291523")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("ok"))
        self.assertIn("Agent chat mode is now active", str(out.get("message") or ""))
        set_chat_mode_mock.assert_called_once_with("8362291523", "agent", source_text="Transfer with agent")

    @patch("scripts.telegram_command_listener._set_chat_mode")
    def test_transfer_request_sentence_switches_into_agent_mode(self, set_chat_mode_mock):
        out = _handle_command("Can you transfer me with an agent ?", {"telegram_listener": {}}, source_chat_id="8362291523")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("ok"))
        self.assertIn("Agent chat mode is now active", str(out.get("message") or ""))
        set_chat_mode_mock.assert_called_once_with("8362291523", "agent", source_text="Can you transfer me with an agent ?")

    @patch("scripts.telegram_command_listener.queue_copilot_request")
    def test_say_copilot_queues_bridge_request(self, queue_mock):
        queue_mock.return_value = {
            "ok": True,
            "request": {"request_id": "copilot_20260523_205900_deadbeef"},
        }

        out = _handle_command("say copilot search the web for gold macro headlines", {"telegram_listener": {}}, source_chat_id="8362291523")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("ok"))
        self.assertIn("Copilot handoff queued", str(out.get("message") or ""))
        self.assertIn("copilot_20260523_205900_deadbeef", str(out.get("message") or ""))
        queue_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()