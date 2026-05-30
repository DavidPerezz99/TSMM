import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.telegram_command_listener as listener


class TelegramAgentModeTests(unittest.TestCase):
    def test_general_sentence_with_back_and_tsmm_does_not_exit_agent_mode(self):
        text = (
            "So what I'm asking you is to access the tsmm app and then look at the analysis summary files "
            "for the current trading jobs we are running. Or for example if I tell you to execute a 7h timeframe "
            "inference would you be able to give me back the output vector ?"
        )

        self.assertFalse(listener._is_agent_mode_exit_request(text))

    def test_background_notifications_are_muted_for_agent_mode_chat(self):
        with patch("scripts.telegram_command_listener._chat_mode", side_effect=lambda chat_id: "agent" if str(chat_id) == "123" else "default"):
            with patch("scripts.telegram_command_listener.send_telegram_notification") as send_mock:
                listener._send_to_chat_ids({"telegram_notifications": {}}, ["123", "456"], "status update")

        send_mock.assert_called_once()
        sent_cfg = send_mock.call_args.args[0]
        self.assertEqual(sent_cfg.get("chat_id"), "456")

    def test_setting_agent_mode_creates_dedicated_session_memory_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "telegram_chat_mode_state.json"
            session_root = root / "llm_session_memories"

            with patch("scripts.telegram_command_listener._chat_mode_state_path", return_value=state_path):
                with patch("scripts.telegram_command_listener._llm_session_memory_root", return_value=session_root):
                    listener._set_chat_mode("123", "agent", source_text="Transfer with agent")
                    payload = listener._chat_mode_payload()
                    entry = payload.get("chats", {}).get("123", {})
                    session_id = str(entry.get("session_id") or "")

                    self.assertTrue(session_id.startswith("llm_session_"))
                    session_file = session_root / "chat_123" / f"{session_id}.jsonl"
                    meta_file = session_root / "chat_123" / f"{session_id}.meta.json"
                    self.assertTrue(session_file.exists())
                    self.assertTrue(meta_file.exists())

                    session_lines = [json.loads(line) for line in session_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                    self.assertEqual(session_lines[0].get("event"), "agent_session_started")

                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    self.assertEqual(meta.get("chat_id"), "123")
                    self.assertEqual(meta.get("source_text"), "Transfer with agent")

    def test_record_llm_session_message_appends_to_active_session_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_path = root / "telegram_chat_mode_state.json"
            session_root = root / "llm_session_memories"

            with patch("scripts.telegram_command_listener._chat_mode_state_path", return_value=state_path):
                with patch("scripts.telegram_command_listener._llm_session_memory_root", return_value=session_root):
                    listener._set_chat_mode("123", "agent", source_text="Transfer with agent")
                    listener._record_llm_session_message("123", "inbound", "Hello there")
                    payload = listener._chat_mode_payload()
                    session_id = str(payload.get("chats", {}).get("123", {}).get("session_id") or "")
                    session_file = session_root / "chat_123" / f"{session_id}.jsonl"

                    session_lines = [json.loads(line) for line in session_file.read_text(encoding="utf-8").splitlines() if line.strip()]
                    self.assertEqual(session_lines[-1].get("direction"), "inbound")
                    self.assertEqual(session_lines[-1].get("text"), "Hello there")

    def test_agent_mode_routes_plain_chat_to_llm_path_only(self):
        with patch("scripts.telegram_command_listener._chat_mode", return_value="agent"):
            with patch("scripts.telegram_command_listener._handle_agent_chat", return_value={"handled": True, "ok": True, "message": "llm only"}) as agent_mock:
                out = listener._handle_command("start trading", {"telegram_listener": {}}, source_chat_id="123")

        self.assertTrue(out.get("handled"))
        self.assertEqual(out.get("message"), "llm only")
        self.assertTrue(out.get("agent_chat_mode"))
        agent_mock.assert_called_once()

    def test_agent_mode_fallback_does_not_emit_ops_summary(self):
        with patch("scripts.telegram_command_listener.load_llm_providers_config", return_value={"providers": {}}):
            with patch("scripts.telegram_command_listener._select_llm_providers_for_chat", return_value=["local_ollama"]):
                with patch("scripts.telegram_command_listener.call_llm", return_value={"ok": False, "text": ""}):
                    out = listener._handle_agent_chat("Are you self aware ?", {"telegram_listener": {}, "llm": {}}, source_chat_id="123")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("agent_chat_mode"))
        self.assertNotIn("TSMM ops agent summary", str(out.get("message") or ""))
        self.assertIn("LLM chat session is active", str(out.get("message") or ""))

    def test_agent_mode_tsmm_question_uses_read_only_context_fallback(self):
        trading_cfg = {"telegram_listener": {}, "llm": {}}
        compact_context = {
            "trading_state": {"status": "agent_b_running"},
            "active_jobs": [
                {
                    "job_id": "job_1",
                    "market_state": "running_position",
                    "decision": "sell",
                    "model": "ulr",
                    "entry": 4526.84,
                    "stop_loss": 4563.05,
                    "take_profit": 4454.41,
                    "mt5_ticket": 310750943,
                    "agent_a_rationale": "Momentum and confluence aligned.",
                    "agent_b_recommendation": "maintain_position",
                    "agent_b_consensus": "sell",
                    "agent_b_score": -0.31,
                    "agent_b_reason": "Trend still favors the downside.",
                }
            ],
        }

        with patch("scripts.telegram_command_listener.load_llm_providers_config", return_value={"providers": {}}):
            with patch("scripts.telegram_command_listener._select_llm_providers_for_chat", return_value=["local_ollama"]):
                with patch("scripts.telegram_command_listener.call_llm", return_value={"ok": False, "text": ""}):
                    with patch("scripts.telegram_command_listener._build_ops_context", return_value={}):
                        with patch("scripts.telegram_command_listener._compact_ops_context", return_value={"trading_state": {"status": "agent_b_running"}}):
                            with patch("scripts.telegram_command_listener._active_job_summaries_for_agent_chat", return_value=compact_context["active_jobs"]):
                                out = listener._handle_agent_chat(
                                    "Tell me about the current trading jobs running in mt5 and the analysis for each signal.",
                                    trading_cfg,
                                    source_chat_id="123",
                                )

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("agent_chat_mode"))
        self.assertIn("read-only mode", str(out.get("message") or ""))
        self.assertIn("job_1", str(out.get("message") or ""))
        self.assertIn("Momentum and confluence aligned", str(out.get("message") or ""))

    def test_agent_mode_builtin_reply_reports_resource_state(self):
        with patch("scripts.telegram_command_listener.read_resource_status", return_value={"cpu": 21.5, "ram": 44.0, "breach_since": 0, "last_relieved_at": 0}):
            out = listener._handle_agent_chat("Can you tell me the current resource state", {"telegram_listener": {}, "llm": {}}, source_chat_id="123")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("builtin_agent_reply"))
        self.assertIn("cpu=21.5%", str(out.get("message") or ""))
        self.assertIn("ram=44.0%", str(out.get("message") or ""))

    def test_agent_mode_builtin_reply_reports_web_search_limitation(self):
        out = listener._handle_agent_chat("Can you search the web ?", {"telegram_listener": {}, "llm": {}}, source_chat_id="123")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("builtin_agent_reply"))
        self.assertIn("web search is not exposed", str(out.get("message") or ""))
        self.assertIn("say copilot", str(out.get("message") or ""))

    def test_agent_mode_builtin_reply_reports_console_limitation(self):
        out = listener._handle_agent_chat("Can you execute console commands ?", {"telegram_listener": {}, "llm": {}}, source_chat_id="123")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("builtin_agent_reply"))
        self.assertIn("arbitrary console execution is not exposed", str(out.get("message") or ""))
        self.assertIn("say copilot", str(out.get("message") or ""))

    def test_agent_mode_builtin_reply_reports_current_time(self):
        trading_cfg = {"agent": {"timezone": "America/Bogota"}, "telegram_listener": {}, "llm": {}}
        out = listener._handle_agent_chat("Can you tell me what time is now ?", trading_cfg, source_chat_id="123")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("builtin_agent_reply"))
        self.assertIn("Current time is", str(out.get("message") or ""))
        self.assertIn("America/Bogota", str(out.get("message") or ""))

    def test_agent_mode_builtin_reply_reports_presence(self):
        out = listener._handle_agent_chat("Are you there ?", {"telegram_listener": {}, "llm": {}}, source_chat_id="123")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("builtin_agent_reply"))
        self.assertIn("agent chat session is active", str(out.get("message") or ""))

    def test_agent_mode_builtin_reply_reports_current_role(self):
        out = listener._handle_agent_chat("What are you doing ?", {"telegram_listener": {}, "llm": {}}, source_chat_id="123")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("builtin_agent_reply"))
        self.assertIn("Telegram agent-chat mode", str(out.get("message") or ""))
        self.assertIn("say copilot", str(out.get("message") or ""))

    def test_agent_mode_builtin_reply_guides_copilot_handoff(self):
        out = listener._handle_agent_chat("Please tell copilot to grab the latest chat history.", {"telegram_listener": {}, "llm": {}}, source_chat_id="123")

        self.assertTrue(out.get("handled"))
        self.assertTrue(out.get("builtin_agent_reply"))
        self.assertIn("say copilot", str(out.get("message") or ""))


if __name__ == "__main__":
    unittest.main()