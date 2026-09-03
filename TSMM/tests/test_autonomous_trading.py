import sys
import unittest
from datetime import datetime, timezone
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.telegram_command_listener import _active_job_display_states, _active_job_ids, _adopt_untracked_live_agent_b_positions, _agent_b_heartbeat_stale, _current_autonomous_session, _handle_weekend_quiet_mode_exit, _is_active_trading_state, _job_registry_path, _job_root_dir, _job_table_row, _maintain_programmed_orders, _programmed_order_maintenance_decision, _reconcile_orphaned_agent_b_jobs, _refresh_job_state_from_mt5, _scheduled_refresh_target, _send_to_chat_ids, _set_runtime_scope_env, _should_auto_reconcile_agent_b_job, _subscriber_path, _weekend_utc_quiet_mode_active
from utils.trading_job import (
    _autonomous_followup_meets_entry_thresholds,
    _programmed_order_expiration_minutes,
)


class AutonomousTradingTests(unittest.TestCase):
    def test_followup_programmed_orders_use_opportunity_expiration(self):
        trading_cfg = {
            "trading_job": {"programmed_order_expiration_minutes": 420},
            "autonomous_trading": {"opportunity_order_expiration_minutes": 90},
        }

        self.assertEqual(_programmed_order_expiration_minutes(trading_cfg), 420)
        self.assertEqual(
            _programmed_order_expiration_minutes(trading_cfg, "mandatory_session"),
            420,
        )
        self.assertEqual(
            _programmed_order_expiration_minutes(trading_cfg, "autonomous_followup"),
            90,
        )

    @patch("scripts.telegram_command_listener.send_telegram_notification")
    @patch("scripts.telegram_command_listener._chat_mode", return_value="tsmm")
    def test_send_to_chat_ids_prefixes_account_label(self, _chat_mode_mock, send_mock):
        _send_to_chat_ids(
            {"runtime": {"profile_label": "FTMO"}, "telegram_notifications": {"enabled": True}},
            ["123"],
            "status ok",
        )

        sent_message = send_mock.call_args.args[1]
        self.assertEqual(sent_message, "[FTMO] status ok")

    def test_listener_runtime_scope_redirects_runtime_files(self):
        previous = os.environ.get("TSMM_RUNTIME_DIR")
        try:
            runtime_root = _set_runtime_scope_env({"runtime": {"namespace": "ftmo"}})
            self.assertEqual(runtime_root.as_posix(), (ROOT / "reports" / "runtime" / "ftmo").as_posix())
            self.assertEqual(_job_registry_path().as_posix(), (ROOT / "reports" / "runtime" / "ftmo" / "trading_job_registry.json").as_posix())
            self.assertEqual(_job_root_dir().as_posix(), (ROOT / "reports" / "runtime" / "ftmo" / "trading_jobs").as_posix())
            self.assertEqual(_subscriber_path().as_posix(), (ROOT / "reports" / "runtime" / "ftmo" / "telegram_subscribers.json").as_posix())
        finally:
            if previous is None:
                os.environ.pop("TSMM_RUNTIME_DIR", None)
            else:
                os.environ["TSMM_RUNTIME_DIR"] = previous

    def test_scheduled_refresh_target_catches_up_to_last_market_close(self):
        trading_cfg = {
            "telegram_listener": {
                "scheduled_model_refresh": {
                    "enabled": True,
                    "market_close_timezone": "America/New_York",
                    "market_close_time": "17:00",
                    "catch_up_missed_market_closes": True,
                }
            }
        }

        now_utc = datetime(2026, 5, 16, 15, 0, 0, tzinfo=timezone.utc)
        target = _scheduled_refresh_target(trading_cfg, now_utc=now_utc)

        self.assertEqual(target.get("target_date"), "2026-05-15")
        self.assertEqual(target.get("scheduled_local"), "2026-05-15 17:00:00")
        self.assertTrue(target.get("due"))

    def test_weekend_quiet_mode_uses_bogota_friday_to_sunday_window(self):
        trading_cfg = {
            "agent": {"timezone": "America/Bogota"},
            "telegram_listener": {
                "scheduled_model_refresh": {
                    "weekend_utc_quiet_mode": True,
                    "weekend_quiet_timezone": "America/Bogota",
                    "weekend_quiet_start_day": 4,
                    "weekend_quiet_start_time": "17:00",
                    "weekend_quiet_end_day": 6,
                    "weekend_quiet_end_time": "17:00",
                }
            }
        }

        friday_before_close_utc = datetime(2026, 5, 15, 21, 59, 0, tzinfo=timezone.utc)
        friday_close_utc = datetime(2026, 5, 15, 22, 0, 0, tzinfo=timezone.utc)
        sunday_before_open_utc = datetime(2026, 5, 17, 21, 59, 0, tzinfo=timezone.utc)
        sunday_open_utc = datetime(2026, 5, 17, 22, 0, 0, tzinfo=timezone.utc)

        self.assertFalse(_weekend_utc_quiet_mode_active(trading_cfg, now_utc=friday_before_close_utc))
        self.assertTrue(_weekend_utc_quiet_mode_active(trading_cfg, now_utc=friday_close_utc))
        self.assertTrue(_weekend_utc_quiet_mode_active(trading_cfg, now_utc=sunday_before_open_utc))
        self.assertFalse(_weekend_utc_quiet_mode_active(trading_cfg, now_utc=sunday_open_utc))

    def test_agent_b_heartbeat_stale_when_last_tick_missing_past_grace(self):
        trading_cfg = {"mode_b": {"poll_seconds": 300}, "telegram_listener": {}}
        state = {
            "status": "agent_b_running",
            "stage": "agent_b",
            "started_at": "2026-05-24 20:00:00",
        }

        self.assertTrue(
            _agent_b_heartbeat_stale(
                trading_cfg,
                state,
                now_utc=datetime(2026, 5, 24, 20, 15, 0, tzinfo=timezone.utc),
            )
        )

    def test_agent_b_heartbeat_not_stale_while_runner_is_within_startup_grace(self):
        trading_cfg = {"mode_b": {"poll_seconds": 300}, "telegram_listener": {}}
        state = {
            "status": "agent_b_running",
            "stage": "agent_b",
            "started_at": "2026-05-21 16:54:36",
            "runner_started_at": "2026-05-24 20:10:00",
        }

        self.assertFalse(
            _agent_b_heartbeat_stale(
                trading_cfg,
                state,
                now_utc=datetime(2026, 5, 24, 20, 14, 0, tzinfo=timezone.utc),
            )
        )

    @patch("scripts.telegram_command_listener._is_pid_alive", return_value=True)
    def test_should_auto_reconcile_agent_b_job_when_runner_alive_but_heartbeat_stale(self, _alive_mock):
        trading_cfg = {"mode_b": {"poll_seconds": 300}, "telegram_listener": {}}
        state = {
            "status": "agent_b_running",
            "stage": "agent_b",
            "runner_pid": 1234,
            "position": {"ticket": 101},
            "started_at": "2026-05-24 20:00:00",
        }

        self.assertTrue(_should_auto_reconcile_agent_b_job(trading_cfg, state))

    @patch("scripts.telegram_command_listener._console_trace")
    @patch("scripts.telegram_command_listener._send_to_chat_ids")
    @patch("scripts.telegram_command_listener._subscriber_chat_ids")
    @patch("scripts.telegram_command_listener._format_active_jobs_digest")
    @patch("scripts.telegram_command_listener._restart_endpoint_service")
    def test_weekend_quiet_mode_exit_restarts_endpoint_and_notifies(
        self,
        restart_mock,
        digest_mock,
        chat_ids_mock,
        send_mock,
        console_mock,
    ):
        restart_mock.return_value = {"ok": True, "pid": 1234}
        digest_mock.return_value = "TSMM active jobs (2)"
        chat_ids_mock.return_value = ["123"]

        result = _handle_weekend_quiet_mode_exit(
            {
                "telegram_listener": {"scheduled_model_refresh": {"weekend_stop_endpoint": True}},
                "telegram_notifications": {"enabled": True},
            },
            default_chat_id="123",
            last_chat_id="",
        )

        self.assertTrue(result["endpoint_restart"]["ok"])
        self.assertEqual(result["digest"], "TSMM active jobs (2)")
        self.assertTrue(result["notified"])
        send_mock.assert_called_once()
        console_mock.assert_called_once()

    def test_session_window_resolution_uses_configured_timezone(self):
        trading_cfg = {
            "agent": {"timezone": "America/Bogota"},
            "autonomous_trading": {
                "enabled": True,
                "timezone": "America/Bogota",
                "session_windows": [
                    {"name": "asia_australia", "start": "17:00", "end": "07:00"},
                    {"name": "london", "start": "07:00", "end": "14:00"},
                    {"name": "new_york", "start": "14:00", "end": "21:00"},
                ],
            },
        }

        now_utc = datetime(2026, 5, 12, 19, 30, 0, tzinfo=timezone.utc)
        session = _current_autonomous_session(trading_cfg, now_utc=now_utc)

        self.assertEqual(session.get("name"), "new_york")
        self.assertEqual(session.get("start_local"), "2026-05-12 14:00:00")
        self.assertEqual(session.get("end_local"), "2026-05-12 21:00:00")

    def test_followup_thresholds_reject_weak_plan(self):
        trading_cfg = {
            "autonomous_trading": {
                "min_success_probability_for_followup": 0.58,
                "min_confidence_for_followup": 0.55,
                "min_cm_accuracy_for_followup": 0.52,
                "max_input_fooling_risk_for_followup": 0.45,
                "require_consensus_alignment_for_followup": True,
            }
        }
        plan = {
            "decision": "buy",
            "success_probability": 0.49,
            "confidence": 0.51,
            "cm_accuracy": 0.5,
            "input_fooling_risk": 0.5,
            "enrichment": {"alignment": "opposed"},
        }

        ok, reason = _autonomous_followup_meets_entry_thresholds(plan, trading_cfg)

        self.assertFalse(ok)
        self.assertIn("success_probability<0.58", reason)
        self.assertIn("confidence<0.55", reason)
        self.assertIn("cm_accuracy<0.52", reason)
        self.assertIn("input_fooling_risk>0.45", reason)
        self.assertIn("consensus_alignment_opposed", reason)

    def test_followup_thresholds_accept_strong_plan(self):
        trading_cfg = {
            "autonomous_trading": {
                "min_success_probability_for_followup": 0.58,
                "min_confidence_for_followup": 0.55,
                "min_cm_accuracy_for_followup": 0.52,
                "max_input_fooling_risk_for_followup": 0.45,
                "require_consensus_alignment_for_followup": True,
            }
        }
        plan = {
            "decision": "sell",
            "success_probability": 0.67,
            "confidence": 0.62,
            "cm_accuracy": 0.6,
            "input_fooling_risk": 0.21,
            "enrichment": {"alignment": "aligned"},
        }

        ok, reason = _autonomous_followup_meets_entry_thresholds(plan, trading_cfg)

        self.assertTrue(ok)
        self.assertEqual(reason, "thresholds_passed")

    @patch("scripts.telegram_command_listener._pending_approval_for_job")
    def test_pending_approval_job_is_treated_as_active(self, pending_mock):
        pending_mock.return_value = {"job_id": "job_test", "title": "Approve trade"}
        state = {
            "job_id": "job_test",
            "status": "completed",
            "stage": "agent_a",
            "ended_at": "2026-05-13 22:00:00",
            "closed_reason": "awaiting_manual_approval",
        }

        self.assertTrue(_is_active_trading_state(state))

    @patch("scripts.telegram_command_listener._send_to_chat_ids")
    @patch("scripts.telegram_command_listener._subscriber_chat_ids")
    @patch("scripts.telegram_command_listener._persist_job_state")
    @patch("scripts.telegram_command_listener._run_cmd_async")
    @patch("scripts.telegram_command_listener._stop_job_runner")
    @patch("scripts.telegram_command_listener._adopt_untracked_live_agent_b_positions")
    @patch("scripts.telegram_command_listener._read_trading_state")
    @patch("scripts.telegram_command_listener._active_job_ids")
    @patch("scripts.telegram_command_listener.MT5Adapter")
    def test_reconcile_keeps_live_orphan_supervision_when_direct_ticket_lookup_misses(
        self,
        adapter_cls,
        active_ids_mock,
        read_state_mock,
        adopt_mock,
        stop_runner_mock,
        run_cmd_async_mock,
        persist_mock,
        chat_ids_mock,
        send_mock,
    ):
        adapter = adapter_cls.return_value
        adapter.connect.return_value = (True, "connected")
        adapter.get_position_by_ticket.return_value = {"ok": True, "position": None}
        adapter.list_open_positions.return_value = {
            "ok": True,
            "positions": [
                {
                    "ticket": 101,
                    "symbol": "XAUUSD",
                    "volume": 0.01,
                    "price_open": 100.0,
                    "price_current": 101.0,
                    "sl": 99.0,
                    "tp": 102.0,
                    "comment": "TSMM programmed ",
                    "magic": 7070001,
                }
            ],
        }
        adapter.get_position_close_outcome.return_value = {"ok": False, "found": False, "ticket": 101}
        adapter.shutdown.return_value = None

        active_ids_mock.return_value = ["job_live"]
        read_state_mock.return_value = {
            "job_id": "job_live",
            "status": "agent_b_running",
            "stage": "agent_b",
            "runner_pid": 0,
            "position": {"ticket": 101, "symbol": "XAUUSD"},
            "state_path": "reports/runtime/trading_jobs/job_live/trading_job_state.json",
            "started_at": "2026-05-24 20:00:00",
        }
        run_cmd_async_mock.return_value = {"ok": True, "pid": 4321}
        chat_ids_mock.return_value = ["123"]
        adopt_mock.return_value = []

        events = _reconcile_orphaned_agent_b_jobs(
            trading_cfg={"broker": {"mt5": {}}, "execution": {"symbol": "XAUUSD"}, "telegram_notifications": {"enabled": True}},
            trading_config_path=ROOT / "config" / "trading_agent.yaml",
            job_cooldowns={},
            default_chat_id="",
            last_chat_id="123",
        )

        resume_events = [event for event in events if event.get("action") == "resume_agent_b"]
        close_events = [event for event in events if event.get("action") == "close_stale_agent_b"]

        self.assertEqual(len(close_events), 0)
        self.assertGreaterEqual(len(resume_events), 1)
        self.assertTrue(resume_events[0]["ok"])
        self.assertEqual(resume_events[0]["ticket"], 101)
        persist_mock.assert_called_once()
        persisted_payload = persist_mock.call_args[0][1]
        self.assertTrue(bool(persisted_payload.get("runner_started_at")))
        self.assertEqual(((persisted_payload.get("position") or {}).get("ticket")), 101)
        adapter.get_position_close_outcome.assert_not_called()
        send_mock.assert_called_once()

    @patch("scripts.telegram_command_listener._pending_approval_for_job")
    def test_job_table_row_labels_programmed_orders_and_pending_approval(self, pending_mock):
        pending_mock.return_value = {}
        trading_cfg = {"agent": {"timezone": "America/Bogota"}}
        programmed_state = {
            "job_id": "job_programmed",
            "status": "agent_a_completed",
            "order": {"order_ticket": 321, "price_open": 100.0, "sl": 99.0, "tp": 102.0},
            "plan": {"decision": "buy", "model": "nbeats"},
            "started_at": "2026-05-13 20:00:00",
        }

        programmed_row = _job_table_row(trading_cfg, programmed_state)

        self.assertEqual(programmed_row[1], "programmed_order")
        self.assertEqual(programmed_row[2], "Agent A pending fill")

        pending_mock.return_value = {"job_id": "job_pending", "title": "Approve trade"}
        pending_state = {
            "job_id": "job_pending",
            "status": "agent_a_completed",
            "plan": {"decision": "buy", "model": "nbeats"},
            "started_at": "2026-05-13 20:05:00",
        }

        pending_row = _job_table_row(trading_cfg, pending_state)

        self.assertEqual(pending_row[1], "pending_approval")
        self.assertEqual(pending_row[2], "Awaiting approval")

    def test_job_table_row_uses_live_position_sl_tp(self):
        trading_cfg = {"agent": {"timezone": "America/Bogota"}}
        live_state = {
            "job_id": "job_live",
            "status": "agent_b_running",
            "plan": {"decision": "buy", "model": "nbeats", "entry": 4700.0, "stop_loss": 4680.0, "take_profit": 4740.0},
            "position": {"ticket": 123, "price_open": 4702.0, "sl": 4691.0, "tp": 4728.0, "profit": 4.5},
            "agent_b_plan": {"next_review_utc": "2026-05-14 15:10:00"},
            "started_at": "2026-05-14 15:00:00",
        }

        row = _job_table_row(trading_cfg, live_state)

        self.assertEqual(row[1], "running_position")
        self.assertEqual(row[4], "4702.0000")
        self.assertEqual(row[5], "4691.0000")
        self.assertEqual(row[6], "4728.0000")

    def test_refresh_job_state_from_mt5_clears_stale_terminal_fields_for_live_order(self):
        class Adapter:
            def get_position_by_ticket(self, _ticket):
                return {"ok": True, "position": None}

            def find_position_by_order(self, _ticket):
                return {"ok": True, "position": None}

            def find_live_position_by_plan(self, **_kwargs):
                return {"ok": True, "position": None}

            def get_pending_order_by_ticket(self, _ticket):
                return {
                    "ok": True,
                    "order": {"order_ticket": 305982540, "price_open": 4710.15, "sl": 4672.47, "tp": 4785.51},
                }

        state = {
            "job_id": "job_live",
            "status": "failed",
            "stage": "agent_a",
            "order_submission_mode": "programmed",
            "ended_at": "2026-05-14 14:45:18",
            "closed_reason": "order_place_failed: unknown",
            "order": {"order_ticket": 305982540},
            "plan": {"symbol": "XAUUSD", "volume": 0.01, "entry": 4710.15, "stop_loss": 4672.47, "take_profit": 4785.51},
        }

        refreshed = _refresh_job_state_from_mt5({"execution": {"symbol": "XAUUSD", "default_volume": 0.01}}, state, Adapter())

        self.assertEqual(refreshed["status"], "agent_a_completed")
        self.assertNotIn("ended_at", refreshed)
        self.assertNotIn("closed_reason", refreshed)
        self.assertEqual(refreshed["order"]["order_ticket"], 305982540)

    def test_refresh_job_state_from_mt5_closes_stale_programmed_order_missing_in_mt5(self):
        class Adapter:
            def get_position_by_ticket(self, _ticket):
                return {"ok": True, "position": None}

            def find_position_by_order(self, _ticket):
                return {"ok": True, "position": None}

            def find_live_position_by_plan(self, **_kwargs):
                return {"ok": True, "position": None}

            def get_pending_order_by_ticket(self, _ticket):
                return {"ok": True, "order": None}

            def find_pending_order_by_plan(self, **_kwargs):
                return {"ok": True, "order": None}

        state = {
            "job_id": "job_missing",
            "status": "agent_a_completed",
            "stage": "agent_a",
            "mode": "mode_a",
            "agent_a_approved": True,
            "order_submission_mode": "programmed",
            "order": {"order_ticket": 307733263, "price_open": 4584.32, "sl": 4620.99, "tp": 4510.97},
            "plan": {"symbol": "XAUUSD", "volume": 0.01, "entry": 4584.32, "stop_loss": 4620.99, "take_profit": 4510.97},
            "programmed_order_expiration_utc": "2030-05-18 21:57:40",
        }

        refreshed = _refresh_job_state_from_mt5({"execution": {"symbol": "XAUUSD", "default_volume": 0.01}}, state, Adapter())

        self.assertEqual(refreshed["status"], "closed")
        self.assertEqual(refreshed["closed_reason"], "programmed_order_missing_in_mt5")
        self.assertNotIn("order", refreshed)
        self.assertTrue(refreshed.get("ended_at"))

    def test_programmed_order_maintenance_decision_cancels_crossed_buy_stop(self):
        trading_cfg = {
            "agent": {"signal_interpretation": "contrarian"},
            "autonomous_trading": {
                "pending_order_maintenance": {
                    "enabled": True,
                    "cancel_opposed_consensus_threshold": 0.25,
                    "entry_cross_tolerance_abs": 0.05,
                }
            },
            "mode_b": {"close_consensus_threshold": 0.25},
        }
        state = {"plan": {"decision": "buy", "signal_interpretation": "contrarian"}}
        order = {"type": 4, "price_open": 4584.32}
        assessment = {"consensus": "buy", "consensus_score": 0.41}

        decision = _programmed_order_maintenance_decision(state, order, assessment, 4584.50, trading_cfg)

        self.assertTrue(decision["cancel"])
        self.assertEqual(decision["reason"], "programmed_order_entry_crossed")

    def test_programmed_order_maintenance_decision_cancels_strategy_mismatch(self):
        trading_cfg = {
            "agent": {"signal_interpretation": "contrarian"},
            "autonomous_trading": {
                "pending_order_maintenance": {
                    "enabled": True,
                    "cancel_opposed_consensus_threshold": 0.25,
                    "entry_cross_tolerance_abs": 0.05,
                }
            },
            "mode_b": {"close_consensus_threshold": 0.25},
        }
        state = {"plan": {"decision": "buy", "signal_interpretation": "momentum"}}
        order = {"type": 4, "price_open": 4584.32}
        assessment = {"consensus": "sell", "consensus_score": -0.41}

        decision = _programmed_order_maintenance_decision(state, order, assessment, 4583.90, trading_cfg)

        self.assertTrue(decision["cancel"])
        self.assertEqual(decision["reason"], "programmed_order_strategy_mismatch")

    @patch("scripts.telegram_command_listener._persist_job_state")
    @patch("scripts.telegram_command_listener._refresh_job_state_from_mt5")
    @patch("scripts.telegram_command_listener._read_trading_state")
    @patch("scripts.telegram_command_listener._active_job_ids")
    @patch("scripts.telegram_command_listener._collect_all_model_assessment_signals")
    @patch("scripts.telegram_command_listener.MT5Adapter")
    def test_maintain_programmed_orders_cancels_invalidated_pending_order(
        self,
        adapter_cls,
        assessment_mock,
        active_ids_mock,
        read_state_mock,
        refresh_mock,
        persist_mock,
    ):
        adapter = adapter_cls.return_value
        adapter.connect.return_value = (True, "connected")
        adapter.cancel_pending_order.return_value = {"ok": True, "order_ticket": 307733263}
        adapter.shutdown.return_value = None
        adapter._mt5 = type("Mt5", (), {"symbol_info_tick": lambda self, _symbol: type("Tick", (), {"ask": 4584.60, "bid": 4584.40})()})()

        assessment_mock.return_value = {"consensus": "buy", "consensus_score": 0.42, "source": "agent_a_enrichment"}
        active_ids_mock.return_value = ["job_live"]
        state = {
            "job_id": "job_live",
            "status": "agent_a_completed",
            "stage": "agent_a",
            "mode": "mode_a",
            "order_submission_mode": "programmed",
            "state_path": "reports/runtime/trading_jobs/job_live/trading_job_state.json",
            "order": {"order_ticket": 307733263, "symbol": "XAUUSD", "price_open": 4584.32, "sl": 4620.99, "tp": 4510.97, "type": 4},
            "plan": {"decision": "buy", "model": "ulr", "signal_interpretation": "contrarian"},
        }
        read_state_mock.return_value = state
        refresh_mock.side_effect = [state, {**state, "order": {}}]

        events = _maintain_programmed_orders(
            {
                "agent": {"signal_interpretation": "contrarian"},
                "broker": {"mt5": {}},
                "mode_b": {"close_consensus_threshold": 0.25},
                "autonomous_trading": {
                    "pending_order_maintenance": {
                        "enabled": True,
                        "assessment_timeout_seconds": 3.0,
                        "cancel_opposed_consensus_threshold": 0.25,
                        "entry_cross_tolerance_abs": 0.05,
                    }
                },
            }
        )

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["reason"], "programmed_order_entry_crossed")
        persist_mock.assert_called_once()
        persisted_payload = persist_mock.call_args[0][1]
        self.assertEqual(persisted_payload["status"], "closed")
        self.assertEqual(persisted_payload["closed_reason"], "programmed_order_entry_crossed")
        self.assertNotIn("order", persisted_payload)
        adapter.shutdown.assert_called_once()

    @patch("scripts.telegram_command_listener._persist_job_state")
    @patch("scripts.telegram_command_listener._refresh_job_state_from_mt5")
    @patch("scripts.telegram_command_listener._read_trading_state")
    @patch("scripts.telegram_command_listener._active_job_ids")
    @patch("scripts.telegram_command_listener._collect_all_model_assessment_signals")
    @patch("scripts.telegram_command_listener.MT5Adapter")
    def test_maintain_programmed_orders_cancels_duplicate_pending_order(
        self,
        adapter_cls,
        assessment_mock,
        active_ids_mock,
        read_state_mock,
        refresh_mock,
        persist_mock,
    ):
        adapter = adapter_cls.return_value
        adapter.connect.return_value = (True, "connected")
        adapter.cancel_pending_order.return_value = {"ok": True, "order_ticket": 307733263}
        adapter.shutdown.return_value = None
        adapter._mt5 = type("Mt5", (), {"symbol_info_tick": lambda self, _symbol: type("Tick", (), {"ask": 4584.00, "bid": 4583.80})()})()

        assessment_mock.return_value = {"consensus": "buy", "consensus_score": 0.42, "source": "agent_a_enrichment"}
        active_ids_mock.return_value = ["job_older", "job_newer"]
        newer_state = {
            "job_id": "job_newer",
            "status": "agent_a_completed",
            "stage": "agent_a",
            "mode": "mode_a",
            "started_at": "2026-05-19 14:41:48",
            "order_submission_mode": "programmed",
            "state_path": "reports/runtime/trading_jobs/job_newer/trading_job_state.json",
            "order": {"order_ticket": 307733999, "symbol": "XAUUSD", "price_open": 4584.32, "sl": 4620.99, "tp": 4510.97, "type": 4},
            "plan": {"decision": "buy", "model": "ulr", "signal_interpretation": "contrarian", "volume": 0.01},
        }
        older_state = {
            "job_id": "job_older",
            "status": "agent_a_completed",
            "stage": "agent_a",
            "mode": "mode_a",
            "started_at": "2026-05-19 14:41:34",
            "order_submission_mode": "programmed",
            "state_path": "reports/runtime/trading_jobs/job_older/trading_job_state.json",
            "order": {"order_ticket": 307733263, "symbol": "XAUUSD", "price_open": 4584.32, "sl": 4620.99, "tp": 4510.97, "type": 4},
            "plan": {"decision": "buy", "model": "ulr", "signal_interpretation": "contrarian", "volume": 0.01},
        }
        read_state_mock.side_effect = [older_state, newer_state]
        refresh_mock.side_effect = [newer_state, older_state, {**older_state, "order": {}}]

        events = _maintain_programmed_orders(
            {
                "agent": {"signal_interpretation": "contrarian"},
                "broker": {"mt5": {}},
                "mode_b": {"close_consensus_threshold": 0.25},
                "autonomous_trading": {
                    "pending_order_maintenance": {
                        "enabled": True,
                        "assessment_timeout_seconds": 3.0,
                        "cancel_opposed_consensus_threshold": 0.25,
                        "entry_cross_tolerance_abs": 0.05,
                    }
                },
            }
        )

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["job_id"], "job_older")
        self.assertEqual(events[0]["reason"], "programmed_order_duplicate")
        persist_mock.assert_called_once()
        persisted_payload = persist_mock.call_args[0][1]
        self.assertEqual(persisted_payload["closed_reason"], "programmed_order_duplicate")
        adapter.cancel_pending_order.assert_called_once_with(307733263)
        adapter.shutdown.assert_called_once()

    @patch("scripts.telegram_command_listener._read_json")
    @patch("scripts.telegram_command_listener._persist_job_state")
    @patch("scripts.telegram_command_listener._refresh_job_state_from_mt5")
    @patch("scripts.telegram_command_listener._read_trading_state")
    @patch("scripts.telegram_command_listener._all_disk_job_ids")
    @patch("scripts.telegram_command_listener._registry_payload")
    @patch("scripts.telegram_command_listener.MT5Adapter")
    def test_active_job_ids_revives_stale_programmed_order_from_mt5(
        self,
        adapter_cls,
        registry_mock,
        disk_ids_mock,
        read_state_mock,
        refresh_mock,
        persist_mock,
        read_json_mock,
    ):
        adapter = adapter_cls.return_value
        adapter.connect.return_value = (True, "connected")
        registry_mock.return_value = {"active_job_ids": [], "jobs": {"job_live": {"job_id": "job_live"}}}
        disk_ids_mock.return_value = []
        read_state_mock.return_value = {
            "job_id": "job_live",
            "status": "failed",
            "stage": "agent_a",
            "mode": "mode_a",
            "order_submission_mode": "programmed",
            "ended_at": "2026-05-14 14:45:18",
            "closed_reason": "order_place_failed: unknown",
            "state_path": "reports/runtime/trading_jobs/job_live/trading_job_state.json",
        }
        refresh_mock.return_value = {
            "job_id": "job_live",
            "status": "agent_a_completed",
            "stage": "agent_a",
            "mode": "mode_a",
            "order_submission_mode": "programmed",
            "state_path": "reports/runtime/trading_jobs/job_live/trading_job_state.json",
            "order": {"order_ticket": 305982540, "price_open": 4710.15, "sl": 4672.47, "tp": 4785.51},
        }
        read_json_mock.return_value = {}

        out = _active_job_ids({"broker": {"mt5": {}}})

        self.assertEqual(out, ["job_live"])
        persist_mock.assert_called_once()
        adapter.shutdown.assert_called_once()

    @patch("scripts.telegram_command_listener._persist_job_state")
    @patch("scripts.telegram_command_listener._refresh_job_state_from_mt5")
    @patch("scripts.telegram_command_listener._read_trading_state")
    @patch("scripts.telegram_command_listener._active_job_ids")
    @patch("scripts.telegram_command_listener.MT5Adapter")
    def test_active_job_display_states_dedupes_duplicate_pending_order_and_includes_orphan_positions(
        self,
        adapter_cls,
        active_ids_mock,
        read_state_mock,
        refresh_mock,
        persist_mock,
    ):
        adapter = adapter_cls.return_value
        adapter.connect.return_value = (True, "connected")
        adapter.list_open_positions.return_value = {
            "ok": True,
            "positions": [
                {"ticket": 101, "symbol": "XAUUSD", "volume": 0.01, "price_open": 4700.0, "sl": 4680.0, "tp": 4740.0, "comment": "TSMM programmed ", "magic": 7070001, "type": 0},
                {"ticket": 102, "symbol": "XAUUSD", "volume": 0.01, "price_open": 4701.0, "sl": 4681.0, "tp": 4741.0, "comment": "TSMM programmed ", "magic": 7070001, "type": 0},
                {"ticket": 103, "symbol": "XAUUSD", "volume": 0.01, "price_open": 4702.0, "sl": 4682.0, "tp": 4742.0, "comment": "TSMM programmed ", "magic": 7070001, "type": 0},
            ],
        }
        adapter._mt5 = type("Mt5", (), {"orders_get": lambda self: [{"order_ticket": 305982540, "symbol": "XAUUSD", "volume": 0.01, "price_open": 4710.15, "sl": 4672.47, "tp": 4785.51, "comment": "TSMM programmed ", "magic": 7070001, "type": 3}]})()
        active_ids_mock.return_value = ["job_a", "job_b"]
        read_state_mock.side_effect = [
            {"job_id": "job_a", "status": "agent_a_completed", "stage": "agent_a", "mode": "mode_a", "state_path": "a.json"},
            {"job_id": "job_b", "status": "agent_a_completed", "stage": "agent_a", "mode": "mode_a", "state_path": "b.json"},
        ]
        refresh_mock.side_effect = [
            {"job_id": "job_a", "status": "agent_a_completed", "stage": "agent_a", "mode": "mode_a", "state_path": "a.json", "order": {"order_ticket": 305982540, "symbol": "XAUUSD", "volume": 0.01, "price_open": 4710.15, "sl": 4672.47, "tp": 4785.51, "comment": "TSMM programmed ", "magic": 7070001, "type": 3}},
            {"job_id": "job_b", "status": "agent_a_completed", "stage": "agent_a", "mode": "mode_a", "state_path": "b.json", "order": {"order_ticket": 305982540, "symbol": "XAUUSD", "volume": 0.01, "price_open": 4710.15, "sl": 4672.47, "tp": 4785.51, "comment": "TSMM programmed ", "magic": 7070001, "type": 3}},
        ]

        states = _active_job_display_states({"broker": {"mt5": {}}, "execution": {"symbol": "XAUUSD"}})

        order_tickets = [int(((state.get("order") or {}).get("order_ticket", 0) or 0)) for state in states]
        position_tickets = [int(((state.get("position") or {}).get("ticket", 0) or 0)) for state in states if (state.get("position") or {})]
        self.assertEqual(order_tickets.count(305982540), 1)
        self.assertEqual(position_tickets, [101, 102, 103])
        self.assertEqual(len(states), 4)
        self.assertEqual(persist_mock.call_count, 2)
        adapter.shutdown.assert_called_once()

    @patch("scripts.telegram_command_listener._send_to_chat_ids")
    @patch("scripts.telegram_command_listener._persist_job_state")
    @patch("scripts.telegram_command_listener._run_cmd_async")
    @patch("scripts.telegram_command_listener._job_for_ticket_on_disk")
    @patch("scripts.telegram_command_listener._subscriber_chat_ids")
    def test_adopt_untracked_positions_resumes_existing_ticket_owner_before_creating_orphan(
        self,
        chat_ids_mock,
        job_for_ticket_mock,
        run_cmd_async_mock,
        persist_mock,
        send_mock,
    ):
        class Adapter:
            def list_open_positions(self_inner):
                return {
                    "ok": True,
                    "positions": [
                        {
                            "ticket": 101,
                            "symbol": "XAUUSD",
                            "volume": 0.01,
                            "price_open": 4700.0,
                            "sl": 4680.0,
                            "tp": 4740.0,
                            "comment": "TSMM programmed ",
                            "magic": 7070001,
                            "type": 0,
                        }
                    ],
                }

        chat_ids_mock.return_value = ["123"]
        existing_state = {
            "job_id": "job_live",
            "status": "agent_b_running",
            "stage": "agent_b",
            "runner_pid": 0,
            "position": {"ticket": 101, "symbol": "XAUUSD"},
            "state_path": "reports/runtime/trading_jobs/job_live/trading_job_state.json",
        }
        job_for_ticket_mock.return_value = ("job_live", Path("reports/runtime/trading_jobs/job_live/trading_job_state.json"), existing_state)
        run_cmd_async_mock.return_value = {"ok": True, "pid": 4321}

        events = _adopt_untracked_live_agent_b_positions(
            trading_cfg={"broker": {"mt5": {}}, "execution": {"symbol": "XAUUSD"}},
            trading_config_path=ROOT / "config" / "trading_agent.yaml",
            job_cooldowns={},
            default_chat_id="",
            last_chat_id="123",
            adapter=Adapter(),
        )

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["action"], "resume_tracked_mt5_position")
        self.assertEqual(events[0]["job_id"], "job_live")
        persist_mock.assert_called_once()
        persisted_payload = persist_mock.call_args[0][1]
        self.assertEqual(persisted_payload["runner_pid"], 4321)
        self.assertTrue(bool(persisted_payload.get("runner_started_at")))
        self.assertEqual(((persisted_payload.get("position") or {}).get("ticket")), 101)
        send_mock.assert_called_once()

    @patch("scripts.telegram_command_listener._send_to_chat_ids")
    @patch("scripts.telegram_command_listener._persist_job_state")
    @patch("scripts.telegram_command_listener._run_cmd_async")
    @patch("scripts.telegram_command_listener._stop_job_runner")
    @patch("scripts.telegram_command_listener._sync_state_runner_pid_from_process")
    @patch("scripts.telegram_command_listener._job_for_ticket_on_disk")
    @patch("scripts.telegram_command_listener._subscriber_chat_ids")
    def test_adopt_untracked_positions_rebinds_existing_non_agent_b_job(
        self,
        chat_ids_mock,
        job_for_ticket_mock,
        _sync_pid_mock,
        stop_runner_mock,
        run_cmd_async_mock,
        persist_mock,
        send_mock,
    ):
        class Adapter:
            def list_open_positions(self_inner):
                return {
                    "ok": True,
                    "positions": [
                        {
                            "ticket": 202,
                            "symbol": "XAUUSD",
                            "volume": 0.01,
                            "price_open": 4705.0,
                            "sl": 4667.0,
                            "tp": 4781.0,
                            "comment": "",
                            "magic": 0,
                            "type": 0,
                        }
                    ],
                }

        chat_ids_mock.return_value = ["123"]
        existing_state = {
            "job_id": "job_existing",
            "status": "agent_a_completed",
            "stage": "agent_a",
            "runner_pid": 999,
            "position": {"ticket": 202, "symbol": "XAUUSD"},
            "plan": {"decision": "buy"},
            "state_path": "reports/runtime/trading_jobs/job_existing/trading_job_state.json",
        }
        job_for_ticket_mock.return_value = (
            "job_existing",
            Path("reports/runtime/trading_jobs/job_existing/trading_job_state.json"),
            existing_state,
        )
        run_cmd_async_mock.return_value = {"ok": True, "pid": 5432}

        events = _adopt_untracked_live_agent_b_positions(
            trading_cfg={
                "broker": {"mt5": {}},
                "execution": {"symbol": "XAUUSD"},
                "telegram_listener": {
                    "adopt_any_running_positions": True,
                    "rebind_existing_non_agent_b": True,
                    "seed_missing_sltp_on_adopt": False,
                },
            },
            trading_config_path=ROOT / "config" / "trading_agent.yaml",
            job_cooldowns={},
            default_chat_id="",
            last_chat_id="123",
            adapter=Adapter(),
        )

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["action"], "rebind_existing_live_position_to_agent_b")
        self.assertEqual(events[0]["job_id"], "job_existing")
        stop_runner_mock.assert_called_once()
        self.assertGreaterEqual(persist_mock.call_count, 2)
        pre_rebind_payload = persist_mock.call_args_list[0][0][1]
        final_payload = persist_mock.call_args_list[-1][0][1]
        self.assertEqual(pre_rebind_payload.get("status"), "agent_b_running")
        self.assertEqual(pre_rebind_payload.get("stage"), "agent_b")
        self.assertEqual(((pre_rebind_payload.get("plan") or {}).get("model")), "manual_entry_auto_adopt")
        self.assertEqual(final_payload.get("runner_pid"), 5432)
        send_mock.assert_called_once()

    @patch("scripts.telegram_command_listener._send_to_chat_ids")
    @patch("scripts.telegram_command_listener._persist_job_state")
    @patch("scripts.telegram_command_listener._run_cmd_async")
    @patch("scripts.telegram_command_listener._job_for_ticket_on_disk")
    @patch("scripts.telegram_command_listener._subscriber_chat_ids")
    def test_adopt_untracked_positions_can_adopt_external_symbol_when_any_running_enabled(
        self,
        chat_ids_mock,
        job_for_ticket_mock,
        run_cmd_async_mock,
        persist_mock,
        send_mock,
    ):
        class Adapter:
            def list_open_positions(self_inner):
                return {
                    "ok": True,
                    "positions": [
                        {
                            "ticket": 303,
                            "symbol": "EURUSD",
                            "volume": 0.02,
                            "price_open": 1.1,
                            "sl": 1.09,
                            "tp": 1.12,
                            "comment": "external trade",
                            "magic": 555,
                            "type": 0,
                        }
                    ],
                }

        chat_ids_mock.return_value = ["123"]
        job_for_ticket_mock.return_value = ("", Path(), {})
        run_cmd_async_mock.return_value = {"ok": True, "pid": 6543}

        events = _adopt_untracked_live_agent_b_positions(
            trading_cfg={
                "broker": {"mt5": {}},
                "execution": {"symbol": "XAUUSD"},
                "telegram_listener": {
                    "adopt_any_running_positions": True,
                    "seed_missing_sltp_on_adopt": False,
                },
            },
            trading_config_path=ROOT / "config" / "trading_agent.yaml",
            job_cooldowns={},
            default_chat_id="",
            last_chat_id="123",
            adapter=Adapter(),
        )

        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]["ok"])
        self.assertEqual(events[0]["action"], "adopt_orphan_mt5_position")
        self.assertEqual(events[0]["adoption_kind"], "external")
        self.assertEqual(events[0]["ticket"], 303)
        self.assertEqual(persist_mock.call_count, 2)
        bootstrap_payload = persist_mock.call_args_list[0][0][1]
        self.assertEqual(bootstrap_payload.get("adoption_kind"), "external")
        self.assertEqual(((bootstrap_payload.get("plan") or {}).get("model")), "any_live_position_auto_adopt")
        self.assertEqual(bootstrap_payload.get("recovery_method"), "listener_any_live_mt5_position")
        send_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
