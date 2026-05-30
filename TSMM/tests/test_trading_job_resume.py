import sys
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.recover_runtime_after_reboot import _reboot_recovery_alert_message, _state_for_job
from utils.trading_job import _account_profile_label, _incoming_account_mirror_context, _job_registry_path, _launch_account_mirror_start, _load_state, _mirror_agent_a_entry_on_peer_preflight_failure, _propagate_mirror_job_action, _recover_agent_b_state_from_live_position, _state_path, _update_job_registry, resume_trading_job


class TradingJobResumeTests(unittest.TestCase):
    def test_account_profile_label_prefers_runtime_profile_label(self):
        self.assertEqual(_account_profile_label({"runtime": {"profile_label": "Pepperstone"}}), "Pepperstone")

    @patch.dict(
        "os.environ",
        {
            "TSMM_ACCOUNT_MIRROR_SOURCE_JOB_ID": "job_SRC_01",
            "TSMM_ACCOUNT_MIRROR_SOURCE_CONFIG_PATH": "config/trading_agent.yaml",
            "TSMM_ACCOUNT_MIRROR_SOURCE_PROFILE": "Pepperstone",
        },
        clear=False,
    )
    def test_incoming_account_mirror_context_reads_source_metadata(self):
        mirror = _incoming_account_mirror_context()

        self.assertEqual(mirror.get("role"), "mirror")
        self.assertEqual(mirror.get("peer_job_id"), "job_SRC_01")
        self.assertEqual(mirror.get("peer_trading_config_path"), "config/trading_agent.yaml")

    @patch("utils.trading_job.subprocess.Popen")
    @patch("utils.trading_job.load_trading_config")
    @patch("utils.trading_job._new_job_id", return_value="job_FTMO_01")
    @patch.dict("os.environ", {"TRADING_CONFIG_PATH": "config/trading_agent.yaml", "CONFIG_PATH": "config/config.yaml", "TSMM_RUNTIME_DIR": "reports/runtime"}, clear=False)
    def test_account_mirror_launch_spawns_peer_start_with_suppression(self, _job_id_mock, load_cfg_mock, popen_mock):
        popen_mock.return_value.pid = 4321
        load_cfg_mock.return_value = {"runtime": {"job_id_prefix": "FTMO", "profile_label": "FTMO"}}

        result = _launch_account_mirror_start(
            app_config={},
            trading_cfg={
                "runtime": {"profile_label": "Pepperstone"},
                "account_mirror": {
                    "enabled": True,
                    "peer_profile": "FTMO",
                    "peer_trading_config_path": "config/trading_agent_ftmo.yaml",
                },
            },
            output_dir="reports",
            logger=None,
            selected_model="nbeats",
            source_job_id="job_SRC_01",
            request_context={"effective_submission_mode": "market", "autonomous_trigger": "listener_manual"},
        )

        self.assertTrue(result.get("ok"))
        command = popen_mock.call_args.args[0]
        env = popen_mock.call_args.kwargs["env"]
        self.assertEqual(command[:6], [sys.executable, "app.py", "trading-job", "start", "--job-id", "job_FTMO_01"])
        self.assertIn("--submission-mode", command)
        self.assertIn("--plan-model", command)
        self.assertEqual(env["TSMM_ACCOUNT_MIRROR_SUPPRESS"], "1")
        self.assertEqual(env["TSMM_ACCOUNT_MIRROR_SOURCE_JOB_ID"], "job_SRC_01")
        self.assertEqual(env["TRADING_CONFIG_PATH"], "config/trading_agent_ftmo.yaml")
        self.assertNotIn("TSMM_RUNTIME_DIR", env)

    @patch("utils.trading_job.stop_trading_job")
    @patch("utils.trading_job.load_trading_config")
    def test_propagate_mirror_job_action_routes_stop_without_recurse(self, load_cfg_mock, stop_mock):
        load_cfg_mock.return_value = {"runtime": {"profile_label": "FTMO"}}
        stop_mock.return_value = {"ok": True, "job_ids": ["job_FTMO_01"]}

        result = _propagate_mirror_job_action(
            "stop",
            "reports",
            {
                "mirror": {
                    "peer_profile": "FTMO",
                    "peer_job_id": "job_FTMO_01",
                    "peer_trading_config_path": "config/trading_agent_ftmo.yaml",
                }
            },
        )

        self.assertTrue(result.get("ok"))
        stop_mock.assert_called_once_with("reports", {"runtime": {"profile_label": "FTMO"}}, job_id="job_FTMO_01", propagate_mirror=False)

    @patch("utils.trading_job._notify", return_value={"channel": {"kind": "order_filled"}})
    @patch("utils.trading_job.MT5Adapter")
    @patch("utils.trading_job.load_trading_config")
    def test_market_mirror_fallback_places_peer_order_after_data_sync_preflight_failure(self, load_cfg_mock, adapter_cls, _notify_mock):
        peer_cfg = {
            "runtime": {"profile_label": "FTMO"},
            "broker": {"mt5": {}},
            "execution": {"symbol": "XAUUSD", "default_volume": 0.01},
            "trading_job": {},
        }
        load_cfg_mock.return_value = peer_cfg

        adapter = adapter_cls.return_value
        adapter.connect.return_value = (True, "connected")
        adapter.list_pending_orders.return_value = {"ok": True, "orders": []}
        adapter.list_open_positions.return_value = {"ok": True, "positions": []}
        adapter.place_market_order.return_value = {
            "ok": True,
            "order_ticket": 7001,
            "retcode": 10009,
            "position": {
                "ticket": 9001,
                "symbol": "XAUUSD",
                "volume": 0.01,
                "price_open": 4500.1,
                "price_current": 4500.1,
                "sl": 4550.0,
                "tp": 4440.0,
                "type": 1,
                "side": "sell",
                "comment": "TSMM market orde",
                "magic": 7070001,
            },
        }
        adapter.find_position_by_order.return_value = {"ok": True, "position": None}
        adapter.shutdown.return_value = None

        source_cfg = {
            "runtime": {"profile_label": "Pepperstone"},
            "broker": {"mt5": {}},
            "execution": {"symbol": "XAUUSD", "default_volume": 0.01},
        }
        source_state = {
            "job_id": "job_SRC_01",
            "order_submission_mode": "market",
            "plan": {
                "decision": "sell",
                "entry": 4500.1,
                "stop_loss": 4550.0,
                "take_profit": 4440.0,
                "volume": 0.01,
            },
            "mirror": {
                "ok": True,
                "peer_profile": "FTMO",
                "peer_job_id": "job_FTMO_01",
                "peer_trading_config_path": "config/trading_agent_ftmo.yaml",
                "source_profile": "Pepperstone",
                "source_job_id": "job_SRC_01",
            },
        }

        with TemporaryDirectory() as tmpdir:
            peer_state_path = Path(_state_path(tmpdir, peer_cfg, "job_FTMO_01"))
            peer_state_path.parent.mkdir(parents=True, exist_ok=True)
            peer_state_path.write_text(
                json.dumps(
                    {
                        "job_id": "job_FTMO_01",
                        "status": "failed",
                        "stage": "preflight",
                        "mode": "mode_a",
                        "closed_reason": "data_sync_failed:2026-05-27 01:51:00",
                    }
                ),
                encoding="utf-8",
            )

            result = _mirror_agent_a_entry_on_peer_preflight_failure(
                app_config={"symbol": "XAUUSD"},
                source_trading_cfg=source_cfg,
                output_dir=tmpdir,
                source_state=source_state,
            )

            self.assertTrue(result.get("ok"))
            updated_peer_state = _load_state(str(peer_state_path))
            self.assertEqual(updated_peer_state.get("status"), "agent_b_running")
            self.assertEqual(updated_peer_state.get("stage"), "agent_b")
            self.assertEqual((((updated_peer_state.get("position") or {}).get("ticket"))), 9001)
            self.assertEqual((((updated_peer_state.get("order") or {}).get("order_ticket"))), 7001)

    @patch("utils.trading_job.MT5Adapter")
    @patch("utils.trading_job.load_trading_config")
    def test_market_mirror_fallback_skips_when_peer_failure_is_not_data_sync(self, load_cfg_mock, adapter_cls):
        peer_cfg = {
            "runtime": {"profile_label": "FTMO"},
            "broker": {"mt5": {}},
            "execution": {"symbol": "XAUUSD", "default_volume": 0.01},
            "trading_job": {},
        }
        load_cfg_mock.return_value = peer_cfg

        source_cfg = {
            "runtime": {"profile_label": "Pepperstone"},
            "broker": {"mt5": {}},
            "execution": {"symbol": "XAUUSD", "default_volume": 0.01},
        }
        source_state = {
            "job_id": "job_SRC_01",
            "order_submission_mode": "market",
            "plan": {"decision": "sell", "entry": 4500.1, "volume": 0.01},
            "mirror": {
                "ok": True,
                "peer_profile": "FTMO",
                "peer_job_id": "job_FTMO_01",
                "peer_trading_config_path": "config/trading_agent_ftmo.yaml",
                "source_profile": "Pepperstone",
                "source_job_id": "job_SRC_01",
            },
        }

        with TemporaryDirectory() as tmpdir:
            peer_state_path = Path(_state_path(tmpdir, peer_cfg, "job_FTMO_01"))
            peer_state_path.parent.mkdir(parents=True, exist_ok=True)
            peer_state_path.write_text(
                json.dumps(
                    {
                        "job_id": "job_FTMO_01",
                        "status": "failed",
                        "stage": "preflight",
                        "mode": "mode_a",
                        "closed_reason": "missing_token_env:TIINGO_API_TOKEN",
                    }
                ),
                encoding="utf-8",
            )

            result = _mirror_agent_a_entry_on_peer_preflight_failure(
                app_config={"symbol": "XAUUSD"},
                source_trading_cfg=source_cfg,
                output_dir=tmpdir,
                source_state=source_state,
            )

            self.assertFalse(result.get("ok"))
            self.assertTrue(result.get("skipped"))
            self.assertEqual(result.get("reason"), "peer_not_data_sync_preflight_failure")
            adapter_cls.assert_not_called()

    def test_runtime_scope_redirects_registry_and_state_alias(self):
        trading_cfg = {"runtime": {"namespace": "ftmo"}, "trading_job": {}}

        self.assertEqual(_job_registry_path("reports", trading_cfg).as_posix(), "reports/runtime/ftmo/trading_job_registry.json")
        self.assertEqual(Path(_state_path("reports", trading_cfg)).as_posix(), "reports/runtime/ftmo/trading_job_state.json")

    @patch("utils.trading_job._notify", return_value={"channel": {"kind": "position_recovered"}})
    @patch("utils.trading_job.MT5Adapter")
    def test_recover_agent_b_state_prefers_explicit_position_ticket_before_plan_match(self, adapter_cls, _notify_mock):
        adapter = adapter_cls.return_value
        adapter.connect.return_value = (True, "connected")
        adapter.get_position_by_ticket.return_value = {
            "ok": True,
            "position": {"ticket": 309849139, "symbol": "XAUUSD", "volume": 0.01, "price_open": 4519.38, "sl": 4483.22, "tp": 4591.69},
        }
        adapter.find_position_by_order.return_value = {"ok": True, "position": None}
        adapter.find_live_position_by_plan.return_value = {
            "ok": True,
            "position": {"ticket": 309848911, "symbol": "XAUUSD", "volume": 0.01, "price_open": 4519.38, "sl": 4483.22, "tp": 4591.69},
        }
        adapter.shutdown.return_value = None

        state = {
            "job_id": "job_mt5_orphan_pos_309849139",
            "status": "agent_b_running",
            "stage": "agent_b",
            "position": {"ticket": 309849139},
            "plan": {"decision": "buy", "symbol": "XAUUSD", "volume": 0.01, "entry": 4519.38, "stop_loss": 4483.22, "take_profit": 4591.69},
        }

        out = _recover_agent_b_state_from_live_position(state, {"broker": {"mt5": {}}, "execution": {"symbol": "XAUUSD"}}, "reports")

        self.assertTrue(out.get("ok"))
        self.assertEqual((out.get("state") or {}).get("recovery_method"), "position_ticket")
        self.assertEqual((((out.get("state") or {}).get("position") or {}).get("ticket")), 309849139)
        adapter.get_position_by_ticket.assert_called_once_with(309849139)
        adapter.find_live_position_by_plan.assert_not_called()

    def test_reboot_recovery_alert_message_summarizes_actions(self):
        message = _reboot_recovery_alert_message(
            {
                "checked_at": "2026-05-22 13:21:44",
                "enabled": True,
                "dry_run": False,
                "actions": [
                    {"kind": "endpoint", "status": "started", "pid": 111},
                    {"kind": "trading_resume", "status": "started", "job_id": "job_live", "pid": 222},
                ],
            },
            {"runtime": {"profile_label": "FTMO"}},
        )

        self.assertIn("[FTMO] TSMM reboot recovery completed", message)
        self.assertIn("endpoint=started pid=111", message)
        self.assertIn("trading_resume=started(job_live) pid=222", message)

    @patch("utils.trading_job._load_state")
    @patch("utils.trading_job._state_path")
    def test_explicit_resume_does_not_fallback_to_legacy_state(self, state_path_mock, load_state_mock):
        state_path_mock.side_effect = ["job_state.json", "legacy_state.json"]
        load_state_mock.return_value = {}

        out = resume_trading_job(
            app_config={},
            trading_cfg={},
            output_dir="reports",
            logger=None,
            job_id="job_explicit",
        )

        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "No trading job state found to resume")
        self.assertEqual(load_state_mock.call_count, 1)

    @patch("scripts.recover_runtime_after_reboot._load_json")
    def test_recovery_ignores_mismatched_job_state(self, load_json_mock):
        load_json_mock.return_value = {
            "job_id": "job_other",
            "state_path": str(ROOT / "reports" / "runtime" / "trading_jobs" / "job_explicit" / "trading_job_state.json"),
            "status": "agent_b_running",
        }
        registry = {
            "jobs": {
                "job_explicit": {
                    "state_path": str(ROOT / "reports" / "runtime" / "trading_jobs" / "job_explicit" / "trading_job_state.json")
                }
            }
        }

        state = _state_for_job("job_explicit", registry)

        self.assertEqual(state, {})

    def test_programmed_agent_a_job_stays_active_in_registry(self):
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "runtime" / "trading_jobs" / "job_live" / "trading_job_state.json"
            payload = {
                "job_id": "job_live",
                "status": "agent_a_completed",
                "stage": "agent_a",
                "mode": "mode_a",
                "started_at": "2026-05-14 14:13:31",
                "runner_pid": 1234,
                "order_submission_mode": "programmed",
                "programmed_order_expiration_utc": "2026-05-14 21:13:31",
                "closed_reason": None,
            }

            _update_job_registry(tmpdir, {"trading_job": {}}, payload, str(state_path))
            registry = _load_state(str(Path(tmpdir) / "runtime" / "trading_job_registry.json"))

            self.assertEqual(registry.get("active_job_ids"), ["job_live"])

    def test_closed_agent_a_job_stays_out_of_active_registry(self):
        with TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "runtime" / "trading_jobs" / "job_closed" / "trading_job_state.json"
            payload = {
                "job_id": "job_closed",
                "status": "agent_a_completed",
                "stage": "agent_a",
                "mode": "mode_a",
                "started_at": "2026-05-14 14:13:31",
                "runner_pid": 0,
                "order_submission_mode": "market",
                "closed_reason": "autonomous_followup_filtered",
                "ended_at": "2026-05-14 14:27:26",
            }

            _update_job_registry(tmpdir, {"trading_job": {}}, payload, str(state_path))
            registry = _load_state(str(Path(tmpdir) / "runtime" / "trading_job_registry.json"))

            self.assertEqual(registry.get("active_job_ids"), [])

    def test_killed_job_does_not_replace_active_latest_job(self):
        with TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "runtime" / "trading_job_registry.json"

            live_state_path = Path(tmpdir) / "runtime" / "trading_jobs" / "job_live" / "trading_job_state.json"
            live_payload = {
                "job_id": "job_live",
                "status": "agent_a_completed",
                "stage": "agent_a",
                "mode": "mode_a",
                "started_at": "2026-05-18 14:57:39",
                "updated_at": "2026-05-18 14:57:40",
                "runner_pid": 1076,
                "order_submission_mode": "programmed",
                "programmed_order_expiration_utc": "2026-05-18 21:57:40",
                "closed_reason": None,
            }
            _update_job_registry(tmpdir, {"trading_job": {}}, live_payload, str(live_state_path))

            killed_state_path = Path(tmpdir) / "runtime" / "trading_jobs" / "job_killed" / "trading_job_state.json"
            killed_payload = {
                "job_id": "job_killed",
                "status": "killed",
                "stage": "agent_a",
                "mode": "mode_a",
                "started_at": "2026-05-16 12:44:47",
                "updated_at": "2026-05-18 16:41:54",
                "runner_pid": 0,
                "order_submission_mode": "programmed",
                "programmed_order_expiration_utc": "2026-05-16 19:44:51",
                "closed_reason": "manual_kill",
                "ended_at": "2026-05-18 16:41:54",
            }
            _update_job_registry(tmpdir, {"trading_job": {}}, killed_payload, str(killed_state_path))

            registry = _load_state(str(registry_path))

            self.assertEqual(registry.get("active_job_ids"), ["job_live"])
            self.assertEqual(registry.get("latest_job_id"), "job_live")


if __name__ == "__main__":
    unittest.main()