import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.trading_job import (
    _agent_b_risk_adjustment,
    _apply_agent_b_risk_adjustment,
    _attempt_agent_b_close,
    _mirror_agent_b_position_action,
)


class _StubAdapter:
    def __init__(self, close_result, position_result):
        self._close_result = close_result
        self._position_result = position_result

    def close_position_by_ticket(self, ticket):
        return dict(self._close_result)

    def get_position_by_ticket(self, ticket):
        return dict(self._position_result)

    def get_position_close_outcome(self, ticket):
        return {"ok": True, "found": False, "ticket": int(ticket)}


class _StubMirrorAdapter:
    def __init__(self, modify_result=None):
        self._modify_result = modify_result or {"ok": True, "position": {"ticket": 22222, "sl": 100.5, "tp": 103.0}}

    def connect(self):
        return True, "connected"

    def shutdown(self):
        return None

    def modify_position_risk(self, ticket, stop_loss=None, take_profit=None):
        payload = dict(self._modify_result)
        payload.setdefault("ticket", int(ticket))
        payload.setdefault("stop_loss", stop_loss)
        payload.setdefault("take_profit", take_profit)
        return payload


class AgentBRiskManagementTests(unittest.TestCase):
    def test_failed_risk_adjustment_is_retried_but_successful_one_is_deduplicated(self):
        class _RiskAdapter:
            def __init__(self):
                self.results = [{"ok": False}, {"ok": True}]
                self.calls = 0

            def modify_position_risk(self, ticket, stop_loss=None, take_profit=None):
                self.calls += 1
                return self.results.pop(0)

        adapter = _RiskAdapter()
        state = {}
        adjustment = {"action": "attach_delayed_stop_loss", "stop_loss": 99.0, "take_profit": 103.0}

        first = _apply_agent_b_risk_adjustment(adapter, state, 123, adjustment)
        second = _apply_agent_b_risk_adjustment(adapter, state, 123, adjustment)
        third = _apply_agent_b_risk_adjustment(adapter, state, 123, adjustment)

        self.assertTrue(first["attempted"])
        self.assertTrue(second["attempted"])
        self.assertFalse(third["attempted"])
        self.assertEqual(adapter.calls, 2)

    @patch("utils.trading_job._notify", return_value={"channel": {"kind": "risk_update"}})
    @patch("utils.trading_job._save_job_state")
    @patch("utils.trading_job._load_state")
    @patch("utils.trading_job.load_trading_config")
    @patch("utils.trading_job.MT5Adapter")
    def test_mirror_agent_b_risk_update_applies_levels_to_peer_job(
        self,
        adapter_cls,
        load_cfg_mock,
        load_state_mock,
        save_state_mock,
        _notify_mock,
    ):
        adapter_cls.return_value = _StubMirrorAdapter(
            modify_result={
                "ok": True,
                "ticket": 22222,
                "stop_loss": 100.5,
                "take_profit": 103.0,
                "position": {"ticket": 22222, "symbol": "XAUUSD", "sl": 100.5, "tp": 103.0},
            }
        )
        load_cfg_mock.return_value = {"runtime": {"profile_label": "FTMO"}, "broker": {"mt5": {}}}
        load_state_mock.return_value = {
            "job_id": "job_FTMO_01",
            "stage": "agent_b",
            "status": "agent_b_running",
            "position": {"ticket": 22222, "symbol": "XAUUSD", "sl": 99.0, "tp": 102.0},
            "notifications": [],
        }

        result = _mirror_agent_b_position_action(
            action="risk_update",
            output_dir="reports",
            source_trading_cfg={
                "runtime": {"profile_label": "Pepperstone"},
                "account_mirror": {"enabled": True, "mirror_agent_b_risk_updates": True},
            },
            source_state={
                "mirror": {
                    "peer_profile": "FTMO",
                    "peer_job_id": "job_FTMO_01",
                    "peer_trading_config_path": "config/trading_agent_ftmo.yaml",
                }
            },
            source_job_id="job_Pepperstone_01",
            risk_adjustment={"action": "trail_stop_loss", "stop_loss": 100.5, "take_profit": 103.0},
        )

        self.assertTrue(result.get("ok"))
        saved_payload = save_state_mock.call_args.args[3]
        self.assertEqual(saved_payload["position"]["ticket"], 22222)
        self.assertEqual(saved_payload["last_risk_adjustment"]["mirrored_from"]["account"], "Pepperstone")
        self.assertEqual(saved_payload["last_risk_adjustment"]["result"]["stop_loss"], 100.5)

    @patch("utils.trading_job._notify", return_value={"channel": {"kind": "system_update"}})
    def test_failed_agent_b_close_keeps_job_active(self, _notify_mock):
        state = {
            "job_id": "job_test",
            "status": "agent_b_running",
            "position": {"ticket": 12345, "symbol": "XAUUSD", "price_open": 100.0},
            "notifications": [],
        }
        adapter = _StubAdapter(
            close_result={"ok": False, "message": "order_send failed retcode=10013", "retcode": 10013},
            position_result={"ok": True, "position": {"ticket": 12345, "symbol": "XAUUSD", "price_open": 100.0}},
        )

        closed = _attempt_agent_b_close(
            adapter=adapter,
            state=state,
            pos_ticket=12345,
            output_dir="reports/runtime",
            trading_cfg={},
            job_id="job_test",
            closed_reason="mode_b_consensus_close(sell,-0.492)",
            final_status="closed",
            failure_message="close failed",
        )

        self.assertFalse(closed)
        self.assertEqual(state["status"], "agent_b_running")
        self.assertEqual(state["pending_close_reason"], "mode_b_consensus_close(sell,-0.492)")
        self.assertEqual(state["pending_close_status"], "closed")
        self.assertNotIn("ended_at", state)
        self.assertEqual(state["last_close_failure"]["result"]["retcode"], 10013)
        self.assertEqual(len(state["notifications"]), 1)

    @patch("utils.trading_job._notify", return_value={"channel": {"kind": "system_update"}})
    @patch("utils.trading_job._mirror_agent_b_position_action", return_value={"ok": True, "action": "close", "peer_profile": "FTMO"})
    def test_successful_agent_b_close_records_mirror_result(self, mirror_mock, _notify_mock):
        state = {
            "job_id": "job_test",
            "status": "agent_b_running",
            "position": {"ticket": 12345, "symbol": "XAUUSD", "price_open": 100.0},
            "notifications": [],
            "mirror": {"peer_profile": "FTMO", "peer_job_id": "job_FTMO_01", "peer_trading_config_path": "config/trading_agent_ftmo.yaml"},
        }
        adapter = _StubAdapter(
            close_result={"ok": True, "ticket": 12345, "retcode": 10009},
            position_result={"ok": True, "position": None},
        )

        closed = _attempt_agent_b_close(
            adapter=adapter,
            state=state,
            pos_ticket=12345,
            output_dir="reports/runtime",
            trading_cfg={"account_mirror": {"enabled": True}, "runtime": {"profile_label": "Pepperstone"}},
            job_id="job_test",
            closed_reason="mode_b_consensus_close(sell,-0.492)",
            final_status="closed",
            failure_message="close failed",
        )

        self.assertTrue(closed)
        self.assertTrue(state["close_result"]["mirror_result"]["ok"])
        mirror_mock.assert_called_once()

    def test_aligned_profitable_buy_trails_stop_and_extends_tp(self):
        state = {
            "plan": {
                "decision": "buy",
                "entry": 100.0,
                "stop_loss": 99.0,
                "take_profit": 102.0,
            }
        }
        position = {
            "price_open": 100.0,
            "price_current": 101.5,
            "sl": 99.0,
            "tp": 102.0,
        }
        current_plan = {
            "position_side": "buy",
            "recommendation": "maintain_position",
            "consensus_score": 0.55,
            "close_threshold": 0.25,
        }
        trading_cfg = {
            "mode_b": {"manage_existing_positions": True},
            "risk": {
                "stop_loss_pct": 1.0,
                "take_profit_pct": 2.0,
                "trailing": {"enabled": True, "trail_pct_base": 0.5},
            },
        }

        out = _agent_b_risk_adjustment(state, position, current_plan, trading_cfg)

        self.assertIsNotNone(out)
        self.assertIn("trail_stop_loss", out["action"])
        self.assertIn("extend_take_profit", out["action"])
        self.assertGreater(out["stop_loss"], 100.0)
        self.assertGreater(out["take_profit"], 102.0)

    def test_request_extension_position_still_trails_stop_when_consensus_aligned(self):
        state = {
            "plan": {
                "decision": "buy",
                "entry": 100.0,
                "stop_loss": 99.0,
                "take_profit": 102.0,
            },
            "pending_close_reason": "extension_not_approved_in_window",
        }
        position = {
            "price_open": 100.0,
            "price_current": 101.5,
            "sl": 99.0,
            "tp": 102.0,
        }
        current_plan = {
            "position_side": "buy",
            "recommendation": "request_extension",
            "consensus": "buy",
            "consensus_score": 0.55,
            "close_threshold": 0.25,
        }
        trading_cfg = {
            "mode_b": {"manage_existing_positions": True},
            "risk": {
                "stop_loss_pct": 1.0,
                "take_profit_pct": 2.0,
                "trailing": {"enabled": True, "trail_pct_base": 0.5},
            },
        }

        out = _agent_b_risk_adjustment(state, position, current_plan, trading_cfg)

        self.assertIsNotNone(out)
        self.assertIn("trail_stop_loss", out["action"])
        self.assertGreater(out["stop_loss"], 100.0)

    def test_defensive_buy_moves_stop_to_break_even(self):
        state = {
            "plan": {
                "decision": "buy",
                "entry": 100.0,
                "stop_loss": 99.0,
                "take_profit": 102.0,
            }
        }
        position = {
            "price_open": 100.0,
            "price_current": 100.8,
            "sl": 99.0,
            "tp": 102.0,
        }
        current_plan = {
            "position_side": "buy",
            "recommendation": "prepare_defensive_exit",
            "consensus_score": 0.18,
            "close_threshold": 0.25,
        }
        trading_cfg = {
            "mode_b": {"manage_existing_positions": True},
            "risk": {
                "stop_loss_pct": 1.0,
                "take_profit_pct": 2.0,
                "trailing": {"enabled": True, "trail_pct_base": 0.5},
            },
        }

        out = _agent_b_risk_adjustment(state, position, current_plan, trading_cfg)

        self.assertIsNotNone(out)
        self.assertEqual(out["action"], "tighten_stop_loss")
        self.assertEqual(out["stop_loss"], 100.0)
        self.assertEqual(out["take_profit"], 102.0)

    @patch("utils.trading_job._opposing_programmed_tp_target")
    def test_aligned_buy_tightens_tp_to_opposing_programmed_entry(self, opposing_target_mock):
        opposing_target_mock.return_value = {"entry": 101.25, "job_id": "job_sell_pending", "side": "sell"}
        state = {
            "job_id": "job_buy_live",
            "state_path": "reports/runtime/trading_jobs/job_buy_live/trading_job_state.json",
            "plan": {
                "decision": "buy",
                "entry": 100.0,
                "stop_loss": 99.0,
                "take_profit": 102.0,
            },
        }
        position = {
            "symbol": "XAUUSD",
            "price_open": 100.0,
            "price_current": 101.0,
            "sl": 99.0,
            "tp": 102.0,
        }
        current_plan = {
            "position_side": "buy",
            "recommendation": "maintain_position",
            "consensus_score": 0.55,
            "close_threshold": 0.25,
        }
        trading_cfg = {
            "mode_b": {"manage_existing_positions": True, "min_sltp_adjust_abs": 0.05},
            "risk": {
                "stop_loss_pct": 1.0,
                "take_profit_pct": 2.0,
                "trailing": {"enabled": True, "trail_pct_base": 0.5},
            },
        }

        out = _agent_b_risk_adjustment(state, position, current_plan, trading_cfg)

        self.assertIsNotNone(out)
        self.assertIn("tighten_take_profit", out["action"])
        self.assertEqual(out["take_profit"], 101.25)
        self.assertIn("job_sell_pending", out["rationale"])


if __name__ == "__main__":
    unittest.main()
