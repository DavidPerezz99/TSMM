import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.operation_feedback_store import (
    log_agent_b_sample_feedback,
    log_notification_feedback,
    log_state_transition_feedback,
    resolve_operation_feedback_root,
)


class OperationFeedbackStoreTests(unittest.TestCase):
    def test_state_transition_writes_daily_and_by_job_logs(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = str(Path(tmpdir) / "reports")
            trading_cfg = {
                "runtime": {"root_dir": str(Path(output_dir) / "runtime")},
                "operation_feedback": {
                    "enabled": True,
                    "write_by_job_logs": True,
                },
            }
            current_state = {
                "job_id": "job_TEST_01",
                "status": "agent_a_completed",
                "stage": "agent_a",
                "started_at": "2026-05-29 00:00:00",
                "plan": {
                    "model": "nbeats",
                    "decision": "buy",
                    "confidence": 0.71,
                    "cm_accuracy": 0.63,
                    "success_probability": 0.66,
                },
                "mode_a": {"backtest": {"win_rate": 0.57, "n_trades": 12}},
            }

            out = log_state_transition_feedback(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                previous_state={},
                current_state=current_state,
            )

            self.assertTrue(out.get("ok"), out)

            feedback_root = resolve_operation_feedback_root(output_dir, trading_cfg)
            daily_files = list(feedback_root.glob("daily/**/operations_*.jsonl"))
            self.assertTrue(daily_files)

            with daily_files[0].open("r", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            self.assertTrue(rows)
            self.assertEqual(rows[0].get("job_id"), "job_TEST_01")
            self.assertEqual(rows[0].get("source"), "state_transition")
            self.assertIn(rows[0].get("event_kind"), {"state_initialized", "state_status_changed", "state_updated"})

            by_job_file = feedback_root / "by_job" / "job_TEST_01.jsonl"
            self.assertTrue(by_job_file.exists())

    def test_state_transition_skips_when_signature_unchanged(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = str(Path(tmpdir) / "reports")
            trading_cfg = {
                "runtime": {"root_dir": str(Path(output_dir) / "runtime")},
                "operation_feedback": {"enabled": True},
            }
            state = {
                "job_id": "job_TEST_02",
                "status": "agent_b_running",
                "stage": "agent_b",
                "plan": {"model": "nbeats", "decision": "sell"},
                "agent_b_plan": {"recommendation": "hold", "consensus": "hold", "should_close": False},
            }

            first = log_state_transition_feedback(output_dir, trading_cfg, {}, state)
            second = log_state_transition_feedback(output_dir, trading_cfg, state, dict(state))

            self.assertTrue(first.get("ok"))
            self.assertTrue(second.get("skipped"))
            self.assertEqual(second.get("reason"), "no_significant_state_change")

    def test_notification_and_agent_b_sample_logs_include_expected_fields(self):
        with TemporaryDirectory() as tmpdir:
            output_dir = str(Path(tmpdir) / "reports")
            trading_cfg = {
                "runtime": {"root_dir": str(Path(output_dir) / "runtime")},
                "operation_feedback": {"enabled": True},
            }
            state = {
                "job_id": "job_TEST_03",
                "status": "agent_b_running",
                "stage": "agent_b",
                "plan": {"model": "ulr", "decision": "buy", "confidence": 0.62},
                "order_submission_mode": "market",
                "started_at": "2026-05-29 00:00:00",
            }
            signal = {
                "consensus": "buy",
                "consensus_score": 0.44,
                "timeframes": {
                    "10m": {"signal": 1, "confidence": 0.61, "model": "nbeats"},
                },
            }
            current_plan = {"recommendation": "hold", "should_close": False}

            notify_out = log_notification_feedback(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                channel="agent_a",
                kind="order_failed",
                message="order failed in test",
                metadata={"reason": "test"},
                state={"job_id": "job_TEST_03", "status": "failed", "stage": "agent_a"},
                job_id="job_TEST_03",
            )
            sample_out = log_agent_b_sample_feedback(
                output_dir=output_dir,
                trading_cfg=trading_cfg,
                state=state,
                signals=signal,
                current_plan=current_plan,
            )

            self.assertTrue(notify_out.get("ok"))
            self.assertTrue(sample_out.get("ok"))

            feedback_root = resolve_operation_feedback_root(output_dir, trading_cfg)
            daily_files = sorted(feedback_root.glob("daily/**/operations_*.jsonl"))
            self.assertTrue(daily_files)
            with daily_files[-1].open("r", encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]

            event_kinds = {str(e.get("event_kind") or "") for e in events}
            self.assertIn("notify_agent_a_order_failed", event_kinds)
            self.assertIn("mode_b_assessment_sample", event_kinds)


if __name__ == "__main__":
    unittest.main()
