import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.operation_feedback_weekly_summary import _drift_summary, _patterns_summary, _terminal_operation_events
from scripts.analyze_operation_feedback import build_job_rows, summarize_events


class OperationFeedbackSummaryTests(unittest.TestCase):
    def test_terminal_events_keep_latest_terminal_snapshot_when_later_event_is_running(self):
        events = [
            {"job_id": "job_1", "timestamp_utc": "2026-05-22 00:00:00", "status": "closed", "outcome_label": "good"},
            {"job_id": "job_1", "timestamp_utc": "2026-05-22 00:01:00", "status": "agent_b_running", "outcome_label": "pending"},
        ]

        terminal = _terminal_operation_events(events)

        self.assertEqual(len(terminal), 1)
        self.assertEqual(terminal[0]["status"], "closed")

    def test_full_history_summary_counts_terminal_outcome_independently_of_latest_state(self):
        events = [
            {"job_id": "job_1", "timestamp_utc": "2026-05-22 00:00:00", "status": "closed", "outcome_label": "good"},
            {"job_id": "job_1", "timestamp_utc": "2026-05-22 00:01:00", "status": "agent_b_running", "outcome_label": "pending"},
        ]

        summary = summarize_events(events, build_job_rows(events))

        self.assertEqual(summary["terminal_operation_count"], 1)
        self.assertEqual(summary["terminal_outcome_counts"], [("good", 1)])

    def test_patterns_summary_returns_good_and_bad_rankings(self):
        events = [
            {
                "job_id": "job_1",
                "outcome_label": "good",
                "status": "closed",
                "close_reason_family": "mode_b_consensus_close",
                "performance_snapshot": {"decision": "buy", "model_name": "nbeats"},
                "execution_snapshot": {"order_submission_mode": "programmed", "close_outcome_profit": 12.3},
                "metadata": {"fallback_attempts": 0},
            },
            {
                "job_id": "job_2",
                "outcome_label": "bad",
                "status": "closed",
                "close_reason_family": "mode_b_consensus_close",
                "performance_snapshot": {"decision": "buy", "model_name": "nbeats"},
                "execution_snapshot": {"order_submission_mode": "programmed", "close_outcome_profit": -4.0},
                "metadata": {"fallback_attempts": 0},
            },
            {
                "job_id": "job_3",
                "outcome_label": "bad",
                "status": "failed",
                "close_reason_family": "order_place_failed",
                "performance_snapshot": {"decision": "sell", "model_name": "ulr"},
                "execution_snapshot": {"order_submission_mode": "market", "close_outcome_profit": -1.5},
                "metadata": {"fallback_attempts": 1},
            },
        ]

        good_patterns, bad_patterns = _patterns_summary(events, min_count=1)

        self.assertTrue(good_patterns)
        self.assertTrue(bad_patterns)
        self.assertIn("good_rate", good_patterns[0])
        self.assertIn("bad_rate", bad_patterns[0])

    def test_drift_summary_flags_metric_degradation(self):
        events = [
            {
                "timestamp_utc": "2026-05-22 00:00:00",
                "outcome_label": "good",
                "performance_snapshot": {
                    "model_name": "nbeats",
                    "confidence": 0.74,
                    "cm_accuracy": 0.71,
                    "success_probability": 0.70,
                    "input_fooling_risk": 0.21,
                },
            },
            {
                "timestamp_utc": "2026-05-23 00:00:00",
                "outcome_label": "good",
                "performance_snapshot": {
                    "model_name": "nbeats",
                    "confidence": 0.72,
                    "cm_accuracy": 0.69,
                    "success_probability": 0.68,
                    "input_fooling_risk": 0.22,
                },
            },
            {
                "timestamp_utc": "2026-05-27 00:00:00",
                "outcome_label": "bad",
                "performance_snapshot": {
                    "model_name": "nbeats",
                    "confidence": 0.60,
                    "cm_accuracy": 0.58,
                    "success_probability": 0.57,
                    "input_fooling_risk": 0.33,
                },
            },
            {
                "timestamp_utc": "2026-05-28 00:00:00",
                "outcome_label": "bad",
                "performance_snapshot": {
                    "model_name": "nbeats",
                    "confidence": 0.58,
                    "cm_accuracy": 0.55,
                    "success_probability": 0.54,
                    "input_fooling_risk": 0.36,
                },
            },
        ]

        drift_rows = _drift_summary(events)
        self.assertTrue(drift_rows)
        first = drift_rows[0]
        self.assertEqual(first.get("model_name"), "nbeats")
        self.assertTrue(first.get("drift_alert"))
        self.assertTrue(first.get("drift_reasons"))


if __name__ == "__main__":
    unittest.main()
