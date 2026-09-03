import unittest
from datetime import datetime
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import yaml

from tools.experiment_session import (
    _best_available_candidate,
    _entry_source,
    _materialize_session,
    build_session_plan,
)
from tools.hypersearch import (
    build_nbeats_variants,
    build_sweep_plan,
    generate_smart_experiments,
    get_global_params,
    unique_experiments,
    SmartSearchEngine,
)
from models.multivariate_models import resolve_interpretable_stacks_config
from utils.tracking import write_run_summary
from utils.experiment_planning import (
    estimate_experiment_memory,
    estimate_max_records,
    next_local_deadline,
)
from utils.experiment_artifacts import best_r2_model, export_worthy_experiment_bundle


ROOT = Path(__file__).resolve().parents[1]


class HypersearchPlanningTests(unittest.TestCase):
    def _yaml(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def test_input_target_sets_are_not_a_global_dimension(self):
        sweep = self._yaml("config/sweeps/sweep_30m_high_return.yaml")
        self.assertNotIn("input_target_sets", get_global_params(sweep))

    def test_30m_sweep_has_expected_unique_count(self):
        base = self._yaml("config_templates/univariate.yaml")
        sweep = self._yaml("config/sweeps/sweep_30m_high_return.yaml")
        plan = build_sweep_plan(base, sweep, ram_limit_gb=20.0)
        self.assertEqual(plan["raw_generated"], 216)
        self.assertEqual(plan["unique_experiments"], 216)
        self.assertEqual(plan["duplicates_removed"], 0)
        self.assertEqual(plan["per_model"], {"nbeats": 192, "ulr": 24})
        self.assertEqual(set(plan["capacity_by_model"]), {"nbeats", "ulr"})

    def test_generated_sweep_uses_nested_nbeats_epochs(self):
        base = self._yaml("config_templates/univariate.yaml")
        sweep = self._yaml("config/sweeps/sweep_30m_high_return.yaml")
        experiments = unique_experiments(
            generate_smart_experiments(base, sweep, verbose=False)
        )
        nbeats = next(
            cfg
            for cfg in experiments
            if cfg["models_to_run"]["univariate"] == ["nbeats"]
        )
        self.assertEqual(nbeats["nbeats"]["epochs"], 60)
        self.assertEqual(nbeats["sql_source_mode"], "cache")
        self.assertEqual(
            nbeats["nbeats"]["stacks_config"][0]["hidden_size"],
            nbeats["nbeats"]["hidden_size"],
        )
        resolved = resolve_interpretable_stacks_config(
            nbeats["nbeats"]["stacks_config"], nbeats["nbeats"]["hidden_size"]
        )
        self.assertEqual(resolved[0]["hidden_size"], nbeats["nbeats"]["hidden_size"])

    def test_nbeats_direct_stacks_config_expands_inner_list_values(self):
        variants = build_nbeats_variants(
            {
                "nbeats.model_type": ["interpretable"],
                "nbeats.hidden_size": [128],
                "nbeats.learning_rate": [0.003],
                "nbeats.stacks_config": [[
                    {"type": "trend", "num_blocks": [3, 6], "degree": 3},
                    {"type": "seasonality", "num_blocks": [3, 6], "num_harmonics": [8, 24]},
                ]],
            }
        )
        self.assertEqual(len(variants), 8)
        self.assertEqual(
            {
                (
                    variant["nbeats"]["stacks_config"][0]["num_blocks"],
                    variant["nbeats"]["stacks_config"][0]["hidden_size"],
                    variant["nbeats"]["stacks_config"][1]["num_blocks"],
                    variant["nbeats"]["stacks_config"][1]["hidden_size"],
                    variant["nbeats"]["stacks_config"][1]["num_harmonics"],
                )
                for variant in variants
            },
            {
                (3, 128, 3, 128, 8),
                (3, 128, 3, 128, 24),
                (3, 128, 6, 128, 8),
                (3, 128, 6, 128, 24),
                (6, 128, 3, 128, 8),
                (6, 128, 3, 128, 24),
                (6, 128, 6, 128, 8),
                (6, 128, 6, 128, 24),
            },
        )

    def test_write_run_summary_marks_nested_metrics_as_success(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            summary_path = write_run_summary(
                config_path="cfg_00001.yaml",
                metrics={
                    "nbeats": {
                        "primary": {"MAE": 1.0, "RMSE": 2.0, "R2": 0.4, "MAPE": 3.0}
                    }
                },
                summary_dir=temp_dir,
            )
            payload = json.loads(Path(summary_path).read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "SUCCESS")

    def test_memory_estimate_is_monotonic_and_record_limit_is_guarded(self):
        config = {
            "records": 10_000,
            "n_steps": 96,
            "m_steps": 1,
            "input_features": ["HIGH", "y_diff", "Price_return"],
            "target_features": ["y_diff", "Price_return"],
            "rolling_windows": [2, 7, 30, 60],
            "models_to_run": {"univariate": ["nbeats"], "multivariate": []},
            "nbeats": {"model_type": "interpretable", "hidden_size": 128},
        }
        small = estimate_experiment_memory(config, records=10_000)
        large = estimate_experiment_memory(config, records=100_000)
        self.assertGreater(large["estimated_peak_gb"], small["estimated_peak_gb"])
        ceiling = estimate_max_records(config, ram_limit_gb=20.0)
        self.assertLessEqual(
            estimate_experiment_memory(config, records=ceiling)["estimated_peak_gb"],
            20.0,
        )

    def test_next_deadline_rolls_to_next_day_after_cutoff(self):
        now = datetime(2026, 8, 16, 22, 0)
        deadline = next_local_deadline(now, "05:00")
        self.assertEqual(deadline, datetime(2026, 8, 17, 5, 0))

    def test_committed_nightly_session_is_ordered_and_bounded(self):
        session = self._yaml("config/experiment_sessions/xauusd_nightly.yaml")
        plan = build_session_plan(session)
        self.assertTrue(plan["manual_start_only"])
        self.assertEqual(plan["deadline_local"], "05:00")
        self.assertEqual(plan["total_experiments"], 8748)
        self.assertEqual(plan["max_records_per_experiment"], 10000)
        self.assertEqual(plan["max_experiments_per_endpoint"], 400)
        self.assertEqual(len(plan["entries"]), 28)
        self.assertEqual(
            [entry["name"] for entry in plan["entries"][:4]],
            [
                "xauusd_10m_open",
                "xauusd_10m_high",
                "xauusd_10m_low",
                "xauusd_10m_close",
            ],
        )
        self.assertEqual(
            {entry["target_col"] for entry in plan["entries"]},
            {"OPEN", "HIGH", "LOW", "CLOSE"},
        )
        self.assertEqual(
            {entry["data_timeframe_minutes"] for entry in plan["entries"]},
            {10, 30, 60, 180, 420, 720, 1440},
        )

    def test_session_target_profiles_keep_target_and_features_paired(self):
        session = self._yaml("config/experiment_sessions/xauusd_nightly.yaml")
        close_entry = next(
            entry for entry in session["experiments"] if entry["name"] == "xauusd_7h_close"
        )
        _, _, base, sweep = _entry_source(close_entry, session)
        configs = list(unique_experiments(generate_smart_experiments(base, sweep, verbose=False)))
        self.assertEqual(len(configs), 324)
        self.assertTrue(all(config["target_col"] == "CLOSE" for config in configs))
        self.assertTrue(
            all(config["input_features"][0] == "CLOSE" for config in configs)
        )

    def test_us500_matrix_covers_every_ohlc_endpoint_with_bounded_searches(self):
        session = self._yaml("config/experiment_sessions/us500_nightly.yaml")
        plan = build_session_plan(session)
        self.assertEqual(len(plan["entries"]), 28)
        self.assertTrue(all(entry["plan"]["unique_experiments"] <= 400 for entry in plan["entries"]))
        self.assertEqual({entry["target_col"] for entry in plan["entries"]}, {"OPEN", "HIGH", "LOW", "CLOSE"})

    def test_best_available_candidate_uses_reliable_rolling_r2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            configs = root / "configs"
            summaries = root / "summaries"
            configs.mkdir()
            summaries.mkdir()
            for index, score in enumerate((0.12, 0.44), 1):
                config_path = configs / f"cfg_{index:05d}.yaml"
                config_path.write_text("records: 1000\n", encoding="utf-8")
                (summaries / f"cfg_{index:05d}__summary.json").write_text(
                    json.dumps(
                        {
                            "config_path": str(config_path),
                            "status": "SUCCESS",
                            "metric": {
                                "nbeats": {
                                    "R2": score,
                                    "sample_count": 120,
                                    "evaluation_protocol": "rolling_origin_one_step",
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            selected = _best_available_candidate(
                {"configs_dir": str(configs), "summaries_dir": str(summaries)}
            )
            self.assertAlmostEqual(selected["r2"], 0.44)
            self.assertEqual(Path(selected["config_path"]).name, "cfg_00002.yaml")

    def test_session_materialization_is_stable_for_resume(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            base_path = root / "base.yaml"
            sweep_path = root / "sweep.yaml"
            session_path = root / "session.yaml"
            base_path.write_text(
                yaml.safe_dump(
                    {
                        "records": 100,
                        "n_steps": 5,
                        "m_steps": 1,
                        "rolling_windows": [2],
                        "input_features": ["HIGH", "y_diff"],
                        "target_features": ["y_diff"],
                        "target_col": "HIGH",
                    }
                ),
                encoding="utf-8",
            )
            sweep_path.write_text(
                yaml.safe_dump(
                    {
                        "smart_generation": True,
                        "models_to_run": {"univariate": ["ulr"], "multivariate": []},
                        "records": [100],
                        "n_steps": [5],
                        "input_target_sets": [
                            {"input_features": ["HIGH", "y_diff"], "target_features": ["y_diff"]}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            session = {
                "session_name": "resume_test",
                "manual_start_only": True,
                "output_root": str(root / "output"),
                "minimum_experiments_before_early_stop_per_endpoint": 40,
                "resources": {"max_process_ram_gb": 20, "max_estimated_experiment_minutes": 90},
                "experiments": [
                    {
                        "name": "first",
                        "base_config": str(base_path),
                        "sweep_config": str(sweep_path),
                        "max_experiments": 2,
                    }
                ],
            }
            session_path.write_text(yaml.safe_dump(session), encoding="utf-8")
            plan = build_session_plan(session)
            first = _materialize_session(session_path, session, plan)
            second = _materialize_session(session_path, session, plan)
            self.assertEqual(first, second)
            self.assertEqual(first["total_experiments"], 1)
            self.assertEqual(
                first["minimum_experiments_before_early_stop_per_endpoint"],
                40,
            )
            configs = list(Path(first["entries"][0]["configs_dir"]).glob("cfg_*.yaml"))
            self.assertEqual(len(configs), 1)

    def test_smart_search_ranks_r2_highest_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_paths = []
            for index, score in enumerate([0.1, 0.8, -0.2], 1):
                config_path = root / f"cfg_{index:05d}.yaml"
                config_path.write_text("records: 100\n", encoding="utf-8")
                config_paths.append(config_path)
                (root / f"cfg_{index:05d}__summary.json").write_text(
                    json.dumps(
                        {
                            "config_path": str(config_path),
                            "status": "SUCCESS",
                            "metric": {"nbeats": {"R2": score, "MAPE": 10 - score}},
                        }
                    ),
                    encoding="utf-8",
                )
            engine = SmartSearchEngine.__new__(SmartSearchEngine)
            engine.archive_dir = root
            engine.top_n = 2
            engine.metric_name = "R2"
            engine.direction = "max"
            self.assertEqual(engine._best_summaries(), [config_paths[1], config_paths[0]])

    def test_best_r2_model_uses_primary_metrics(self):
        name, score = best_r2_model({
            "ulr": {"metrics": {"R2": 0.61}},
            "nbeats": {"metrics": {"R2": 0.73}},
            "broken": {"metrics": {"R2": "nan"}},
        })
        self.assertEqual((name, score), ("nbeats", 0.73))

    @patch("utils.experiment_artifacts.save_best_model")
    def test_worthy_bundle_is_strictly_gated(self, save_model):
        def fake_save(models, evaluation, model_dir, logger, config=None):
            path = Path(model_dir) / "nbeats_high_10m_20260825_010203.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")
            return str(path)

        save_model.side_effect = fake_save
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg_path = root / "cfg_00001.yaml"
            cfg_path.write_text("data_path: missing.sqlite\n", encoding="utf-8")
            common = {
                "models": {"nbeats": {"model": object()}},
                "future_forecasts": {"nbeats": {"future": [1.0]}},
                "config": {"data_path": "missing.sqlite", "records": 4000},
                "config_path": str(cfg_path), "artifact_root": str(root / "worthy"),
                "r2_threshold": 0.6, "logger": object(),
            }
            self.assertIsNone(export_worthy_experiment_bundle(
                evaluation={"nbeats": {"metrics": {"MAE": 1.0, "R2": 0.6}}}, **common
            ))
            accepted = export_worthy_experiment_bundle(
                evaluation={"nbeats": {"metrics": {"MAE": 1.0, "R2": 0.600001}}}, **common
            )
            self.assertTrue((accepted / "manifest.json").exists())
            self.assertTrue((accepted / "experiment_config.yaml").exists())
            manifest = json.loads((accepted / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["qualification"]["operator"], ">")

            fallback = export_worthy_experiment_bundle(
                evaluation={
                    "nbeats": {
                        "metrics": {
                            "MAE": 1.0,
                            "R2": 0.41,
                            "sample_count": 120,
                            "evaluation_protocol": "rolling_origin_one_step",
                        }
                    }
                },
                selection_tier="fallback_best_available",
                force_export=True,
                **common,
            )
            fallback_manifest = json.loads(
                (fallback / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertFalse(fallback_manifest["qualification"]["qualified_candidate"])
            self.assertEqual(
                fallback_manifest["qualification"]["selection_tier"],
                "fallback_best_available",
            )


if __name__ == "__main__":
    unittest.main()
