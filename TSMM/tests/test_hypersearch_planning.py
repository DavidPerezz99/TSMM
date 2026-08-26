import asyncio
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hypersearch import (  # noqa: E402
    BulkSearchEngine,
    build_nbeats_variants,
    build_sweep_plan,
    generate_smart_experiments,
    get_global_params,
    unique_experiments,
)
from utils.tracking import write_run_summary  # noqa: E402
from utils.experiment_artifacts import best_r2_model, export_worthy_experiment_bundle  # noqa: E402


class HypersearchPlanningTests(unittest.TestCase):
    def _yaml(self, relative_path):
        with (ROOT / relative_path).open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    def test_input_target_sets_are_paired_not_a_global_dimension(self):
        sweep = self._yaml("config/sweeps/sweep_30m_high_return.yaml")
        self.assertNotIn("input_target_sets", get_global_params(sweep))

    def test_csv_30m_sweep_has_exact_smart_count(self):
        base = self._yaml("config_templates/univariate.yaml")
        sweep = self._yaml("config/sweeps/sweep_30m_high_return.yaml")
        plan = build_sweep_plan(base, sweep)
        self.assertEqual(plan["raw_generated"], 312)
        self.assertEqual(plan["unique_experiments"], 312)
        self.assertEqual(plan["duplicates_removed"], 0)
        self.assertEqual(
            plan["per_model"],
            {"nbeats": 192, "svr": 96, "ulr": 24},
        )

    def test_generation_preserves_csv_source_and_does_not_mutate_base(self):
        base = self._yaml("config_templates/univariate.yaml")
        original = deepcopy(base)
        sweep = self._yaml("config/sweeps/sweep_30m_high_return.yaml")
        configs = list(
            unique_experiments(
                generate_smart_experiments(base, sweep, verbose=False)
            )
        )
        self.assertEqual(base, original)
        self.assertTrue(
            all(config["data_path"].endswith("xauusd_30m_2009.csv") for config in configs)
        )
        self.assertTrue(all("input_target_sets" not in config for config in configs))
        nbeats_hidden = {
            config["nbeats"]["hidden_size"]
            for config in configs
            if config["models_to_run"]["univariate"] == ["nbeats"]
        }
        self.assertEqual(nbeats_hidden, {128, 256})

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
                    variant["nbeats"]["stacks_config"][1]["num_blocks"],
                    variant["nbeats"]["stacks_config"][1]["num_harmonics"],
                )
                for variant in variants
            },
            {
                (3, 3, 8),
                (3, 3, 24),
                (3, 6, 8),
                (3, 6, 24),
                (6, 3, 8),
                (6, 3, 24),
                (6, 6, 8),
                (6, 6, 24),
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

    def test_materialized_file_count_matches_plan_and_cap_is_preflighted(self):
        base = self._yaml("config_templates/univariate.yaml")
        sweep = self._yaml("config/sweeps/sweep_30m_high_return.yaml")
        with tempfile.TemporaryDirectory() as temp_dir, patch("hypersearch.clear_cache"):
            root = Path(temp_dir)
            engine = BulkSearchEngine(
                base,
                sweep,
                root / "exact",
                asyncio.Semaphore(1),
                max_experiments=312,
            )
            paths = engine.materialize_configs()
            self.assertEqual(len(paths), 312)
            self.assertEqual(len(list((root / "exact").glob("cfg_*.yaml"))), 312)

            capped = BulkSearchEngine(
                base,
                sweep,
                root / "capped",
                asyncio.Semaphore(1),
                max_experiments=311,
            )
            with self.assertRaises(RuntimeError):
                capped.materialize_configs()
            self.assertEqual(list((root / "capped").glob("cfg_*.yaml")), [])

    def test_best_r2_model_uses_primary_metrics(self):
        name, score = best_r2_model(
            {
                "ulr": {"metrics": {"R2": 0.61}},
                "nbeats": {"metrics": {"R2": 0.73}},
                "broken": {"metrics": {"R2": "nan"}},
            }
        )
        self.assertEqual(name, "nbeats")
        self.assertEqual(score, 0.73)

    @patch("utils.experiment_artifacts.save_best_model")
    def test_worthy_bundle_is_strictly_gated_and_self_contained(self, save_model):
        def fake_save(models, evaluation, model_dir, logger):
            path = Path(model_dir) / "nbeats_20260825_010203.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"model")
            return str(path)

        save_model.side_effect = fake_save
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cfg_path = root / "cfg_00001.yaml"
            cfg_path.write_text("data_path: missing.csv\n", encoding="utf-8")
            common = {
                "models": {"nbeats": {"model": object()}},
                "future_forecasts": {"nbeats": {"future": [1.0]}},
                "config": {"data_path": "missing.csv", "records": 4000},
                "config_path": str(cfg_path),
                "artifact_root": str(root / "worthy"),
                "r2_threshold": 0.6,
                "logger": object(),
            }
            rejected = export_worthy_experiment_bundle(
                evaluation={"nbeats": {"metrics": {"MAE": 1.0, "R2": 0.6}}},
                **common,
            )
            self.assertIsNone(rejected)
            self.assertFalse((root / "worthy").exists())

            accepted = export_worthy_experiment_bundle(
                evaluation={"nbeats": {"metrics": {"MAE": 1.0, "R2": 0.600001}}},
                **common,
            )
            self.assertIsNotNone(accepted)
            self.assertTrue((accepted / "manifest.json").exists())
            self.assertTrue((accepted / "experiment_config.yaml").exists())
            self.assertTrue((accepted / "evaluation.json").exists())
            self.assertTrue((accepted / "forecasts.json").exists())
            self.assertEqual(len(list((accepted / "model_files").glob("*.joblib"))), 1)
            manifest = json.loads((accepted / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["qualification"]["operator"], ">")
            self.assertEqual(manifest["qualification"]["threshold"], 0.6)


if __name__ == "__main__":
    unittest.main()
