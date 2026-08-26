from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import yaml

from utils.investing_agent import (
    _discover_agent_a_enrichment_candidates,
    _discover_endpoint_specs,
)


def _write_model_config(path: Path, *, n_steps: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "n_steps": n_steps,
                "m_steps": 1,
                "horizon": 6,
                "input_features": ["HIGH", "y_diff"],
                "target_features": ["y_diff"],
                "target_col": "HIGH",
            }
        ),
        encoding="utf-8",
    )


class ModelEndpointSelectionTests(unittest.TestCase):
    def test_explicit_config_is_authoritative_over_higher_scored_runner_up(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            promoted = root / "high10mResults" / "nbeats" / "top1_06000.yaml"
            runner_up = root / "high10mResults" / "nbeats" / "top2_09999.yaml"
            _write_model_config(promoted)
            _write_model_config(runner_up, n_steps=72)

            specs = _discover_endpoint_specs(
                {
                    "10m": {
                        "url": "http://127.0.0.1:8000/predict/10m",
                        "model": "nbeats",
                        "config_path": str(promoted),
                    }
                },
                config_root=str(root),
            )

            self.assertEqual(Path(specs["10m"]["config_path"]), promoted)
            self.assertEqual(specs["10m"]["n_steps"], 6)

    def test_repository_promotions_are_selected_and_legacy_v1_is_ignored(self):
        project_root = Path(__file__).resolve().parents[1]
        trading_cfg = yaml.safe_load(
            (project_root / "config" / "trading_agent.yaml").read_text(encoding="utf-8")
        )

        candidates = {
            (item["timeframe"], item["family"]): Path(item["config_path"])
            for item in _discover_agent_a_enrichment_candidates(trading_cfg)
        }

        self.assertEqual(candidates[("7h", "low")].name, "top1_07400.yaml")
        self.assertEqual(candidates[("7h", "high")].name, "top1_07000.yaml")
        self.assertEqual(candidates[("10m", "high")].name, "top1_06000.yaml")
        for key in (("7h", "low"), ("7h", "high"), ("10m", "high")):
            self.assertNotIn("legacy", candidates[key].parts)


if __name__ == "__main__":
    unittest.main()
