from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import joblib
import yaml

from utils.investing_agent import _discover_endpoint_specs
from utils.model_deployment import (
    activate_deployment,
    install_bundle,
    resolve_active_deployment,
    validate_bundle,
)


class SerializableModel:
    def predict(self, values):
        return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_bundle(root: Path, *, score: float = 0.71) -> Path:
    bundle = root / "cfg_00001__nbeats__r2_0_710000__20260829T010000Z"
    model_dir = bundle / "model_files"
    model_dir.mkdir(parents=True)
    config = {
        "timeframe": "10m",
        "data_timeframe_minutes": 10,
        "target_col": "HIGH",
        "n_steps": 12,
        "m_steps": 1,
        "horizon": 6,
        "input_features": ["HIGH", "y_diff"],
        "target_features": ["y_diff"],
    }
    (bundle / "experiment_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (bundle / "evaluation.json").write_text(
        json.dumps({"nbeats": {"metrics": {"R2": score, "MAE": 0.1}}}), encoding="utf-8"
    )
    (bundle / "data_manifest.json").write_text(
        json.dumps({"first_index": "2025-01-01 00:00:00", "last_index": "2026-08-14 23:50:00"}),
        encoding="utf-8",
    )
    model_path = model_dir / "nbeats_high_10m_20260829_010000.joblib"
    artifacts_path = model_dir / "nbeats_artifacts_high_10m_20260829_010000.joblib"
    joblib.dump(SerializableModel(), model_path)
    joblib.dump({"scaler_X": None, "scaler_y": None}, artifacts_path)
    checksums = {
        str(path.relative_to(bundle)).replace(os.sep, "/"): _sha256(path)
        for path in sorted(bundle.rglob("*")) if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "qualification": {"metric": "R2", "model": "nbeats", "score": score, "operator": ">", "threshold": 0.6},
        "checksums_sha256": checksums,
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


class ModelDeploymentTests(unittest.TestCase):
    def test_bundle_install_activation_and_endpoint_discovery_use_exact_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = _create_bundle(root)
            deployment_root = root / "deployments"
            active_path = deployment_root / "active.json"

            validated = validate_bundle(bundle, "10m_high")
            self.assertEqual(validated["model"], "nbeats")
            deployment = install_bundle(bundle, "10m_high", deployment_root=deployment_root)
            with patch.dict(os.environ, {"TSMM_ACTIVE_MODEL_MANIFEST": str(active_path)}):
                activated = activate_deployment(
                    deployment,
                    metrics={"holdout_r2": 0.71},
                    deployment_root=deployment_root,
                )
                resolved = resolve_active_deployment("10m_high", deployment_root=deployment_root)
                self.assertEqual(resolved["deployment_id"], activated["deployment_id"])
                self.assertTrue(Path(resolved["model_path"]).is_file())
                self.assertEqual(resolved["training_data_last_index"], "2026-08-14 23:50:00")

                legacy = root / "config" / "high10mResults" / "nbeats" / "top1_09999.yaml"
                legacy.parent.mkdir(parents=True)
                legacy.write_text(
                    yaml.safe_dump({"target_col": "HIGH", "n_steps": 99, "input_features": ["HIGH"]}),
                    encoding="utf-8",
                )
                specs = _discover_endpoint_specs(
                    {"10m": {"url": "http://localhost/predict/10m", "config_path": str(legacy)}},
                    config_root=str(root / "config"),
                )
                self.assertEqual(specs["10m"]["n_steps"], 12)
                self.assertEqual(Path(specs["10m"]["config_path"]), Path(resolved["config_path"]))
                self.assertEqual(Path(specs["10m"]["model_path"]), Path(resolved["model_path"]))

    def test_checksum_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = _create_bundle(Path(temp_dir))
            (bundle / "experiment_config.yaml").write_text("target_col: LOW\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validate_bundle(bundle, "10m_high", load_artifacts=False)

    def test_endpoint_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = _create_bundle(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "does not match endpoint family"):
                validate_bundle(bundle, "10m_low", load_artifacts=False)

    def test_zip_bundle_is_supported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bundle = _create_bundle(root)
            archive = root / "qualified_bundle.zip"
            with zipfile.ZipFile(archive, "w") as target:
                for path in bundle.rglob("*"):
                    if path.is_file():
                        target.write(path, arcname=str(Path(bundle.name) / path.relative_to(bundle)))
            deployment = install_bundle(
                archive,
                "10m_high",
                deployment_root=root / "deployments",
            )
            self.assertTrue(Path(deployment["model_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
