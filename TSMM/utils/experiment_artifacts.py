"""Reproducible artifact bundles for exceptional Hypersearch experiments."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from utils.evaluator import save_best_model


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def best_r2_model(evaluation: Dict[str, Any]) -> Tuple[Optional[str], Optional[float]]:
    """Return the model with the highest finite primary-target R2 score."""
    best_name: Optional[str] = None
    best_score: Optional[float] = None
    for model_name, payload in (evaluation or {}).items():
        metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
        raw_score = metrics.get("R2") if isinstance(metrics, dict) else None
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        if best_score is None or score > best_score:
            best_name = str(model_name)
            best_score = score
    return best_name, best_score


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_file_fingerprint(path: Path) -> Dict[str, Any]:
    """Fingerprint a potentially large dataset without rereading all of it."""
    if not path.exists() or not path.is_file():
        return {"exists": False}
    size = path.stat().st_size
    sample_bytes = 1024 * 1024
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        if size > sample_bytes:
            stream.seek(max(size - sample_bytes, 0))
            digest.update(stream.read(sample_bytes))
    return {
        "exists": True,
        "size_bytes": size,
        "modified_at_utc": datetime.fromtimestamp(
            path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "sampled_sha256": digest.hexdigest(),
        "sample_definition": "first_and_last_1MiB_plus_file_size",
    }


def _git_revision(project_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def export_worthy_experiment_bundle(
    *,
    models: Dict[str, Any],
    evaluation: Dict[str, Any],
    future_forecasts: Dict[str, Any],
    config: Dict[str, Any],
    config_path: str,
    artifact_root: str,
    r2_threshold: float,
    logger: Any,
    dataframe: Any = None,
) -> Optional[Path]:
    """Persist a bundle only when the best primary-target R2 is above the gate."""
    model_name, r2_score = best_r2_model(evaluation)
    if model_name is None or r2_score is None or r2_score <= float(r2_threshold):
        return None

    root = Path(artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc)
    score_slug = f"{r2_score:.6f}".replace("-", "neg_").replace(".", "_")
    bundle_name = (
        f"{Path(config_path).stem}__{model_name}__r2_{score_slug}__"
        f"{created_at.strftime('%Y%m%dT%H%M%SZ')}"
    )
    staging = root / f".staging_{bundle_name}_{uuid.uuid4().hex}"
    destination = root / bundle_name
    staging.mkdir(parents=True, exist_ok=False)

    try:
        model_dir = staging / "model_files"
        selected_models = {model_name: models[model_name]}
        selected_evaluation = {model_name: evaluation[model_name]}
        save_parameters = inspect.signature(save_best_model).parameters
        if "config" in save_parameters:
            saved_path = save_best_model(
                selected_models,
                selected_evaluation,
                str(model_dir),
                logger,
                config=config,
            )
        else:
            saved_path = save_best_model(
                selected_models, selected_evaluation, str(model_dir), logger
            )
        saved_files = [path for path in model_dir.rglob("*") if path.is_file()]
        if not saved_path or not saved_files:
            raise RuntimeError(f"Model persistence failed for qualifying model {model_name}")

        with (staging / "experiment_config.yaml").open("w", encoding="utf-8") as stream:
            yaml.safe_dump(_jsonify(config), stream, sort_keys=False)
        (staging / "evaluation.json").write_text(
            json.dumps(_jsonify(selected_evaluation), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        (staging / "forecasts.json").write_text(
            json.dumps(
                _jsonify({model_name: (future_forecasts or {}).get(model_name)}),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        project_root = Path(__file__).resolve().parents[1]
        requirements_path = project_root / "requirements.txt"
        if requirements_path.exists():
            shutil.copy2(requirements_path, staging / "requirements.txt")

        configured_data_path = Path(str(config.get("data_path") or ""))
        if not configured_data_path.is_absolute():
            configured_data_path = project_root / configured_data_path
        data_manifest: Dict[str, Any] = {
            "configured_path": str(config.get("data_path") or ""),
            "resolved_path": str(configured_data_path.resolve()),
            "file_fingerprint": _sampled_file_fingerprint(configured_data_path),
            "records_requested": config.get("records"),
        }
        if dataframe is not None:
            data_manifest.update(
                {
                    "rows_loaded_for_experiment": int(len(dataframe)),
                    "columns": [str(column) for column in dataframe.columns],
                    "first_index": str(dataframe.index[0]) if len(dataframe) else None,
                    "last_index": str(dataframe.index[-1]) if len(dataframe) else None,
                }
            )
        (staging / "data_manifest.json").write_text(
            json.dumps(_jsonify(data_manifest), indent=2, allow_nan=False),
            encoding="utf-8",
        )

        checksums = {
            str(path.relative_to(staging)).replace(os.sep, "/"): _sha256(path)
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "schema_version": 1,
            "created_at_utc": created_at.isoformat(),
            "experiment_id": Path(config_path).stem,
            "source_config_path": str(Path(config_path).resolve()),
            "qualification": {
                "metric": "R2",
                "model": model_name,
                "score": r2_score,
                "operator": ">",
                "threshold": float(r2_threshold),
            },
            "code": {
                "git_commit": _git_revision(project_root),
                "python": sys.version,
                "platform": platform.platform(),
            },
            "random_seed": config.get("random_seed", config.get("seed")),
            "checksums_sha256": checksums,
        }
        (staging / "manifest.json").write_text(
            json.dumps(_jsonify(manifest), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        staging.rename(destination)
        return destination
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
