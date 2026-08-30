"""Validated, versioned deployment of reproducible forecasting bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPLOYMENT_ROOT = PROJECT_ROOT / "model_files" / "deployments"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def active_manifest_path(deployment_root: Optional[Path] = None) -> Path:
    override = str(os.environ.get("TSMM_ACTIVE_MODEL_MANIFEST") or "").strip()
    if override:
        path = Path(override)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    return (deployment_root or DEFAULT_DEPLOYMENT_ROOT).resolve() / "active.json"


def _parse_endpoint(endpoint: str) -> Tuple[str, str]:
    raw = str(endpoint or "").strip().lower()
    if "_" not in raw:
        raise ValueError("Endpoint must use the <timeframe>_<family> form, for example 10m_high")
    timeframe, family = raw.rsplit("_", 1)
    if not timeframe or family not in {"open", "high", "low", "close"}:
        raise ValueError(f"Invalid forecasting endpoint: {endpoint}")
    return timeframe, family


def _timeframe_from_config(config: Dict[str, Any]) -> str:
    explicit = str(config.get("timeframe") or "").strip().lower()
    if explicit:
        return explicit
    try:
        minutes = int(config.get("data_timeframe_minutes") or 0)
    except (TypeError, ValueError):
        minutes = 0
    if minutes <= 0:
        return ""
    if minutes % 10080 == 0:
        return f"{minutes // 10080}w"
    if minutes % 1440 == 0:
        return f"{minutes // 1440 * 24}h"
    if minutes % 60 == 0:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def _safe_extract_zip(archive: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive, "r") as source:
        destination_root = destination.resolve()
        for member in source.infolist():
            candidate = (destination / member.filename).resolve()
            try:
                candidate.relative_to(destination_root)
            except ValueError as exc:
                raise ValueError(f"Unsafe path in bundle archive: {member.filename}") from exc
        source.extractall(destination)
    roots = [path for path in destination.iterdir()]
    if len(roots) == 1 and roots[0].is_dir() and not (destination / "manifest.json").exists():
        return roots[0]
    return destination


def _bundle_root(bundle: Path, scratch: Path) -> Path:
    source = bundle.resolve()
    if source.is_dir():
        return source
    if source.is_file() and source.suffix.lower() == ".zip":
        return _safe_extract_zip(source, scratch)
    raise FileNotFoundError(f"Bundle must be a directory or .zip archive: {source}")


def validate_bundle(bundle: Path, endpoint: str, *, load_artifacts: bool = True) -> Dict[str, Any]:
    """Validate checksums, endpoint identity, and serialized model compatibility."""
    timeframe, family = _parse_endpoint(endpoint)
    with tempfile.TemporaryDirectory(prefix="tsmm_bundle_validate_") as temp_dir:
        root = _bundle_root(Path(bundle), Path(temp_dir))
        manifest_path = root / "manifest.json"
        config_path = root / "experiment_config.yaml"
        evaluation_path = root / "evaluation.json"
        if not manifest_path.exists() or not config_path.exists() or not evaluation_path.exists():
            raise ValueError("Bundle is missing manifest.json, experiment_config.yaml, or evaluation.json")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        data_manifest_path = root / "data_manifest.json"
        data_manifest = (
            json.loads(data_manifest_path.read_text(encoding="utf-8"))
            if data_manifest_path.exists() else {}
        )
        checksums = manifest.get("checksums_sha256") or {}
        if not isinstance(checksums, dict) or not checksums:
            raise ValueError("Bundle manifest has no file checksums")
        for relative, expected in checksums.items():
            candidate = (root / str(relative)).resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError as exc:
                raise ValueError(f"Unsafe checksum path in bundle: {relative}") from exc
            if not candidate.is_file():
                raise ValueError(f"Bundle checksum target is missing: {relative}")
            actual = _sha256(candidate)
            if actual.lower() != str(expected).lower():
                raise ValueError(f"Bundle checksum mismatch: {relative}")

        config_timeframe = _timeframe_from_config(config)
        config_family = str(config.get("target_col") or "").strip().lower()
        if config_timeframe != timeframe:
            raise ValueError(
                f"Bundle timeframe {config_timeframe or '<missing>'} does not match endpoint {timeframe}"
            )
        if config_family != family:
            raise ValueError(
                f"Bundle target {config_family or '<missing>'} does not match endpoint family {family}"
            )

        qualification = manifest.get("qualification") or {}
        model_name = str(qualification.get("model") or "").strip().lower()
        if not model_name:
            raise ValueError("Bundle manifest does not identify the qualifying model")
        model_dir = root / "model_files"
        model_files = sorted(
            path for path in model_dir.glob("*.joblib")
            if "_artifacts_" not in path.name.lower()
        )
        artifact_files = sorted(path for path in model_dir.glob("*.joblib") if "_artifacts_" in path.name.lower())
        if len(model_files) != 1:
            raise ValueError(f"Bundle must contain exactly one fitted .joblib model; found {len(model_files)}")
        if len(artifact_files) > 1:
            raise ValueError(f"Bundle contains multiple scaler artifact files; found {len(artifact_files)}")
        model_path = model_files[0]
        artifacts_path = artifact_files[0] if artifact_files else None
        if not model_path.name.lower().startswith(f"{model_name}_"):
            raise ValueError("Serialized model filename does not match the qualifying model in manifest.json")

        if load_artifacts:
            joblib.load(model_path)
            if artifacts_path is not None:
                artifacts = joblib.load(artifacts_path)
                if artifacts is not None and not isinstance(artifacts, dict):
                    raise ValueError("Scaler artifact file must contain a mapping")

        return {
            "endpoint": f"{timeframe}_{family}",
            "timeframe": timeframe,
            "family": family,
            "model": model_name,
            "qualification": qualification,
            "training_data_first_index": data_manifest.get("first_index"),
            "training_data_last_index": data_manifest.get("last_index"),
            "bundle_root": str(root),
            "bundle_name": root.name,
            "config_relative_path": str(config_path.relative_to(root)).replace(os.sep, "/"),
            "model_relative_path": str(model_path.relative_to(root)).replace(os.sep, "/"),
            "artifacts_relative_path": (
                str(artifacts_path.relative_to(root)).replace(os.sep, "/") if artifacts_path else None
            ),
            "source_manifest": manifest,
        }


def install_bundle(
    bundle: Path, endpoint: str, *, deployment_root: Optional[Path] = None,
    load_artifacts: bool = True,
) -> Dict[str, Any]:
    """Copy a validated bundle into an immutable endpoint version directory."""
    root = (deployment_root or DEFAULT_DEPLOYMENT_ROOT).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tsmm_bundle_install_") as temp_dir:
        extracted = _bundle_root(Path(bundle), Path(temp_dir) / "extracted")
        details = validate_bundle(extracted, endpoint, load_artifacts=load_artifacts)
        endpoint_key = str(details["endpoint"])
        source_manifest = details["source_manifest"]
        identity_payload = json.dumps(
            {
                "endpoint": endpoint_key,
                "checksums": source_manifest.get("checksums_sha256") or {},
            },
            sort_keys=True,
        ).encode("utf-8")
        identity = hashlib.sha256(identity_payload).hexdigest()[:16]
        deployment_id = f"{Path(extracted).name}__{identity}"
        destination = root / endpoint_key / deployment_id
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = destination.parent / f".staging_{deployment_id}_{uuid.uuid4().hex}"
            try:
                shutil.copytree(extracted, staging)
                os.replace(staging, destination)
            finally:
                if staging.exists():
                    shutil.rmtree(staging, ignore_errors=True)

        installed = validate_bundle(destination, endpoint_key, load_artifacts=load_artifacts)
        record = {
            "schema_version": 1,
            "endpoint": endpoint_key,
            "deployment_id": deployment_id,
            "deployment_dir": str(destination.resolve()),
            "config_path": str((destination / installed["config_relative_path"]).resolve()),
            "model_path": str((destination / installed["model_relative_path"]).resolve()),
            "artifacts_path": (
                str((destination / installed["artifacts_relative_path"]).resolve())
                if installed.get("artifacts_relative_path") else None
            ),
            "model": installed["model"],
            "timeframe": installed["timeframe"],
            "family": installed["family"],
            "qualification": installed["qualification"],
            "training_data_first_index": installed.get("training_data_first_index"),
            "training_data_last_index": installed.get("training_data_last_index"),
            "source_bundle": str(Path(bundle).resolve()),
            "installed_at_utc": _utc_now(),
        }
        _atomic_json(destination / "deployment.json", record)
        return record


def load_active_manifest(deployment_root: Optional[Path] = None) -> Dict[str, Any]:
    path = active_manifest_path(deployment_root)
    if not path.exists():
        return {"schema_version": 1, "endpoints": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Active model manifest is unreadable: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("endpoints", {}), dict):
        raise ValueError(f"Active model manifest has an invalid structure: {path}")
    return payload


def _deployment_files_exist(record: Dict[str, Any]) -> bool:
    required = [record.get("config_path"), record.get("model_path")]
    if not all(value and Path(str(value)).is_file() for value in required):
        return False
    artifacts = record.get("artifacts_path")
    return not artifacts or Path(str(artifacts)).is_file()


def resolve_active_deployment(endpoint: str, deployment_root: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    timeframe, family = _parse_endpoint(endpoint)
    record = dict((load_active_manifest(deployment_root).get("endpoints") or {}).get(f"{timeframe}_{family}") or {})
    if not record:
        return None
    if not _deployment_files_exist(record):
        raise FileNotFoundError(f"Active deployment files are incomplete for {timeframe}_{family}")
    if str(record.get("timeframe") or "").lower() != timeframe:
        raise ValueError(f"Active deployment timeframe mismatch for {timeframe}_{family}")
    if str(record.get("family") or "").lower() != family:
        raise ValueError(f"Active deployment family mismatch for {timeframe}_{family}")
    return record


def deployment_model_spec(deployment: Dict[str, Any]) -> Dict[str, Any]:
    """Build the exact inference specification carried by an installed bundle."""
    config_path = Path(str(deployment.get("config_path") or ""))
    if not config_path.is_file():
        raise FileNotFoundError(f"Deployment config is missing: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    metrics = deployment.get("metrics") or {}
    qualification = deployment.get("qualification") or {}
    return {
        "family": str(deployment.get("family") or config.get("target_col") or "").lower(),
        "timeframe": str(deployment.get("timeframe") or _timeframe_from_config(config)).lower(),
        "model": str(deployment.get("model") or qualification.get("model") or "").lower(),
        "config_path": str(config_path.resolve()),
        "model_path": str(Path(str(deployment.get("model_path") or "")).resolve()),
        "artifacts_path": (
            str(Path(str(deployment["artifacts_path"])).resolve()) if deployment.get("artifacts_path") else None
        ),
        "deployment_id": deployment.get("deployment_id"),
        "r2": float(metrics.get("holdout_r2", qualification.get("score", 0.0)) or 0.0),
        "refreshed_r2": metrics.get("holdout_r2", qualification.get("score")),
        "validation_status": "installed_bundle",
        "training_data_first_index": deployment.get("training_data_first_index"),
        "training_data_last_index": deployment.get("training_data_last_index"),
        "n_steps": int(config.get("n_steps", 1) or 1),
        "m_steps": int(config.get("m_steps", 1) or 1),
        "horizon": int(config.get("horizon", config.get("m_steps", 1)) or 1),
        "input_features": [str(value) for value in (config.get("input_features") or [])],
        "target_features": [str(value) for value in (config.get("target_features") or ["y_diff"])],
        "target_col": str(config.get("target_col") or deployment.get("family") or "HIGH").upper(),
        "rolling_windows": [int(value) for value in (config.get("rolling_windows") or [2, 7, 30, 60])],
    }


def activate_deployment(
    deployment: Dict[str, Any], *, metrics: Optional[Dict[str, Any]] = None,
    deployment_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Atomically switch one endpoint to an already validated deployment."""
    endpoint = str(deployment.get("endpoint") or "")
    timeframe, family = _parse_endpoint(endpoint)
    if not _deployment_files_exist(deployment):
        raise ValueError(f"Deployment files are incomplete for {endpoint}")
    deployment_dir = Path(str(deployment.get("deployment_dir") or ""))
    if not deployment_dir.is_dir():
        raise ValueError(f"Deployment directory is missing for {endpoint}")
    validate_bundle(deployment_dir, endpoint, load_artifacts=False)
    payload = load_active_manifest(deployment_root)
    payload.setdefault("schema_version", 1)
    record = dict(deployment)
    record["endpoint"] = f"{timeframe}_{family}"
    record["activated_at_utc"] = _utc_now()
    if metrics is not None:
        record["metrics"] = metrics
    payload.setdefault("endpoints", {})[record["endpoint"]] = record
    payload["updated_at_utc"] = record["activated_at_utc"]
    _atomic_json(active_manifest_path(deployment_root), payload)
    return record


def restore_active_manifest(snapshot: Dict[str, Any], deployment_root: Optional[Path] = None) -> None:
    _atomic_json(active_manifest_path(deployment_root), snapshot)
