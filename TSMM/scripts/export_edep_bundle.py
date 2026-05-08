import argparse
import re
from pathlib import Path

import joblib
import yaml


def _infer_timeframe(config_path: Path, cfg: dict) -> str:
    timeframe = str(cfg.get("timeframe", "")).strip().lower()
    if timeframe:
        return timeframe

    data_path = str(cfg.get("data_path", "")).lower()
    match = re.search(r"(\d+[mhw])", data_path)
    if match:
        return match.group(1)

    match = re.search(r"high(\d+[mhw])results", str(config_path).lower())
    if match:
        return match.group(1)

    return "unknown"


def _infer_model_type(cfg: dict, override: str) -> str:
    if override:
        return override.strip().lower()

    univariate = list(((cfg.get("models_to_run") or {}).get("univariate") or []))
    if univariate:
        return str(univariate[0]).strip().lower()
    return "unknown"


def parse_args():
    parser = argparse.ArgumentParser(description="Export one TSMM predictor into an eDep-ready single bundle")
    parser.add_argument("--model-file", required=True, help="Path to the saved TSMM model file")
    parser.add_argument("--artifacts-file", required=True, help="Path to the TSMM scaler artifacts file")
    parser.add_argument("--config-file", required=True, help="Path to the matching top config YAML")
    parser.add_argument("--output-file", required=True, help="Output bundle path, preferably ending in .pkl")
    parser.add_argument("--model-type", default="", help="Optional override, e.g. ulr or nbeats")
    parser.add_argument("--payload-field", default="window", help="Object field name for matrix input")
    parser.add_argument("--vector-field", default="vector", help="Object field name for flattened input")
    parser.add_argument("--signal-feature", default="y_diff", help="Target feature used to derive buy/sell/hold")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_file)
    artifacts_path = Path(args.artifacts_file)
    config_path = Path(args.config_file)
    output_path = Path(args.output_file)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not artifacts_path.exists():
        raise FileNotFoundError(f"Artifacts file not found: {artifacts_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    model = joblib.load(model_path)
    artifacts = joblib.load(artifacts_path)
    if "scaler_X" not in artifacts or "scaler_y" not in artifacts:
        raise ValueError("Artifacts file must contain scaler_X and scaler_y")

    spec = {
        "model_type": _infer_model_type(cfg, args.model_type),
        "timeframe": _infer_timeframe(config_path, cfg),
        "n_steps": int(cfg.get("n_steps", 1) or 1),
        "input_features": list(cfg.get("input_features") or []),
        "target_features": list(cfg.get("target_features") or []),
        "payload_field": str(args.payload_field),
        "vector_field": str(args.vector_field),
        "signal_feature": str(args.signal_feature),
        "source_config": str(config_path),
        "source_model": str(model_path),
        "source_artifacts": str(artifacts_path),
    }

    bundle = {
        "bundle_version": "1.0.0",
        "model": model,
        "artifacts": {
            "scaler_X": artifacts["scaler_X"],
            "scaler_y": artifacts["scaler_y"],
        },
        "spec": spec,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)

    print(
        {
            "ok": True,
            "output_file": str(output_path),
            "model_type": spec["model_type"],
            "timeframe": spec["timeframe"],
            "n_steps": spec["n_steps"],
            "n_features": len(spec["input_features"]),
        }
    )


if __name__ == "__main__":
    main()