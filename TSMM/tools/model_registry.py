"""Inspect, promote, and roll back forecasting champions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.model_deployment import (
    activate_deployment,
    install_bundle,
    load_active_manifest,
    restore_active_manifest,
    validate_bundle,
)
from utils.model_governance import assess_challenger, load_registry, promote, rollback


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Champion/challenger forecasting registry")
    parser.add_argument("--registry", default="output/model_registry/registry.json")
    sub = parser.add_subparsers(dest="command", required=True)
    assess = sub.add_parser("assess")
    assess.add_argument("--metrics", required=True)
    assess.add_argument("--champion-metrics")
    promote_cmd = sub.add_parser("promote")
    promote_cmd.add_argument("--endpoint", required=True)
    promote_cmd.add_argument("--bundle", required=True)
    promote_cmd.add_argument("--metrics", required=True)
    validate_cmd = sub.add_parser("validate-bundle")
    validate_cmd.add_argument("--endpoint", required=True)
    validate_cmd.add_argument("--bundle", required=True)
    install_cmd = sub.add_parser("install")
    install_cmd.add_argument("--endpoint", required=True)
    install_cmd.add_argument("--bundle", required=True)
    rollback_cmd = sub.add_parser("rollback")
    rollback_cmd.add_argument("--endpoint", required=True)
    sub.add_parser("show")
    args = parser.parse_args()
    registry_path = (ROOT / args.registry).resolve() if not Path(args.registry).is_absolute() else Path(args.registry)

    if args.command == "show":
        result = load_registry(registry_path)
    elif args.command == "validate-bundle":
        result = validate_bundle(Path(args.bundle), args.endpoint)
        result.pop("source_manifest", None)
    elif args.command == "install":
        result = install_bundle(Path(args.bundle), args.endpoint)
    elif args.command == "rollback":
        registry = load_registry(registry_path)
        endpoint_record = ((registry.get("endpoints") or {}).get(args.endpoint) or {})
        history = endpoint_record.get("history") or []
        if not history:
            raise ValueError(f"No rollback generation is available for {args.endpoint}")
        previous = history[-1]
        deployment = previous.get("deployment")
        if not deployment:
            deployment = install_bundle(Path(previous["bundle"]), args.endpoint)
        active_snapshot = load_active_manifest()
        try:
            activated = activate_deployment(deployment, metrics=previous.get("metrics"))
            result = rollback(registry_path, args.endpoint)
            result["deployment"] = activated
        except Exception:
            restore_active_manifest(active_snapshot)
            raise
    else:
        candidate = _load(args.metrics)
        champion = _load(args.champion_metrics) if getattr(args, "champion_metrics", None) else None
        assessment = assess_challenger(candidate, champion)
        if args.command == "assess":
            result = assessment
        else:
            current = ((load_registry(registry_path).get("endpoints") or {}).get(args.endpoint) or {}).get("champion")
            assessment = assess_challenger(candidate, (current or {}).get("metrics"))
            if not assessment.get("approved"):
                raise ValueError(f"Challenger did not pass promotion gates: {assessment.get('failures')}")
            deployment = install_bundle(Path(args.bundle), args.endpoint)
            active_snapshot = load_active_manifest()
            try:
                activated = activate_deployment(deployment, metrics=candidate)
                result = promote(
                    registry_path, args.endpoint, args.bundle, candidate, assessment,
                    deployment=activated,
                )
            except Exception:
                restore_active_manifest(active_snapshot)
                raise
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
