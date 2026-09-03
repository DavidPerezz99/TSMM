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
    parser.add_argument(
        "--asset",
        default="xauusd",
        help="Asset namespace. Non-XAU assets use model_files/deployments/<asset> by default.",
    )
    parser.add_argument("--deployment-root", default=None, help="Explicit immutable deployment root")
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
    activate_cmd = sub.add_parser(
        "activate",
        help="Install and deliberately activate a package without challenger promotion",
    )
    activate_cmd.add_argument("--endpoint", required=True)
    activate_cmd.add_argument("--bundle", required=True)
    activate_cmd.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Required when activating a labelled below-threshold best-available package",
    )
    rollback_cmd = sub.add_parser("rollback")
    rollback_cmd.add_argument("--endpoint", required=True)
    sub.add_parser("show")
    args = parser.parse_args()
    asset = str(args.asset or "xauusd").strip().lower()
    if args.deployment_root:
        deployment_root = Path(args.deployment_root)
        if not deployment_root.is_absolute():
            deployment_root = (ROOT / deployment_root).resolve()
    else:
        deployment_root = ROOT / "model_files" / "deployments"
        if asset not in {"", "xauusd"}:
            deployment_root = deployment_root / asset
    registry_value = str(args.registry)
    if registry_value == "output/model_registry/registry.json" and asset not in {"", "xauusd"}:
        registry_value = f"output/model_registry/{asset}_registry.json"
    registry_path = (ROOT / registry_value).resolve() if not Path(registry_value).is_absolute() else Path(registry_value)

    if args.command == "show":
        result = load_registry(registry_path)
    elif args.command == "validate-bundle":
        result = validate_bundle(Path(args.bundle), args.endpoint)
        result.pop("source_manifest", None)
    elif args.command == "install":
        result = install_bundle(Path(args.bundle), args.endpoint, deployment_root=deployment_root)
    elif args.command == "activate":
        deployment = install_bundle(
            Path(args.bundle), args.endpoint, deployment_root=deployment_root
        )
        qualification = dict(deployment.get("qualification") or {})
        is_fallback = (
            str(qualification.get("selection_tier") or "qualified_candidate")
            != "qualified_candidate"
            or not bool(qualification.get("qualified_candidate", True))
        )
        if is_fallback and not bool(args.allow_fallback):
            raise ValueError(
                "This is a labelled best-available fallback. Re-run with "
                "--allow-fallback to acknowledge that it did not pass the R2 gate."
            )
        score = float(qualification.get("score", 0.0) or 0.0)
        result = activate_deployment(
            deployment,
            metrics={
                "holdout_r2": score,
                "selection_tier": qualification.get("selection_tier"),
                "qualified_candidate": bool(
                    qualification.get("qualified_candidate", not is_fallback)
                ),
            },
            deployment_root=deployment_root,
        )
    elif args.command == "rollback":
        registry = load_registry(registry_path)
        endpoint_record = ((registry.get("endpoints") or {}).get(args.endpoint) or {})
        history = endpoint_record.get("history") or []
        if not history:
            raise ValueError(f"No rollback generation is available for {args.endpoint}")
        previous = history[-1]
        deployment = previous.get("deployment")
        if not deployment:
            deployment = install_bundle(Path(previous["bundle"]), args.endpoint, deployment_root=deployment_root)
        active_snapshot = load_active_manifest(deployment_root)
        try:
            activated = activate_deployment(
                deployment, metrics=previous.get("metrics"), deployment_root=deployment_root
            )
            result = rollback(registry_path, args.endpoint)
            result["deployment"] = activated
        except Exception:
            restore_active_manifest(active_snapshot, deployment_root=deployment_root)
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
            deployment = install_bundle(
                Path(args.bundle), args.endpoint, deployment_root=deployment_root
            )
            active_snapshot = load_active_manifest(deployment_root)
            try:
                activated = activate_deployment(
                    deployment, metrics=candidate, deployment_root=deployment_root
                )
                result = promote(
                    registry_path, args.endpoint, args.bundle, candidate, assessment,
                    deployment=activated,
                )
            except Exception:
                restore_active_manifest(active_snapshot, deployment_root=deployment_root)
                raise
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
