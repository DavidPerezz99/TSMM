"""Inspect, promote, and roll back forecasting champions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    rollback_cmd = sub.add_parser("rollback")
    rollback_cmd.add_argument("--endpoint", required=True)
    sub.add_parser("show")
    args = parser.parse_args()
    registry_path = (ROOT / args.registry).resolve() if not Path(args.registry).is_absolute() else Path(args.registry)

    if args.command == "show":
        result = load_registry(registry_path)
    elif args.command == "rollback":
        result = rollback(registry_path, args.endpoint)
    else:
        candidate = _load(args.metrics)
        champion = _load(args.champion_metrics) if getattr(args, "champion_metrics", None) else None
        assessment = assess_challenger(candidate, champion)
        if args.command == "assess":
            result = assessment
        else:
            current = ((load_registry(registry_path).get("endpoints") or {}).get(args.endpoint) or {}).get("champion")
            assessment = assess_challenger(candidate, (current or {}).get("metrics"))
            result = promote(registry_path, args.endpoint, args.bundle, candidate, assessment)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
