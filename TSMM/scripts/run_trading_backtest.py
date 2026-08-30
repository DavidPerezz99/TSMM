"""Run an accelerated, model-backed TSMM trading strategy evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.model_deployment import deployment_model_spec, install_bundle
from utils.strategy_backtest import (
    ConsoleProgressBar,
    discover_replay_model_specs,
    run_historical_strategy_backtest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay TSMM over historical one-minute data using the current fitted forecasting artifacts, "
            "configured trading sessions, programmed entries, and five-minute Agent B checks."
        )
    )
    period = parser.add_mutually_exclusive_group()
    period.add_argument("--previous-month", action="store_true", help="Evaluate the previous local calendar month (default when no dates are supplied)")
    period.add_argument("--start", dest="start_date", help="Start date or timestamp in the trading timezone")
    parser.add_argument("--end", dest="end_date", help="End date or timestamp; required with --start")
    parser.add_argument("--trading-config", default="config/trading_agent.yaml", help="Trading configuration to replay")
    parser.add_argument("--market-source", default=None, help="One-minute SQLite/CSV source; defaults to dashboard.master_table_path")
    parser.add_argument("--output-dir", default=None, help="Run output directory; a timestamped reports/backtests folder is used by default")
    parser.add_argument("--initial-balance", type=float, default=100000.0)
    parser.add_argument("--contract-size", type=float, default=100.0, help="Currency P/L multiplier per 1.0 lot and 1.0 price move")
    parser.add_argument("--poll-minutes", type=int, default=None, help="Override Agent B replay interval")
    parser.add_argument("--candidate-bundle", help="Bundle directory or .zip to replay without activating it")
    parser.add_argument("--candidate-endpoint", help="Endpoint replaced by the candidate, for example 10m_high")
    parser.add_argument("--max-ticks", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(str(path or ""))
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def main() -> int:
    args = parse_args()
    config_path = _resolve(args.trading_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Trading config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        trading_cfg = yaml.safe_load(handle) or {}

    market_source = args.market_source or ((trading_cfg.get("dashboard") or {}).get("master_table_path")) or "data/market_data.sqlite"
    market_source_path = _resolve(str(market_source))
    output_dir = _resolve(
        args.output_dir
        or f"reports/backtests/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    print("Preparing historical market data and loading current forecasting models...", flush=True)
    progress = ConsoleProgressBar()
    replay_specs = None
    if bool(args.candidate_bundle) != bool(args.candidate_endpoint):
        raise ValueError("--candidate-bundle and --candidate-endpoint must be supplied together")
    if args.candidate_bundle:
        deployment = install_bundle(_resolve(args.candidate_bundle), args.candidate_endpoint)
        candidate_spec = deployment_model_spec(deployment)
        replay_specs = discover_replay_model_specs(trading_cfg)
        candidate_key = (candidate_spec["timeframe"], candidate_spec["family"])
        replay_specs = [
            spec for spec in replay_specs
            if (str(spec.get("timeframe")), str(spec.get("family")).lower()) != candidate_key
        ]
        replay_specs.append(candidate_spec)
        print(
            f"Candidate package loaded for {args.candidate_endpoint}: {deployment['deployment_id']}",
            flush=True,
        )

    result = run_historical_strategy_backtest(
        market_source=str(market_source_path),
        trading_cfg=trading_cfg,
        output_dir=str(output_dir),
        start_date=args.start_date,
        end_date=args.end_date,
        previous_month=bool(args.previous_month or not args.start_date),
        initial_balance=float(args.initial_balance),
        contract_size=float(args.contract_size),
        poll_minutes=args.poll_minutes,
        tick_progress_cb=progress,
        max_ticks=args.max_ticks,
        specs=replay_specs,
    )
    if not result.get("ok"):
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 1

    overall = (result.get("summary") or {}).get("overall") or {}
    print(
        json.dumps(
            {
                "ok": True,
                "result_grade": ((result.get("summary") or {}).get("validity") or {}).get("result_grade"),
                "trades": overall.get("n_trades"),
                "win_rate": overall.get("win_rate"),
                "net_pnl": overall.get("net_pnl"),
                "return_pct": overall.get("total_return_pct"),
                "max_drawdown_pct": overall.get("max_drawdown_pct"),
                "report": result.get("report_path"),
                "summary": result.get("summary_path"),
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
