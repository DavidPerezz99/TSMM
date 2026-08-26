# TSMM

Model recalibration, champion/challenger promotion, shadow evaluation, and small-account risk sizing are documented in [docs/MODEL_GOVERNANCE_AND_SMALL_ACCOUNT.md](docs/MODEL_GOVERNANCE_AND_SMALL_ACCOUNT.md).

TSMM is a time-series forecasting and managed-trading system with model
training, signal generation, MT5 execution, operational feedback, autonomous
trade supervision, and validation dashboards.

## Project Structure

```text
TSMM/
|-- app.py                  Main forecasting and trading-job CLI
|-- apps/                   Dash monitoring and configuration applications
|-- assets/                 Static project assets
|-- config/                 Runtime, trading, provider, and sweep configuration
|-- config_templates/       Reusable model configuration templates
|-- docs/                   Manuals, runbooks, references, and agent handoffs
|-- models/                 Forecasting model implementations
|-- scripts/                Runtime services, maintenance, and analysis commands
|-- tests/                  Automated tests
|-- tools/                  Modeling, hypersearch, and development utilities
|-- utils/                  Shared runtime and trading modules
|-- requirements.txt        Python dependencies
`-- tsmm_toggle.bat         Windows service launcher
```

Runtime and generated content belongs under `data/`, `logs/`, `model_files/`,
`output/`, and `reports/`. These directories are intentionally separate from
source code and mostly ignored by Git.

## Common Commands

```powershell
# Forecasting or trading CLI
python app.py
python app.py trading-job start

# Model-backed historical strategy evaluation (no MT5 connection)
python app.py backtest --previous-month
python app.py backtest --start-date 2026-07-01 --end-date 2026-07-31

# User interfaces
python scripts/start_all_uis.py

# Operational feedback analysis
python scripts/analyze_operation_feedback.py

# Tests
python -m unittest discover -s tests

# Hyperparameter search
python tools/hypersearch.py --help
python tools/hypersearch.py plan --base-config config_templates/univariate.yaml --param-grid config/sweeps/sweep_30m_high_return.yaml --ram-limit-gb 20

# Manual, resumable overnight experiment session
python tools/experiment_session.py plan --config config/experiment_sessions/xauusd_nightly.yaml
python tools/experiment_session.py prepare-data --config config/experiment_sessions/xauusd_nightly.yaml
python tools/experiment_session.py capacity --config config/experiment_sessions/xauusd_nightly.yaml
python tools/experiment_session.py run --config config/experiment_sessions/xauusd_nightly.yaml
python tools/experiment_session.py status --config config/experiment_sessions/xauusd_nightly.yaml
```

## Documentation

- [Coding agent notes](AGENTS.md)
- [User manual](docs/USER_MANUAL.md)
- [CLI runbook](docs/RUNBOOK_CLI.md)
- [Testing guide](docs/TESTING_GUIDE.md)
- [Predictor endpoint formats](docs/PREDICTOR_ENDPOINT_FORMATS.md)
- [Trading-analysis enhancement handoff](docs/TRADING_ANALYSIS_ENHANCEMENTS_HANDOFF.md)

Delayed stop-loss protection introduced by the trading-analysis branch remains
disabled by default. Review the enhancement handoff before changing that
configuration.
