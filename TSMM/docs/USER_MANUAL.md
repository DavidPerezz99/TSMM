# TSMM User Manual

This manual is for day-to-day use of TSMM: setup, running forecasts, dashboards, validation, and hypersearch.

## 1) What TSMM does

TSMM trains time-series forecasting models from YAML config and produces:
- Forecast report files (PDF/CSV/Parquet)
- Metrics JSON files
- Optional dashboard and UI workflows
- Experiment summaries for hyperparameter search

Main runtime entry points:
- `app.py`: single forecasting run
- `tools/search_mode.py`: one experiment run (often called by hypersearch)
- `tools/hypersearch.py`: bulk sweep execution
- `apps/ui.py`: config editor UI (Dash)
- `apps/dashboard.py`: live dashboard (Dash)

## 2) Requirements

- Windows PowerShell (recommended)
- Python 3.10+ (3.11 is recommended in this repo)
- Internet access only if pulling live market data

## 3) First-time setup

From the project root (`TSMM/`):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If your shell blocks activation scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4) Configure the app

Primary config file:
- `config/config.yaml`

Important keys to verify before running:
- `data_path`
- `data_timeframe_minutes` when `data_path` points to `data/market_data.sqlite`
- `date_col`
- `target_col`
- `problem_type` (`univariate` or `multivariate`)
- `models_to_run`
- `output.format` (`pdf`, `csv`, `parquet`)
- `output.directory`

SQLite-backed example:

```yaml
data_path: data/market_data.sqlite
data_timeframe_minutes: 420
date_col: DATE
target_col: HIGH
records: 5000
```

Common `data_timeframe_minutes` values:
- `1` for the 1-minute master series
- `10` for 10-minute grouped candles
- `30` for 30-minute grouped candles
- `420` for 7-hour grouped candles

Trading-mode related config:
- `config/trading_agent.yaml`

Sweep config:
- `config/sweep_definition.yaml`

## 5) Run a normal forecasting session

```powershell
python app.py
```

What happens:
- Runs startup master sync from Tiingo first (if enabled in trading config)
- Loads config from `config/config.yaml` (or `CONFIG_PATH` env var)
- Loads and preprocesses data
- Trains selected models
- Evaluates models
- Saves output report/table and metrics

Startup sync is controlled in `config/trading_agent.yaml` under `dashboard`:
- `startup_sync_enabled`
- `startup_max_pulls`
- `startup_freshness_lag_minutes`
- `startup_status_path`

At startup, backend writes a summary JSON to `startup_status_path` (default `reports/runtime/startup_sync_status.json`).

Typical outputs:
- `reports/forecast_report_<timestamp>.<pdf|csv|parquet>`
- `reports/all_models_metrics_<timestamp>.json`
- logs under `logs/`

Use a custom config path:

```powershell
$env:CONFIG_PATH="config/config.yaml"
python app.py
```

## 5.1) Run trading jobs (Agent A -> Agent B)

The app now supports a dedicated trading-job flow with persistence and manual stop.

Start trading job:

```powershell
python app.py trading-job start
```

Start trading job with explicit model for Agent A plan:

```powershell
python app.py trading-job start --plan-model ulr
```
 
Resume an in-progress trading job (from saved runtime state):

```powershell
python app.py trading-job resume
```

Manual safety stop (writes stop flag checked by Agent B loop):

```powershell
python app.py trading-job stop
```

Runtime state files are configured in `config/trading_agent.yaml` under `trading_job`.

## 6) Use the Config UI

Start UI:

```powershell
python apps/ui.py
```

Open in browser:
- `http://127.0.0.1:8051`

UI tabs let you:
- Edit and save main config
- Edit and save trading agent config
- Edit and save sweep definition
- Edit and save LLM providers config
- Trigger master table update (if Tiingo token is set)
- Interrupt/resume Mode B flag control

LLM provider config file:
- `config/llm_providers.yaml`

Trading-agent LLM switch is in:
- `config/trading_agent.yaml` under `llm`

Notes:
- Use `env:YOUR_ENV_VAR` for secret fields to avoid hardcoding keys.
- Agent A can attach optional LLM reasoning when `mode_a.use_llm_explanation: true`.
- Agent B can request LLM assistance periodically (default every 5 minutes).

## 7) Run the Live Dashboard

Start dashboard:

```powershell
python apps/dashboard.py
```

Open in browser:
- `http://127.0.0.1:8050`

Dashboard is for monitoring market data/state and visual analytics.

Startup sync panel in dashboard:
- A `Startup Sync Summary` panel is shown under controls.
- It reads backend startup status from `dashboard.startup_status_path` in `config/trading_agent.yaml`.
- It displays:
	- Pull attempts
	- Latest aligned timestamp
	- Whether weekend relaxation was applied
	- Last status write timestamp
	- Optional reason for skipped/failed startup sync

Important: this panel updates after running backend startup at least once (`python app.py` or `python app.py trading-job start`), because backend writes the status file.

## 8) Validate data and disruption modules

Quick validator from source dataset:

```powershell
python scripts/validate_source_dataset.py --config config/config.yaml
```

Full validation suite (unit tests + source validator):

```powershell
python scripts/run_validation_suite.py --config config/config.yaml
```

Unit tests only:

```powershell
python -m unittest discover -s tests -v
```

## 9) Run one search experiment

```powershell
python tools/search_mode.py --config config/config.yaml
```

This writes experiment summaries under `experiments/`.

Useful flags:

```powershell
python tools/search_mode.py --config config/config.yaml --summary-dir experiments
python tools/search_mode.py --config config/config.yaml --bulk-search
```

## 10) Run bulk hyperparameter search

Preview the exact unique count and conservative per-experiment RAM estimate
before creating any files or training models:

```powershell
python tools/hypersearch.py plan --base-config config_templates/univariate.yaml --param-grid config/sweeps/sweep_30m_high_return.yaml --ram-limit-gb 20
```

Example:

```powershell
python tools/hypersearch.py bulk_search --base-config config_templates/univariate.yaml --param-grid config/sweep_definition.yaml --output-dir generated_cfgs --max-parallel 2
```

If memory is limited:

```powershell
python tools/hypersearch.py bulk_search --base-config config_templates/univariate.yaml --param-grid config/sweep_definition.yaml --output-dir generated_cfgs --max-parallel 1
```

Use `--max-experiments` as a hard safety limit. The command refuses to start if
the corrected unique plan is larger:

```powershell
python tools/hypersearch.py bulk_search --base-config config_templates/univariate.yaml --param-grid config/sweeps/sweep_30m_high_return.yaml --output-dir generated_cfgs --max-parallel 1 --max-experiments 250 --cpu-threads-per-experiment 6
```

Shortlist completed configurations by the highest out-of-sample R2 (the new
default):

```powershell
python tools/hypersearch.py smart_search --from-experiments generated_cfgs/SESSION_FOLDER --top-n 20 --metric R2
```

For error metrics, `auto` uses the lower value, for example `--metric RMSE`.

### 10.1) Manual overnight experiment sessions

The committed XAUUSD session is:

- `config/experiment_sessions/xauusd_nightly.yaml`
- all 28 independent combinations of OPEN/HIGH/LOW/CLOSE and 10m, 30m, 1h,
  3h, 7h, 12h, and 24h;
- 10,476 unique experiments in the complete campaign, with no individual
  OHLC/timeframe sweep exceeding 432 experiments;
- execution order: shortest timeframe first, then OPEN, HIGH, LOW, and CLOSE;
- algorithms: ULR plus interpretable and black-box N-BEATS;
- hard local stop: 5:00 AM;
- one experiment process at a time;
- 20 GB per-process RAM limit, a 2 GB emergency system reserve, and six
  numerical-library CPU threads;
- generated configs, summaries, logs, and resume state under
  `output/hypersearch_sessions/xauusd_all_ohlc_timeframes_v1/`.

Nothing schedules or starts this session automatically. Every run must be
started manually.

Inspect the plan without creating or running anything:

```powershell
python tools/experiment_session.py plan --config config/experiment_sessions/xauusd_nightly.yaml
```

Refresh all seven SQL views and materialized cache tables before a new
recalibration campaign or after correcting historical data:

```powershell
python tools/experiment_session.py prepare-data --config config/experiment_sessions/xauusd_nightly.yaml
```

Show the exact cached history and the RAM-, CPU-, and data-limited record
ceiling for every model family:

```powershell
python tools/experiment_session.py capacity --config config/experiment_sessions/xauusd_nightly.yaml
```

Start or resume and continue until 5:00 AM:

```powershell
python tools/experiment_session.py run --config config/experiment_sessions/xauusd_nightly.yaml
```

Run exactly one pending experiment:

```powershell
python tools/experiment_session.py run --config config/experiment_sessions/xauusd_nightly.yaml --max-experiments 1
```

Run a manually sized batch, for example ten experiments:

```powershell
python tools/experiment_session.py run --config config/experiment_sessions/xauusd_nightly.yaml --max-experiments 10
```

Show completion by ordered sweep:

```powershell
python tools/experiment_session.py status --config config/experiment_sessions/xauusd_nightly.yaml
```

The runner never starts a new experiment inside the final 30 minutes before
the deadline. If a running experiment reaches 5:00 AM, its process tree is
terminated and marked `INTERRUPTED`; the next manual run retries it. A process
that exceeds 20 GB, or leaves less than 2 GB available system RAM, is terminated
with `RESOURCE_LIMIT`. Completed experiments are never rerun during resume.

The planner reports separate RAM- and CPU-duration-limited record ceilings and
then takes the minimum of those limits and the actual cached history. On this
32 GB Ryzen 5 machine, with a 20 GB process cap and a 90-minute budget for the
slowest configured shape, the August 17, 2026 capacity snapshot is:

| Timeframe | Cached candles | RAM ceiling | 90-minute CPU ceiling | Effective ceiling | Largest configured experiment |
|---|---:|---:|---:|---:|---:|
| 10m | 617,182 | 826,132 | 15,415 | 15,415 | 15,000 |
| 30m | 206,540 | 332,014 | 11,181 | 11,181 | 10,000 |
| 1h | 103,880 | 589,697 | 13,829 | 13,829 | 12,000 |
| 3h | 35,982 | 2,727,638 | 20,267 | 20,267 | 16,000 |
| 7h | 16,213 | 2,911,245 | 20,473 | 16,213 | 16,000 |
| 12h | 9,921 | 3,808,536 | 21,244 | 9,921 | 9,000 |
| 24h | 5,417 | 5,505,398 | 22,109 | 5,417 | 5,000 |

These figures are conservative planning estimates, not measured promises. The
runner records actual duration and peak process RAM for every experiment and
uses matching-shape duration history for later deadline decisions.

The table is the ceiling that remains safe for **every** configured shape; its
CPU column is therefore governed by the slowest N-BEATS black-box variant. The
`capacity` command also prints algorithm-specific limits. Under the analytical
estimate, ULR can use the complete cached history at every timeframe, including
all 617,182 10m candles, while the worst-case N-BEATS limits match the effective
short-timeframe ceilings shown above. The committed grids intentionally remain
smaller so the first campaign can replace estimates with measured peak RAM and
duration before expanding any individual search.

The XAUUSD sweeps use the SQLite market database and materialized cache tables
for every configured timeframe. These tables avoid reparsing CSV data and
avoid recomputing the full dynamic view for every experiment. The live minute
tail is still merged when a cache is read. SQL reduces ingestion and resampling
overhead, but feature engineering, overlapping sequence creation, model
training, recursive inference, and evaluation still happen in memory, so the
RAM and deadline guards remain required.

The lower-level migration command remains available when rebuilding from a
different database or symbol:

```powershell
python scripts/migrate_market_data_to_sqlite.py --db-path data/market_data.sqlite --symbol XAUUSD --views 10,30,60,180,420,720,1440 --create-cache-tables
```

The multi-timeframe cache builder reads the minute master once and atomically
replaces the seven indexed cache tables. Routine new minute rows do not require
a rebuild because cache reads merge the mutable live tail.

## 11) Live data loop

Set token first:

```powershell
$env:TIINGO_API_TOKEN="<your_token>"
```

Run updater loop:

```powershell
python scripts/live_data_loop.py --config config/config.yaml --every-seconds 60
```

## 12) Common workflow for new users

1. Create venv and install requirements.
2. Open and adjust `config/config.yaml`.
3. Run `python app.py`.
4. Check generated files in `reports/`.
5. Run validator/tests before any serious experiment or deployment.
6. Use `tools/search_mode.py` and `tools/hypersearch.py` when tuning parameters.

## 13) Troubleshooting

### A) Torch / N-BEATS import errors

Symptom:
- Errors related to torch/cudnn when using N-BEATS.

Fix options:
- Install CPU torch build if GPU stack is not configured.
- Or switch `models_to_run` to models not requiring N-BEATS.

### B) Missing package errors

```powershell
pip install -r requirements.txt
```

### C) Config path issues

- Make sure paths are relative to project root or absolute.
- Confirm `data_path` exists.

### D) Empty or failed outputs

- Check logs under `logs/`.
- Run source validator to confirm dataset quality.
- Reduce `records` and/or `--max-parallel` for memory pressure.

## 14) File map (quick reference)

- `app.py`: main forecasting pipeline
- `apps/ui.py`: config editor UI
- `apps/dashboard.py`: live dashboard
- `apps/validation_dashboard.py`: agent validation dashboard
- `scripts/run_trading_backtest.py`: model-backed historical strategy evaluation
- `tools/search_mode.py`: single experiment run
- `tools/hypersearch.py`: bulk sweep engine
- `scripts/validate_source_dataset.py`: disruption validation from source data
- `scripts/run_validation_suite.py`: unit tests + source validation
- `config/config.yaml`: main forecasting config
- `config/trading_agent.yaml`: trading/agent config
- `config/sweep_definition.yaml`: sweep definitions

The validation dashboard's model-backed mode and `python app.py backtest`
replay the configured trading sessions with current forecasting artifacts,
programmed-order fills, five-minute Agent B checks, costs, closures, and daily
equity reporting. The generated report explicitly warns when current fitted
models overlap the historical period, because that case is a retrospective
stress test rather than a clean walk-forward estimate.

## 15) Recommended safe baseline

For first successful run:
- Keep `problem_type: univariate`
- Use one or two models in `models_to_run`
- Keep `records` limited (for example 2000-5000)
- Set `output.format: pdf`
- Run validation suite after forecast run

## 16) Before testing checklist

Use this checklist before running any tests or validation suite:

1. Confirm DB-backed source in `config/config.yaml`:
	- `data_path: data/market_data.sqlite`
	- `data_timeframe_minutes` matches your target test horizon (for example `10`, `30`, or `420`).
2. Confirm startup sync config in `config/trading_agent.yaml`:
	- `dashboard.startup_sync_enabled: true`
	- `dashboard.startup_max_pulls: 2`
	- `dashboard.startup_status_path` points to `reports/runtime/startup_sync_status.json` (or your custom path).
3. Export Tiingo token in terminal:
	- PowerShell: `$env:TIINGO_API_TOKEN="<your_token>"`
4. Run backend once to create/refresh startup sync status:
	- `python app.py`
5. Open dashboard and verify `Startup Sync Summary`:
	- `Pull attempts` is shown
	- `Latest aligned timestamp` is not `N/A` for normal operation
	- `Weekend relaxation applied` matches market calendar context
6. Run test command(s) only after steps 1-5 pass.

## 17) Endpoint reference

Use this section as the quick source of truth for currently configured endpoints.

Trading model endpoints (from `config/trading_agent.yaml`):
This app consumes these endpoints from a separate forecasting API app. It does not expose `/predict/*` routes itself.
- `10m`: `http://127.0.0.1:8000/predict/10m`
- Other timeframes should be added here only after they are deployed in your separate endpoint-serving app.

Local UI and dashboard services:
- Trading dashboard: `http://127.0.0.1:8050`
- Config UI: `http://127.0.0.1:8051`
- Validation dashboard: `http://127.0.0.1:8052`

LLM connector endpoint patterns:
- OpenAI-compatible: `{base_url}{chat_endpoint}`
- Anthropic: `{base_url}/v1/messages`
- Hugging Face: `https://api-inference.huggingface.co/models/{model}` (or configured `inference_endpoint`)
- Ollama: `{base_url}/api/generate`

External market data endpoint pattern:
- Tiingo FX prices: `https://api.tiingo.com/tiingo/fx/{symbol}/prices`

Notes:
- Some endpoints are configuration targets and may not be running unless their services are started.
- If you change ports/hosts, update this section and `config/trading_agent.yaml` together.
- Current baseline in this repo is only `10m` configured unless you explicitly add more deployed routes.

---

If you want, the next improvement is a role-based quick start section (Analyst / Research / Live Ops) with copy-paste commands for each role.
