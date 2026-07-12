# TSMM Testing Guide (From Source Dataset)

This guide starts from your source CSV and validates both disruption detection and full pipeline behavior.

## 1) Open project root
Use this as working directory:

`TSMM/`

## 2) (Optional) Create and activate environment
PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## 3) Install dependencies

```powershell
pip install -r requirements.txt
```

## 4) Validate source dataset + disruption modules only (fast)
This command loads your dataset through `utils/data_loader.py`, then runs:
- momentum overlay
- volatility targeting
- regime classifier
- rupture forecaster

```powershell
python scripts/validate_source_dataset.py --config config/config.yaml
```

Expected output: JSON summary with keys `momentum`, `vol_target`, `regime`, `rupture`.

## 5) Run automated tests (unit + synthetic disruption)

```powershell
python -m unittest discover -s tests -v
```

Or run everything (tests + source dataset validator) in one command:

```powershell
python scripts/run_validation_suite.py --config config/config.yaml
```

What this covers:
- `tests/test_overlays.py`: schema/range checks for momentum/vol-target/regime.
- `tests/test_disruption_detection.py`: synthetic market regime shift and rupture behavior checks.

## 6) Run full app on source dataset (single run)
Generates report/metrics using configured model list.

```powershell
python app.py
```

Separate command modes:

```powershell
# Forecast report + metrics only
python app.py forecast-report

# Trading plan only
python app.py trading-plan

# Trading plan with manual model selection
python app.py trading-plan --plan-model ulr
```

If you need a specific config file:

```powershell
$env:CONFIG_PATH="config/config.yaml"
python app.py
```

Outputs are under `reports/` (PDF + metrics JSON depending on config).

Report naming now includes target and timestamp:
- `forecast_report_<target_col>_<YYYYMMDD_HHMMSS>.pdf` (or csv/parquet)

## 6.1) Deterministic investing-agent report (Mode A)
By default, the app also reads `config/trading_agent.yaml` and generates a
trading plan PDF report per session:

```powershell
python app.py
```

Output example:
- `reports/trading_plans/trading_plan_<target_col>_<YYYYMMDD_HHMMSS>.pdf`

To use a different trading config file:

```powershell
$env:TRADING_CONFIG_PATH="config/trading_agent.yaml"
python app.py
```

## 6.2) Enable Mode B scaffold (MT5-first)
Edit `config/trading_agent.yaml`:
- `agent.mode: mode_b`
- `mode_b.enabled: true`
- `broker.mt5.enabled: true`

Then run:

```powershell
python app.py
```

Notes:
- Decision logic remains deterministic.
- Mode B checks model endpoints and MT5 connection readiness.
- Live execution is gated by `agent.confirm_live_execution`.

## 6.3) Plot data export for dashboard replay
The app exports source data used by report charts to:
- `output/plot_data/<model_name>/...`
- `output/plot_data/rupture/...`

Controlled by `plot_data_export` in `config/config.yaml`.

## 7) Run search mode for one generated config

```powershell
python tools/search_mode.py --config config/config.yaml
```

This writes experiment summary JSON including governance metrics under `_governance`.

## 8) Run bulk hypersearch
Use the config path under `config/` (important).

```powershell
python tools/hypersearch.py bulk_search --base-config config_templates/univariate.yaml --param-grid config/sweep_definition.yaml --output-dir generated_cfgs --max-parallel 2
```

If memory is tight, reduce parallel workers:

```powershell
python tools/hypersearch.py bulk_search --base-config config_templates/univariate.yaml --param-grid config/sweep_definition.yaml --output-dir generated_cfgs --max-parallel 1
```

## 9) Suggested pass/fail checks before live/paper mode
- No crashes in steps 4–8.
- Rupture metrics exist and are not degenerate (all zeros).
- Regime output has valid `state` and policy.
- Report includes Momentum/Risk Overlay page and Rupture page.
- Search summaries include `_governance` while keeping `SUCCESS/NO_METRICS` behavior.
- Trading plan report is generated with deterministic entry/SL/TP rationale.
- Backtest section in trading report shows operation lifecycle metrics.

## 10) Troubleshooting
- If a command fails with missing packages, reinstall with step 3.
- If hypersearch fails due path, verify `--param-grid config/sweep_definition.yaml`.
- If runtime memory spikes, use `--max-parallel 1` first.

## 11) Live data update loop (1-minute)
Set API token first:

```powershell
$env:TIINGO_API_TOKEN="<your_token>"
```

Run:

```powershell
python scripts/live_data_loop.py --config config/config.yaml --every-seconds 60
```

## 12) Run live dashboard

```powershell
python apps/dashboard.py
```

Open browser at `http://127.0.0.1:8050`.

## 13) Predictor endpoint POST formats

For the exact POST body format and `ingestion_pipeline.py` templates for each currently available `top1` predictor artifact, see:

- [PREDICTOR_ENDPOINT_FORMATS.md](PREDICTOR_ENDPOINT_FORMATS.md)
