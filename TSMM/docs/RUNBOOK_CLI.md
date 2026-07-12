# TSMM CLI Runbook

This runbook documents reliable command-line operations for deployment, trading, endpoint service, and Telegram command control.

## 1) End-to-end deployment

Dry run first:

```powershell
python scripts/deploy_agent_pipeline.py --pipeline-config config/agent_pipeline.yaml --dry-run
```

Full deploy and start trading job:

```powershell
python scripts/deploy_agent_pipeline.py --pipeline-config config/agent_pipeline.yaml
```

Force refresh stage:

```powershell
python scripts/deploy_agent_pipeline.py --pipeline-config config/agent_pipeline.yaml --refresh
```

Skip refresh stage:

```powershell
python scripts/deploy_agent_pipeline.py --pipeline-config config/agent_pipeline.yaml --no-refresh
```

Deploy without starting trading job:

```powershell
python scripts/deploy_agent_pipeline.py --pipeline-config config/agent_pipeline.yaml --no-start-job
```

Retrain selected targets during deploy:

```powershell
python scripts/deploy_agent_pipeline.py --pipeline-config config/agent_pipeline.yaml --retrain 7h:ulr,3h:nbeats
```

FTMO profile deploy using the live FTMO account already connected through the local MT5 terminal:

```powershell
$env:MT5_FTMO_LOGIN="531158622"
$env:MT5_FTMO_PASSWORD="<set-your-ftmo-password>"
python scripts/deploy_agent_pipeline.py --pipeline-config config/agent_pipeline_ftmo.yaml --dry-run
```

## 2) Trading job operations

Start trading job:

```powershell
python app.py trading-job start
```

Start with explicit plan model:

```powershell
python app.py trading-job start --plan-model ulr
```

Resume Agent B session from state:

```powershell
python app.py trading-job resume
```

Stop trading job gracefully:

```powershell
python app.py trading-job stop
```

Stop one specific trading job gracefully:

```powershell
python app.py trading-job stop --job-id YOUR_JOB_ID
```

Hard-kill all active trading jobs:

```powershell
python app.py trading-job kill
```

Hard-kill one specific trading job:

```powershell
python app.py trading-job kill --job-id YOUR_JOB_ID
```

Runtime state file:

- reports/runtime/trading_job_state.json

## 3) Endpoint service operations

Default policy is on-demand startup (not always-on). Keep deploy endpoint stage disabled unless explicitly required.

Start service manually:

```powershell
python scripts/local_signal_endpoint_service.py
```

Health check:

```powershell
python -c "import requests; print(requests.get('http://127.0.0.1:8000/health', timeout=5).json())"
```

## 4) Telegram command listener

Start listener (always-on process):

```powershell
python scripts/telegram_command_listener.py --trading-config config/trading_agent.yaml
```

Start the FTMO listener in parallel with its own runtime scope and command prefix:

```powershell
$env:MT5_FTMO_LOGIN="531158622"
$env:MT5_FTMO_PASSWORD="<set-your-ftmo-password>"
python scripts/telegram_command_listener.py --trading-config config/trading_agent_ftmo.yaml
```

Start listener in a visible popup console (Windows):

```powershell
Start-Process -FilePath "cmd.exe" -WorkingDirectory "C:\Users\artur\3D Objects\tsmm\TSMM_beginnerVersion\TSMM" -ArgumentList "/k C:\Users\artur\AppData\Local\Programs\Python\Python311\python.exe scripts\telegram_command_listener.py --trading-config config\trading_agent.yaml"
```

Supported Telegram commands (from allowed chat IDs):

- /tsmm help
- /tsmm commands
- /tsmm ?
- /tsmm status
- /tsmm deploy --refresh
- /tsmm deploy --no-refresh --no-start-job
- /tsmm deploy --dry-run
- /tsmm deploy stop
- /tsmm trading start
- /tsmm trading start --plan-model ulr

When both listeners are running on the same Telegram bot, use `/tsmm ...` for the default Pepperstone profile and `/ftmo ...` for the FTMO profile.
- /tsmm trading resume
- /tsmm trading status
- /tsmm trading stop
- /tsmm endpoint restart
- /tsmm ui start
- /tsmm ui stop
- /tsmm resource status
- /tsmm resource relieve

Telegram agent-chat bridge:

- Use `transfer to agent` to switch the chat into dedicated LLM mode.
- Use `say copilot <prompt>` to queue a request for the active VS Code Copilot session.
- The request is stored under `reports/runtime/copilot_bridge/requests/` and can be inspected from VS Code with `python scripts/copilot_bridge.py list`.
- Reply back to Telegram from this VS Code session with `python scripts/copilot_bridge.py reply --request-id <id> --message "..."`.

Async request tracking behavior:

- Latest async request (deploy, trading start/resume) status is posted every 2 minutes.
- A completion notification is sent once the latest async request exits.

Listener configuration is in config/trading_agent.yaml under telegram_listener.

Optional auth hardening:

- Set telegram_listener.require_secret: true
- Set telegram_listener.secret_env to an env var name (default TSMM_TELEGRAM_COMMAND_SECRET)
- Include marker #<secret> in chat commands when enabled

Audit file for all handled chat commands:

- reports/runtime/telegram_command_audit.jsonl

Steward governance instructions:

- docs/AGENT_STEWARD_INSTRUCTIONS.md

## 5) UI lifecycle (minimal-by-default)

Do not keep UIs running permanently. Launch only when explicitly requested.

Start all UIs:

```powershell
python scripts/start_all_uis.py
```

Force restart all UIs:

```powershell
python scripts/start_all_uis.py --force-restart
```

Stop all UIs:

```powershell
python scripts/stop_all_uis.py
```

## 6) Stage logs and output artifacts

Deployment summary JSON:

- reports/runtime/deployment_pipeline_last.json

Per-stage JSONL log stream:

- reports/runtime/deployment_pipeline_stage_log.jsonl

Trading reports and state:

- reports/trading_plans/
- reports/runtime/

## 7) Resource guard and recovery policy

Configured in config/trading_agent.yaml under resource_guard.

Behavior:

- If CPU or RAM is >= 95% for >= 240 seconds (4 minutes), guard triggers relief actions.
- Relief actions can stop local endpoint service and active UIs.
- Guard event log file: reports/runtime/resource_guard_events.jsonl
- Guard state file: reports/runtime/resource_guard_state.json

Post-reboot runtime recovery:

```powershell
python scripts/recover_runtime_after_reboot.py
```

Dry-run the recovery plan:

```powershell
python scripts/recover_runtime_after_reboot.py --dry-run
```

## 8) Notes on Agent A fallback behavior

If Agent A returns hold/no-trade, fallback attempts are evaluated from config/trading_agent.yaml:

- agent_a_fallback.attempts[0]
- agent_a_fallback.attempts[1]
- agent_a_fallback.attempts[2]

Automatic fallback discovery is also supported:

- agent_a_fallback.auto_discover: true
- agent_a_fallback.target_families: [high, low, close, open]
- agent_a_fallback.prefer_timeframes: ordered list for shorter-timeframe checks
- agent_a_fallback.max_attempts: cap for auto-generated attempts

Each attempt can override CONFIG_PATH and selected model. Attempt outcomes are persisted into trading job state under agent_a_fallback_attempts.

## 9) Operation feedback datastore and weekly summary

Operation feedback is enabled by default in `config/trading_agent.yaml` and `config/trading_agent_ftmo.yaml` under `operation_feedback`.

Purpose:

- Capture time-sensitive lifecycle evidence for each operation (Agent A planning, approval/order flow, Agent B supervision samples, close outcomes).
- Preserve performance snapshots for drift analysis (`confidence`, `cm_accuracy`, `success_probability`, `input_fooling_risk`, backtest priors).
- Keep date-partitioned logs and per-job logs for fast scanning by date or job id.

Default storage layout (runtime-scoped):

- Daily JSONL: `reports/runtime[/ftmo]/operation_feedback/daily/YYYY/MM/DD/operations_YYYYMMDD.jsonl`
- Per-job JSONL: `reports/runtime[/ftmo]/operation_feedback/by_job/<job_id>.jsonl`

Generate weekly summary (JSON + Markdown):

```powershell
python scripts/operation_feedback_weekly_summary.py --days 7
```

Optional explicit range:

```powershell
python scripts/operation_feedback_weekly_summary.py --start-date 2026-05-22 --end-date 2026-05-28
```

Default summary outputs:

- `reports/runtime[/ftmo]/operation_feedback/weekly/operation_feedback_weekly_YYYYMMDD.json`
- `reports/runtime[/ftmo]/operation_feedback/weekly/operation_feedback_weekly_YYYYMMDD.md`

## 10) Multi-asset master tables (US500 integration)

The SQLite market layer now supports symbol-scoped master tables and views in one DB family.

Naming convention:

- XAUUSD (backward compatible): `ohlc_1m`, `ohlc_10m`, `ohlc_30m`, ...
- Other assets: `ohlc_1m_<symbol_lower>`, `ohlc_10m_<symbol_lower>`, `ohlc_30m_<symbol_lower>`, ...

Import US500 HISTDATA folder tree (recursive):

```powershell
python scripts/migrate_market_data_to_sqlite.py --master-dir "C:/Users/USUARIO/Documents/DataBuild" --master-glob "**/*.csv" --db-path data/market_data_us500.sqlite --symbol US500 --views 10,30,60,180,420,720,1440,10080
```

Optional trading config update during import:

```powershell
python scripts/migrate_market_data_to_sqlite.py --master-dir "C:/Users/USUARIO/Documents/DataBuild" --db-path data/market_data_us500.sqlite --symbol US500 --views 10,30,60,180,420,720,1440,10080 --update-trading-config --trading-config config/trading_agent.yaml
```

Config keys for symbol selection:

- `dashboard.master_table_path`: SQLite DB path for the runtime/profile.
- `dashboard.sql_symbol`: symbol namespace used when reading from SQLite.
- `config/config.yaml -> sql_symbol`: app-level symbol for status/snapshot queries.
