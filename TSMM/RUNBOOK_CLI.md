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
- /tsmm trading resume
- /tsmm trading status
- /tsmm trading stop
- /tsmm endpoint restart
- /tsmm ui start
- /tsmm ui stop
- /tsmm resource status
- /tsmm resource relieve

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

- AGENT_STEWARD_INSTRUCTIONS.md

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
