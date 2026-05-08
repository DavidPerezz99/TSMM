# TSMM Steward Agent Instructions

Purpose: governance and execution policy for command and natural-language backend operations.

## 1) Safety and authorization policy

- Accept commands only from allowed chat IDs configured in config/trading_agent.yaml under telegram_listener.allowed_chat_ids.
- If telegram_listener.require_secret is true, enforce secret marker token in each command using env var defined by telegram_listener.secret_env.
- Reject unknown root commands not listed in telegram_listener.allowed_commands.
- Never execute arbitrary shell commands from chat.

## 2) Command scope and behavior

Allowed root commands:

- status
- deploy
- trading
- endpoint
- ui
- resource

Behavior:

- deploy: run scripts/deploy_agent_pipeline.py with safe flags only.
- trading: allow start/resume/stop via app.py CLI.
- endpoint: restart local signal endpoint only.
- ui: start/stop UIs on explicit user request only.
- resource: inspect resource status or trigger immediate relief check.

## 3) Natural-language interface policy

- Accept non-prefixed plain-language messages when telegram_listener.allow_natural_language is true.
- Map natural-language intents only to approved command tails under allowed root commands.
- Never treat free-form chat as arbitrary shell input.
- For ambiguous messages, respond with a clarification and examples instead of executing actions.
- Preserve explicit command behavior (/tsmm prefix) without regression.

Examples of valid natural-language intents:

- start trading -> trading start
- resume trading -> trading resume
- stop trading -> trading stop
- deploy with refresh -> deploy --refresh
- stop deploy -> deploy stop
- show trading status -> trading status
- restart endpoint -> endpoint restart
- start ui / stop ui
- resource status / resource relieve

## 4) Audit and traceability

- Log each handled command to reports/runtime/telegram_command_audit.jsonl with timestamp, chat_id, raw text, parsed text, decision outcome.
- Record resource relief actions in reports/runtime/resource_guard_events.jsonl.
- Persist pipeline stage logs in reports/runtime/deployment_pipeline_stage_log.jsonl.
- Log conversational exchanges in reports/runtime/telegram_conversation_log.jsonl.
- If a message is parsed from natural language, include parsed marker and routed command tail in audit fields when available.

## 5) Async request status and completion policy

- Long-running actions (at minimum deploy, trading start, trading resume) should run asynchronously.
- Track the latest async request metadata (pid, command, start time, runtime paths) in listener state.
- Send progress updates for the latest request at telegram_listener.latest_request_status_interval_seconds cadence.
- Send one completion notification when the tracked request finishes.

## 6) Resource governance policy

- Use resource_guard thresholds from config/trading_agent.yaml.
- If CPU or RAM sustains threshold breach for configured duration, automatically relieve load by shedding non-critical services:
  - local endpoint service
  - UI processes
- Do not auto-launch UIs after relief; require explicit request.

## 7) Endpoint lifecycle policy

- Local endpoint service is on-demand.
- Do not keep endpoint always-on unless explicitly configured for a run.
- If endpoint call is needed and service is down, start service lazily and retry request.

## 8) Operational expectations

- Keep trading runtime deterministic for execution logic; LLM remains advisory.
- Backtests must remain technical-only.
- Prefer reversible actions and clear operator visibility through logs.
