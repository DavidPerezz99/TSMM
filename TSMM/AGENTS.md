# Coding Agent Notes

## Repository Organization

Keep the project root intentionally small. Root files are reserved for the main
CLI, dependency/bootstrap files, Windows launchers, and top-level project
documentation.

Place new work according to these ownership boundaries:

- `apps/`: interactive Dash applications and other user-facing app entry points;
- `config/`: runtime, trading, provider, validation, and sweep configuration;
- `docs/`: manuals, runbooks, references, architecture notes, and handoffs;
- `models/`: forecasting model implementations;
- `scripts/`: operational services, maintenance jobs, migrations, and analysis
  commands;
- `tests/`: automated tests;
- `tools/`: developer-facing modeling, Hypersearch, inspection, and verification
  utilities;
- `tools/legacy/`: retained older entry points that are not part of the primary
  runtime;
- `utils/`: shared application, trading, broker, data, and reporting modules.

Do not move secondary scripts or documentation back into the repository root.
When moving an entry point, update subprocess commands, documentation, launcher
process matching, project-root discovery, and asset/config paths together.

## Runtime and Generated Content

Keep generated or operational content under `data/`, `logs/`, `model_files/`,
`output/`, and `reports/`. Do not commit runtime feedback, broker state, generated
reports, caches, or temporary diagnostics unless explicitly requested.

Use descriptive temporary locations outside the root. Root `.tmp*` files were
removed and are ignored. `.pytest_cache/` is also ignored.

The untracked `chat_sessions/` directory may contain user-owned session data.
Do not move, delete, or stage it without explicit approval.

## Current Trading Enhancement Branch

The branch `trading-analysis-enhancements` includes operational feedback
analysis and guarded delayed stop-loss support. Delayed protection is disabled
by default and must not be enabled in production without paper/demo validation,
MAE/MFE analysis, calibrated entry confidence, and account-level loss guards.

Read these files before changing that behavior:

- `docs/TRADING_ANALYSIS_ENHANCEMENTS_HANDOFF.md`
- `README.md`
- `config/trading_agent.yaml`
- `config/trading_agent_ftmo.yaml`
- `utils/trading_job.py`
- `utils/investing_agent.py`

`scripts/full_horizon_report.py` is a recurring inference path, not a training
report. Keep its market refresh, live cache-tail merge, latest-`n_steps` input
selection, atomic output, and freshness metadata intact. `m_steps` is the number
of future outputs and must never be used to offset the inference input window.
`horizon` is the requested future path length; when it exceeds `m_steps`, use
`utils/recursive_inference.py` to recursively produce every step. Do not flatten
auxiliary target features into the primary time horizon or reduce the configured
six-step path to the model's one-step output.
Keep `r2_train`, `inference_strength`, and `r2_live_rolling` semantically
separate. Live R2 must use only matured `y_diff` forecasts matched to completed
future candles. Preserve both lineage-level rolling R2 across retrains and the
separate exact-artifact metric.

The tracked deployment and operational audit are documented in
`docs/TRADING_ANALYSIS_ENHANCEMENTS_HANDOFF.md`. Health figures in that document
are dated snapshots, not permanent truths; re-run live checks before acting.
Important current cautions include contaminated job-registry state, workers that
may survive programmed-order cancellation, and a parity process that can become
a no-op when source and target resolve to the same MT5 account.

Do not use `scripts/recover_runtime_after_reboot.py` as a routine restart while
the registry is contaminated because it may resume a stale job. Before any
runtime restart, verify MT5 positions and pending orders directly, then restart
only the intended services. The full test suite can also contaminate production
runtime state unless tests use an isolated runtime root.

Poor training/live R2 does not currently hard-block the base trading plan:
`mode_a.block_on_confidence_thresholds` and
`mode_a.block_on_input_fooling_risk` are both false. Observed missing entries
were caused by consensus-based pending-order cancellation followed by disabled
autonomous follow-up. Preserve that distinction when diagnosing trading
behavior.

The active entry policy was expanded on 2026-07-14. Opposing countertrades are
policy exceptions that bypass approval even when they are auto-created. The
primary profile permits up to three concurrent autonomous jobs per session and
up to three Agent B-supervised follow-up scans, but follow-up plans still pass
the configured quality filters and account risk checks. Agent B does not submit
orders itself; `mode_b.allow_open_new_positions` permits it to trigger a new
Agent A scan through the listener.

Manual broker positions are human-owned. Both trading profiles set
`telegram_listener.adopt_manual_entries` and
`telegram_listener.adopt_any_running_positions` to `false`. Do not adopt,
modify, seed SL/TP on, close, or otherwise manage a position unless it is
clearly TSMM-tagged or the user explicitly transfers ownership to TSMM.

## Verification

After structural changes, verify that no stale paths remain and run at least:

```powershell
.venv\Scripts\python.exe -m py_compile <changed-python-files>
.venv\Scripts\python.exe -m unittest discover -s tests
git diff --check
```

The full test suite currently has known runtime-scope/shared-state isolation
failures. Do not silently attribute those baseline failures to unrelated work;
run affected test modules independently to distinguish regressions from existing
suite contamination.
