# TSMM Trading Analysis Enhancements Handoff

For the 2026-09-02 R2 evaluation correction, bounded recalibration campaigns,
audited US500 continuation, and lag-safe cross-asset experiments, see
`docs/MODEL_RELIABILITY_AND_RECALIBRATION.md` and
`docs/US500_AND_CROSS_ASSET_MODELS.md`.

## 2026-09-02 US500, search fallback, and opportunity-policy update

- `scripts/refresh_market_assets.py` now maintains XAUUSD and an audited
  SPY-derived US500 continuation. Raw SPY remains namespaced and every derived
  US500 interval is recorded in `market_data_provenance`.
- `config/inference_us500.yaml` isolates US500 deployments, inference metrics,
  service port, and runtime output from XAUUSD.
- The US500 and bidirectional cross-asset experiment sessions cover all 28 OHLC
  endpoints, enforce 10,000-record/400-experiment/20-GB limits, and use
  backward-only timestamp joins.
- A search that does not reach R2 > 0.60 preserves the best reliable
  rolling-origin package after its endpoint budget as
  `fallback_best_available`. It is not automatically a champion and requires
  explicit fallback acknowledgement to activate.
- Agent A entry direction now uses a weighted 3h/7h/12h/24h ensemble; 10m/30m/1h
  confirm timing. Agent B has a separate short-horizon-weighted management
  consensus. Bounded hourly opportunity scans use programmed range entries and
  backtest reports now expose entry rejection and pending-terminal counts.
- Both broker profiles remain disabled. No terminal or autonomous service was
  started as part of this update.

### Offline replay evidence

The final 15-trading-day replay for 2026-08-12 through 2026-09-01, using a
$5,000 starting balance, produced 23 filled trades from 299 policy attempts:
16 wins, 7 losses, 69.57% win rate, $14.16 net P&L, 0.283% return, 1.139 profit
factor, and 0.964% maximum drawdown. The same period had previously produced no
fills because conviction scaling reduced 0.01-lot plans below the broker's
minimum. The guard now floors such a plan to exactly the minimum lot only when
that lot still fits the monetary risk allowance.

This result is graded `exploratory retrospective current-model replay`, not
out-of-sample evidence: the current model packages lack pre-period training
cutoffs and may have learned from some of the replayed history. Keep autonomous
trading disabled until recalibrated packages with explicit lineage pass strict
point-in-time or walk-forward evaluation and paper/demo validation.

## Scope

This document describes the work performed on branch
`trading-analysis-enhancements` to improve TSMM operational analysis,
Agent B risk management, delayed stop-loss safety, and repository hygiene.

The objective was to use the accumulated operation-feedback data to identify
weaknesses in trade selection and management while preparing a guarded version
of delayed stop-loss protection. The delayed-stop feature is implemented but
remains disabled by default.

## Branch And Deployment State

The tracked implementation is on `trading-analysis-enhancements`, with no
`codex` prefix. The original analysis, risk-management, cleanup, and repository
organization work was committed as `5ebd11a`. Live inference freshness,
recursive horizons, and online R2 evaluation were committed and pushed as
`7aba69e` on 2026-07-13.

At the end of that deployment, local `HEAD` and
`origin/trading-analysis-enhancements` both resolved to `7aba69e`. The local
signal endpoint and primary Telegram listener were restarted from that checkout
and passed live readiness checks. Future documentation-only commits may move the
branch tip, so always verify the deployed branch and commit rather than assuming
this snapshot remains current.

Preserve unrelated user work and inspect `git status` before staging. The
untracked `chat_sessions/` directory is not part of this implementation and
must not be staged, moved, or deleted without explicit approval.

The pre-existing edit near `_launch_followup_agent_a_start` in
`utils/trading_job.py` was already present before this work. Do not revert it as
part of this feature.

## Operational Analysis Added

### Full-history analyzer

Added `scripts/analyze_operation_feedback.py`.

The script reads operation-feedback JSONL files and writes:

- one job-level CSV;
- one machine-readable summary JSON;
- one Markdown summary.

It reports event/source counts, latest job states, terminal outcomes, Agent A
metric distributions, Agent B recommendations, confirmed risk actions, close
reasons, and good/bad terminal metric comparisons.

Example commands:

```powershell
.venv\Scripts\python.exe scripts\analyze_operation_feedback.py `
  --feedback-root reports\runtime\operation_feedback `
  --out-dir reports\analysis\operation_feedback `
  --prefix pepperstone

.venv\Scripts\python.exe scripts\analyze_operation_feedback.py `
  --feedback-root reports\runtime\ftmo\operation_feedback `
  --out-dir reports\analysis\operation_feedback `
  --prefix ftmo
```

Generated reports are under `reports/analysis/operation_feedback/`. The entire
`reports/` tree is ignored by Git.

### Terminal-outcome correction

Updated `scripts/operation_feedback_weekly_summary.py` and the new analyzer so
terminal results are selected from the latest terminal event for each job. A
later running or notification event can no longer erase an earlier completed
outcome from the analysis.

### Agent B feedback correction

Updated `utils/operation_feedback_store.py` to record:

- `should_close` and `close_reason` consistently;
- `risk_action_proposed` from the current management plan;
- `risk_action` only when the broker update was successful;
- previous, requested, and proposed SL/TP levels.

This distinction matters because Agent B feedback is emitted before the current
broker modification is attempted. Treating the proposal as an applied action
would corrupt future training labels.

## Delayed Stop-Loss Protection

### Configuration

Added the following under `execution` in both trading configurations:

```yaml
delayed_stop_loss:
  enabled: false
  allowed_conviction_modes:
  - no_sl
  min_conviction: 0.8
  max_volume_multiplier: 1.0
  max_unprotected_seconds: 900
  max_unprotected_adverse_pct: 0.25
```

The FTMO profile uses tighter limits of 300 seconds and 0.15% adverse movement.
Both profiles keep `enabled: false`.

### Order behavior

The helpers in `utils/trading_job.py` now:

- convert a missing SL to the normal percentage-based fallback when delayed
  protection is disabled or the plan is ineligible;
- allow an initial broker SL of `0.0` only when delayed protection is enabled,
  the conviction mode is allowed, and conviction meets the configured minimum;
- cap delayed-protection volume at
  `execution.default_volume * max_volume_multiplier`;
- persist requested/effective volume and planned protection metadata in job
  state so Agent B can recover it after a restart.

This intentionally prevents the existing high-conviction `no_sl` mode from
combining an unprotected interval with its original 1.5x volume increase.

### Agent B behavior

Agent B can attach the planned/fallback stop when any of these conditions occur:

- maximum unprotected time is reached;
- maximum adverse movement is reached;
- the current assessment becomes defensive.

A fail-safe check runs before market-data synchronization. Therefore a data-sync
failure does not bypass the time/adverse protection limits.

Broker risk modifications now have explicit attempt tracking:

- failed modifications are retried on later Agent B cycles;
- successfully applied duplicate modifications are suppressed;
- successful fail-safe changes are mirrored to the paired account when account
  mirroring is configured.

### MT5 adapter hardening

`utils/investing_agent.py` now accepts optional SL/TP values for market and
programmed orders. `None` becomes MT5's `0.0` no-level value.

`modify_position_risk` normalizes requested levels against live bid/ask,
symbol digits, broker stop distance, and freeze distance. This handles the case
where price has already crossed the originally planned delayed stop. If no live
price is available, normalization is skipped rather than accidentally clearing
the existing protection.

## Repository Cleanup

Added `.tmp*` to `.gitignore` and removed all 37 `.tmp*` entries from the
repository root after verifying that none were referenced by runtime code,
startup scripts, tests, or configuration.

Most were one-off broker audits, hard-coded ticket/job diagnostics, listener
restart helpers, generated snapshots, and an obsolete NVIDIA installer helper.
`.tmp_adapter_postfix_smoke.py` was tracked, so its deletion appears in Git.

### Root directory organization

The project root was subsequently reorganized to separate concerns:

- Dash applications moved to `apps/`;
- Hypersearch, search-mode, result-selection, and verification utilities moved
  to `tools/`;
- the legacy Snowflake forecasting entry point moved to `tools/legacy/`;
- manuals, runbooks, references, command notes, and handoffs moved to `docs/`;
- `test_sweep.yaml` moved to `config/test_sweep.yaml`;
- an empty `.requirements_to_install.txt` marker and generated `.pytest_cache/`
  were removed;
- a root `README.md` now explains the directory structure and common commands.
- a root `AGENTS.md` records the directory conventions and safety notes for
  future coding agents.

All moved Python entry points explicitly resolve the project root. UI launchers,
Hypersearch child-process paths, documentation commands, and Dash asset-folder
configuration were updated for the new locations. Common commands now include
`python apps/dashboard.py`, `python apps/ui.py`, and
`python tools/hypersearch.py`.

## Tests Added or Updated

Added `tests/test_delayed_stop_loss.py` to cover:

- standard fallback when delayed protection is disabled;
- zero initial SL for an eligible delayed-protection plan;
- minimum-conviction enforcement;
- volume capping;
- Agent B attachment after adverse movement.

Updated tests for:

- failed risk-update retry and successful deduplication;
- MT5 normalization of a stop already crossed by live price;
- proposed versus broker-confirmed risk feedback;
- terminal-event selection in both analyzers;
- Agent B close-outcome adapter behavior.

Focused verification command set:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_delayed_stop_loss.py"
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_agent_b_risk_management.py"
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_operation_feedback_store.py"
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_operation_feedback_summary.py"
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_mt5_adapter.py"
```

All 24 focused tests pass. Python compilation for the modified modules also
passes.

The full suite was run once: 114 of 121 tests passed. The failures were in
existing runtime-scope and shared-state isolation behavior. In isolation,
`test_trading_job_resume.py` passes; `test_autonomous_trading.py` retains two
failures involving FTMO runtime-scope resolution and stale terminal-state
refresh. These were not caused or fixed by this change set.

## Important Limitations

Do not enable delayed protection in production yet.

- A Python process, MT5 terminal, network, or machine failure during the
  unprotected interval still leaves no broker-side stop.
- Current conviction is a heuristic score, not a calibrated realized-win
  probability. Historical success probability is close to 0.50 and observed
  confusion-matrix accuracy is below 0.50 on average.
- MAE/MFE paths and counterfactual stop-out labels have not been reconstructed.
  There is no evidence yet that delayed protection improves net expectancy
  after spread, slippage, and tail losses.
- An account-level floating-loss and daily-loss circuit breaker should exist
  before considering delayed protection for FTMO.
- The root temporary files are gone, but generated reports and runtime evidence
  must remain under their ignored data directories.

## Full-Horizon Inference Freshness

`scripts/full_horizon_report.py` was corrected after the original enhancement
commit to ensure its recurring report is independent of model retraining.

Two stale-data defects were found:

- `query_ohlc()` preferred materialized timeframe cache tables that were built
  during retraining but never refreshed with new minute rows;
- inference discarded the newest `m_steps` rows, incorrectly treating forecast
  horizon length as an input lag.

The market query now merges cached history with a live aggregation beginning at
the cache's final bucket. Inference always consumes the newest `n_steps`; the
model still emits `m_steps` future values. The local endpoint uses the same
window rule and returns its complete horizon plus inference timestamps.

The primary Telegram listener launches the report every 300 seconds by default
through `listener.inference_report_interval_seconds`. The FTMO profile disables
its duplicate launcher so both listeners do not write the same report at once.

Each report now records:

- report start/completion time;
- minute-source `data_as_of` time;
- timeframe bucket and input-window bounds;
- an input fingerprint;
- model path and model modification time;
- dynamic confidence and static training R2 semantics;
- market-refresh result.

The report's `horizon` is the configured future path, not the flattened set of
target features from one model call. Production configs currently use
`m_steps: 1` and `horizon: 6`, so inference recursively generates six candles:
each predicted `y_diff` and auxiliary target feature is fed into the next input
window. `feature_horizons` preserves the same six-step path for every predicted
target feature. Keep the report and local signal endpoint on the shared
`utils/recursive_inference.py` implementation.

Online inference evaluation is persisted in
`reports/runtime/full_horizon_metrics.sqlite`. Every issued primary `y_diff`
forecast is matured only after its corresponding future timeframe candle is
complete. The report exposes:

- `r2_train`: static R2 from the newest completed training evaluation;
- `inference_strength`: dynamic forecast-magnitude versus volatility heuristic;
- `r2_live_rolling`: rolling R2 from matured live forecasts;
- `r2_live_samples`: distinct origin candles currently supporting live R2;
- `r2_live_delta`: movement from the previous metric snapshot.

Primary rolling live R2 follows the model lineage (timeframe, family, and model
type) across retraining generations so higher timeframes can accumulate enough
samples despite daily retraining. `r2_live_current_model` and its sample count
show the exact current artifact separately. Repeated five-minute inferences
within one partial timeframe candle are retained for audit, but only the latest
forecast from that origin candle is included in R2 to avoid overweighting one
realized outcome. The current defaults require 10 matured origin candles and use
a 100-sample window.

Writes are atomic, and `reports/runtime/full_horizon_report_status.json` records
running, completed, or failed state. R2 remains static until a newer completed
training evaluation exists. Confidence is recomputed every inference, although
its numeric value may repeat when forecast magnitude and recent volatility have
not materially changed.

## Deployment Verification

The 2026-07-13 deployment used the following sequence:

1. Verify `trading-analysis-enhancements` is checked out and tracks the branch
   with the same name on `origin`.
2. Compile all changed inference modules and run focused tests.
3. Stage only tracked enhancement files; exclude `chat_sessions/` and runtime
   outputs.
4. Commit and push the enhancement branch.
5. Confirm MT5 has zero open positions and zero pending orders before restarting
   runtime processes.
6. Stop the old endpoint, listener, and canceled session workers.
7. Start `scripts/local_signal_endpoint_service.py` and
   `scripts/telegram_command_listener.py` from the deployed checkout.
8. Verify `/health`, Telegram network connectivity, MT5 connectivity, and a real
   full-horizon report.

The post-deployment report produced 28 successful model/family rows, zero
errors, six future steps per row, and a healthy online evaluator. MT5 was
authenticated with zero positions and zero pending orders. The endpoint loaded
all seven configured timeframes. Process IDs are intentionally omitted because
they are ephemeral.

Focused verification for the inference deployment passed eight tests:

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_inference_window_freshness.py"
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_inference_performance.py"
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_market_db_multi_asset.py"
```

## Operational Health Snapshot

The following findings are a point-in-time snapshot from 2026-07-13 and must be
re-measured before making a new operational decision.

Healthy components at inspection time:

- market minute data was current and Tiingo refreshes were succeeding;
- the market and inference-metrics SQLite databases passed integrity checks;
- the endpoint, Telegram listener, Ollama, MT5 terminal, scheduled model
  refresh, and report scheduler were running;
- MT5 authentication succeeded and there was no live exposure;
- the recurring report and online forecast maturation were current.

Trading-quality concerns at inspection time:

- only 6 of 28 latest training R2 values were positive;
- live rolling R2 was available for 16 model/family lines and 15 were negative;
- several retraining logs warned that the discriminator had insufficient
  samples;
- R2 is a regression-quality diagnostic, not direct proof of trade P/L, but the
  breadth of negative scores is a material warning against unrestricted
  autonomous execution.

Current policy does not directly block a base 7h plan when model safeguards are
breached. In `config/trading_agent.yaml`, both
`mode_a.block_on_confidence_thresholds` and
`mode_a.block_on_input_fooling_risk` are `false`. Do not claim that poor R2 by
itself is why no entry was placed.

The observed no-entry sequence was:

- the London session created a programmed entry near 4018.10;
- pending-order maintenance canceled it when refreshed consensus changed;
- the New York session created another programmed entry near 4005.28;
- maintenance canceled that order after another consensus invalidation;
- `autonomous_trading.followup_enabled` was `false`, so neither cancellation
  received a replacement entry.

The practical causal chain was weak or unstable predictions, followed by a
consensus reversal, order cancellation, and no follow-up. Model quality
contributed indirectly by making consensus unstable.

## Known Runtime Issues

- `reports/runtime/trading_job_registry.json` contained hundreds of entries,
  including temporary test-runtime paths and stale jobs. The global
  `trading_job_state.json` also pointed at an obsolete failed job while newer
  jobs existed. Treat active-job monitoring as unreliable until registry scope
  and cleanup are repaired.
- Two canceled autonomous-session workers remained alive and idle after their
  pending orders were removed. They were stopped during deployment, but the
  lifecycle defect remains in code: a canceled programmed order should make its
  worker exit and close its state cleanly.
- `scripts/recover_runtime_after_reboot.py` can resume jobs that the contaminated
  registry labels active. Do not use it for a routine service restart until the
  registry has been audited; restart endpoint/listener explicitly after checking
  broker exposure.
- The parity enforcer was launched with `config/trading_agent.yaml` as source
  and `config/trading_agent_ftmo.yaml` as target, but both resolved to the same
  FTMO account. It therefore compared FTMO to itself and continuously grew a
  large no-op log. Disable it when only one account is intended, or restore a
  genuinely distinct source account.
- The full test suite has shared production-runtime contamination failures. Run
  focused tests, or set an isolated runtime root before running the entire suite
  on a machine hosting live TSMM services.

No safety configuration was changed during the health audit or deployment.
Delayed stop protection remained disabled, and model-quality blocks remained at
their existing values.

## Entry Throughput And Manual-Trade Ownership

The 2026-07-14 entry-policy update made these behavioral changes:

- auto-created opposing countertrades now evaluate their explicit auto-approval
  policy before the generic follow-up approval rule, so they do not wait for a
  Telegram approval when `auto_approve_opposing_countertrade` is enabled;
- follow-up Agent A scans no longer require explicit approval in the two tracked
  trading profiles;
- the active profile permits three concurrent autonomous jobs per session and
  up to three Agent B-supervised follow-up scans;
- `mode_b.allow_open_new_positions` is now enforced by the listener and permits
  Agent B supervision to trigger a new Agent A scan; Agent B still does not
  submit broker orders directly;
- autonomous follow-up plans still pass the configured success-probability,
  confidence, confusion-matrix accuracy, input-fooling-risk, consensus, and
  account-level risk checks before submission;
- both profiles explicitly exclude manual and otherwise untracked broker
  positions from automatic adoption.

Manual positions are intentionally managed by the human operator. Agents must
not attach SL/TP, rebind, close, or adopt a magic-0/unlabeled position unless the
operator explicitly transfers that individual trade to TSMM. These policy
changes do not modify positions or pending orders that already exist at the
broker, and delayed stop protection remains disabled.

## Recommended Next Work

1. Build per-trade lifecycle records joining Agent A plan, order submission,
   fill, Agent B decisions, broker modifications, and final realized P/L.
2. Reconstruct MAE, MFE, time-to-MAE, time-to-MFE, and counterfactual outcomes
   from stored market data.
3. Backtest standard, volatility/structure-based, and delayed-stop policies on
   identical entry signals, including spread and slippage.
4. Calibrate a meta-label model that estimates whether to take a signal; do not
   use raw model confidence as realized probability.
5. Add account-level risk guards and an external watchdog that can attach a
   broker stop independently of Agent B's main process.
6. Run delayed protection only in paper/demo mode, then compare expectancy,
   tail loss, drawdown, and stop-out recovery against the standard policy.

## Joint OHLC Entry Upgrade (2026-08-30)

The live Agent A path and model-backed replay now share an optional
`signal_policy` configured in all three tracked trading profiles. It replaces
the former single-HIGH entry side with a weighted 7h OPEN/HIGH/LOW/CLOSE vote,
requires two of 10m/30m/1h to confirm timing, uses the predicted HIGH/LOW range
for programmed entry placement, and derives bounded SL/TP distances from 10m
ATR and realized volatility.

Programmed-order market fallback is restricted to configured triggers. Before
conversion it reassesses the joint policy, requires the same side and minimum
score, confirms pending-order cancellation, and re-runs the normal account and
prop-firm guard. No service was restarted and no broker position or order was
modified while this code was implemented.

Historical reports now include MAE/MFE timing and aggregate path diagnostics.
`--require-point-in-time-models` rejects an evaluation unless every package has
an explicit training cutoff before the period; artifact modification time is no
longer accepted as training lineage. See `docs/TRADING_SIGNAL_POLICY.md`.

The detailed logic assessment and research notes are in
`reports/analysis/trading_logic_state_upgrade_plan_20260710.md` on this
workspace, but that file is ignored by Git. This handoff is the tracked summary
that should travel with the branch.
