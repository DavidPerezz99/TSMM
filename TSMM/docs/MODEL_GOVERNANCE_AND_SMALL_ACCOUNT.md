# Model governance and small-account safety

TSMM now separates four decisions that used to be easy to confuse:

1. A bulk experiment is **worth preserving** when its primary-target R2 is strictly greater than `0.60`.
2. A preserved challenger is **statistically credible** only after chronological walk-forward validation.
3. A credible challenger is **tradeable** only after an after-cost strategy replay passes risk/return gates.
4. A tradeable challenger becomes the **champion** only when it beats the current champion; the previous champion remains available for rollback.

An R2-qualified experiment is not automatically promoted and daily refresh jobs do not replace the champion.

## Domino / bulk-search artifacts

Bulk search and resume commands accept `--worthy-r2-threshold` (default `0.6`). Runs at exactly `0.6` are not retained. Qualifying bundles are written below the session's `worthy_artifacts/` directory and include:

- the exact experiment YAML;
- the fitted model and scaler artifacts;
- evaluation metrics and forecasts;
- Python/platform, Git commit, requirements, and random seed metadata;
- dataset path/range/fingerprint metadata;
- SHA-256 checksums.

The forecasting-only `main` branch implements the same behavior without importing any trading modules.

## Leakage-safe validation

N-BEATS preprocessing is fitted only on its training period. Early stopping uses a later chronological validation block, and reported evaluation uses a separate untouched holdout of at least 20 observations. Training uses a fixed seed and restores the best validation checkpoint.

Run expanding-window retraining for a challenger:

```powershell
.venv\Scripts\python.exe tools\walk_forward_validate.py `
  --config config\high10mResults\nbeats\top1_06000.yaml `
  --folds 3 `
  --test-rows 60 `
  --output output\validation\10m_high_walk_forward.json
```

Every fold expands the historical training window, leaves a horizon gap, retrains, and scores only the later fold.

## Bounded nightly recalibration

The session remains manual-start-only, respects the local 05:00 deadline and RAM guard, and resumes its ordered endpoint list. When an experiment for an endpoint clears R2 > 0.60, its bundle is preserved and the runner advances to the next endpoint instead of spending the rest of the night searching the already-satisfied endpoint.

```powershell
.venv\Scripts\python.exe tools\experiment_session.py run `
  --config config\experiment_sessions\xauusd_nightly.yaml
```

## Champion/challenger promotion

First run the model-backed backtest with costs. Then combine its summary with the walk-forward evidence and the bundle's untouched holdout:

```powershell
.venv\Scripts\python.exe tools\build_candidate_metrics.py `
  --bundle output\hypersearch_sessions\SESSION\ENTRY\worthy_artifacts\BUNDLE `
  --walk-forward output\validation\10m_high_walk_forward.json `
  --backtest-summary reports\backtests\RUN\summary.json `
  --output output\validation\10m_high_candidate.json

.venv\Scripts\python.exe tools\model_registry.py assess `
  --metrics output\validation\10m_high_candidate.json

.venv\Scripts\python.exe tools\model_registry.py promote `
  --endpoint 10m_high `
  --bundle output\hypersearch_sessions\SESSION\ENTRY\worthy_artifacts\BUNDLE `
  --metrics output\validation\10m_high_candidate.json
```

Promotion fails closed unless all evidence exists. Defaults require non-negative holdout and median fold R2, no fold below `-0.25`, directional accuracy of at least `0.52`, profit factor of at least `1.10`, positive expectancy, drawdown no higher than `15%`, at least 30 trades, and an R2 improvement over the incumbent.

Rollback is atomic:

```powershell
.venv\Scripts\python.exe tools\model_registry.py rollback --endpoint 10m_high
```

## Trading admission and a USD 100 account

Forecasts are evidence, not orders. Refreshed R2 replaces the historical filename score when available; a negative refreshed R2 gives that model zero consensus weight. The hybrid gate also requires enough qualified models, meaningful consensus, a hard stop, positive expected value after spread/slippage, and a sane forecast range. Failure means `hold`.

At order time, TSMM reads live equity and the broker's minimum/step volume, estimates the loss at the stop for one lot, and rounds the permitted size **down**. If the broker minimum lot would exceed the configured percentage risk, the order is skipped. Conviction may reduce size but cannot remove or widen the hard stop.

Set `shadow_mode.enabled: true` to record the complete plan and terminate the job without connecting to or submitting anything to MT5. This should be used before enabling a Pepperstone demo or live profile.

No software can make trades that “mostly” profit or guarantee that USD 100 becomes USD 1,000. For XAUUSD in particular, a 0.01-lot broker minimum will often be too large for a USD 100 account under a 0.5% risk budget; skipping those trades is intentional. A low-spread major FX pair may fit the size better, but it still requires its own data, models, walk-forward results, and paper validation.
