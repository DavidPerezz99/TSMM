# Model reliability and bounded recalibration

## Why the former R2 values were unreliable

The legacy production YAML files commonly reserve only three observations for
testing. R2 computed from three points is not a stable estimate of future model
quality. A second mismatch made bulk-search scores unnecessarily pessimistic:
the evaluator recursively generated the entire holdout even though production
endpoints receive newly observed candles between one-step calls. ULR also fitted
its feature and target scalers before its chronological split, allowing holdout
distribution information into preprocessing.

The corrected protocol is `rolling_origin_one_step`:

1. Keep at least 120 latest observations untouched.
2. Fit preprocessing and the model only on earlier observations.
3. For every holdout timestamp, predict from only the actual rows available
   immediately before that timestamp.
4. Record R2 together with sample count, direction accuracy, zero-change MAE,
   and MAE skill versus that zero-change baseline.

A bundle is retained only when R2 is strictly above 0.60, it contains at least
60 holdout samples, it uses the corrected protocol, and it beats the zero-change
MAE baseline. Passing this gate creates a candidate, not an automatic production
promotion; walk-forward and after-cost strategy validation still apply.
Bulk runs keep each confidence discriminator inside its candidate artifact
package; they no longer write experimental discriminators into the active
global model directory.

If an endpoint reaches its configured search budget without crossing 0.60,
TSMM ranks only successful rolling-origin runs with at least 60 samples,
reproduces the best finite-R2 run, and exports it as
`fallback_best_available`. The bundle manifest keeps its true score and says
`qualified_candidate: false`; it is never silently promoted as a champion.
This prevents a long search from losing its best reproducible result while
preserving the difference between "best seen" and "good enough." A fallback
requires an explicit `model_registry activate --allow-fallback` action before
the trading runtime can use it.

## Resource boundaries

`config/experiment_sessions/xauusd_nightly.yaml` enforces the current machine
policy:

- no more than 10,000 records in any experiment;
- no more than 400 generated experiments per endpoint;
- one experiment process at a time;
- six numerical-library CPU threads;
- a 20 GB process RAM ceiling and 2 GB system reserve;
- manual start and resumable state;
- a local 05:00 hard stop.

The current baseline plan contains 28 XAUUSD OHLC endpoints and 8,748 possible
experiments. Each endpoint stops early as soon as a qualifying bundle is found.
The large total is therefore a ceiling, not the expected executed count.

US500 and XAUUSD/US500 comparison sessions are also bounded to 10,000 records,
400 experiments per endpoint, and preserve the best available package after
160 unsuccessful qualified-candidate attempts. They run at least 40 successful
experiments per endpoint before early stopping so baseline and cross-asset
recipes are both sampled. Their compact matrices produce
28 endpoints and 5,832 planned configurations each; resolving an endpoint early
marks its remaining configurations as intentionally skipped in progress output.

Preview and resume the campaign from the repository root:

```powershell
.venv\Scripts\python.exe tools\experiment_session.py plan --config config\experiment_sessions\xauusd_nightly.yaml
.venv\Scripts\python.exe tools\experiment_session.py run --config config\experiment_sessions\xauusd_nightly.yaml --max-experiments 10
```

Omit `--max-experiments` for a manual run that continues until 05:00, all
endpoints finish, or a resource guard stops it. The console displays completion,
peak RAM, and ETA. Reliable rankings can be inspected with:

```powershell
.venv\Scripts\python.exe tools\selector.py output\hypersearch_sessions\xauusd_all_ohlc_timeframes_v1 --topk 10
```

## First controlled results (2026-09-02 UTC)

After appending 2,852 XAUUSD minute rows, the master ended at
`2026-09-02 03:34:00` UTC and all 10m through 24h caches were rebuilt. A small
7h HIGH ULR diagnostic using 2,000 records and a 120-row holdout produced R2
0.4394, 65.0% direction agreement, and 18.79% MAE skill over zero change. A
three-epoch, 1,000-row N-BEATS smoke test produced R2 0.1596 and 2.95% MAE
skill. These are diagnostics, not deployable champions.

The guarded campaign completed the first 37 runs. The 10m OPEN ULR candidate
scored R2 0.9987 and was packaged successfully. OPEN is deliberately assigned
zero strategy weight because its near-deterministic relationship to the prior
candle can produce excellent forecast metrics without providing useful HIGH/LOW
range information. The first 36 10m HIGH runs peaked at R2 0.54745, so no 10m
HIGH candidate was promoted yet. The session state is resumable and retains all
completed configurations.

## US500 exogenous-data implementation

`scripts/refresh_market_assets.py` maintains both databases. Tiingo IEX SPY
minutes are stored under their own `SPY` symbol and can continue the broker-like
US500 series only after a robust overlap ratio passes dispersion and seam-jump
limits. Every derived US500 interval is recorded in
`market_data_provenance`; the raw and derived namespaces are never confused.
Once a proxy continuation exists, later refreshes continue to calibrate against
the immutable native overlap rather than against previously derived prices.

Cross-asset recipes use an exact or bounded backward-only as-of join. Baseline
recipes skip the cross source entirely, so they retain their full trading-day
history. Compare baseline and exogenous variants on the same rolling holdout;
retain the exogenous version only when its model and after-cost strategy metrics
improve. See `US500_AND_CROSS_ASSET_MODELS.md` for commands and deployment paths.

The notebook `XAU_USD_hourly_prices_DL.ipynb` is useful as historical provenance,
but it contains duplicated exploratory cells, failed provider calls, and an
embedded credential. Rotate that credential and keep replacements only in
environment variables such as `TIINGO_API_TOKEN`.
