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

The current plan contains 28 XAUUSD OHLC endpoints and 8,748 possible
experiments. Each endpoint stops early as soon as a qualifying bundle is found.
The large total is therefore a ceiling, not the expected executed count.

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

## US500 exogenous-data decision

The SQLite schema already supports namespaced US500 tables. However, the US500
database currently ends at `2026-04-30 23:58:00`, while XAUUSD reaches September
2026. It must not be forward-filled across that four-month gap or joined using
future timestamps. US500 is therefore not enabled as an XAUUSD exogenous input
yet.

Before enabling it:

1. refresh US500 from a supported provider without a manually scaled SPY proxy;
2. verify timestamp, timezone, market-hours, duplicate, and missing-gap quality;
3. join with a backward-only as-of rule and a strict staleness tolerance;
4. compare the same XAUUSD sweep with and without US500 on identical holdouts;
5. keep the exogenous version only if walk-forward and after-cost results improve.

The notebook `XAU_USD_hourly_prices_DL.ipynb` is useful as historical provenance,
but it contains duplicated exploratory cells, failed provider calls, and an
embedded credential. Rotate that credential and keep replacements only in
environment variables such as `TIINGO_API_TOKEN`.
