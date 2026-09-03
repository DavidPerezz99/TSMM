# US500 and cross-asset modeling

## Data maintenance

The old notebook is historical provenance, not the production updater. It
contains an embedded Tiingo credential; rotate that credential and expose its
replacement only through an environment variable. The maintained profiles keep
the asset pools separate: XAUUSD uses `TIINGO_API_TOKEN`, while US500 uses
`TIINGO_API_TOKEN_ALT`. Do not place either asset's variable in the other
asset's `token_envs` list.

Run the maintained pipeline from the repository root:

```powershell
.venv\Scripts\python.exe scripts\refresh_market_assets.py --asset us500
```

The pipeline stores raw Tiingo SPY minutes in `ohlc_1m_spy`. It derives new
`ohlc_1m_us500` rows only after calibrating SPY to the native US500 overlap with
a robust median ratio. It fails closed if ratio dispersion or the native/proxy
seam exceeds configured limits, records every derived interval in
`market_data_provenance`, then rebuilds 10m through 24h cache tables. SPY is a
proxy, not the same tradable instrument as a broker US500 CFD; provenance must
remain visible in evaluation reports.

## Model searches

US500 baseline and XAUUSD-enriched variants:

```powershell
.venv\Scripts\python.exe tools\experiment_session.py plan --config config\experiment_sessions\us500_nightly.yaml
.venv\Scripts\python.exe tools\experiment_session.py prepare-data --config config\experiment_sessions\us500_nightly.yaml
.venv\Scripts\python.exe tools\experiment_session.py run --config config\experiment_sessions\us500_nightly.yaml --max-experiments 10
```

XAUUSD baseline and US500-enriched variants use a separate session so an
existing XAUUSD campaign manifest remains reproducible:

```powershell
.venv\Scripts\python.exe tools\experiment_session.py run --config config\experiment_sessions\xauusd_us500_exogenous_nightly.yaml --max-experiments 10
```

Both sessions cover OPEN, HIGH, LOW, and CLOSE over 10m, 30m, 1h, 3h, 7h, 12h,
and 24h. They are manual, resumable, stop at 05:00 local time, use at most
10,000 records and 20 GB process RAM, and generate no more than 400 variants per
endpoint. Cross inputs are joined only from the same or an earlier candle. When
the source market is closed, price levels can be carried forward within the
configured staleness limit, but return features become zero until a new source
candle arrives. This prevents one old move from being counted repeatedly and
keeps XAUUSD baseline/enriched holdouts comparable.
Even if an early configuration crosses R2 0.60, each endpoint runs at least 40
successful configurations so both baseline and cross-asset recipes receive a
comparison sample before the endpoint advances.

## Package placement and inference

Use `tools/model_registry.py --asset us500` to install or promote downloaded
packages. The asset flag isolates them under:

```text
model_files/deployments/us500/<timeframe>_<family>/<deployment-id>/
```

The active map is `model_files/deployments/us500/active.json`. Do not manually
copy a model into the generic `model_files` directory; the immutable package
contains the exact config, fitted model, scalers, evaluation, data lineage, and
checksums needed by the US500 endpoint service.

Qualified and fallback packages remain distinct. A fallback can be installed
for inspection, but activation requires the explicit `--allow-fallback`
acknowledgement. Its actual R2 remains attached to the endpoint and controls its
trading vote weight.
