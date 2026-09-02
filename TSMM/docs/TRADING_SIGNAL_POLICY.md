# Joint OHLC Trading Signal Policy

TSMM no longer treats one HIGH model as the trading side while using the other
OHLC models only as supporting commentary. When `signal_policy.enabled` is
true, live Agent A and the historical replay use the same decision policy:

1. Qualified HIGH and LOW models are the primary direction/range families.
   CLOSE has a small supporting weight and OPEN has zero trading weight.
2. TSMM scores 10m, 30m, 1h, 3h, 7h, 12h, and 24h candidates using current
   endpoint confidence and model-quality weight, then uses the strongest
   non-neutral HIGH/LOW timeframe as the direction anchor. This means a strong
   10m model can lead while a stale or weak 7h model cannot retain authority by
   configuration alone.
3. The remaining configured timeframes independently confirm entry timing. The
   default policy requires two supporting timeframes to agree.
   Models below the configured R2 floor have zero voting power; refreshed R2
   takes precedence over the historical selection score when it is available.
   Legacy scores that have not yet passed the new rolling-origin protocol retain
   only 35% of their nominal voting weight until they are revalidated.
4. Qualified HIGH and LOW forecast levels on the selected anchor timeframe form
   a projected range. An unqualified model, a collapsed range, or a HIGH forecast
   below LOW causes abstention instead of silently swapping the levels. A buy is placed
   toward the lower part of that range and a sell toward the upper part, with a
   configured maximum distance from the current market.
5. Recent 10m ATR and realized volatility replace a single fixed-percentage
   stop/target with bounded volatility-aware distances.
6. The probability estimate is computed before the hybrid admission gate, so
   expected value uses the estimate instead of silently substituting raw
   confidence. A single-endpoint fallback cannot bypass a joint-policy HOLD.
7. A programmed order that remains unfilled near expiry may become a market
   order only for configured triggers. TSMM re-runs all model assessments,
   requires the joint policy to still support the original side, confirms the
   pending order was cancelled, and runs the normal account/prop-firm guard
   before submitting the market order.

The policy can abstain. Requiring one broker order regardless of model quality
would defeat the confirmation and risk controls; `mandatory_session` means an
analysis attempt is mandatory, not that a weak signal must become a trade.

Agent B now respects the trailing enable flag for both stop and target changes.
When enabled, its ratchet uses the larger of the configured price gap and an ATR
gap, never loosens an existing stop, moves defensive protection beyond entry by
the estimated round-trip cost after sufficient favorable movement, and extends
take profit only when consensus and configured confidence both pass.

## Backtest validity

Normal historical replay continues to allow current/legacy artifacts and labels
the result exploratory. For evidence-grade evaluation, run:

```powershell
python app.py backtest `
  --start-date 2026-08-01 `
  --end-date 2026-08-15 `
  --require-point-in-time-models
```

Strict mode refuses to run unless every model package declares a
`training_data_last_index` earlier than the evaluated period. File modification
time is not accepted as training lineage. This creates a genuine out-of-sample
frozen-model window.

For an automated walk-forward replay across immutable deployment versions, use:

```powershell
python app.py backtest `
  --start-date 2026-08-01 `
  --end-date 2026-08-15 `
  --walk-forward-deployments model_files/deployments
```

At every replay tick this mode selects, independently for each timeframe/OHLC
endpoint, the newest installed package whose `training_data_last_index` is
strictly earlier than that tick. It refuses to start if deployment history is
missing a configured endpoint or no pre-period version is eligible.

Each result now includes per-trade MAE, MFE, time-to-MAE, time-to-MFE, aggregate
winner/loser path diagnostics, and MFE capture. Those values help distinguish
bad direction from bad entry timing and poor exit capture.
