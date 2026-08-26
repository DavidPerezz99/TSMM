# Upcomers Oracle 5K Profile

The isolated Oracle profile is `config/trading_agent_oracle.yaml`. It uses its
own runtime directory, job prefix, Telegram command prefix, credentials, and
MT5 terminal path. It does not mirror, adopt, modify, or close positions from
another account.

## Verified program rules

Upcomers' current Oracle help page lists these rules for a $5,000 account:

- 5% equity-based Dynamic Risk Shield, initially $4,750, trailing upward and
  locking at the $5,000 initial balance;
- 4% daily drawdown, or $200, measured from 00:00 to 23:59 UTC;
- 2% maximum loss on one trade idea, or $100, including slippage;
- five valid trading days, each with at least 0.5% ($25) realized profit;
- a 20% Best Day Rule for reward eligibility;
- no tick scalping (trades under two minutes), martingale, grid trading, HFT,
  unrelated-account copying, reverse/group hedging, or news straddling.

The public Oracle page currently advertises a 99% profit split. The signed
agreement and account dashboard remain authoritative if the purchased account
shows a different split or add-on.

Official references:

- https://help.upcomers.com/en/articles/13649871-upcomers-oracle-low-cost-instant-funding
- https://help.upcomers.com/en/articles/8703143-what-trading-strategies-are-prohibited-at-upcomers
- https://help.upcomers.com/en/articles/12762068-metatrader-5-mt5-download-setup-troubleshooting

## TSMM safety envelope

The profile intentionally operates well inside the firm's breach limits:

- XAUUSD volume is fixed at 0.01 lot;
- every order requires a broker-side stop loss;
- broker-calculated planned stop loss is capped at $20;
- projected realized daily loss plus the new trade risk is capped at $50;
- a conservative $100 account-equity buffer is maintained;
- any existing position or pending order, including a manual one, blocks a new
  TSMM order without adopting or modifying that exposure;
- one position/order is allowed, stacking and countertrades are disabled;
- autonomous sessions and follow-up entries are disabled;
- all initial entries require explicit approval;
- delayed stop-loss protection is disabled;
- model-driven Agent B closes are disabled to avoid systematic sub-two-minute
  closes. Broker SL/TP remains active at all times.

These safeguards reduce risk but cannot eliminate gaps, slippage, terminal or
network failures, broker liquidation, or a rules interpretation in the signed
agreement that differs from the public help center.

## Local secret variables

The repository contains only environment-variable references. The local user
environment must provide:

- `MT5_UPCOMERS_ORACLE_LOGIN`
- `MT5_UPCOMERS_ORACLE_PASSWORD`
- `MT5_UPCOMERS_TERMINAL_PATH`

Never put the password into YAML, documentation, command history, Telegram, or
version control.

## Operation

Validate the deployment without starting a trade:

```powershell
.venv\Scripts\python.exe scripts\deploy_agent_pipeline.py `
  --pipeline-config config\agent_pipeline_oracle.yaml `
  --dry-run
```

Start the isolated listener only after read-only MT5 verification reports the
expected server, a $5,000 balance/equity, and zero positions/orders:

```powershell
.venv\Scripts\python.exe scripts\telegram_command_listener.py `
  --trading-config config\trading_agent_oracle.yaml
```

Use `/oracle ...` for this account. Do not use recovery or parity scripts for
the Oracle profile.
