# AiVora — automated Nifty / Bank Nifty options trading

AiVora is an end-to-end research and execution skeleton for trading
index options.  It covers everything from raw historical data
ingestion to a backtested LightGBM model with a P&L curve.

```
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ DhanHQ REST API  │ → │ ETL pipeline     │ → │ Training Parquet │ → │ LightGBM +       │
│ (spot + per-     │   │ (clean, FE, DB)  │   │                  │   │ Optuna +         │
│  expiry options) │   │                  │   │                  │   │ backtest         │
└──────────────────┘   └──────────────────┘   └──────────────────┘   └──────────────────┘
```

The historical loader walks every weekly expiry in the configured
date range, and fetches 5-minute ATM CE/PE intraday OHLC + OI/IV
directly from DhanHQ's expired-options endpoint — no scrip-master
lookup or manual security-ID resolution required. No CSV inputs are
needed — DhanHQ is the sole market-data source.

**Verified data depth:** DhanHQ's expired-options endpoint has been
empirically confirmed to return real 5-minute NIFTY ATM option data
(sane strikes matching actual historical NIFTY levels) as far back as
**60 months (5 years)** — their docs' "up to 5 years" claim checks out
in practice, it's not just marketing copy. Requires an active
**Data APIs subscription** on the account (~₹499+tax/month) — without
it every data call fails with error `DH-902`.

## Project layout

```
AiVora/
├── config.yaml                 ← single source of truth for tunables
├── requirements.txt
├── .env.example                ← copy to .env and fill in Dhan creds
├── data/
│   ├── raw/                    ← raw historical Parquet cache
│   ├── processed/              ← training Parquet
│   └── db/aivora.sqlite        ← SQLite store
├── models/                     ← serialized boosters + registry JSON
├── reports/                    ← classification reports + plots
├── logs/                       ← rotating run logs
├── scripts/
│   ├── run_pipeline.py         ← ETL entrypoint (historical | daily | rebuild)
│   ├── run_historical_load.py  ← dedicated multi-month cold-start
│   ├── train_model.py          ← train + evaluate + backtest
│   ├── retrain.py              ← daily-update + retrain
│   └── refresh_dhan_token.py   ← write a fresh 24-hour token to .env
├── src/aivora/
│   ├── pipeline/
│   │   ├── dhan_client.py      ← DhanHQ adapter (spot intraday + expired-options + live option chain)
│   │   ├── data_ingestion.py   ← live/daily Dhan helpers
│   │   ├── data_cleaning.py    ← winsorising, gap filling
│   │   ├── feature_engineering.py
│   │   ├── database.py         ← SQLite schema + upsert
│   │   └── pipeline.py         ← orchestrator
│   ├── ml/
│   │   ├── dataset.py          ← splits + walk-forward folds
│   │   ├── train.py            ← LightGBM + Optuna
│   │   ├── evaluate.py         ← reports + plots
│   │   └── registry.py         ← JSON model registry
│   ├── backtest/backtester.py  ← options P&L simulator
│   └── utils/                  ← config, logger, calendar
└── tests/                      ← pytest smoke tests
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # PowerShell — or `source .venv/Scripts/activate` in Git Bash
pip install -r requirements.txt
cp .env.example .env            # fill in DhanHQ credentials
```

### DhanHQ credentials

1. Open <https://web.dhan.co/> → Profile → **Access DhanHQ APIs**.
2. Click **Generate Access Token** (valid 24 hours) — no TOTP/2FA
   needed for this manual flow, since you're already logged in.
3. Paste your `client_id` and `access_token` into `.env`, or run
   `python -m scripts.refresh_dhan_token` for an interactive prompt
   that writes them for you.
4. On the same **DhanHQ Trading APIs** page, open the **Data APIs**
   tab and subscribe (~₹499+tax/month) — every data call fails with
   `DH-902` until this is active.

Static-IP whitelisting is **only** needed for order-placement
endpoints. All market-data calls used in this project (intraday
candles, expired-options data, option chain, expiry list) work from
any IP.

### Historical range

The cold-start range lives under `historical:` in `config.yaml`:

```yaml
historical:
  start_date: "2026-01-05"   # ~6 months back by default
  end_date: ""               # blank → today
```

Override at the CLI with `--start` / `--end` on either
`scripts.run_pipeline` or `scripts.run_historical_load`. Going further
back than ~6 months is unverified — test a small range first if you
push the window wider.

## End-to-end run

```bash
# 1. cold start — multi-month DhanHQ pull → SQLite → training Parquet
python -m scripts.refresh_dhan_token            # if you don't have a token yet
python -m scripts.run_historical_load           # or: run_pipeline --mode historical

# 2. fit the model + evaluate + backtest
python -m scripts.train_model

# 3. daily — fetch yesterday/today's candles from DhanHQ and rebuild
python -m scripts.refresh_dhan_token            # if previous token expired
python -m scripts.run_pipeline --mode daily

# 4. weekly retrain (incremental pull + new model version)
python -m scripts.retrain
```

A 6-month load walks ~26 weekly expiries × 2 symbols × 2 sides (CE/PE)
— roughly 100+ throttled calls against the expired-options endpoint
(it rate-limits harder than the general data API; the client retries
on `DH-904` with backoff), plus a couple of chunked spot-history
calls. Expect this to take several minutes, not seconds.

Outputs land in:

* `models/lgbm_model.pkl` — joblib-serialised booster
* `models/lgbm_model.json` — chosen hyperparameters + val metrics
* `models/model_registry.json` — append-only version log
* `reports/classification_report.txt` — sklearn report + directional metrics
* `reports/plots/confusion_matrix.png`, `feature_importance.png`, `equity_curve.png`
* `reports/trades.csv` — every simulated trade
* `logs/aivora.log` — rotating run logs

## Historical-data caveats

* **Dhan's own docs and SDK examples are inconsistent** (and in one
  case simply wrong — the README's `expired_options_data` example
  uses `instrument_type="INDEX"`, which silently returns empty data;
  it must be `"OPTIDX"`). Depth-wise though, the marketed "5 years" is
  real — verified empirically at 6, 9, 12, 18, 24, 30, 36, 48, and 60
  months back, all returning sane strikes matching NIFTY's actual
  historical levels at each point in time.

* **`expiry_code=0` is broken.** Despite docs listing `0` as a valid
  "current/near expiry" value, the server rejects it outright
  (`expiryCode is required`). The client uses `1` ("next") and `2`
  ("far") instead, bracketing the request window tightly around the
  target expiry week — that's what makes the right contract resolve.

* **OI / IV in history** — unlike the old scrip-master-based
  approach, the expired-options endpoint *does* return OI and IV
  historically (via `required_data`), not just OHLCV.

* **Static-IP whitelisting** is required only for order-placement
  endpoints, not for any of the data calls used here.

## Configuration

All knobs live in `config.yaml`.  Edit and re-run; nothing in the
code expects you to override paths or thresholds programmatically.

| Section | What it controls |
|---|---|
| `project` | Capital, timezone, name |
| `paths` | Where every artefact is read / written |
| `instruments` | Lot size, strike step, Dhan security ID / segment per symbol |
| `market` | Session window + candle interval |
| `historical` | Cold-start date range for the historical loader |
| `dhan` | Data-API rate, option-chain gap, expired-options gap, retry budget |
| `zerodha` | (Kept for later live trading; unused by the data pipeline) |
| `features` | RSI / BB / MA periods, NaN handling |
| `labels` | Forward horizon + UP/DOWN thresholds |
| `model` | Train fraction, Optuna trials, CV folds |
| `backtest` | Probability threshold, risk limits |

## Testing

```bash
pytest -q
```

The smoke tests use synthetic data — no network or credentials
required.  They cover:

* End-to-end cleaning + feature engineering
* Look-ahead leakage check on every engineered feature
* SQLite round-trip with deduplication

## Notes

* The backtest's option P&L model is intentionally simple
  (delta + linear theta).  Replace `_estimate_entry_premium` /
  `_exit_premium` with a Black-Scholes path or actual fills once
  you have real option price history.
* Class imbalance is left to LightGBM — if FLAT dominates badly
  after labelling, consider `is_unbalance=True` or supply
  `sample_weight` in `train.py`.
* The code path will not place a real order anywhere — there is
  no execution adapter by design.  Add one only after you trust
  the backtest.
