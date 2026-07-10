# AiVora — data flow reference

Everything in AiVora starts and ends at the SQLite store. This
document walks through how bytes get from the DhanHQ API into the
`spot_futures` and `options_chain` tables, and out into the
training Parquet.

Anchor files:

- Client: [src/aivora/pipeline/dhan_client.py](../src/aivora/pipeline/dhan_client.py)
- Ingestion: [src/aivora/pipeline/data_ingestion.py](../src/aivora/pipeline/data_ingestion.py)
- Cleaning: [src/aivora/pipeline/data_cleaning.py](../src/aivora/pipeline/data_cleaning.py)
- Storage: [src/aivora/pipeline/database.py](../src/aivora/pipeline/database.py)
- Orchestrator: [src/aivora/pipeline/pipeline.py](../src/aivora/pipeline/pipeline.py)
- Feature engineering: [src/aivora/pipeline/feature_engineering.py](../src/aivora/pipeline/feature_engineering.py)

---

## 1. How DhanHQ's intraday API returns candles

The pipeline uses `dhanhq.intraday_minute_data(...)` via
`DhanClient.spot_intraday`. Three things worth knowing:

1. **You pass a date range**, not an "N candles" count. Dhan returns
   every 5-minute OHLCV bar that closed inside the window.
2. **A single call is capped at ~90 days.** `spot_intraday` chunks
   the request into 80-day windows (`chunk = timedelta(days=80)`
   in `dhan_client.py`) to leave headroom and to keep progress logs
   readable.
3. **Only completed candles are returned.** If you call at 10:33,
   you get up to the 10:30 candle. The 10:30→10:34:59 window is
   still forming; it is not returned until 10:35.

Response shape (parsed by `_normalise_candles`):

```
{"data": {"open": [...], "high": [...], "low": [...],
          "close": [...], "volume": [...], "timestamp": [...]}}
```

Each array is aligned by index. Timestamps are epoch seconds in IST.

---

## 2. What the daily update actually does

Entry point: `python -m scripts.run_pipeline --mode daily`
Function: `aivora.pipeline.pipeline.run_daily_update()`

For each instrument in `config.yaml → instruments`:

1. Read `MAX(datetime)` from `spot_futures` for that symbol
   (`database.last_loaded_timestamp`). Just informational — the
   fetch window is anchored to the wall clock, not to this value.
2. Call `fetch_recent_spot(days_back=2)` — which computes
   `start = previous_trading_day(today - 6 days)` and asks Dhan
   for that window. In practice that means **the last ~5 trading
   days** get pulled every run.
3. Apply the cleaning pipeline (see §4).
4. Upsert into `spot_futures` (see §5).
5. Attempt one live ATM option-chain snapshot; failures are logged
   and swallowed.

After both symbols are processed, `build_training_dataset()`
re-runs feature engineering against the DB and rewrites
`data/processed/training_dataset.parquet`. The Parquet file is a
pure derivative of the DB — it can be rebuilt at any time.

---

## 3. What "completed candle" means, precisely

Given wall-clock time `T`:

```
last_closed_5min = floor(T to nearest 5-min boundary) - 5m  if T is on a boundary
                 = floor(T to nearest 5-min boundary)       otherwise
```

Examples (IST, on a trading day):

| Wall clock T | Last candle Dhan returns |
|---|---|
| 09:14:30 | none (pre-open) |
| 09:20:00 | 09:15 candle just closed → 09:15 |
| 09:23:45 | 09:15 |
| 09:25:00 | 09:20 |
| 15:29:59 | 15:25 |
| 15:30:00 | 15:25 (the 15:30 candle closes at 15:30) |
| 15:30:01 | 15:30 |

So the pipeline never picks up the currently-forming candle. If
you re-run the daily pipeline five minutes later, you get exactly
one more row per symbol.

---

## 4. Cleaning stages applied to every row

From `data_cleaning.clean()`, in order:

1. `_round_to_5min` — snap timestamps to the 5-min grid so
   09:20:01 becomes 09:20:00. Dhan is normally precise but this
   guards against edge cases.
2. `drop_non_trading_days` — remove weekends and NSE holidays via
   the calendar in `utils/calendar.py`.
3. `restrict_to_session` — keep only 09:15 ≤ time ≤ 15:30.
4. `deduplicate` — `(symbol, datetime)` dedup, keeping the
   freshest value on conflicts.
5. `fill_small_gaps` — forward-fill runs of ≤ 3 consecutive
   missing candles per column, and mark the filled rows with
   `is_filled = 1`.
6. `drop_unfillable_rows` — drop rows where the four essential
   spot OHLC columns are still NaN.
7. `winsorise` — clip OI / LTP / volume outliers at the 0.1 % /
   99.9 % per-symbol quantiles.

Result: a dataframe that always has timestamps on a clean grid, only
inside trading hours, with sane price ranges.

---

## 5. Upsert = idempotent write

`spot_futures` schema:

```sql
CREATE TABLE spot_futures (
    datetime    TEXT NOT NULL,
    symbol      TEXT NOT NULL,
    spot_open   REAL,
    ...
    is_filled   INTEGER DEFAULT 0,
    PRIMARY KEY (symbol, datetime)
);
```

Writes are:

```sql
INSERT OR REPLACE INTO spot_futures (...) VALUES (?, ?, ...)
```

Because the primary key is `(symbol, datetime)`:

- Re-running the pipeline **cannot** produce duplicate rows.
- A row with the same key is overwritten with the fresher value.
- The row-count check in
  [scripts/test_pipeline_robustness.py](../scripts/test_pipeline_robustness.py)
  T1 is a live proof of this invariant.

The `options_chain` table has the same pattern with a wider PK:
`(symbol, datetime, strike, type)`.

---

## 6. Recovering missing data

### Recent gap (≤ 5 days old)

Just re-run the daily pipeline. `fetch_recent_spot(days_back=2)`
already covers the last ~5 trading days, and the upsert will
back-fill the deleted rows.

```bash
python -m scripts.run_pipeline --mode daily
```

### Older gap (> 5 days old)

Daily mode will NOT touch older data — its query window doesn't
reach that far. Use the historical mode with an explicit range:

```bash
python -m scripts.run_pipeline --mode historical \
        --start 2026-05-10 --end 2026-05-16
```

This calls `DhanClient.load_full_historical_data(start, end)`
which:

1. Chunks the range into ~80-day windows for spot.
2. Walks every weekly expiry inside the range and pulls the
   ATM CE + PE 5-min OHLC/OI/IV series via the
   expired-options endpoint (`expired_option_intraday`).
3. Cleans and upserts into `spot_futures` + `options_chain`.

The historical mode is **also idempotent** — running it over a
range you already have won't create duplicates, but it *will*
overwrite rows with the latest Dhan data. That's usually what
you want.

Both modes conclude by calling `build_training_dataset()`, so the
Parquet is refreshed automatically.

---

## 7. When to run what

| Situation | Command | What runs |
|---|---|---|
| Fresh setup, no data | `run_historical_load` or `run_pipeline --mode historical` | Full multi-month pull, spot + expired options |
| Cold Parquet, warm DB | `run_pipeline --mode rebuild` | Feature engineering only, no API calls |
| Weekend / holiday | Nothing needed | Data pipeline is a no-op on non-trading days |
| Every trading day after 15:35 | `run_pipeline --mode daily` | Appends today's rows, refreshes Parquet |
| Suspected data gaps | `scripts.audit_data`, then targeted historical run | Read-only audit → surgical backfill |
| Suspected regression in the ETL | `scripts.test_pipeline_robustness` | Backup, four scenarios, restore |

---

## 8. Verification tools

- **`python -m scripts.audit_data`** — read-only DB scan. Reports
  duplicate rows (should always be 0), missing candles per day,
  price outliers, unaligned timestamps, and NaN counts. Writes
  a timestamped report under `logs/`.

- **`python -m scripts.test_pipeline_robustness`** — four
  live tests (idempotency, recent-gap recovery, mid-day behaviour,
  old-gap recovery). Every test is bracketed by a backup and
  restore of the `spot_futures` table, so nothing persists after
  the run. Uses `--restore-only` if a run is interrupted.

Both scripts share the project's logger and config, so their
output is consistent with the rest of the pipeline.
