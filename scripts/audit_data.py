"""Data-quality audit for the AiVora SQLite store.

Read-only.  Never writes to the database.  For each configured
symbol it reports:

* Volume: total rows, calendar date range, rows-per-day stats.
* Structural integrity:
    - Duplicate (symbol, datetime) rows.
    - Missing 5-minute candles inside 09:15 - 15:30 on trading days.
    - Timestamps not aligned to the 5-minute grid.
* Value sanity:
    - Non-positive prices.
    - Out-of-range prices (Nifty > 30 000, BankNifty > 60 000).
    - Adjacent-candle % change > 5 %.
* Storage:
    - NaN counts per column.

Output goes to two places:
    - Console (short summary).
    - ``logs/audit_report_YYYYMMDD_HHMMSS.txt`` (full detail).

Usage::

    python -m scripts.audit_data
    python -m scripts.audit_data --symbol NIFTY --spike-pct 0.03
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aivora.pipeline.database import connect  # noqa: E402
from aivora.utils.calendar import is_trading_day  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.utils.logger import get_logger  # noqa: E402

log = get_logger("scripts.audit_data")

# Reasonable per-symbol price ceilings.  These are intentionally
# generous - the check is meant to catch obviously broken ticks
# (e.g. spot showing "0" or "3 000 000"), not to police normal
# multi-year drift.
DEFAULT_MAX_PRICE = {
    "NIFTY":     30_000.0,
    "BANKNIFTY": 75_000.0,
}


# =============================================================
#  DB reads
# =============================================================
def _load_symbol(symbol: str) -> pd.DataFrame:
    """Read the spot_futures rows for one symbol.

    Kept in a tiny helper so the audit can be run standalone
    without pulling in any of the write-path code.
    """
    with connect() as conn:
        df = pd.read_sql(
            "SELECT * FROM spot_futures WHERE symbol = ? ORDER BY datetime",
            conn, params=(symbol,),
        )
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def _load_duplicates(symbol: str) -> pd.DataFrame:
    """PK is (symbol, datetime) so this SHOULD always be empty -
    running the check anyway proves the invariant still holds."""
    with connect() as conn:
        return pd.read_sql(
            """
            SELECT symbol, datetime, COUNT(*) AS n
            FROM spot_futures
            WHERE symbol = ?
            GROUP BY symbol, datetime
            HAVING COUNT(*) > 1
            """,
            conn, params=(symbol,),
        )


# =============================================================
#  Expected-grid helpers
# =============================================================
def _expected_5min_grid(day: date) -> List[pd.Timestamp]:
    """The 76 timestamps we expect for one trading day.

    09:15 - 15:30 inclusive at 5-minute steps.  Older rows in this
    project sometimes only run to 15:25 (75 stamps); the missing-
    candle check treats a lone missing 15:30 as informational
    rather than a hard error.
    """
    start = datetime.combine(day, time(9, 15))
    end = datetime.combine(day, time(15, 30))
    stamps: List[pd.Timestamp] = []
    cur = start
    while cur <= end:
        stamps.append(pd.Timestamp(cur))
        cur += timedelta(minutes=5)
    return stamps


# =============================================================
#  Individual checks
# =============================================================
def _rows_per_day(df: pd.DataFrame) -> pd.Series:
    return df.groupby(df["datetime"].dt.date).size().sort_index()


def _missing_candles(df: pd.DataFrame) -> Dict[date, List[pd.Timestamp]]:
    """Return {trading_day: [missing timestamps]}.

    Only trading days (per the NSE calendar) that already have
    *any* row are considered - a day with zero rows is reported
    separately by :func:`_missing_days`.
    """
    out: Dict[date, List[pd.Timestamp]] = {}
    have = df.groupby(df["datetime"].dt.date)["datetime"].apply(set)
    for d, present in have.items():
        if not is_trading_day(d):
            continue
        expected = set(_expected_5min_grid(d))
        missing = sorted(expected - set(present))
        if missing:
            out[d] = missing
    return out


def _missing_days(df: pd.DataFrame) -> List[date]:
    """Trading days that have zero rows between first and last present dates."""
    if df.empty:
        return []
    first = df["datetime"].dt.date.min()
    last = df["datetime"].dt.date.max()
    have = set(df["datetime"].dt.date.unique())
    out: List[date] = []
    cur = first
    while cur <= last:
        if is_trading_day(cur) and cur not in have:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def _price_range_violations(df: pd.DataFrame, symbol: str, max_price: float) -> pd.DataFrame:
    """Rows whose spot_close is non-positive or wildly out-of-range."""
    mask = (df["spot_close"].isna()) | (df["spot_close"] <= 0) | (df["spot_close"] > max_price)
    return df.loc[mask, ["datetime", "spot_open", "spot_high", "spot_low", "spot_close"]]


def _price_spikes(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Adjacent-candle close-to-close moves larger than ``threshold``.

    Boundary between trading days is excluded (a real overnight
    gap is not the same as an intraday spike).
    """
    if len(df) < 2:
        return df.iloc[0:0]
    d = df.sort_values("datetime").reset_index(drop=True).copy()
    d["prev_close"] = d["spot_close"].shift(1)
    d["prev_date"] = d["datetime"].dt.date.shift(1)
    d["same_day"] = d["prev_date"] == d["datetime"].dt.date
    d["pct_change"] = (d["spot_close"] - d["prev_close"]) / d["prev_close"]
    mask = d["same_day"] & (d["pct_change"].abs() > threshold)
    return d.loc[mask, ["datetime", "prev_close", "spot_close", "pct_change"]]


def _misaligned_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Timestamps not on a 5-minute boundary (or with non-zero seconds)."""
    ts = df["datetime"]
    bad = (ts.dt.minute % 5 != 0) | (ts.dt.second != 0)
    return df.loc[bad, ["datetime", "spot_close"]]


def _nan_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()}


# =============================================================
#  Per-symbol audit
# =============================================================
def audit_symbol(
    symbol: str,
    max_price: float,
    spike_pct: float,
    include_missing_15_30: bool,
) -> Dict:
    """Run every check on one symbol and return a structured result."""
    log.info("audit: symbol=%s", symbol)
    df = _load_symbol(symbol)
    result: Dict = {"symbol": symbol, "rows": len(df)}
    if df.empty:
        result["error"] = "No rows in spot_futures for this symbol."
        return result

    rpd = _rows_per_day(df)
    result["date_range"] = (df["datetime"].min(), df["datetime"].max())
    result["trading_days"] = int(len(rpd))
    result["rows_per_day"] = {
        "min": int(rpd.min()), "median": float(rpd.median()),
        "mean": float(rpd.mean()), "max": int(rpd.max()),
    }
    # Days that don't match the modal expected count (76 or 75).
    modes = rpd.mode()
    modal = int(modes.iloc[0]) if not modes.empty else 76
    result["modal_rows_per_day"] = modal
    result["non_modal_days"] = int((rpd != modal).sum())

    # ---- structural ----
    dupes = _load_duplicates(symbol)
    result["duplicate_rows"] = int(dupes["n"].sum()) if not dupes.empty else 0
    result["duplicate_sample"] = dupes.head(5).to_dict(orient="records") if not dupes.empty else []

    missing = _missing_candles(df)
    if not include_missing_15_30:
        # Strip the 15:30 marker so the "old data ends at 15:25" case
        # doesn't drown the report in a false-positive per day.
        missing = {
            d: [t for t in ts if t.time() != time(15, 30)]
            for d, ts in missing.items()
        }
        missing = {d: ts for d, ts in missing.items() if ts}
    result["days_with_missing_candles"] = len(missing)
    result["total_missing_candles"] = sum(len(v) for v in missing.values())
    result["missing_candles_sample"] = {
        str(d): [t.isoformat() for t in ts[:5]] for d, ts in list(missing.items())[:10]
    }

    zero_days = _missing_days(df)
    result["missing_trading_days"] = [d.isoformat() for d in zero_days[:20]]
    result["missing_trading_days_count"] = len(zero_days)

    misaligned = _misaligned_timestamps(df)
    result["misaligned_timestamps"] = int(len(misaligned))
    result["misaligned_sample"] = misaligned.head(5).to_dict(orient="records")

    # ---- values ----
    oor = _price_range_violations(df, symbol, max_price)
    result["price_range_violations"] = int(len(oor))
    result["price_range_sample"] = oor.head(5).to_dict(orient="records")

    spikes = _price_spikes(df, spike_pct)
    result["price_spikes"] = int(len(spikes))
    result["price_spikes_sample"] = spikes.head(10).to_dict(orient="records")

    result["nan_counts"] = _nan_counts(df)
    return result


# =============================================================
#  Report rendering
# =============================================================
def _fmt_section(title: str) -> List[str]:
    return ["", title, "-" * len(title)]


def render_report(results: List[Dict], args) -> str:
    lines: List[str] = [
        "=" * 70,
        "AiVora - data quality audit",
        f"Timestamp   : {datetime.now().isoformat(timespec='seconds')}",
        f"DB          : {get_config().paths['sqlite_path']}",
        f"Spike thr.  : {args.spike_pct:.1%} adjacent-candle % change",
        f"Ignore 15:30 missing: {not args.include_15_30}",
    ]
    for r in results:
        lines.extend(_fmt_section(f"Symbol: {r['symbol']}"))
        if r.get("error"):
            lines.append(f"  ERROR: {r['error']}")
            continue
        lines.append(f"  Rows                       : {r['rows']:,}")
        lines.append(f"  Date range                 : {r['date_range'][0]} -> {r['date_range'][1]}")
        lines.append(f"  Trading days present       : {r['trading_days']:,}")
        lines.append(
            f"  Rows/day                   : min={r['rows_per_day']['min']}  "
            f"median={r['rows_per_day']['median']:.1f}  "
            f"mean={r['rows_per_day']['mean']:.2f}  "
            f"max={r['rows_per_day']['max']}"
        )
        lines.append(
            f"  Modal rows/day             : {r['modal_rows_per_day']} "
            f"(non-modal days: {r['non_modal_days']})"
        )
        lines.append("")
        lines.append(f"  Duplicate rows             : {r['duplicate_rows']}")
        if r["duplicate_sample"]:
            for d in r["duplicate_sample"]:
                lines.append(f"    - {d}")

        lines.append(f"  Days with missing candles  : {r['days_with_missing_candles']}")
        lines.append(f"  Total missing candles      : {r['total_missing_candles']}")
        for d, ts in r["missing_candles_sample"].items():
            preview = ", ".join(ts)
            lines.append(f"    - {d}: {preview}")

        lines.append(f"  Missing whole trading days : {r['missing_trading_days_count']}")
        for d in r["missing_trading_days"]:
            lines.append(f"    - {d}")

        lines.append(f"  Timestamps off 5-min grid  : {r['misaligned_timestamps']}")
        for row in r["misaligned_sample"]:
            lines.append(f"    - {row}")

        lines.append("")
        lines.append(f"  Price-range violations     : {r['price_range_violations']}")
        for row in r["price_range_sample"]:
            lines.append(f"    - {row}")

        lines.append(f"  Adjacent-candle spikes >{args.spike_pct:.1%} : {r['price_spikes']}")
        for row in r["price_spikes_sample"]:
            lines.append(f"    - {row}")

        if r["nan_counts"]:
            lines.append("")
            lines.append("  NaN counts by column:")
            for col, n in sorted(r["nan_counts"].items(), key=lambda x: -x[1]):
                lines.append(f"    {col:15s} {n:>10,}")
    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def _console_summary(results: List[Dict]) -> None:
    """One-liner per symbol so the CLI stays readable."""
    for r in results:
        if r.get("error"):
            log.error("%s: %s", r["symbol"], r["error"])
            continue
        log.info(
            "%s: rows=%d days=%d dup=%d missing_candles=%d spikes=%d oor=%d",
            r["symbol"], r["rows"], r["trading_days"],
            r["duplicate_rows"], r["total_missing_candles"],
            r["price_spikes"], r["price_range_violations"],
        )


# =============================================================
#  CLI
# =============================================================
def main() -> int:
    cfg = get_config()

    ap = argparse.ArgumentParser(description="AiVora data-quality audit")
    ap.add_argument("--symbol", choices=[i["symbol"] for i in cfg.instruments],
                    default=None, help="Restrict audit to one symbol")
    ap.add_argument("--spike-pct", type=float, default=0.05,
                    help="Threshold for adjacent-candle %% change (default 5%%)")
    ap.add_argument("--include-15-30", action="store_true",
                    help="Report missing 15:30 candle (older data may end at 15:25)")
    ap.add_argument("--max-price-nifty", type=float,
                    default=DEFAULT_MAX_PRICE["NIFTY"])
    ap.add_argument("--max-price-banknifty", type=float,
                    default=DEFAULT_MAX_PRICE["BANKNIFTY"])
    args = ap.parse_args()

    symbols = [args.symbol] if args.symbol else [i["symbol"] for i in cfg.instruments]
    max_price_by_sym = {
        "NIFTY":     args.max_price_nifty,
        "BANKNIFTY": args.max_price_banknifty,
    }

    results: List[Dict] = []
    for sym in symbols:
        results.append(audit_symbol(
            symbol=sym,
            max_price=max_price_by_sym.get(sym, 1e6),
            spike_pct=args.spike_pct,
            include_missing_15_30=args.include_15_30,
        ))

    report = render_report(results, args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path: Path = cfg.paths["logs_dir"] / f"audit_report_{stamp}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    _console_summary(results)
    log.info("Full report -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
