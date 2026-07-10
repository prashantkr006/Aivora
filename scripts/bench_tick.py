"""End-to-end tick benchmark for MarketDataCache.refresh_if_stale.

Uses a stub Kite client so the timing is deterministic (no network
variance) — measures ONLY the DB read + feature engineering +
inference path we control.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from aivora.webapp.trading_engine import MarketDataCache  # noqa: E402


class _StubKite:
    """No-op Kite client — pretend we already have the latest candles."""
    def fetch_recent_spot(self, symbol: str, days_back: int = 2) -> pd.DataFrame:
        return pd.DataFrame()  # empty ⇒ no upsert


def main() -> int:
    kite = _StubKite()
    now = datetime.now()

    # Warm the cache (first-tick path)
    MarketDataCache._reset()
    t0 = time.perf_counter()
    MarketDataCache.refresh_if_stale(kite, now)
    t_first = time.perf_counter() - t0
    print(f"first tick (cold cache):  {t_first:.2f}s")

    # Force a subsequent tick by resetting _last_fetch
    MarketDataCache._last_fetch = None
    t0 = time.perf_counter()
    MarketDataCache.refresh_if_stale(kite, now)
    t_hot = time.perf_counter() - t0
    print(f"hot tick (warm frozen model): {t_hot:.2f}s")

    # Confirm predictions are non-empty
    preds = MarketDataCache.snapshot()
    print(f"predictions produced for: {list(preds.keys())}")
    for sym, r in preds.items():
        print(f"  {sym}: p_up={r.p_up:.4f} p_down={r.p_down:.4f} vr={r.vol_regime_pct:.3f}")

    verdict = "PASS ✅" if t_hot < 5.0 else "FAIL ❌ (>5s)"
    print(f"\nVERDICT: {verdict}  (target <5s per tick)")
    return 0 if t_hot < 5.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
