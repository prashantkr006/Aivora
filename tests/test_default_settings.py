"""A new portfolio must start on the config that was actually validated.

Settings live per portfolio in the database. Realigning the existing
portfolios to the canonical config on 2026-08-03 therefore never touched
``default_settings()`` in the code, and nobody noticed: the running
portfolios were right, so nothing looked wrong.

It surfaced when a second instance was stood up on 2026-08-05. Both of its
fresh portfolios came up with the volatility filter ON — 0.15/0.90 — which
the 55-month walk-forward had already measured and rejected: Rs 8.82L with
the filter off against Rs 7.22L with it on.

Every portfolio created between those dates started on a config that had
been tested and found worse.
"""

from __future__ import annotations

import pytest

from aivora.live.portfolio import default_settings

# The founder-confirmed 55-month walk-forward config (2026-08-03).
CANONICAL = {
    "prob_threshold_up": 0.55,
    "prob_threshold_down": 0.55,
    "take_profit_pct": 0.60,
    "stop_loss_pct": 0.30,
    "min_minutes_since_open": 0,
    "max_minutes_since_open": 300,
    "vol_regime_min": 0.0,
    "vol_regime_max": 999.0,
    "max_trades_per_day": 10,
    "horizon_candles": 12,
    "risk_per_trade_pct": 0.02,
}


@pytest.mark.parametrize("key,expected", sorted(CANONICAL.items()))
def test_default_matches_the_validated_config(key, expected):
    assert default_settings()[key] == expected


def test_the_volatility_filter_is_off():
    """The specific one that drifted. A range this wide admits everything,
    which is the point — the filter was measured and rejected."""
    s = default_settings()
    assert s["vol_regime_min"] <= 0.0
    assert s["vol_regime_max"] >= 999.0


def test_a_fresh_portfolio_gets_the_canonical_config(tmp_path):
    """Not just the dict — what a new portfolio actually starts with."""
    import time

    from aivora.webapp import portfolios, users

    u = users.register(f"cfg_{time.time()}@example.com", "TestPassword_123")
    for mode in ("paper", "live"):
        s = portfolios.UserPortfolio(u.id, mode).load()["settings"]
        drift = {k: (s.get(k), v) for k, v in CANONICAL.items() if s.get(k) != v}
        assert not drift, f"{mode} portfolio drifted: {drift}"
