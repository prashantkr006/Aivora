"""NSE trading calendar helpers.

We try ``nsepython`` first for the authoritative holiday list.
If the package or network is unavailable we fall back to a static
``KNOWN_HOLIDAYS`` set so unit tests and offline pipelines still
behave correctly.  The fallback is intentionally minimal — extend
it manually if you operate fully offline.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from typing import Iterable, Set

import pandas as pd

from .logger import get_logger

log = get_logger(__name__)

# Static fallback list — extend as NSE announces holidays each year.
# Source: nseindia.com/resources/exchange-communication-holidays
#
# Cross-check against the audit's "missing whole trading days" report
# before adding a new year — mismatches usually mean either a new
# holiday to add here or a genuine data-loss day to backfill.
KNOWN_HOLIDAYS: Set[date] = {
    # ---------- 2021 ----------
    date(2021, 1, 26),   # Republic Day
    date(2021, 3, 11),   # Mahashivratri
    date(2021, 3, 29),   # Holi
    date(2021, 4, 2),    # Good Friday
    date(2021, 4, 14),   # Dr. Ambedkar Jayanti
    date(2021, 4, 21),   # Ram Navami
    date(2021, 5, 13),   # Eid-ul-Fitr
    date(2021, 7, 21),   # Bakri Id (Eid-ul-Adha)
    date(2021, 8, 19),   # Muharram
    date(2021, 9, 10),   # Ganesh Chaturthi
    date(2021, 10, 15),  # Dussehra
    date(2021, 11, 5),   # Diwali Balipratipada
    date(2021, 11, 19),  # Guru Nanak Jayanti

    # ---------- 2022 ----------
    date(2022, 1, 26),   # Republic Day
    date(2022, 3, 1),    # Mahashivratri
    date(2022, 3, 18),   # Holi
    date(2022, 4, 14),   # Dr. Ambedkar Jayanti / Mahavir Jayanti
    date(2022, 4, 15),   # Good Friday
    date(2022, 5, 3),    # Eid-ul-Fitr
    date(2022, 8, 9),    # Muharram
    date(2022, 8, 15),   # Independence Day
    date(2022, 8, 31),   # Ganesh Chaturthi
    date(2022, 10, 5),   # Dussehra
    date(2022, 10, 24),  # Diwali/Laxmi Pujan (regular session closed; Muhurat only)
    date(2022, 10, 26),  # Diwali Balipratipada
    date(2022, 11, 8),   # Guru Nanak Jayanti

    # ---------- 2023 ----------
    date(2023, 1, 26),   # Republic Day
    date(2023, 3, 7),    # Holi
    date(2023, 3, 30),   # Ram Navami
    date(2023, 4, 4),    # Mahavir Jayanti
    date(2023, 4, 7),    # Good Friday
    date(2023, 4, 14),   # Dr. Ambedkar Jayanti
    date(2023, 5, 1),    # Maharashtra Day
    date(2023, 6, 29),   # Bakri Id
    date(2023, 8, 15),   # Independence Day
    date(2023, 9, 19),   # Ganesh Chaturthi
    date(2023, 10, 2),   # Gandhi Jayanti
    date(2023, 10, 24),  # Dussehra
    date(2023, 11, 14),  # Diwali Balipratipada
    date(2023, 11, 27),  # Guru Nanak Jayanti
    date(2023, 12, 25),  # Christmas

    # ---------- 2024 ----------
    date(2024, 1, 22),   # Ram Mandir Pran Pratishtha (one-off)
    date(2024, 1, 26),   # Republic Day
    date(2024, 3, 8),    # Mahashivratri
    date(2024, 3, 25),   # Holi
    date(2024, 3, 29),   # Good Friday
    date(2024, 4, 11),   # Eid-ul-Fitr
    date(2024, 4, 17),   # Ram Navami
    date(2024, 5, 1),    # Maharashtra Day
    date(2024, 5, 20),   # Lok Sabha election (Mumbai constituency)
    date(2024, 6, 17),   # Bakri Id
    date(2024, 7, 17),   # Muharram
    date(2024, 8, 15),   # Independence Day
    date(2024, 10, 2),   # Gandhi Jayanti
    date(2024, 11, 1),   # Diwali/Laxmi Pujan (Muhurat trading only)
    date(2024, 11, 15),  # Guru Nanak Jayanti
    date(2024, 11, 20),  # Maharashtra Assembly election
    date(2024, 12, 25),  # Christmas

    # ---------- 2025 ----------
    date(2025, 1, 26),   # Republic Day (Sun — kept for reference)
    date(2025, 2, 26),   # Mahashivratri
    date(2025, 3, 14),   # Holi
    date(2025, 3, 31),   # Eid-ul-Fitr
    date(2025, 4, 10),   # Mahavir Jayanti
    date(2025, 4, 14),   # Dr. Ambedkar Jayanti
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 1),    # Maharashtra Day
    date(2025, 8, 15),   # Independence Day
    date(2025, 8, 27),   # Ganesh Chaturthi
    date(2025, 10, 2),   # Gandhi Jayanti / Dussehra
    date(2025, 10, 21),  # Diwali/Laxmi Pujan
    date(2025, 10, 22),  # Diwali Balipratipada
    date(2025, 11, 5),   # Guru Nanak Jayanti
    date(2025, 12, 25),  # Christmas
}


@lru_cache(maxsize=1)
def nse_holidays() -> Set[date]:
    """Return NSE equity-segment holidays.

    Always seeded by the static ``KNOWN_HOLIDAYS`` set (which covers
    2021-2025) and *augmented* by whatever nsepython returns — which
    is typically only the current year.  Merging both means historical
    days like 2022-01-26 are correctly flagged as holidays even when
    nsepython only knows about the current year.
    """
    holidays: Set[date] = set(KNOWN_HOLIDAYS)
    try:
        from nsepython import holiday_master  # type: ignore

        raw = holiday_master()
        equities = raw.get("CM", []) or raw.get("Equities", [])
        for row in equities:
            try:
                holidays.add(datetime.strptime(row["tradingDate"], "%d-%b-%Y").date())
            except (KeyError, ValueError):
                continue
    except Exception as exc:  # broad: nsepython failures are non-fatal
        log.warning("nsepython unavailable (%s) — using static list only.", exc)
    return holidays


def is_trading_day(d: date) -> bool:
    """A weekday that isn't on the NSE holiday list."""
    if d.weekday() >= 5:  # Saturday / Sunday
        return False
    return d not in nse_holidays()


def previous_trading_day(d: date) -> date:
    """Walk back at most 10 days until we hit a trading day."""
    cur = d - timedelta(days=1)
    for _ in range(10):
        if is_trading_day(cur):
            return cur
        cur -= timedelta(days=1)
    raise RuntimeError(f"No trading day found within 10 days before {d}")


def trading_days_between(start: date, end: date) -> Iterable[date]:
    """Yield each trading day in ``[start, end]`` inclusive."""
    cur = start
    while cur <= end:
        if is_trading_day(cur):
            yield cur
        cur += timedelta(days=1)


def market_session_filter(df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
    """Drop rows outside the NSE cash-segment session window."""
    open_t = time(9, 15)
    close_t = time(15, 30)
    times = pd.to_datetime(df[dt_col]).dt.time
    mask = (times >= open_t) & (times <= close_t)
    dropped = (~mask).sum()
    if dropped:
        log.info("market_session_filter: dropped %d off-session rows", dropped)
    return df.loc[mask].copy()
