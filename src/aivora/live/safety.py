"""Safety rails.

Every entry to a live-order function goes through
:func:`assert_can_trade_live`.  Any failed check raises and the
caller MUST NOT retry silently.

Rails covered:
    - Master switch must be ON.
    - Daily loss must be within the configured cap.
    - We must be inside the configured session window on a
      trading day.
    - Kite credentials must be present.
    - The frozen model files must exist (a stale model day would
      otherwise re-use last month's predictions unnoticed).
"""

from __future__ import annotations

from datetime import datetime

from ..utils.calendar import is_trading_day
from ..utils.config import get_config
from ..utils.logger import get_logger

log = get_logger(__name__)


class SafetyError(RuntimeError):
    pass


def _assert_creds(kite) -> None:
    """Credential pre-flight shared by the entry and exit paths."""
    if kite is not None:
        creds = getattr(kite, "creds", None)
        if creds is None or not creds.api_key or not creds.access_token:
            raise SafetyError(
                "Kite credentials missing for this account — reconnect Kite "
                "from the Profile page"
            )
    else:
        creds = get_config().kite_credentials()
        if not creds.api_key or not creds.access_token:
            raise SafetyError("Kite credentials missing in .env")


def assert_can_exit_live(portfolio, kite=None) -> None:
    """Pre-flight for CLOSING a live position.

    Deliberately far weaker than the entry check.  Exiting must not inherit
    gates that exist to limit *new* risk: the master switch, the entry time
    window and the daily loss cap.  Applying those to exits stranded open
    positions — worst of all, the daily-loss-cap check blocked closing losers
    at precisely the moment the cap was breached, and ``emergency_square_off``
    routes through here too, so the panic button was blocked as well.

    Only two things genuinely prevent an exit: the market being shut, and not
    having usable credentials to send the order.
    """
    state = portfolio.load()

    if state["mode"] != "live":
        raise SafetyError("Portfolio is not in live mode")

    now = datetime.now()
    if not is_trading_day(now.date()):
        raise SafetyError(f"{now.date()} is not a trading day")

    _assert_creds(kite)


def assert_can_trade_live(portfolio, kite=None) -> None:
    """Called on every live order path.  Raises ``SafetyError`` on any failure.

    ``kite`` is the client that will actually place the order.  In multi-user
    mode its credentials come from the caller's encrypted broker record, not
    from the process environment, so the credential pre-flight below must
    inspect *those* credentials.  Checking ``.env`` instead used to reject
    perfectly valid per-user sessions with "Kite credentials missing in .env".

    It stays optional so the legacy single-user scheduler path, which has no
    per-user client, keeps working against the environment credentials.
    """
    state = portfolio.load()

    if not state.get("master_switch"):
        raise SafetyError("Master switch is OFF")

    if state["mode"] != "live":
        raise SafetyError("Portfolio is not in live mode")

    now = datetime.now()
    if not is_trading_day(now.date()):
        raise SafetyError(f"{now.date()} is not a trading day")

    settings = state["settings"]
    start_min = int(settings["min_minutes_since_open"])
    end_min = int(settings["max_minutes_since_open"])
    hh_mm = now.hour * 60 + now.minute
    msoo = hh_mm - (9 * 60 + 15)
    if not (start_min <= msoo <= end_min):
        raise SafetyError(
            f"Outside configured trading window ({start_min}-{end_min} min after open); "
            f"current msoo={msoo}"
        )

    # Daily loss cap.
    today = now.date().isoformat()
    today_realized = sum(
        float(t.get("realized_pnl") or 0.0)
        for t in state["trades"]
        if str(t.get("exit_time", ""))[:10] == today
    )
    cap_pct = float(settings.get("daily_loss_limit_pct", 0.05))
    cap = -cap_pct * float(state["initial_capital"])
    if today_realized <= cap:
        raise SafetyError(
            f"Daily loss cap breached: today_realized={today_realized:.2f} <= cap={cap:.2f}"
        )

    # Kite creds present — on the client that will place the order.
    _assert_creds(kite)

    # Frozen model present?
    models_dir = get_config().paths["models_dir"]
    for name in ("current_up.pkl", "current_down.pkl"):
        if not (models_dir / name).exists():
            raise SafetyError(f"Frozen model missing: {models_dir / name}")


def check_ip_whitelist_hint() -> str:
    """Return a human-friendly reminder about IP-whitelisting for order APIs.

    We can't actually verify Zerodha's whitelist from here — this
    is a message string the UI displays as a persistent warning.
    """
    return (
        "Zerodha requires static-IP whitelisting for order-placement APIs. "
        "If your public IP changes (mobile network, hotel WiFi), orders will "
        "be rejected with a 'not whitelisted' error. Confirm from your "
        "current network before enabling live mode."
    )
