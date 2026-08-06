"""Check an instance is ready to trade tomorrow, before it has to.

Everything here is something that has actually gone wrong on a live box:

* a TOTP seed that was never valid, failing at 6:15am for three days into a
  log nobody read while the user reconnected by hand each morning;
* fresh portfolios coming up with the volatility filter on — a config the
  55-month walk-forward had already measured and rejected;
* live capital left at the Rs 1,00,000 default, so the first cash sync
  books a withdrawal that never happened;
* a second user pasting the first user's api_key, which would send their
  orders to the wrong account.

Read-only. Prints a line per check and exits non-zero if anything is
seriously wrong, so it can go in a cron of its own.

    python -m scripts.preflight
    python -m scripts.preflight --user 27
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivora.live.kite_auth import check_totp_secret  # noqa: E402
from aivora.utils.config import get_config  # noqa: E402
from aivora.webapp import brokers, db as webapp_db, portfolios  # noqa: E402

OK, WARN, BAD = "  ok  ", " warn ", " FAIL "

CANONICAL = {
    "prob_threshold_up": 0.55, "prob_threshold_down": 0.55,
    "take_profit_pct": 0.60, "stop_loss_pct": 0.30,
    "min_minutes_since_open": 0, "max_minutes_since_open": 300,
    "vol_regime_min": 0.0, "vol_regime_max": 999.0,
    "max_trades_per_day": 10, "horizon_candles": 12,
}

_fails: List[str] = []
_warns: List[str] = []


def say(level: str, msg: str) -> None:
    print(f"[{level}] {msg}")
    if level is BAD:
        _fails.append(msg)
    elif level is WARN:
        _warns.append(msg)


# =============================================================
def check_instance() -> None:
    print("\n=== instance ===")
    cfg = get_config()

    lots = {i["symbol"]: int(i["lot_size"]) for i in cfg.instruments}
    say(OK, f"config lot sizes: {lots} (live reads Kite's own value anyway)")

    models = cfg.paths["models_dir"]
    missing = [f for f in ("current_up.pkl", "current_down.pkl")
               if not (models / f).exists()]
    if missing:
        say(BAD, f"frozen model missing: {', '.join(missing)}")
    else:
        import json
        meta = models / "current_model.json"
        when = json.loads(meta.read_text())["frozen_at"] if meta.exists() else "?"
        say(OK, f"frozen model present (frozen_at {when})")

    # Features need roughly 60 days of history before the first tick.
    from aivora.pipeline import database
    try:
        cutoff = datetime.now() - timedelta(days=70)
        spot = database.load_spot_futures_since(cutoff)
        if spot.empty:
            say(BAD, "no spot data in the last 70 days — the tick cannot "
                     "build features")
        else:
            last = spot["datetime"].max()
            age = (datetime.now() - last.to_pydatetime()).days
            level = OK if age <= 4 else WARN
            say(level, f"market data to {last} ({age}d old, {len(spot):,} rows)")
    except Exception as exc:  # noqa: BLE001
        say(BAD, f"could not read spot data: {exc}")


def check_users(only: int | None) -> None:
    with webapp_db.connect() as c:
        users = [dict(r) for r in c.execute(
            "SELECT id, email FROM users ORDER BY id")]
    if only is not None:
        users = [u for u in users if u["id"] == only]
    if not users:
        say(BAD, "no users on this instance")
        return

    # A shared api_key means one user's orders can land in another's
    # account. Worth checking before it does.
    by_key: Dict[str, list] = defaultdict(list)

    for u in users:
        uid, email = u["id"], u["email"]
        z = brokers.get(uid, "ZERODHA")

        # Somebody who registered and stopped there is not a fault. Somebody
        # who armed the master switch without credentials is.
        armed = []
        for mode in ("paper", "live"):
            try:
                if portfolios.UserPortfolio(uid, mode).load()["master_switch"]:
                    armed.append(mode)
            except Exception:  # noqa: BLE001
                pass

        if not z or not z.api_key or not z.api_secret:
            if armed:
                print(f"\n=== user {uid} · {email} ===")
                say(BAD, f"{'/'.join(armed)} switch is ON but no Kite "
                         "api_key/api_secret is saved")
            elif only is not None:
                print(f"\n=== user {uid} · {email} ===")
                say(WARN, "registered but Kite not configured")
            continue

        print(f"\n=== user {uid} · {email} ===")
        by_key[z.api_key].append(uid)
        say(OK, f"api_key {z.api_key[:6]}… · client {z.client_id or '?'}")

        if not z.access_token:
            say(BAD, "no access token — connect Kite from the Profile page")
        else:
            ts = z.token_updated_at
            try:
                age_h = (datetime.now(timezone.utc)
                         - datetime.fromisoformat(str(ts))).total_seconds() / 3600
                level = OK if age_h < 24 else WARN
                say(level, f"access token set ({age_h:.1f}h old)")
            except Exception:  # noqa: BLE001
                say(OK, f"access token set (updated {ts})")

        # TOTP auto-login is optional, but silence about it is not helpful:
        # without it somebody has to click Connect every single morning.
        if not z.password or not z.totp_secret:
            miss = [n for n, v in (("password", z.password),
                                   ("totp_secret", z.totp_secret)) if not v]
            say(WARN, f"no morning auto-login — missing {', '.join(miss)}; "
                      "this user must connect by hand daily")
        else:
            why = check_totp_secret(z.totp_secret)
            say(BAD if why else OK,
                f"TOTP seed rejected — {why}" if why else "TOTP seed valid")

        for mode in ("paper", "live"):
            try:
                st = portfolios.UserPortfolio(uid, mode).load()
            except Exception:  # noqa: BLE001
                say(WARN, f"{mode}: no portfolio")
                continue

            drift = {k: (st["settings"].get(k), v)
                     for k, v in CANONICAL.items() if st["settings"].get(k) != v}
            say(BAD if drift else OK,
                f"{mode}: settings drifted — " + ", ".join(
                    f"{k}={h} (want {w})" for k, (h, w) in drift.items())
                if drift else f"{mode}: settings match the validated config")

            cap, sw = st["current_capital"], st["master_switch"]
            open_n = sum(1 for t in st["trades"] if not t.get("exit_time"))
            say(OK, f"{mode}: capital Rs {cap:,.2f} · switch "
                    f"{'ON' if sw else 'off'} · {open_n} open")
            if mode == "live" and sw and abs(cap - 100_000.0) < 0.01:
                say(WARN, "live capital is exactly the Rs 1,00,000 default — "
                          "if that is not the real balance, the first cash "
                          "sync will book a movement that never happened")

    for key, uids in by_key.items():
        if len(uids) > 1:
            say(BAD, f"users {uids} share one api_key — orders could land in "
                     "the wrong account")


# =============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, default=None)
    a = ap.parse_args()

    print("=" * 66)
    print(f"AiVora preflight · {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 66)

    check_instance()
    check_users(a.user)

    print("\n" + "=" * 66)
    if _fails:
        print(f"{len(_fails)} problem(s) that will stop trading:")
        for m in _fails:
            print(f"  · {m}")
    if _warns:
        print(f"{len(_warns)} thing(s) worth knowing:")
        for m in _warns:
            print(f"  · {m}")
    if not _fails and not _warns:
        print("All clear.")
    print("\nNot checked here: the HOST timezone, which crontab uses and a "
          "container cannot see.\nRun `date` on the host — it must say IST.")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
