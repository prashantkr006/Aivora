"""AiVora live-trading dashboard (Streamlit).

Single-page layout.  Every number comes from the ``Portfolio``
JSON so what you see is exactly what the scheduler wrote.

Run:

    python -m scripts.run_dashboard
    # or
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aivora.live import kite_auth
from aivora.live import scheduler as sched
from aivora.live.inference import LiveInference
from aivora.live.portfolio import Portfolio, default_settings
from aivora.live.position_tracker import emergency_square_off
from aivora.live.safety import check_ip_whitelist_hint
from aivora.utils.config import get_config

st.set_page_config(page_title="AiVora — live", layout="wide", page_icon="📈")


# =============================================================
#  State
# =============================================================
def _get_portfolio() -> Portfolio:
    mode = st.session_state.get("mode", "paper")
    return Portfolio(mode=mode)


def _refresh_now():
    """Trigger a manual tick.  Returns the tick's report dict."""
    p = _get_portfolio()
    return sched.run_tick(p)


# =============================================================
#  Kite auth (top of the app — before anything else runs)
# =============================================================
def _handle_kite_redirect() -> None:
    """If Kite redirected back with ?request_token=…, exchange it
    for an access_token and clear the URL.

    Called once at the top of main() so the user sees the outcome
    on the very next render, no manual paste required.
    """
    params = st.query_params
    rq = params.get("request_token")
    if not rq:
        return
    # Streamlit returns a str (or list of str depending on version).
    if isinstance(rq, list):
        rq = rq[0]
    try:
        kite_auth.exchange_request_token(rq)
        st.toast("✅ Kite access token updated.", icon="✅")
        # Persist a hint in session_state so it survives the URL clear.
        st.session_state["_kite_last_msg"] = ("success",
            "Kite login complete — token written to .env.")
    except Exception as exc:
        st.session_state["_kite_last_msg"] = ("error",
            f"Kite token exchange failed: {exc}")
    # Clear the query params so a browser refresh doesn't re-exchange
    # a one-shot token (Kite request_tokens are single-use).
    try:
        st.query_params.clear()
    except Exception:
        # Older Streamlit versions expose experimental_set_query_params.
        try:
            st.experimental_set_query_params()
        except Exception:
            pass


def kite_auth_panel() -> None:
    """Sidebar section: token status, browser-login button, TOTP button."""
    st.sidebar.markdown("## 🔑 Kite authentication")
    status = kite_auth.token_status()
    if status.present:
        icon = "🟢"
    else:
        icon = "🔴"
    st.sidebar.markdown(f"{icon} **{status.hint}**")
    if status.last_modified:
        st.sidebar.caption(
            f"token last written: {status.last_modified.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # Show any success/error message from the last redirect exchange.
    msg = st.session_state.pop("_kite_last_msg", None)
    if msg:
        (st.sidebar.success if msg[0] == "success" else st.sidebar.error)(msg[1])

    # --- Browser flow ---
    try:
        url = kite_auth.login_url()
    except Exception as exc:
        st.sidebar.error(f"Login URL error: {exc}")
        url = None
    if url:
        st.sidebar.markdown(
            f"[🔑 Login to Kite (opens Zerodha)]({url})",
            unsafe_allow_html=False,
        )
        st.sidebar.caption(
            "After Zerodha login you'll be redirected back to this app; "
            "the token will be written automatically. "
            "Ensure your app's registered redirect URL on "
            "developers.kite.trade matches the URL this dashboard runs on."
        )

    # --- Manual paste fallback (in case redirect URL isn't set up yet) ---
    with st.sidebar.expander("Paste request_token / redirect URL manually"):
        pasted = st.text_input(
            "Paste ?request_token=… (or the whole redirect URL)",
            key="_kite_manual_paste", value="",
        )
        if st.button("Exchange", key="_kite_manual_btn") and pasted:
            rq = kite_auth.extract_request_token(pasted)
            if not rq:
                st.error("Couldn't find a request_token in that string.")
            else:
                try:
                    kite_auth.exchange_request_token(rq)
                    st.success("Access token updated.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Exchange failed: {exc}")

    # --- Optional TOTP one-click ---
    with st.sidebar.expander("Auto-login (TOTP)  — advanced"):
        st.caption(
            "Runs the full login server-side using KITE_USER_ID + "
            "KITE_PASSWORD + KITE_TOTP_SECRET from .env.  No browser popup."
        )
        if st.button("🤖 Auto-login now", key="_kite_totp_btn"):
            with st.spinner("Logging in via TOTP…"):
                try:
                    kite_auth.totp_auto_login()
                    st.success("Auto-login OK — token refreshed.")
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))


# =============================================================
#  Sidebar — mode + capital + settings
# =============================================================
def sidebar():
    # Kite auth first — a stale token is the single most common cause
    # of every other button silently failing, so surface it up top.
    kite_auth_panel()
    st.sidebar.divider()

    st.sidebar.markdown("## Mode")
    st.session_state.setdefault("mode", "paper")
    mode = st.sidebar.radio(
        "Portfolio mode",
        ["paper", "live"],
        index=(0 if st.session_state["mode"] == "paper" else 1),
        format_func=lambda m: "📊 Paper" if m == "paper" else "💰 Live",
    )
    if mode != st.session_state["mode"]:
        st.session_state["mode"] = mode
        st.rerun()

    p = _get_portfolio()
    summary = p.summary()

    st.sidebar.markdown("## Capital")
    st.sidebar.metric(
        "Initial capital",
        f"₹{summary['initial_capital']:,.0f}",
    )
    new_cap = st.sidebar.number_input(
        "Set new initial capital (only when no trades)",
        min_value=1000.0, max_value=10_000_000.0,
        value=float(summary["initial_capital"]),
        step=1000.0,
    )
    if st.sidebar.button("Apply capital"):
        try:
            p.set_initial_capital(new_cap)
            st.sidebar.success("Capital updated.")
            st.rerun()
        except RuntimeError as exc:
            st.sidebar.error(str(exc))

    st.sidebar.markdown("## Strategy settings")
    settings = summary["settings"]
    with st.sidebar.expander("Thresholds & filters", expanded=False):
        thr_up = st.slider("Prob threshold UP", 0.30, 0.90, float(settings["prob_threshold_up"]), 0.01)
        thr_dn = st.slider("Prob threshold DOWN", 0.30, 0.90, float(settings["prob_threshold_down"]), 0.01)
        tp = st.slider("Take profit %", 0.10, 2.0, float(settings["take_profit_pct"]), 0.05)
        sl = st.slider("Stop loss %", 0.10, 1.0, float(settings["stop_loss_pct"]), 0.05)
        max_tr = st.slider("Max trades / day", 1, 6, int(settings["max_trades_per_day"]))
        vr_min = st.slider("Vol regime min", 0.0, 0.5, float(settings.get("vol_regime_min") or 0.15), 0.05)
        vr_max = st.slider("Vol regime max", 0.5, 1.0, float(settings.get("vol_regime_max") or 0.90), 0.05)
        if st.button("Save settings"):
            p.update_settings({
                "prob_threshold_up": thr_up,
                "prob_threshold_down": thr_dn,
                "take_profit_pct": tp,
                "stop_loss_pct": sl,
                "max_trades_per_day": max_tr,
                "vol_regime_min": vr_min,
                "vol_regime_max": vr_max,
            })
            st.success("Settings saved.")
            st.rerun()

    st.sidebar.markdown("## Danger zone")
    if st.sidebar.button("Reset portfolio (blow away trades)"):
        p.reset()
        st.sidebar.warning("Portfolio reset.")
        st.rerun()

    if mode == "live":
        st.sidebar.warning(check_ip_whitelist_hint())


# =============================================================
#  Top bar
# =============================================================
def top_bar():
    p = _get_portfolio()
    summary = p.summary()
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
    with col1:
        st.markdown(f"**Mode:** {'📊 PAPER' if summary['mode'] == 'paper' else '💰 LIVE'}")
    with col2:
        last = summary.get("last_data_update")
        if last:
            age = (datetime.now() - datetime.fromisoformat(last)).total_seconds()
            st.markdown(f"**Last data:** {last[-8:]} ({age:.0f}s ago)")
        else:
            st.markdown("**Last data:** —")
    with col3:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(f"**IST clock:** {now}")
    with col4:
        # Kite-based tick — used during market hours as the live heartbeat.
        # Guarded against off-session use so a stray click at 20:00 IST
        # doesn't waste an API call.
        if st.button("🔄 Refresh (Kite)", help="Live tick via Kite — market hours only"):
            with st.spinner("Fetching from Kite…"):
                try:
                    r = _refresh_now()
                    if r.get("skipped"):
                        st.info(f"Skipped: {r['skipped']}")
                    else:
                        st.success("Data refreshed.")
                except Exception as exc:
                    st.error(f"Refresh failed: {exc}")
    with col5:
        # Dhan-based backfill — works any time, including after market
        # close.  Uses the same daily-update path as scripts.run_pipeline.
        if st.button("📥 Backfill (Dhan)", help="Fetch last ~5 days from Dhan — safe any time"):
            with st.spinner("Fetching from Dhan (this can take 10-20 s)…"):
                try:
                    from aivora.pipeline import pipeline as pipe_mod
                    pipe_mod.run_daily_update(record_options=False)
                    p.set_last_data_update(datetime.now())
                    p.append_log("Manual Dhan backfill completed.", "info")
                    st.success("Backfilled via Dhan. Reload to see the new candles.")
                except Exception as exc:
                    p.append_log(f"Dhan backfill failed: {exc}", "error")
                    st.error(f"Backfill failed: {exc}")


# =============================================================
#  Master switch + controls
# =============================================================
def controls():
    p = _get_portfolio()
    summary = p.summary()
    col1, col2, col3 = st.columns([2, 2, 3])
    with col1:
        cur = summary["master_switch"]
        label = "🟢 Trading ON — click to STOP" if cur else "🔴 Trading OFF — click to START"
        if st.button(label, use_container_width=True):
            # Live requires explicit confirmation.
            if summary["mode"] == "live" and not cur:
                st.session_state["_confirm_live"] = True
            else:
                p.set_master_switch(not cur)
                st.rerun()
    with col2:
        if st.button("🛑 EMERGENCY SQUARE OFF", use_container_width=True, type="primary"):
            try:
                # Spot map from the latest parquet row per symbol.
                inf = LiveInference()
                spot_map = {}
                for inst in get_config().instruments:
                    r = inf.latest_prediction(inst["symbol"])
                    if r is not None:
                        spot_map[inst["symbol"]] = r.spot_close
                n = emergency_square_off(p, datetime.now(), spot_map)
                st.warning(f"Closed {n} position(s).")
            except Exception as exc:
                st.error(f"Square-off failed: {exc}")
            st.rerun()
    with col3:
        if st.button("♻️ Force refresh model (freeze last 12 months)"):
            import subprocess
            with st.spinner("Freezing model — this can take a couple minutes…"):
                r = subprocess.run(
                    [sys.executable, "-m", "scripts.freeze_model"],
                    cwd=str(_ROOT), capture_output=True, text=True,
                )
            if r.returncode == 0:
                st.success("Model refreshed.")
            else:
                st.error(r.stderr[-2000:])

    # Live-mode confirm dialog.
    if st.session_state.get("_confirm_live"):
        st.error(
            "You're about to enable LIVE trading. Real money at risk. "
            "Type CONFIRM below and press Enter."
        )
        typed = st.text_input("Type CONFIRM to enable live trading", key="_confirm_txt")
        if typed.strip() == "CONFIRM":
            p.set_master_switch(True)
            st.session_state["_confirm_live"] = False
            st.session_state["_confirm_txt"] = ""
            st.success("Live trading ARMED.")
            st.rerun()


# =============================================================
#  Metric cards + trades table + equity curve
# =============================================================
def metric_cards():
    p = _get_portfolio()
    s = p.summary()
    c = st.columns(5)
    c[0].metric("Today P&L", f"₹{s['today_pnl']:,.0f}",
                delta=f"{s['today_pnl']/s['initial_capital']:+.2%}")
    c[1].metric("Trades today", f"{s['trades_today']}",
                delta=f"open: {s['n_open_trades']}")
    c[2].metric("Win rate today", f"{s['win_rate_today']:.0%}")
    c[3].metric("Drawdown", f"{s['drawdown_pct']:.2%}")
    c[4].metric("Capital",
                f"₹{s['current_capital']:,.0f}",
                delta=f"unrl {s['unrealized_pnl_total']:+.0f}")


def trades_table():
    p = _get_portfolio()
    state = p.load()
    if not state["trades"]:
        st.info("No trades yet.")
        return
    df = pd.DataFrame(state["trades"])
    # Neaten columns for display.
    df["Time"] = pd.to_datetime(df["entry_time"]).dt.strftime("%H:%M")
    df["Date"] = pd.to_datetime(df["entry_time"]).dt.strftime("%Y-%m-%d")
    df["Type"] = df["side"].map({"CE": "CALL", "PE": "PUT"})
    df["Strike"] = df["strike"].astype(int)
    df["Lots"] = df["lots"]
    df["Entry"] = df["entry_premium"].round(2)
    df["Current"] = df["current_premium"].astype(float).round(2)
    df["P&L"] = (
        df["realized_pnl"].fillna(df["unrealized_pnl"]).astype(float).round(2)
    )
    df["Status"] = df.apply(
        lambda r: r.get("exit_reason") or ("OPEN" if pd.isna(r.get("exit_time")) else "closed"),
        axis=1,
    )
    show = df[["Date", "Time", "symbol", "Type", "Strike", "Lots",
               "Entry", "Current", "P&L", "Status"]]
    st.dataframe(
        show.sort_values(["Date", "Time"], ascending=[False, False]),
        use_container_width=True, height=350,
    )


def equity_curve():
    p = _get_portfolio()
    state = p.load()
    if not state["trades"]:
        return
    df = pd.DataFrame(state["trades"])
    df = df[df["realized_pnl"].notna()].copy()
    if df.empty:
        return
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df = df.sort_values("exit_time")
    df["equity"] = df["realized_pnl"].astype(float).cumsum() + float(state["initial_capital"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["exit_time"], y=df["equity"], mode="lines+markers",
        name="Equity", line=dict(color="#3366cc"),
    ))
    fig.add_hline(y=float(state["initial_capital"]), line_dash="dash", line_color="grey")
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        height=280, title="Equity curve (realised P&L)",
        yaxis_title="Equity (INR)",
    )
    st.plotly_chart(fig, use_container_width=True)


def event_log():
    p = _get_portfolio()
    state = p.load()
    with st.expander("Event log", expanded=False):
        rows = list(reversed(state.get("log", [])))
        if not rows:
            st.write("(empty)")
            return
        for r in rows[:40]:
            level = r.get("level", "info")
            icon = {"error": "❌", "warn": "⚠️", "info": "ℹ️"}.get(level, "•")
            st.text(f"{icon}  {r['ts']}  {r['msg']}")


# =============================================================
#  Main
# =============================================================
def main():
    # Must run BEFORE we touch anything that needs a Kite token —
    # this is where a redirect back from Zerodha lands.
    _handle_kite_redirect()

    st.title("AiVora — automated Nifty / Bank Nifty options")
    sidebar()
    top_bar()
    st.divider()
    controls()
    st.divider()
    metric_cards()
    st.divider()
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Trades")
        trades_table()
    with right:
        st.subheader("Equity")
        equity_curve()
    event_log()

    # Kick off the background scheduler if it isn't running.
    try:
        sched.ensure_started(_get_portfolio(), interval_seconds=300)
    except Exception as exc:
        st.warning(f"Scheduler not started: {exc}")

    # Soft-auto-refresh the page every 30 seconds so timestamps and
    # open-trade marks keep moving without user interaction.
    time.sleep(0.05)  # yield to Streamlit's own event loop
    st.caption("Page auto-refreshes every 30 seconds.")
    st_autorefresh_key = "_autorefresh_placeholder"
    st.session_state[st_autorefresh_key] = st.session_state.get(st_autorefresh_key, 0) + 1
    st.query_params.update(_="tick")
    time.sleep(0)  # noop keep-alive


if __name__ == "__main__":
    main()
