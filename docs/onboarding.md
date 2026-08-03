# Getting Started with AiVora

A step-by-step guide to connecting your Zerodha account and running your first
trade.

Read the whole page once before you start. Two of the steps involve Zerodha
approvals that can take a few hours, so it helps to know what is coming.

---

## What AiVora actually does

AiVora watches NIFTY and BANKNIFTY on 5-minute candles. Every five minutes it
runs a machine-learning model over ~92 market features and asks one question:
*is there a high-conviction move about to happen?*

If the model's confidence crosses your threshold, AiVora buys an at-the-money
option — a CALL if it expects the index up, a PUT if down — sizes the position
from your capital, and manages the exit automatically (target, stop loss, or a
time-based exit after roughly an hour).

You do not need to understand machine learning to use it. You do need to
understand risk, which is why the rest of this page spends more time on safety
than on features.

**Two modes:**

- **Paper** — simulated money, real market data and real prices. Nothing
  reaches your broker.
- **Live** — real orders, real money, through your own Zerodha account.

They run independently. You can have both on at once, and you should start with
paper only.

---

## Before you start

You will need:

| | |
|---|---|
| A Zerodha trading account | with F&O enabled |
| A Kite Connect subscription | ₹500/month, from Zerodha |
| Some capital | ₹50,000 is a workable minimum — see *How much capital?* below |
| About 30 minutes | plus waiting time for Zerodha to activate things |

**Important:** every AiVora user needs their **own** Kite Connect app. You
cannot share someone else's. Zerodha ties each app to a single trading account,
and your API credentials are stored separately and encrypted per user.

---

## Step 1 — Create your AiVora account

Go to the AiVora dashboard and click **Create account**. Enter your email, a
display name, and a password of at least 8 characters.

That is the whole signup. You are now logged in, but AiVora cannot see any
market data yet — that comes from your broker connection.

---

## Step 2 — Create your Kite Connect app

This is the step that costs money and takes the longest, so do it early.

1. Go to **[developers.kite.trade](https://developers.kite.trade)** and log in
   with your Zerodha credentials.
2. Click **Create new app**.
3. Fill in:
   - **App name** — anything, e.g. `AiVora`
   - **Redirect URL** — this must be exactly:
     ```
     https://auth.aivora-self.com/kite/callback
     ```
     Get this wrong and login will fail with a redirect mismatch.
   - **Postback URL** — leave blank
   - **Zerodha Client ID** — your own, e.g. `BF1234`
4. Pay the subscription (₹500/month).

Once created, the app page shows your **API key**. Click **Show API secret** to
reveal the secret. You will need both in Step 4.

> **Historical data:** if you also want the historical-data add-on it is a
> separate ₹500/month. AiVora's live trading does not require it — that is only
> for running your own backtests.

---

## Step 3 — Whitelist the server IP

**Do not skip this.** Zerodha blocks order placement from any server whose IP
is not on your allow-list. Market data works without it, which makes this
failure confusing: everything looks fine until the moment you try to trade.

1. On [developers.kite.trade](https://developers.kite.trade), go to
   **Profile** — not the app page. The IP list lives on your profile, which is
   why it is easy to miss.
2. Add this IP:
   ```
   43.205.204.155
   ```
3. Save.

Activation is usually quick but can take a few hours. If you skip this, your
first live order fails with:

```
No IPs configured for this app. Add allowed IPs on the Kite developer console.
```

Add only that one IPv4 address. Do not add an IPv6 address — the trading engine
does not use one.

---

## Step 4 — Connect Zerodha inside AiVora

In AiVora, go to **Profile → Zerodha (Kite Connect)** and fill in:

- **Client ID** — your Zerodha user id, e.g. `BF1234`
- **API Key** — from Step 2
- **API Secret** — from Step 2

Click **Save Zerodha credentials**.

Both secrets are encrypted before they touch the database. They are never
written to logs, and no one — including an administrator — can read them back
out through the interface.

Now click the **Login to Kite** link that appears. This sends you to Zerodha,
you approve access, and you land back in AiVora with a green *"Kite connected"*
message.

### About the daily token

Zerodha access tokens **expire every morning, around 7:30 AM**. When that
happens AiVora logs `Kite disconnected. Reconnect via Profile page` and stops
trading until you reconnect.

Two options:

- **Manual** — click the Kite login link each morning. Takes ten seconds.
- **Automatic** — also save your Zerodha password and TOTP secret in the same
  form. AiVora then refreshes the token by itself.

The TOTP secret is the long seed string shown when you set up two-factor
authentication, **not** the 6-digit code that changes every 30 seconds. If you
no longer have it, you can re-run TOTP setup in your Zerodha account to get a
fresh one.

Automatic refresh is convenient but means storing your password. Encrypted, but
stored. Choose according to how comfortable you are with that.

---

## Step 5 — Set up your paper portfolio

**Start here. Do not start with live.**

In the sidebar, set the mode toggle to **paper**, then set your starting
capital. Use the same number you actually intend to trade with later — if you
plan to trade ₹50,000, paper-trade ₹50,000. Paper-trading ₹10 lakh and then
going live with ₹50,000 will teach you nothing useful, because percentage
returns and drawdowns both depend on the capital base.

### How much capital?

One at-the-money NIFTY or BANKNIFTY option lot costs roughly **₹7,000–8,000** in
premium. With `max_trades_per_day` at 10 and several positions potentially open
at once, you want room for a handful of concurrent trades.

- **Below ~₹40,000** — you will frequently be unable to take signals
- **₹50,000** — workable, roughly 6 concurrent positions
- **₹1,00,000+** — comfortable

There is one thing worth knowing about scaling: position size is computed as
`risk_per_trade_pct × capital`, converted to whole lots. Because one lot already
costs about ₹7,300, **everything from ₹50,000 up to about ₹7.3 lakh trades
exactly one lot**. Adding capital in that range makes you safer (the same rupee
loss is a smaller percentage) but does not increase your rupee profit.

---

## Step 6 — Check your settings

Open **Settings** and confirm these values. They come from a 55-month
walk-forward backtest and are the configuration the system was validated on:

| Setting | Value | What it does |
|---|---|---|
| `prob_threshold_up` | 0.55 | Confidence needed for a CALL |
| `prob_threshold_down` | 0.55 | Confidence needed for a PUT |
| `min_minutes_since_open` | 0 | Start trading at 9:15 |
| `max_minutes_since_open` | 300 | No new entries after 14:15 |
| `vol_regime_min` | 0.0 | Volatility filter off |
| `vol_regime_max` | 999.0 | Volatility filter off |
| `max_trades_per_day` | 10 | Daily cap |
| `risk_per_trade_pct` | 0.02 | 2% of capital per trade |
| `horizon_candles` | 12 | Time exit after ~60 minutes |
| `daily_loss_limit_pct` | 0.05 | Stop entering after a 5% down day |

**Keep paper and live identical.** If they differ, your paper results tell you
nothing about what live would have done. This is a real trap — a mismatched
`min_minutes_since_open` once caused a system to take live signals at 9:15 while
paper silently ignored everything before 9:45, and the two books diverged for
weeks before anyone noticed.

---

## Step 7 — Turn it on

In the sidebar, under **Trading control**, click the master switch to start.

The sidebar shows both engines at a glance:

```
Engines: 🟢 paper ON  ·  ⚪ live OFF
```

The switch only ever toggles the mode you currently have selected, so check that
line rather than assuming.

Within five minutes you should see entries appear in the activity log.

---

## Reading the activity log

AiVora writes one summary line per five-minute tick, tagged with the mode that
produced it.

**Nothing to trade** — the normal state, most of the day:
```
✅ 11:45 — checked, nothing to trade | NIFTY 🚫 DOWN conviction 0.16 — needs 0.55 (short by 0.39)
```
It tells you exactly how far the model was from firing.

**A trade opened:**
```
🚀 BANKNIFTY PUT — trade opened
OPEN BANKNIFTY PE strike=57700 lots=1 @ ₹641.95
```

**An entry was tried but did not fill:**
```
⚠️ BANKNIFTY PUT — entry attempted but NOT filled (see error above)
```
Look at the error line just above it. This is not a position — nothing was
opened.

**Other things you will see:**

| Message | Meaning |
|---|---|
| `🔁 already holding a trade` | One position per symbol at a time |
| `🔒 daily trade limit reached` | Hit `max_trades_per_day` |
| `⏸️ paused (master switch OFF)` | Engine is off |
| `🚫 volatility X outside [a, b]` | Volatility filter blocked it |
| `⚠️ Kite disconnected` | Token expired — reconnect |

---

## Going live

Only after your paper account has run for at least a few weeks.

Before you flip the switch, be honest with yourself about three things:

1. **Has paper actually traded?** A paper account that never took a trade has
   told you nothing.
2. **Do the paper results look like the backtest?** If paper is much worse, live
   will be worse still — live adds slippage that paper does not model.
3. **Can you afford the drawdown?** Not "will it happen" — assume it will.

Then: switch the mode toggle to **live**, set your capital, and turn the master
switch on. AiVora will ask you to confirm, because this one spends real money.

### Compare your first live fills against paper

This matters more than anything else in this guide.

Both modes see the same signals. When both are on with identical settings, the
only difference between them is the **fill price** — paper assumes a clean fill,
live gets whatever the market actually gave you.

That difference is slippage, and it is the single largest unknown when moving
from a backtest to real money. On at-the-money weekly options the bid-ask spread
is real. If your average profit per trade is a few hundred rupees and slippage
takes ₹50 of it, you have lost 20% of your edge — and no backtest will warn you.

After 10–20 live trades, compare the two. If live fills track paper closely,
your numbers are trustworthy. If they do not, fix that before increasing size.

---

## Troubleshooting

**`Kite credentials missing for this account — reconnect Kite from the Profile page`**
Your access token expired or was never set. Go to Profile and log in to Kite
again.

**`No IPs configured for this app`**
Step 3 was skipped, or not yet active. Add `43.205.204.155` to the IP list on
your Kite Connect **Profile** page and wait a few hours.

**`⚠️ Kite disconnected. Reconnect via Profile page.`**
The daily token expiry. Reconnect, or set up TOTP auto-refresh.

**Everything looks fine but no trades for days**
Usually normal. This strategy is selective — it trades only when the model is
genuinely confident, and quiet stretches of several days are expected. Check the
log: if it says `conviction 0.16 — needs 0.55`, the market simply has not offered
a setup.

**Paper trades but live does not (or the reverse)**
Their settings differ. Compare both, mode by mode. This is the most common cause
of "it works in paper but not live".

**`entry attempted but NOT filled` repeatedly**
The order is reaching Zerodha and being rejected. Read the error above the line —
usually IP whitelisting, insufficient margin, or an expired token.

---

## What this system is and is not

Some honesty, because you are about to point real money at this.

**What it is:** a strategy validated on a 55-month walk-forward backtest with
transaction costs and taxes modelled, running the same code path in paper and
live.

**What it is not:** a guarantee. Specifically:

- **Backtest fills are modelled, not observed.** Real slippage on ATM weekly
  options is the biggest unvalidated assumption in the whole system.
- **The tested period is finite.** 2022–2026 contains particular market regimes.
  Future markets will differ.
- **The strategy can lose.** It wins roughly 43% of trades and makes money
  because wins are larger than losses. Losing streaks of 15+ trades occur in the
  historical record. If a run of losses will make you turn the system off at the
  worst moment, trade smaller.
- **An edge is not permanent.** Strategies decay as markets adapt. Watch for
  live performance drifting away from expectation.

Start with paper. Move to live with an amount you can afford to lose entirely.
Increase size only after live data — not backtest data — justifies it.

---

## Costs

| Item | Cost |
|---|---|
| Kite Connect API | ₹500/month |
| Historical data add-on | ₹500/month (optional, only for your own backtests) |
| AiVora | see your plan |
| Brokerage & taxes | Zerodha's standard F&O rates, already modelled in the backtest |

---

## Quick reference

| | |
|---|---|
| Kite redirect URL | `https://auth.aivora-self.com/kite/callback` |
| Server IP to whitelist | `43.205.204.155` |
| Token expires | daily, ~7:30 AM |
| Tick frequency | every 5 minutes, 20 seconds past |
| Trading window | 9:15 to 14:15 for new entries |
| Cost per option lot | ~₹7,000–8,000 |
| Suggested minimum capital | ₹50,000 |
