# Running AiVora for someone else

Zerodha's support confirmed what the error message implies: a static IP can
be linked to **one** Kite Connect developer account. Family members can share
one developer account — separate apps under it, all client IDs added, and the
declaration that the IP is used only by you and your immediate family.

Anyone else needs their **own** IP, which means their own machine.

So this guide sets up a second, independent AiVora for one other person.
Their account, their API key, their server, their IP. Nothing shared that
matters.

If the person is immediate family, stop here — you do not need any of this.
Add their client ID under your existing developer account instead.

---

## What must not be shared

| | Why |
|---|---|
| `AIVORA_MASTER_KEY` | Encrypts every stored Kite credential. One key for both instances means either can decrypt the other's. **Generate a new one.** |
| `data/db/webapp.sqlite` | Users, credentials, portfolios, trades. Never copy it. |
| `.env` | Contains the master key. |

## What is fine to copy

| | Why |
|---|---|
| `data/db/aivora.sqlite` | NSE market data — index candles and option snapshots. Nobody's private data, and the new instance needs ~60 days of history before it can compute a single feature. |
| `models/` | The frozen model. Saves an hour of retraining on a fresh box. |

---

## 1. The instance

In Lightsail, create an instance in **Mumbai (ap-south-1)** — same region as
the exchange, and the same region your existing one is in.

Size it the same as your current instance. The workload is identical: one
feature rebuild every five minutes over a 60-day window. Guessing smaller to
save a few dollars means the tick starts running long, and a tick that runs
long stops being a five-minute tick.

Then **Networking → Create static IP → attach it to the instance**.

Two things worth knowing:

- The IP is assigned by AWS. You do not choose it. Whatever it shows —
  `13.201.x.x` or similar — that is the number your friend whitelists.
- A static IP is free while attached to a running instance and billed while
  it sits unattached. If you tear the instance down, release the IP too.
- Default quota is 5 static IPs per region per account, one of which your
  existing instance already holds. More than four of these and you will need
  a quota increase from AWS.

## 2. DNS

Kite's OAuth callback has to land on a public HTTPS URL that exactly matches
the Redirect URL in your friend's developer app. An IP will not do.

Add two A records pointing at the new static IP:

```
friend.yourdomain.com        A    <new static IP>
auth.friend.yourdomain.com   A    <new static IP>
```

Any subdomain works — this is just a label. Wait for it to resolve before
deploying, or Caddy will fail to get a certificate.

## 3. Deploy

SSH into the new instance:

```bash
git clone https://github.com/prashantkr006/Aivora.git aivora && cd aivora
cp .env.example .env
```

Generate a **new** master key and put it in `.env`:

```bash
docker run --rm python:3.12-slim sh -c "pip install -q cryptography && python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
```

Leave every `KITE_*` line blank — those are per-user now and live encrypted
in the database, not in `.env`.

From your **existing** instance, copy the market data and the model:

```bash
scp -i <key.pem> ~/aivora/data/db/aivora.sqlite ubuntu@<new-ip>:~/aivora/data/db/
scp -i <key.pem> -r ~/aivora/models ubuntu@<new-ip>:~/aivora/
```

Then, on the new instance:

```bash
AIVORA_DOMAIN=friend.yourdomain.com bash scripts/deploy.sh
```

It installs Docker, Caddy and the firewall, builds the training parquet from
the copied database if it is missing, starts the three containers and sets up
the nightly backup.

## 4. The two crons deploy.sh does not set

Set these up yourself — the backup cron is the only one it writes.

```bash
crontab -e
```

```cron
# Kite token, every trading morning at 07:30
30 7 * * 1-5 cd /home/ubuntu/aivora && docker compose exec -T worker python -m scripts.auto_refresh_kite_tokens >> logs/token_refresh.log 2>&1

# Monthly retrain — see the note below about which Saturday
0 20 * * 6 [ $(date +\%d) -ge 8 ] && [ $(date +\%d) -le 14 ] && cd /home/ubuntu/aivora && docker compose exec -T dashboard python -m scripts.monthly_retrain --tag monthly >> logs/retrain.log 2>&1
```

**On the retrain date.** Your existing instance runs this on the *first*
Saturday. That turns out to be the worst choice: `freeze_model` reserves the
newest month in the parquet for validation and trains on complete months
before it, so a run on 1 August held out July and trained only to 30 June.
It added no new training data at all. The second Saturday puts the previous
month into training instead. Worth fixing on your own instance too.

## 5. What your friend does

Send them the dashboard URL and these steps. Do **not** send them your IP —
they need theirs.

1. Register at `https://friend.yourdomain.com`. The first account registered
   on a fresh instance becomes admin, so let them register first.
2. Sign up for Kite Connect at
   [developers.kite.trade](https://developers.kite.trade) with **their own**
   Zerodha account and create an app.
3. Set the app's Redirect URL to exactly:
   `https://auth.friend.yourdomain.com/kite/callback`
4. On their **Profile** page in the developer console — not the app page —
   add the static IP of the new instance.
5. In AiVora: Profile → Zerodha → paste api_key and api_secret → Connect.
6. Start in **paper** mode.

Two things to warn them about:

- **Zerodha allows one IP modification per calendar week.** Get the IP right
  the first time or wait seven days.
- Whitelisting takes effect some hours after saving. Market data works
  without it, so everything looks fine until the first live order fails —
  which is exactly the confusing failure the onboarding guide describes.

## 6. Check it before they trade

```bash
cd ~/aivora && docker compose ps
cd ~/aivora && docker compose logs worker --tail 40
```

The worker should register a tick within a minute of them turning the switch
on. If the log says the spot window is empty, `data/db/aivora.sqlite` did not
copy across.

---

## What this does not settle

This is the technical arrangement. Whether providing an algo to someone else
— free or paid — needs registration under SEBI's retail algo rules is a
separate question, and one for Zerodha and a compliance professional rather
than for this document. Ask before it becomes a paid service.
