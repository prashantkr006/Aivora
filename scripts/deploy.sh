#!/usr/bin/env bash
# =============================================================
# AiVora — one-shot production deploy on a fresh Ubuntu 22.04/24.04 VPS.
#
# Usage on the VPS (as a non-root sudoer):
#   1. git clone <your-repo> aivora && cd aivora
#   2. cp .env.example .env  &&  edit .env  (fill AIVORA_MASTER_KEY etc)
#   3. bash scripts/deploy.sh
#
# Idempotent — safe to re-run after upgrades.
# =============================================================
set -euo pipefail

DOMAIN="${AIVORA_DOMAIN:-aivora-self.com}"
APP_HOST="app.${DOMAIN}"
AUTH_HOST="auth.${DOMAIN}"

echo "==> Deploying AiVora for ${DOMAIN}"

# ---- 1. Required tooling ----
if ! command -v docker >/dev/null 2>&1; then
    echo "==> Installing docker + caddy + ufw"
    sudo apt-get update -y
    sudo apt-get install -y docker.io docker-compose-plugin caddy ufw fail2ban git curl
    sudo usermod -aG docker "$USER"
    echo "!! Log out + back in so your user picks up the docker group, then re-run this script."
    exit 0
fi

# ---- 2. Firewall ----
echo "==> Configuring UFW"
sudo ufw allow OpenSSH >/dev/null
sudo ufw allow 80/tcp  >/dev/null
sudo ufw allow 443/tcp >/dev/null
sudo ufw --force enable >/dev/null

# ---- 3. .env sanity ----
if [[ ! -f .env ]]; then
    echo "!! .env missing. Copy .env.example → .env and fill it in first."
    exit 1
fi
if ! grep -q "^AIVORA_MASTER_KEY=..*" .env; then
    echo "!! AIVORA_MASTER_KEY not set in .env — deploy aborted."
    exit 1
fi

# Force the public URLs to match this domain (idempotent overwrites).
sed -i "s|^AIVORA_DASHBOARD_URL=.*|AIVORA_DASHBOARD_URL=https://${APP_HOST}|" .env
sed -i "s|^AIVORA_AUTH_URL=.*|AIVORA_AUTH_URL=https://${AUTH_HOST}|" .env
grep -q "^AIVORA_DASHBOARD_URL=" .env || echo "AIVORA_DASHBOARD_URL=https://${APP_HOST}" >> .env
grep -q "^AIVORA_AUTH_URL=" .env      || echo "AIVORA_AUTH_URL=https://${AUTH_HOST}"   >> .env

# ---- 4. Caddy reverse proxy (auto-TLS via Let's Encrypt) ----
echo "==> Writing Caddyfile"
sudo tee /etc/caddy/Caddyfile > /dev/null <<EOF
${APP_HOST} {
    reverse_proxy localhost:8501
}
${AUTH_HOST} {
    reverse_proxy localhost:8502
}
${DOMAIN} {
    redir https://${APP_HOST}{uri} permanent
}
EOF
sudo systemctl enable caddy
sudo systemctl restart caddy

# ---- 5. Build + verify data present ----
mkdir -p data/db data/processed logs models backups
docker compose build

if [[ ! -f data/db/aivora.sqlite ]]; then
    echo "!! data/db/aivora.sqlite missing on the VPS."
    echo "!! Upload your local DB from your Windows machine first:"
    echo "!!   scp -i \$HOME/.ssh/aivora-key.pem D:/work/AiVora/data/db/aivora.sqlite ubuntu@<VPS_IP>:~/aivora/data/db/"
    echo "!! Then re-run this script."
    exit 1
fi
if [[ ! -f models/current_up.pkl ]] || [[ ! -f models/current_down.pkl ]]; then
    echo "==> Freezing 92-feature model on the shipped DB"
    docker compose run --rm dashboard python -m scripts.freeze_model
fi

# ---- 6. Start services ----
docker compose up -d
sleep 4
docker compose ps

# ---- 7. Nightly backup cron ----
CRON_LINE="0 22 * * * cd $(pwd) && tar czf backups/nightly_\$(date +\\%Y\\%m\\%d).tar.gz data/db/webapp.sqlite data/db/aivora.sqlite models/ .env >/dev/null 2>&1"
( crontab -l 2>/dev/null | grep -v "aivora.*nightly"; echo "$CRON_LINE  # aivora nightly" ) | crontab -

echo ""
echo "==> DONE"
echo "    Dashboard:  https://${APP_HOST}"
echo "    OAuth:      https://${AUTH_HOST}/callback/kite"
echo ""
echo "Next:"
echo "  1. Open https://${APP_HOST} in a browser."
echo "  2. Register the FIRST account — it auto-becomes admin."
echo "  3. Profile → Zerodha → paste api_key + api_secret → Connect Zerodha (OAuth)."
echo "  4. Toggle Trading control → START (in PAPER mode) and wait for the next 5-min tick."
