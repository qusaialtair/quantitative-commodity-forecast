# DEPLOYMENT.md — Altair MK1 on Oracle Cloud Always Free (AMD Micro)

**Target:** Oracle Linux 9 · 1 vCPU · 1 GB RAM · `opc` user  
**Stack:** nginx → Next.js (standalone) + FastAPI (uvicorn) via Docker Compose

---

## Pre-flight Checklist

Before you SSH in, confirm these are done in the Oracle Cloud Console:

- [ ] **Ingress rules** on your VCN security list: allow TCP port **80** from `0.0.0.0/0`
- [ ] **OS firewall** (handled in Phase 1 below)
- [ ] **SSH key** loaded locally: `ssh -i ~/.ssh/your_key.pem opc@YOUR_IP`

---

## Phase 1 — 4 GB Swap File (Critical — do this first)

> The Next.js build (`npm run build`) routinely spikes to 1.2–1.5 GB.  
> Without swap, the OOM killer will terminate the Docker build mid-way.  
> Swap is on a 50 GB block volume — I/O is fast enough for build-time use.

```bash
# 1. Create a 4 GB swap file (dd writes at ~100 MB/s, takes ~40 seconds)
sudo dd if=/dev/zero of=/swapfile bs=128M count=32 status=progress

# 2. Lock down permissions (readable only by root — mandatory for mkswap)
sudo chmod 600 /swapfile

# 3. Format as swap
sudo mkswap /swapfile

# 4. Activate immediately (takes effect now, before reboot)
sudo swapon /swapfile

# 5. Verify it's active
swapon --show
free -h
# Expected: Swap line shows ~4.0G total

# 6. Persist across reboots
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# 7. Tune swappiness — use swap only under memory pressure, not eagerly
sudo sysctl vm.swappiness=10
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
```

---

## Phase 2 — Docker CE Installation (Oracle Linux 9 / dnf)

> Oracle Linux 9 does NOT ship with Docker CE by default.  
> We add the official Docker repository and install via `dnf`.

```bash
# 1. Remove any stale podman/docker conflicts that ship with OL9
sudo dnf remove -y docker docker-client docker-client-latest \
     docker-common docker-latest docker-latest-logrotate \
     docker-logrotate docker-engine podman runc 2>/dev/null || true

# 2. Install prerequisites
sudo dnf install -y dnf-plugins-core curl

# 3. Add Docker CE repository
sudo dnf config-manager --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo

# 4. Install Docker CE + Compose plugin
#    (The Compose plugin ships as 'docker-compose-plugin' — gives you `docker compose`)
sudo dnf install -y docker-ce docker-ce-cli containerd.io \
     docker-buildx-plugin docker-compose-plugin

# 5. Start Docker and enable it to survive reboots
sudo systemctl enable --now docker

# 6. Verify Docker is running
sudo systemctl status docker --no-pager

# 7. Grant the `opc` user Docker access (avoids needing sudo for every command)
sudo usermod -aG docker opc

# ── IMPORTANT: log out and back in for the group change to take effect ──────
exit
# → SSH back in
ssh -i ~/.ssh/your_key.pem opc@YOUR_IP

# 8. Smoke-test (should print Docker version without sudo)
docker version
docker compose version
```

---

## Phase 3 — Open the Firewall Port

> Oracle Linux 9 runs `firewalld` by default. Docker adds its own iptables  
> rules, but the OS firewall must also allow port 80.

```bash
# Open HTTP (port 80) permanently
sudo firewall-cmd --permanent --add-service=http

# Reload firewall rules
sudo firewall-cmd --reload

# Verify
sudo firewall-cmd --list-all | grep services
# Expected output includes: services: ... http ...
```

---

## Phase 4 — Clone the Repository

```bash
# Clone to the opc home directory
git clone https://github.com/qusaialtair/quantitative-commodity-forecast.git

# Enter the project directory (all commands below run from here)
cd quantitative-commodity-forecast

# Confirm structure is correct
ls -la
# Expected: Dockerfile.backend, Dockerfile.frontend, docker-compose.yml,
#           nginx.conf, requirements-api.txt, api/, scripts/, etc.
```

---

## Phase 5 — Inject Credentials Securely

> **Never** paste credentials into the terminal history.  
> Use `nano` to write the `.env` file interactively — it never touches `~/.bash_history`.

```bash
# Create .env using nano (opens a blank interactive editor)
nano .env
```

Paste the following template into nano, **replacing every placeholder** with your real values.  
Then press **`Ctrl+O` → Enter** to save, and **`Ctrl+X`** to exit.

```dotenv
# ── AI / LLM ─────────────────────────────────────────────────────────────────
DEEPSEEK_API_KEY=your_deepseek_key_here
PERPLEXITY_API_KEY=your_perplexity_key_here
GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# ── Market Data ───────────────────────────────────────────────────────────────
FRED_API_KEY=your_fred_key_here

# ── Brokerage ─────────────────────────────────────────────────────────────────
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here

# ── Telegram Telemetry ────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# ── Admin Token (required for AUTHORIZE / HALT endpoints) ─────────────────────
# Generate a strong random value: openssl rand -hex 32
QCTF_ADMIN_TOKEN=generate_a_strong_random_token_here
NEXT_PUBLIC_QCTF_ADMIN_TOKEN=same_value_as_above

# ── Dashboard ─────────────────────────────────────────────────────────────────
# "PRODUCTION AUTOMATED" → polls backend every 3s (live data)
# "RECRUITER SANDBOX"    → renders mock data only (safe public default)
NEXT_PUBLIC_DASHBOARD_MODE=PRODUCTION AUTOMATED

# ── Execution ─────────────────────────────────────────────────────────────────
EXECUTION_MODE=paper_internal
TREASURY_HEDGE_MODE=SIGNAL_ONLY
TREASURY_SHARIA_CLEARED=false
TREASURY_HEDGE_MAX_PCT=20

# ── Market data ingest worker (FastAPI background task) ───────────────────────
MARKET_DATA_INGEST_ENABLED=true
MARKET_DATA_INGEST_INTERVAL_SEC=300
MARKET_DATA_INGEST_CONTINUOUS=false
```

Lock down permissions so only root and opc can read the file:

```bash
chmod 600 .env

# Verify no secrets are world-readable
ls -la .env
# Expected: -rw------- 1 opc opc ... .env
```

---

## Phase 6 — Build & Boot the Engine

```bash
# Build all images (slow step — ~5–10 min on the Micro instance)
# The Next.js build uses swap heavily; this is normal.
docker compose build --no-cache

# Watch memory pressure during the build (open a second SSH session)
watch -n 2 'free -h && docker stats --no-stream'

# Once build completes, launch all services in detached (background) mode
docker compose up -d

# Verify all three containers are running
docker compose ps
# Expected:
#   qctf_nginx     Up    0.0.0.0:80->80/tcp
#   qctf_backend   Up    (healthy)
#   qctf_frontend  Up

# Tail logs from all services (Ctrl+C exits log view — does NOT stop containers)
docker compose logs -f --tail=50
```

---

## Phase 7 — Verify It's Live

```bash
# Test from the server itself
curl -s http://localhost/nginx-health
# Expected: ok

curl -s http://localhost/api/health | python3 -m json.tool
# Expected: {"status": "ok", ...}

# Test from your local machine (replace with your actual IP)
curl -s http://YOUR_ORACLE_IP/nginx-health
```

Open **`http://YOUR_ORACLE_IP`** in a browser — the Altair MK1 terminal should load.

---

## Operations Reference

### Daily Commands

```bash
# Check container status + memory usage
docker compose ps
docker stats --no-stream

# Restart a single service
docker compose restart backend

# Pull latest code and rebuild backend only
git pull
docker compose build --no-cache backend
docker compose up -d --no-deps backend

# View real-time logs
docker compose logs -f backend
docker compose logs -f frontend
```

### Full Redeployment

```bash
docker compose down
git pull
docker compose build --no-cache
docker compose up -d
```

### Emergency Stop

```bash
# Stop containers (keeps images and volumes)
docker compose down
```

---

## Troubleshooting

| Symptom | Diagnosis | Fix |
|---------|-----------|-----|
| Build killed mid-way | OOM — swap not active | `sudo swapon /swapfile && docker compose build` |
| `port 80 already in use` | Another process on port 80 | `sudo ss -tlnp \| grep :80` → kill it |
| Frontend shows `DISCONNECTED` | Backend not healthy | `docker compose logs backend` |
| `docker: command not found` after `usermod` | Group change not applied | Log out and SSH back in |
| Firewall blocks port 80 | Missing ingress/firewall rule | Repeat Phase 3 commands |
| Container restarts in a loop | OOM at runtime | `docker stats` — check which container hits `mem_limit` |
| `.env` variables not loaded | Missing .env file | `ls -la .env` — must exist in same dir as `docker-compose.yml` |

---

## Memory Budget Reference

| Service | `mem_limit` | `memswap_limit` | Notes |
|---------|------------|-----------------|-------|
| `nginx` | 32 MB | 32 MB | Static proxy — never swaps |
| `backend` | 350 MB | 512 MB | FastAPI + uvicorn + pandas |
| `frontend` | 250 MB | 400 MB | Next.js standalone Node.js |
| **OS + kernel** | ~350 MB | — | Oracle Linux 9 idle footprint |
| **Total** | ~982 MB | — | Fits within 1 GB with headroom |

> The 4 GB swap file absorbs the build-time spike only.  
> At steady-state, all three containers fit comfortably within 1 GB physical RAM.
