# ─────────────────────────────────────────────────────────────────────────────
#  QCTF Model — Production Container
#
#  Two runtime targets:
#    (default)  Streamlit dashboard  →  port 8501
#    scheduler  Daily pipeline cron  →  runs run_daily_pipeline.py at 22:00 UTC
#
#  Build:
#    docker build -t qctf-model .
#
#  Run (dashboard):
#    docker run --env-file .env -p 8501:8501 -v $(pwd)/data:/app/data qctf-model
#
#  Run (scheduler):
#    docker run --env-file .env -v $(pwd)/data:/app/data qctf-model scheduler
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

# System deps: LightGBM needs libgomp, git for pip installs, cron for scheduler
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        git \
        cron \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        openai>=1.30.0 \
        hmmlearn>=0.3.0 \
        optuna>=3.6.0 \
        shap>=0.45.0 \
        certifi>=2024.2.2 \
        fredapi>=0.5.2 \
        alpaca-py>=0.21.0

# ── Application code ────────────────────────────────────────────────────────
COPY . .

# ── Data volume mount point ─────────────────────────────────────────────────
# Pipeline artefacts, model weights, logs, and the virtual vault all live
# under /app/data. Mount a host volume here so state survives container
# restarts: -v $(pwd)/data:/app/data
VOLUME /app/data

# ── Streamlit config (headless, no CORS restrictions for reverse proxy) ─────
RUN mkdir -p /root/.streamlit
RUN cat > /root/.streamlit/config.toml <<'TOML'
[server]
headless = true
port = 8501
address = "0.0.0.0"
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 50

[browser]
gatherUsageStats = false
TOML

EXPOSE 8501

# ── Entrypoint: dashboard (default) or scheduler ────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["dashboard"]
