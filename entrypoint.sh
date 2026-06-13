#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  QCTF Model — Container Entrypoint
#
#  Modes:
#    dashboard   (default)  Start Streamlit on port 8501
#    scheduler              Install crontab + start cron daemon + tail logs
#    pipeline               Run the daily pipeline once and exit
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="${1:-dashboard}"

case "$MODE" in
  dashboard)
    echo "[entrypoint] Starting Streamlit dashboard on :8501"
    exec python3 -m streamlit run ui/app.py \
      --server.port=8501 \
      --server.address=0.0.0.0
    ;;

  scheduler)
    echo "[entrypoint] Installing crontab and starting cron daemon"

    # Pass container env vars into cron jobs (cron starts with a bare env).
    printenv | grep -E '^[A-Z_]+=' > /etc/environment

    # 22:00 UTC = 02:00 GST (Gulf Standard Time, UTC+4).
    # US markets close 20:00 UTC (4 PM ET). Two hours of settlement buffer.
    cat > /etc/cron.d/qctf-pipeline <<'CRON'
# QCTF Model — nightly equity pipeline
# 22:00 UTC = 02:00 AM GST = 6:00 PM ET (2h after US market close)
0 22 * * 1-5  root  cd /app && /usr/local/bin/python3 run_daily_pipeline.py >> /app/data/logs/cron_pipeline.log 2>&1
CRON
    chmod 0644 /etc/cron.d/qctf-pipeline
    crontab /etc/cron.d/qctf-pipeline

    # Start cron in the foreground; tail the log so `docker logs` shows output.
    mkdir -p /app/data/logs
    touch /app/data/logs/cron_pipeline.log
    cron
    echo "[entrypoint] Cron daemon running. Pipeline fires at 22:00 UTC (02:00 GST) Mon-Fri."
    exec tail -F /app/data/logs/cron_pipeline.log
    ;;

  pipeline)
    echo "[entrypoint] Running daily pipeline (one-shot)"
    exec python3 run_daily_pipeline.py
    ;;

  *)
    echo "[entrypoint] Unknown mode: $MODE"
    echo "  Usage: docker run <image> [dashboard|scheduler|pipeline]"
    exit 1
    ;;
esac
