#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Worldwide AI Metals — Production Launcher
#  Binds to 0.0.0.0 so Tailscale can route traffic to any device on the network
# ─────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="$SCRIPT_DIR/ui/app.py"
PORT=8501

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║          WORLDWIDE AI METALS — PRODUCTION LAUNCH            ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Fetch Tailscale IP ────────────────────────────────────────────────────────
TAILSCALE_IP=""
# Tailscale CLI: check PATH first, then macOS App Store location
TAILSCALE_CLI=""
if command -v tailscale &>/dev/null; then
    TAILSCALE_CLI="tailscale"
elif [ -x "/Applications/Tailscale.app/Contents/MacOS/Tailscale" ]; then
    TAILSCALE_CLI="/Applications/Tailscale.app/Contents/MacOS/Tailscale"
fi

if [ -n "$TAILSCALE_CLI" ]; then
    TAILSCALE_IP=$($TAILSCALE_CLI ip -4 2>/dev/null | head -n1)
fi

if [ -n "$TAILSCALE_IP" ]; then
    echo "  🌐  Tailscale IP detected: $TAILSCALE_IP"
    echo ""
    echo "  ┌──────────────────────────────────────────────────────────────┐"
    echo "  │  Worldwide AI Metals is live on the Private Network!         │"
    echo "  │                                                              │"
    echo "  │  Type this exact URL into your iPhone Safari browser:        │"
    echo "  │                                                              │"
    echo "  │    http://$TAILSCALE_IP:$PORT"
    echo "  │                                                              │"
    echo "  └──────────────────────────────────────────────────────────────┘"
else
    echo "  ⚠️   Tailscale IP not found — is Tailscale running?"
    echo "      Run: tailscale up"
    echo "      Then re-launch this script."
    echo ""
    LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}')
    if [ -n "$LOCAL_IP" ]; then
        echo "  📡  Fallback — Local network URL (Wi-Fi only):"
        echo "      http://$LOCAL_IP:$PORT"
    fi
fi

echo ""
echo "  ⏳  Starting Streamlit on 0.0.0.0:$PORT ..."
echo "  ─────────────────────────────────────────────────────────────────"
echo ""

# ── Load .env if present ──────────────────────────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ── Launch ────────────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"
streamlit run "$APP_PATH" \
    --server.address 0.0.0.0 \
    --server.port $PORT \
    --server.headless true \
    --browser.gatherUsageStats false
