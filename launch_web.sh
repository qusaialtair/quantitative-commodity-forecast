#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Web Stack Launcher  (Phase XIV)
#    1. starts FastAPI backend on :8000
#    2. starts Next.js dev server on :3000
#  Use this for the new professional frontend.
# ─────────────────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Kill any existing
pkill -f "uvicorn api.server" 2>/dev/null || true
pkill -f "next dev"           2>/dev/null || true
sleep 1

echo "▸ starting FastAPI backend on :8000"
python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload &> logs/api.log &
API_PID=$!

echo "▸ starting Next.js dev server on :3000"
cd web && npm run dev &> ../logs/web.log &
WEB_PID=$!
cd ..

echo ""
echo "  ╔══════════════════════════════════════════════════════════════╗"
echo "  ║          GOLD TRADING AI — WEB STACK LIVE                   ║"
echo "  ╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "    Frontend  →  http://localhost:3000"
echo "    Backend   →  http://localhost:8000"
echo "    API docs  →  http://localhost:8000/docs"
echo ""
echo "    Backend log: tail -f logs/api.log"
echo "    Web log    : tail -f logs/web.log"
echo ""
echo "    Stop both : ./stop_web.sh   (or Ctrl-C this terminal)"
echo ""

# Trap ctrl-c to kill children
trap "kill $API_PID $WEB_PID 2>/dev/null; exit 0" INT TERM
wait
