#!/bin/bash
pkill -f "uvicorn api.server" 2>/dev/null && echo "stopped backend"
pkill -f "next dev"           2>/dev/null && echo "stopped frontend"
