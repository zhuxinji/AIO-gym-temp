#!/usr/bin/env bash
# Serve the local static frontend from this directory.
#   ./frontend/run.sh            -> http://127.0.0.1:8000
#   PORT=9000 ./frontend/run.sh
set -e
cd "$(dirname "$0")"
exec python3 serve.py "${PORT:-8000}"
