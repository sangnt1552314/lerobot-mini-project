#!/bin/bash
set -euo pipefail

export HF_HOME=/scratch/e1583535/cache

# Run from this script's directory so server.py and server.log resolve here.
cd "$(dirname "$0")"

python server.py > server.log 2>&1 &

~/bin/ngrok http 8000 --url https://default.internal
