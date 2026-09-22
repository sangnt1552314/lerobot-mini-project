#!/bin/bash
set -euo pipefail

HOME_PATH="/home/n/ntasang"

# Hugging Face cache on persistent home storage so model downloads survive
# across jobs (SLURM_TMPDIR is wiped when the job ends).
export HF_HOME=${HOME_PATH}/cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
# export HF_HUB_OFFLINE=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Scratch tmp on local disk.
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER}/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p "$TMPDIR"

# Run from this script's directory so server.py and server.log resolve here.
cd "$(dirname "$0")"

PORT=8000

# Free the port if a previous run is still holding it, otherwise uvicorn dies
# with "[Errno 98] address already in use".
free_port() {
    local port=$1 pids=""

    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -t -i :"$port" -sTCP:LISTEN 2>/dev/null || true)
    elif command -v fuser >/dev/null 2>&1; then
        pids=$(fuser -n tcp "$port" 2>/dev/null || true)
    elif command -v ss >/dev/null 2>&1; then
        pids=$(ss -lptnH "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 || true)
    fi

    [ -z "$pids" ] && return 0

    echo "Port $port in use by PID(s): $pids -- killing"
    kill $pids 2>/dev/null || true
    for _ in $(seq 10); do
        sleep 0.5
        kill -0 $pids 2>/dev/null || return 0
    done
    kill -9 $pids 2>/dev/null || true
    sleep 1
}

free_port "$PORT"

python server.py > server.log 2>&1 &

~/bin/ngrok http "$PORT"
