# Running the server in interactive mode

Useful for debugging so you can see errors directly instead of digging through
batch log files. Pick the section for your cluster, then follow the shared
"Install ngrok" / "Start the server" steps at the bottom.

---

## Option A — PBS (NUS HPC)

### 1. Request an interactive GPU node

```bash
qsub -I -l select=1:ngpus=1 -l walltime=02:00:00 -P CFP05-CF-002
```

### 2. Enter the PyTorch container

```bash
module load singularity

singularity exec -e --nv \
  --env HF_HOME=/scratch/e1583535/cache \
  --env HF_HUB_OFFLINE=1 \
  --env TMPDIR=/scratch/e1583535/tmp \
  --env TEMP=/scratch/e1583535/tmp \
  --env TMP=/scratch/e1583535/tmp \
  /app1/common/singularity-img/hopper/pytorch/pytorch_2.6.0_cuda_12.8.sif bash
```

### 3. Activate the venv

The venv is built with `--system-site-packages` so it uses the container's
CUDA-matched torch (see the project README for one-time setup).

```bash
mkdir -p /scratch/e1583535/tmp
source /scratch/e1583535/virtualenvs/my_lebot/bin/activate
cd /scratch/e1583535/projects/lerobot-mini-project

# sanity check: GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Option B — SLURM (NUS SoC)

No singularity here — this cluster uses a plain venv that already has
`torch`/`torchvision` + `requirements_server.txt` installed.

### 1. Request an interactive GPU node

```bash
srun --gres=gpu:h100-47:1 --time=02:00:00 --cpus-per-task=8 --mem=100G --pty bash
```

### 2. Activate the venv and set caches

```bash
source ~/py312/bin/activate
cd ~/projects/lerobot-mini-project

export HF_HOME=$HOME/cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TMPDIR=${SLURM_TMPDIR:-/tmp/$USER}/tmp
export TEMP=$TMPDIR TMP=$TMPDIR
mkdir -p "$HF_HOME" "$TMPDIR"

# sanity check: GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Install ngrok (one-time)

```bash
mkdir -p ~/bin
cd ~/bin

curl -fsSL https://bin.ngrok.com/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | tar -xz

./ngrok version
```

## Start the server + tunnel

From the project root (activated venv):

```bash
cd server

python server.py > server.log 2>&1 &
~/bin/ngrok http 8000 --url https://default.internal
```

`server.py` runs in the background (logging to `server.log`); ngrok runs in the
foreground. Watch the log live in another shell with:

```bash
tail -f server.log
```

## Health check

```bash
curl http://localhost:8000/health   # expect: {"status":"ready"}
```

## Stopping

`Ctrl+C` stops ngrok (foreground). The server keeps running in the background —
stop it explicitly:

```bash
ps aux | grep -E "server.py|uvicorn" | grep -v grep
pkill -f server.py
```