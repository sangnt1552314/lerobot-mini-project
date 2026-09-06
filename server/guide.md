# Running the server in interactive mode

Useful for debugging so you can see errors directly instead of digging through
PBS log files.

## 1. Request an interactive GPU node

```bash
qsub -I -l select=1:ngpus=1 -l walltime=02:00:00 -P CFP05-CF-002
```

## 2. Enter the PyTorch container

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

## 3. Activate the venv

The venv is built with `--system-site-packages` so it uses the container's
CUDA-matched torch (see the project README for one-time setup).

```bash
mkdir -p /scratch/e1583535/tmp
source /scratch/e1583535/virtualenvs/my_lebot/bin/activate

# sanity check: GPU is visible
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. Start the server + tunnel

```bash
cd /scratch/e1583535/projects/lerobot-mini-project/server

python server.py > server.log 2>&1 &
~/bin/ngrok http 8000 --url https://default.internal
```

`server.py` runs in the background (logging to `server.log`); ngrok runs in the
foreground. Watch the log live in another shell with:

```bash
tail -f server.log
```

## 5. Health check

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