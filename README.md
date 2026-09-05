# SO101 Remote Inference Prototype (V0)

## WHAT RUNS WHERE

```
LOCAL SO101 COMPUTER
--------------------
local/client.py
local/cameras.py

NUS HPC PBS COMPUTE NODE
------------------------
server/server.py
ngrok tunnel -> localhost:8000
```

There is only **one** application server: `server/server.py`, running inside
the PBS job on the HPC compute node. `local/client.py` is a client, not a
server. ngrok is not a second server either — it is just a tunnel that
forwards a public HTTPS URL to `server.py` on port 8000.

```
SO101 PC
   |
   | HTTPS
   v
https://<ngrok-url>
   |
   | ngrok tunnel
   v
PBS compute node:8000
   |
   v
server.py
```

This V0 only tests that communication path. **No robot movement and no real
model inference happen yet:**

```
two camera images
+ fake robot state
+ instruction
        |
        v
remote PBS server
        |
        v
fake zero action
        |
        v
print locally
```

---

## Real deployment

### On PBS (NUS HPC)

#### Server environment (one-time setup)

The server relies on the CUDA-matched `torch` shipped inside the Singularity
image (`pytorch_2.6.0_cuda_12.8.sif`); it must **not** install its own torch.
For that to work, the venv has to be created with `--system-site-packages` so
it can see the container's `torch`/`torchvision`, while `requirements_server.txt`
(which deliberately omits torch) layers the rest on top.

Create it once, from inside the container:

```bash
module load singularity
singularity exec -e --nv \
  /app1/common/singularity-img/hopper/pytorch/pytorch_2.6.0_cuda_12.8.sif bash

python3.12 -m venv --system-site-packages /scratch/e1583535/virtualenvs/my_lebot
source /scratch/e1583535/virtualenvs/my_lebot/bin/activate
pip install -r requirements_server.txt

# sanity check: container torch is visible and CUDA initializes
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Do not `pip install torch` or `torchvision` into this venv — a mismatched
CUDA build (e.g. cu130 on a CUDA 12.9 driver) will break `torch.cuda` init.

Submit the job:

```bash
qsub pbs/run_server.pbs
```

The job runs:

```bash
python server/server.py
```

The server listens on `0.0.0.0:8000` inside the compute node.

An ngrok agent must be forwarding a public URL to port 8000 on this same
node. `pbs/run_server.pbs` does **not** start ngrok itself — see the comment
above the `python server/server.py` line in that file:

- if the `SERVER_URL` you'll use on the local computer already has an active
  tunnel managed elsewhere (e.g. started by you in another session, or by
  someone else), you only need this job running `server.py`;
- if this PBS job is also responsible for the tunnel, you must additionally
  run an ngrok agent on the same node, forwarding to `localhost:8000`
  (e.g. `ngrok http 8000`).

For initial debugging, it can be easier to request an **interactive** PBS
job and run `python server/server.py` manually so you can see errors
directly.

### On the local SO101 computer

Set the server URL to your ngrok public URL:

```bash
export SERVER_URL="https://<my-ngrok-url>"
```

First check the server is reachable:

```bash
curl "$SERVER_URL/health"
```

Expected:

```json
{"status":"ready"}
```

Then run the client:

```bash
cd local
pip install -r ../requirements_local.txt
python client.py
```

Expected output (repeats about once per second, Ctrl+C to stop):

```
observation=0 latency=142.3 ms actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...]
observation=1 latency=138.9 ms actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...]
```

You may need to change the camera IDs in `client.py`:

```python
cameras = CameraManager(0, 1)
```

On Linux, inspect available camera devices with:

```bash
ls /dev/video*
```

> Before relying on ngrok on NUS HPC, confirm that the compute node can make
> the required outbound connection and that using a reverse tunnel is
> permitted by NUS HPC policy.

---

## Optional: local-only test (no HPC, no ngrok)

Useful if you just want to sanity-check the client/server code on one
machine before touching the HPC at all.

### Terminal 1 — server

```bash
cd server
pip install -r ../requirements_server.txt
python server.py
```

```bash
curl http://localhost:8000/health
```

Expected: `{"status":"ready"}`

### Terminal 2 — client

`SERVER_URL` defaults to `http://localhost:8000` if unset, so you can just
run:

```bash
cd local
pip install -r ../requirements_local.txt
python client.py
```

---

## What comes later

After this communication test works, the next steps will be:

1. replace dummy joint state with real SO101 state
2. replace fake action with model inference
3. add local robot safety checks
4. only then execute actions
5. optionally move from simple HTTP requests to a persistent connection if
   latency becomes a problem

None of these are implemented yet.
