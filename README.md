# SO101 Remote Inference

Run a MolmoAct2 policy on the NUS HPC while the SO101 arm and cameras stay on
the local computer.

```
SO101 PC (local/client.py)
   |
   | HTTPS via ngrok tunnel
   v
PBS compute node:8000 (server/server.py)
```

`server/server.py` is the only server. `local/client.py` is the client. ngrok
is just a tunnel that forwards a public HTTPS URL to port 8000 on the node.

The client captures two camera frames + arm state + an instruction, sends them
to the server, receives an action chunk, and (in control mode) executes it with
local safety checks.

---

## Server (NUS HPC)

### One-time environment setup

The server uses the CUDA-matched `torch` inside the Singularity image, so the
venv must be created with `--system-site-packages` and must **not** install its
own torch. `requirements_server.txt` omits torch on purpose.

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

Do not `pip install torch` or `torchvision` into this venv.

### Run

```bash
qsub pbs/run_server.pbs
```

The job runs `python server/server.py`, listening on `0.0.0.0:8000`. It does
**not** start ngrok. If this job also owns the tunnel, run an ngrok agent on the
same node forwarding to `localhost:8000` (`ngrok http 8000`).

For debugging, request an interactive PBS job and run `python server/server.py`
manually to see errors directly.

---

## Client (local SO101 computer)

```bash
export SERVER_URL="https://<my-ngrok-url>"
export FOLLOWER_PORT="/dev/ttyACM0"        # your SO101 follower port
```

Check the server is reachable:

```bash
curl "$SERVER_URL/health"     # -> {"status":"ready"}
```

Install and run:

```bash
cd local
pip install -r ../requirements_local.txt
python client.py --mode preview     # print actions only, never moves the arm
python client.py --mode control     # execute actions with local safety checks
```

### Optional environment variables

| Variable        | Default                 | Purpose                                             |
| --------------- | ----------------------- | --------------------------------------------------- |
| `SERVER_URL`    | `http://localhost:8000` | Server / ngrok URL.                                 |
| `FOLLOWER_PORT` | required                | SO101 follower serial port.                         |
| `FOLLOWER_ID`   | `home_follower`         | Follower identifier.                                |
| `MODEL_REPO_ID` | server default          | Choose the model to run, e.g. a fine-tuned repo.    |
| `SAVE_FRAMES`   | unset                   | `1` saves the sent frames to `local/debug_frames/`. |

Example: run a fine-tuned model instead of the server default.

```bash
export MODEL_REPO_ID="tsangb34/molmoact2-so101-soccer-red_bowl-40episodes"
python client.py --mode preview
```

Camera IDs are set in `client.py` (`CameraManager(wrist_id, third_id)`). On
Linux, list devices with `ls /dev/video*`.

---

## Local-only test (no HPC, no ngrok)

Sanity-check the client/server on one machine. `SERVER_URL` defaults to
`http://localhost:8000`.

```bash
# terminal 1
cd server && pip install -r ../requirements_server.txt && python server.py

# terminal 2
cd local && pip install -r ../requirements_local.txt && python client.py
```

---

> Before relying on ngrok on NUS HPC, confirm the compute node can make the
> required outbound connection and that reverse tunnels are permitted by policy.
