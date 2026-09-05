# SO101 Remote Inference Prototype (V0)

This is a minimal prototype that only tests **communication** between:

- a local computer with the SO101 robot and two cameras
- a remote model server running inside an NUS HPC PBS job

**The robot is not moved in this version.** The server returns a fake
(all-zero) action, and the local client only prints it.

```
wrist camera
      \
       \
        -> local/client.py
       /
third camera
      |
      | HTTP/HTTPS
      v
server/server.py
      |
      v
fake zero action
      |
      v
printed by client.py
```

V0 is successful once you see output like this on the local computer, with
**no robot movement**:

```
observation=0 latency=12.4 ms actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...]
```

---

## A. First test everything on one computer

This confirms the client and server work before involving the HPC or ngrok.

### Terminal 1 — this is the SERVER (pretend it's the HPC for now)

```bash
cd so101_remote/server
pip install -r ../requirements_server.txt
python server.py
```

Test it's alive:

```bash
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ready"}
```

### Terminal 2 — this is the LOCAL COMPUTER

```bash
cd so101_remote/local
pip install -r ../requirements_local.txt
python client.py
```

You may need to change the camera IDs in `client.py`:

```python
cameras = CameraManager(0, 1)
```

On Linux, you can inspect available camera devices with:

```bash
ls /dev/video*
```

Expected client output (repeats about once per second, Ctrl+C to stop):

```
observation=0 latency=12.4 ms actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...]
observation=1 latency=10.1 ms actions=[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ...]
```

---

## B. Test server inside PBS

Now move only the server to the HPC. The client keeps running on the local
computer.

```
LOCAL COMPUTER:
client.py

NUS HPC PBS NODE:
server.py
```

Submit the job:

```bash
qsub pbs/run_server.pbs
```

For initial debugging it is often easier to request an **interactive** PBS
job instead, and run the server manually so you can see errors directly:

```bash
python server/server.py
```

Note: `pbs/run_server.pbs` has a placeholder to activate your Python
environment (venv/conda) — edit that before submitting.

---

## C. Test with ngrok

Once the server runs correctly inside a PBS job, expose it to the local
computer with ngrok.

```
SO101 local PC
      |
      | HTTPS
      v
ngrok public URL
      |
      v
FastAPI :8000
inside PBS node
```

On the PBS compute node, after `server.py` is running:

```bash
ngrok http 8000
```

ngrok will print a public URL similar to:

```
https://xxxx.ngrok.app
```

Installing/authenticating ngrok depends on your environment (it is not
assumed to be pre-installed) — follow ngrok's own setup instructions for
whatever account/token you use.

On the local computer, edit `local/client.py` and change:

```python
SERVER_URL = "http://localhost:8000"
```

to:

```python
SERVER_URL = "https://xxxx.ngrok.app"
```

Then run:

```bash
python local/client.py
```

> **Note:** Before relying on ngrok on NUS HPC, confirm that the compute
> node can make the required outbound connection and that using a reverse
> tunnel is permitted by NUS HPC policy.

---

## Run order summary

1. **Localhost test** (section A) — client and server both on your local
   computer.
2. **PBS server test** (section B) — server moves to an HPC PBS job, client
   stays local, still connecting over plain HTTP (e.g. via an SSH tunnel or
   same network).
3. **ngrok test** (section C) — server on PBS node exposed via ngrok, client
   connects over the public HTTPS URL.

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
