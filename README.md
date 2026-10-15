# SO101 Remote Inference

Run a MolmoAct2 policy on the NUS HPC while the SO101 arm and cameras stay on
the local computer. (The ACT cube-pickup policy runs locally, without a server:
see [ACT: red-cube pickup](#act-red-cube-pickup).)

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

---

## ACT: red-cube pickup

A small ACT policy (52M parameters) runs on the laptop itself: no server, no ngrok. The model is downloaded from
Hugging Face on the first run and cached.

```bash
cd local
python run_act.py                                      # preview: arm + cameras + model, the arm never moves
python run_act.py --mode control --start-pose=-9,-74,88,54,0,95
python run_act.py --checkpoint ../outputs/act_my_run   # a checkpoint you fine-tuned (local dir or HF repo id)
```

Default model: [`omiya239532/so101_act_cotrain`](https://huggingface.co/omiya239532/so101_act_cotrain): pick a 4 cm
cube and place it on a white sheet, trained on sim + 10 real demos on the author's rig, which has a **top-down**
third-person camera ([`docs/act_reference_rig.jpg`](docs/act_reference_rig.jpg)). It expects that view; a front/side
webcam is a different input, so zero-shot may reach but not grasp.

Options worth knowing (all in `python run_act.py --help`):
- `--wrist-roll-offset`, `--gripper-open`, `--gripper-on-cube`: only for an arm calibrated differently from the
  checkpoint author's. The home arm was measured on 2026-10-14 (`local/act_calibration/home_follower.json`):
  wrist_roll -1.0 in the grasp orientation, gripper 97.1 open / 15.6 on the cube, folded pose within 5 deg of the
  author's, so the defaults fit it. The start pose above is the median start of the author's real demos.
- `--gripper-floor 8` for the office arm (default 15, the home arm), `--max-speed` [45 deg/s], `--hold-wrist-roll`,
  `--seconds`, `--n-action-steps` [30: re-plan every second].

Control mode asks for `yes`, refuses to start far outside the model's training poses, stops if a target leaves the
model's training range, limits joint speed, stops if a joint is blocked, and holds the pose on Ctrl+C.

### Public datasets (checked 2026-10-14)

| dataset | what it is | use |
|---|---|---|
| `omiya239532/so101_cube_dr` | simulated pick-and-place, 50 ep @ 30 fps, cameras world/wrist, radians | same format as the checkpoint (it trained on it): smoke test only |
| `Li-2025hang/so101-pick-lift` | toy simulation, 50 ep x 600 frames @ 50 fps, cameras overhead/wrist, the cube is always straight ahead | incompatible (fps, camera names); not used despite the name |
| `omiya239532/so101_mixed` | 404 | not available |

None of them is your rig, so the fastest route to a real pickup is to **record 20-40 demos yourself and fine-tune the
checkpoint on them** (the author: sim-pretrained ACT + 10 real demos = 17/20 real successes, the 10 demos alone 5/20).

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=$FOLLOWER_ID --robot.use_degrees=true \
  --robot.cameras="{ world: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30} }" \
  --teleop.type=so101_leader --teleop.port=$LEADER_PORT --teleop.id=my_leader --teleop.use_degrees=true \
  --dataset.repo_id=<hf_user>/so101_redcube_lift --dataset.single_task="pick up the red cube and lift it" \
  --dataset.num_episodes=30 --dataset.episode_time_s=20 --dataset.reset_time_s=10 --dataset.push_to_hub=true
```

`--robot.use_degrees=true` matters: LeRobot's default stores -100..100 instead of degrees. Camera names `world` and
`wrist` match the checkpoint. Check any dataset before training (stats, units, real vs sim, grasps, sample frames):

```bash
python scripts/check_act_dataset.py --repo-id <hf_user>/so101_redcube_lift --checkpoint omiya239532/so101_act_cotrain
```

### Fine-tuning on Hopper

```bash
qsub pbs/train_act.pbs                                                               # 1000-step smoke test
qsub -v DATASET_REPO_ID=<hf_user>/so101_redcube_lift,STEPS=10000 pbs/train_act.pbs   # then STEPS=30000 continues it
qsub -v DATASET_REPO_ID=<hf_user>/so101_redcube_lift,PRETRAINED=none,STEPS=30000 pbs/train_act.pbs   # from scratch
```

Uses the existing LeRobot 0.4.2 venv (`sim/setup_hpc_train.sh`, image `pytorch_2.7.0_cuda_12.8.sif`). The job checks
the GPU and video decoder, downloads the dataset if the node has internet (otherwise run the `check_act_dataset.py`
line above on the login node via `bash sim/hpc_env.sh env HF_HUB_OFFLINE=0 python ...` with
`--root /scratch/e1583535/datasets/lerobot/<repo id>`), validates it against the checkpoint, fine-tunes (batch 8, lr
1e-5, checkpoint every 5000 steps, resumes an existing run), and test-loads the result. Settings: `DATASET_REPO_ID`,
`PRETRAINED`, `STEPS`, `BATCH_SIZE`, `LR`, `SAVE_FREQ`, `OUTPUT_DIR` (see `scripts/train_act.sh`). Log:
`logs/train_act/<jobid>.log`. Copy `outputs/<run>/checkpoints/last/pretrained_model` to the laptop and pass it to
`run_act.py --checkpoint`.

### Sim-to-real ACT on the home rig (`home2`)

Real photos -> MuJoCo profile `home2` (cameras fitted to the vision_pick demos of 2026-10-15) -> scripted IK demos with
randomized appearance -> ACT on Hopper -> ACT evaluated in the sim on unseen seeds -> the real arm. Details of the
profile: [`sim_mujoco/README.md`](sim_mujoco/README.md).

1. Check the rig has not moved (laptop, `conda activate lerobot`). Torque is off: pose the arm by hand, Enter = save,
   `q` + Enter = quit. Then compare a capture with the sim:

   ```bash
   python local/capture_setup.py --name home2_check            # home arm: port/id/cameras are the defaults
   python sim_mujoco/tools/overlay_photo.py --profile home2 --photo local/setup_captures/home2_check/0_front.png \
       --qpos-deg <arm frame printed for capture 0> --out overlay_front.png      # --camera wrist with 0_wrist.png
   ```

   Useful captures: the folded start pose, the open gripper just above the cube, the fingertips touching the cube at
   3-4 spots, 2-3 other reachable poses, the empty table. The robot must overlap in the blend; the table may look
   different (it is randomized).

2. Sim checks on the laptop (no GPU): `python sim_mujoco/scripts/preview.py --profile home2 --episodes 20`
   (expert success + what ACT sees in `sim_mujoco/debug/preview/`), and a 10-demo dataset:
   `python sim_mujoco/scripts/generate.py --num_demos 10 --profile home2 --repo_id local/so101_home2_cube_smoke --root /tmp/home2_smoke`
   then `python sim/tools/inspect_dataset.py --repo-id local/so101_home2_cube_smoke --root /tmp/home2_smoke --video-backend pyav`.

3. Hopper (repo root, after `git pull`): 100 demos -> ACT 20k steps (batch 8, checkpoints every 5k), a 50-episode sim
   eval on unseen seeds every 5k with 10 videos:

   ```bash
   qsub -v PROFILE=home2,OBJECT=cube,NUM_DEMOS=100,STEPS=20000,EVAL_FREQ=5000,EVAL_VIDEOS=10 pbs/mujoco_pipeline.pbs
   ```

   Dataset `/scratch/e1583535/datasets/lerobot/local/so101_home2_cube`, checkpoints `outputs/act_home2_cube`, results
   `outputs/act_home2_cube/eval/step_<step>.json` + videos, log `logs/mujoco_pipeline/<jobid>.log`. Re-submitting
   resumes; `STEPS=40000` continues the same run. Fine-tuning instead of training from scratch: add
   `POLICY_PATH=<a pretrained_model dir with front/wrist cameras, e.g. outputs/act_home_cube/checkpoints/last/pretrained_model>`
   (only before the first submission of that OUTPUT_DIR).

4. Real arm: copy `outputs/act_home2_cube` from Hopper to the laptop's `outputs/`, put the cube 15-20 cm in front of
   the base (up to 5 cm to its right, 7 cm to its left), then

   ```bash
   bash local/run_home.sh act_home2_cube preview      # cameras + model; prints targets, never moves the arm
   bash local/run_home.sh act_home2_cube              # control: Enter to start, homes to the start pose, runs the policy
   ```

   Preview saves what the policy sees to `local/debug_frames/act_{front,wrist}.png`: check front = the third-person
   view. Camera indices and joint offsets per arm are `ARMS` in `local/run_act_policy.py` (home arm: wrist 1, front 0).

### Pickup without learning: `local/vision_pick`

Red-cube detection (OpenCV) + a camera->table homography calibrated by touching the cube with the gripper + top-down
inverse kinematics, run locally on the real arm. From the repo root: `python -m local.vision_pick --mode
detect|calibrate|preview|control [--arm home|office]`. Full steps: [local/vision_pick/README.md](local/vision_pick/README.md).
(`local/cube_fallback.py` is the earlier single-file version; vision_pick reuses its kinematics.)

### ACT demonstrations from vision_pick (no leader arm)

`local/collect_act.py` records demonstrations with the vision_pick controller. You place the cube and press Enter;
the arm picks and lifts it; you press `s` to keep the episode. It needs a vision_pick calibration
(`python -m local.vision_pick --mode calibrate`). Start the first session with the arm folded at rest.

```bash
python local/collect_act.py --episodes 50     # resumes where the last session stopped; never overwrites
python local/collect_act.py --step            # debug: Enter before each planned move, nothing recorded
```

**Each session**
- Type `yes` once to enable control mode.
- Each Enter then runs one whole episode, recorded at every control step: detect → plan → above the cube → open →
  down → close → lift → hold 1 s.
- After it: `s` save, `r` discard and retry at the same spot, `f` mark failed, `q` quit. Either way the arm puts the
  cube back and returns to the start pose.
- **Ctrl+C** stops at once with the arm holding. A blocked joint stops the episode, holds, then backs up to the start
  pose on Enter.

**Start pose:** the arm's pose when the first session starts, saved in `collect/session.json` and reused for every
episode and every later session.

**Variety:** a 3 × 3 count of saved cube positions points to the emptiest area. A cube within 1.5 cm of a saved
position, or outside the calibrated area, asks before running.

**Recording rate:** each episode reports the rate it recorded at. Below 27 fps or with a long gap, retry it. Dim
light lowers webcam fps.

**Dataset:** `~/.cache/huggingface/lerobot/local/so101_redcube_lift`, LeRobot v3.0 at 30 fps.
- `observation.state`: measured joints. `action`: the command actually sent after the safety checks. Both in LeRobot
  degrees with the gripper 0..100, named `shoulder_pan.pos` … `gripper.pos`.
- `observation.images.world` (third-person) and `observation.images.wrist`: 480×640, prepared exactly as
  `local/run_act.py` feeds the policy.
- Task: "Pick up the red cube and lift it."
- Every saved episode is finalized on disk immediately, so a crash loses at most the episode being recorded.
- `collect/attempts.jsonl` logs every attempt (cube position, outcome, recorded fps), with images of failures.
- `collect/split.json` holds the 45/5 split; validation is episodes 9, 19, 29, 39, 49.

Checked with a mock arm and fake cameras (not yet on the real arm): sessions with save / retry / failed / quit,
resume, a blocked joint, step mode. `scripts/check_act_dataset.py --strict` gives PASS against
`omiya239532/so101_act_cotrain`.

**After collecting** (the script prints these with your values):

1. **Validate.** The "every arm value is within ±100" warning is the checker's heuristic; this dataset is in
   degrees.
   ```bash
   python scripts/check_act_dataset.py --repo-id local/so101_redcube_lift --checkpoint omiya239532/so101_act_cotrain --strict
   ```
2. **Copy to Hopper.** The dataset was written with LeRobot 0.6.1 on the laptop and Hopper trains with 0.4.2, so
   run the check there once before submitting:
   ```bash
   rsync -av ~/.cache/huggingface/lerobot/local/so101_redcube_lift/ <you>@<hopper login>:/scratch/e1583535/datasets/lerobot/local/so101_redcube_lift/
   # on the Hopper login node, in the repo:
   bash sim/hpc_env.sh env HF_HUB_OFFLINE=0 python scripts/check_act_dataset.py --repo-id local/so101_redcube_lift \
       --root /scratch/e1583535/datasets/lerobot/local/so101_redcube_lift --checkpoint omiya239532/so101_act_cotrain --strict
   ```
3. **Fine-tune on the 45 training episodes.** Running the same command with `STEPS=30000` continues the run. The
   output goes to `outputs/act_so101_redcube_lift_finetune`.
   ```bash
   qsub -v DATASET_REPO_ID=local/so101_redcube_lift,EPISODES_FILE=/scratch/e1583535/datasets/lerobot/local/so101_redcube_lift/collect/split.json,STEPS=10000 pbs/train_act.pbs
   ```
4. **Check on the 5 held-out episodes.** Copy `outputs/act_so101_redcube_lift_finetune/checkpoints/last/pretrained_model`
   back to the same path on the laptop, then run the command below. It compares the policy's action 1 s ahead with
   simply holding still.
   ```bash
   python scripts/eval_act_offline.py --checkpoint outputs/act_so101_redcube_lift_finetune
   ```
5. **Evaluate on the robot.** On the home rig the third-person camera is index 0 and the wrist camera index 1. Use
   the start pose from `collect/session.json`. Score real pickups over about 10 cube positions inside the calibrated
   area.
   ```bash
   cd local && python run_act.py --checkpoint ../outputs/act_so101_redcube_lift_finetune --mode control \
       --front-cam 0 --wrist-cam 1 --gripper-floor 15 --start-pose=<start_pose from collect/session.json>
   ```
