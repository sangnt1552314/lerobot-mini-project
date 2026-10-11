# SO-101 LiftCube in simulation: scripted demos → LeRobot dataset → ACT → sim eval

> **SIMULATION ONLY.** No physical robot is controlled by these scripts. Nothing under
> `sim/` imports or talks to the real SO-101, the leader arm, or the MolmoAct client/server.

```
LeIsaac dual-camera LiftCube   (Isaac Sim 5.1 / Isaac Lab 2.3, SO-101 + table + red cube)
        │  front cam 640x480@30 + wrist cam 640x480@30 + 6 joint positions
        ▼
scripted state-machine expert  (approach → hover → descend → grasp → lift)
        │  6-D absolute joint targets, only successful episodes kept
        ▼
LeRobotDataset v3              observation.images.front / .wrist, observation.state (6), action (6), task
        ▼
LeRobot ACT                    lerobot-train --policy.type=act
        ▼
LeIsaac evaluation             ACT drives the sim arm; success = cube 20 cm above the robot base
```

Task string: **"Lift the red cube."** Gym id: **`LeIsaac-SO101-LiftCube-DualCam-v0`**.

The whole workflow, from the repository root:

```bash
bash sim/setup.sh
bash sim/preview.sh
bash sim/generate.sh 20
bash sim/train_act.sh
bash sim/eval_act.sh outputs/act_liftcube/checkpoints/last/pretrained_model
```

---

## 1. Requirements

| | |
|---|---|
| OS | Linux (Ubuntu 22.04/24.04 tested by Isaac Sim). **macOS cannot run Isaac Sim.** |
| GPU | NVIDIA RTX with ≥ 8 GB VRAM (16 GB+ recommended), recent driver (`nvidia-smi` must work) |
| Disk | ~30 GB for Isaac Sim + caches, plus datasets (AV1 video; check `du -sh sim/data` after 20 demos) |
| Tools | `conda` (Miniconda), `git`, `curl`, `unzip` |

No physical SO-101 and no leader arm are needed.

Pinned versions (LeIsaac's installation guide at commit `24d3bcd`, verified against its source):
Python 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.0, PyTorch 2.7.0 + CUDA 12.8, LeRobot 0.4.2,
numpy 1.26.0.

## 2. Installation

```bash
cp sim/.env.example sim/.env      # optional: edit names, steps, paths
bash sim/setup.sh
```

`setup.sh` creates a **separate** conda env (`leisaac_sim`) so the real-robot environment is
untouched. It then:

1. checks Linux and `nvidia-smi`,
2. installs the CUDA 12.8 toolkit, FFmpeg 7.1.1 (LeRobot needs it to decode videos) and PyTorch 2.7.0 cu128,
3. clones LeIsaac into `sim/third_party/leisaac` at the pinned commit,
4. installs `leisaac[isaaclab]` (pulls Isaac Sim 5.1 + Isaac Lab 2.3 from `pypi.nvidia.com`) and `leisaac[lerobot]`,
5. downloads the SO-101 USD (HuggingFace `LightwheelAI/leisaac_env`) and Table with Cube (LeIsaac release `v0.1.2`),
6. validates imports, CUDA, torchcodec/FFmpeg, task registration and the config.

The first Isaac Sim launch asks you to accept the NVIDIA Omniverse EULA (or set
`OMNI_KIT_ACCEPT_EULA=YES` in `sim/.env`) and compiles shaders, which can take 5–10 minutes.

Static checks that run anywhere (also on a Mac): `bash sim/check.sh --static`.
Full check on the GPU machine: `bash sim/check.sh`.

## 3. Preview the scene

```bash
bash sim/preview.sh            # headless check, then the Isaac Sim window
bash sim/preview.sh --no-gui   # remote machine: headless check only
```

Step 1 launches the env headless and prints what the policy and dataset will see, e.g.

```
LeRobot dataset features the recorder will write:
  action                       (6,)
  observation.state            (6,)
  observation.images.front     (480, 640, 3)
  observation.images.wrist     (480, 640, 3)
  task                         'Lift the red cube.'
```

(these are read from the live env, not hard-coded). It also checks that the cube rests on the
table, which means physics is running, and saves `sim/debug/observations/front.png` and
`wrist.png`. **Open those two images.** They are exactly what ACT will see.

Step 2 opens the Isaac Sim window and runs the scripted expert in a loop without recording.
Each attempt prints `SUCCESS` / `FAILED`. Close the window or press Ctrl+C to stop.

## 4. Generate 20 demos

```bash
bash sim/generate.sh 20
```

- The argument is the number of **successful** demos. Failed attempts are dropped, never saved.
- Data is written directly as a LeRobot dataset by LeIsaac's LeRobot recorder (no HDF5 step) to
  `sim/data/lerobot/<HF_USER>/<SIM_DATASET_NAME>` (default `sim/data/lerobot/local/so101_liftcube_dualcam_sim`).
- It stops after `3*N+10` attempts if the expert keeps failing, and prints the expert's success rate.
- The script refuses to overwrite an existing dataset. Delete it or change `SIM_DATASET_NAME`.
- `GENERATE_GUI=1 bash sim/generate.sh 20` records with the window open (slower).

`generate.sh` finishes by running the dataset inspector (next section) and fails if fewer
than N episodes were saved.

## 5. Inspect the demos

```bash
source sim/common.sh && load_env && activate_env     # sets HF_LEROBOT_HOME for this shell
python sim/tools/inspect_dataset.py --repo-id local/so101_liftcube_dualcam_sim
```

Prints episodes, fps, frames, the state/action/front/wrist shapes, task strings, per-joint
min/max and episode lengths. It exits non-zero if a camera is missing, a dimension is wrong,
there are NaNs, the actions never change, or there are zero episodes. It saves
`sim/debug/dataset/ep0_{first,middle,last}_{front,wrist}.png`; the `last` frames should show
the cube in the air.

## 6. Generate 100 / 500 demos

```bash
SIM_DATASET_NAME=so101_liftcube_dualcam_sim_100 bash sim/generate.sh 100
SIM_DATASET_NAME=so101_liftcube_dualcam_sim_500 bash sim/generate.sh 500
```

(or set `SIM_DATASET_NAME` / `NUM_DEMOS` in `sim/.env`). Each episode lasts about 7 s of sim time
(roughly 200 frames at 30 fps). Wall-clock speed depends on the GPU, so time the 20-demo run and
scale from there.

## 7. Train ACT

```bash
bash sim/train_act.sh            # 20000 steps (TRAIN_STEPS)
bash sim/train_act.sh 10000      # smoke run
```

This runs the installed LeRobot's trainer:

```bash
lerobot-train --dataset.repo_id=local/so101_liftcube_dualcam_sim --dataset.root=sim/data/lerobot/local/... \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --output_dir=outputs/act_liftcube --job_name=act_so101_liftcube_sim \
  --batch_size=8 --steps=20000 --save_freq=5000 --wandb.enable=true --wandb.project=so101_liftcube_sim
```

ACT reads the camera, state and action shapes from the dataset, and its hyperparameters are
LeRobot's defaults on purpose. `--policy.push_to_hub=false` is needed because LeRobot 0.4.2
otherwise demands a `policy.repo_id`. If you are not logged in to W&B (`wandb login`), the
script logs offline instead of failing. It refuses to reuse an existing `OUTPUT_DIR`; use
`RESUME=1 bash sim/train_act.sh 40000` to continue a run, or set a new `OUTPUT_DIR`. Anything
after the step count is passed to `lerobot-train`.

## 8. Evaluate ACT in simulation

```bash
bash sim/eval_act.sh outputs/act_liftcube/checkpoints/last/pretrained_model        # 20 episodes
bash sim/eval_act.sh outputs/act_liftcube/checkpoints/last/pretrained_model 50
EVAL_GUI=1 bash sim/eval_act.sh outputs/act_liftcube/checkpoints/last/pretrained_model
```

```
========== ACT simulation evaluation ==========
successes      17
failures       3
total episodes 20
success rate   85.0%
===============================================
```

(illustrative numbers). Success is the task's own termination: the cube centre more than
0.20 m above the robot base. Failure is a time-out after `task.episode_length_s` (20 s). A JSON
summary is written next to the checkpoint (`eval_sim_<timestamp>.json`).

**Smoke test of everything:** `bash sim/run_all.sh` runs check → 20 demos → inspect → 10k
training steps → eval, using timestamped names so it never touches your main dataset or
checkpoints. Expect a low success rate from such a short run; it is a plumbing test.

## 9. Expected file structure

```
sim/
├── README.md
├── .env.example            copy to .env (gitignored) to change names/steps/paths
├── common.sh               shared helpers (env loading, conda activation, checks)
├── setup.sh  check.sh  preview.sh  generate.sh  train_act.sh  eval_act.sh  run_all.sh
├── configs/
│   └── lift_cube_dualcam.yaml   ALL tunables: cameras, randomization, robot gains, expert
├── leisaac_ext/            out-of-tree LeIsaac task (nothing patched inside LeIsaac)
│   ├── __init__.py         registers LeIsaac-SO101-LiftCube-DualCam-v0 (gymnasium only)
│   ├── config.py           YAML loader + validation
│   ├── act_policy.py       in-process ACT inference (LeRobot code path)
│   └── lift_cube_dualcam/
│       ├── __init__.py     gym.register(...)
│       ├── env_cfg.py      the env: LiftCube scene with BOTH cameras kept
│       ├── events.py       camera jitter anchored to the spawn pose
│       └── state_machine.py   scripted expert
├── scripts/
│   ├── generate.py         adapted LeIsaac datagen loop (TASK_REGISTRY + our task)
│   ├── eval_act.py         adapted LeIsaac policy_inference loop with ACT in-process
│   └── convert_hdf5.py     fallback only: HDF5 → LeRobot via LeIsaac's converter
├── tools/
│   ├── check_observations.py
│   └── inspect_dataset.py
├── third_party/leisaac/    (created by setup.sh, gitignored) LeIsaac + assets/
├── data/lerobot/...        (gitignored) recorded datasets
└── debug/                  (gitignored) saved camera frames and reports
outputs/act_liftcube/       (gitignored) ACT checkpoints: checkpoints/<step>/pretrained_model
```

### Where things are configured

- **`sim/configs/lift_cube_dualcam.yaml`**: camera poses and intrinsics (match these to the real
  cameras later), cube randomization (±5 cm, ±15° yaw), success height, physics/control rate,
  robot gains, and the expert's offsets and timing. Override the path with `SIM_CONFIG=...`.
- **`sim/.env`**: names, paths, steps, batch size, device, eval episodes, W&B.

### Design notes (why it is built this way)

- **Dual camera.** Upstream `LiftCubeEnvCfg` deletes the template's wrist camera and wrist
  observation. Our env is assembled from the same LeIsaac pieces (SingleArm template, Table with
  Cube asset, `cube_height_above_base` success term) without those deletions.
- **No LiftCube expert exists upstream.** At commit `24d3bcd` LeIsaac ships only
  `PickOrangeStateMachine`. `state_machine.py` follows the same `StateMachineBase` contract and
  PickOrange's recipe (DLS IK on the `gripper` body, world-fixed orientation, the same grasp
  offsets).
- **Actions are joint positions.** PickOrange sends an 8-D end-effector pose to an IK action
  term, so its dataset records poses. Our expert runs the same IK step itself and sends 6-D
  joint targets through the `so101leader` joint-position action space. The dataset `action`
  is therefore joint positions in LeRobot's SO-101 units (arm −100..100, gripper 0..100), the
  same space used by ACT evaluation and by the physical arm.
- **Why our own `generate.py`.** Upstream's `TASK_REGISTRY` is a literal dict inside a script
  that launches Isaac Sim at import, so an out-of-tree task cannot be added without editing
  the LeIsaac checkout. The copy keeps the upstream loop and LeIsaac's own recorder managers.
- **Why ACT runs in-process for eval.** LeIsaac's `--policy_type=lerobot-act` path uses a gRPC
  client written for LeRobot 0.3.3. It pickles LeIsaac's own `RemotePolicyConfig`, which the
  LeRobot 0.4.2 server rejects (`isinstance` check), and its `lerobot-async` extra pins
  protobuf 6 against Isaac Lab's protobuf 3. LeIsaac's recorder needs LeRobot 0.4.2, so
  `eval_act.py` loads the checkpoint with the same LeRobot code that trained it.
- **Identical dynamics in datagen and eval.** The upstream state-machine device disables robot
  gravity and raises joint damping at runtime only during datagen. We set both in the env
  config, so ACT is evaluated on the same dynamics it was trained on.
- **30 Hz everywhere.** 60 Hz physics with decimation 2 means one env step = one camera frame =
  one dataset frame (upstream steps at 60 Hz while labelling the dataset 30 fps).

## 10. Common errors

| Symptom | Fix |
|---|---|
| `Isaac Sim needs Linux with an NVIDIA RTX GPU` | Run on a Linux GPU machine (e.g. a lab workstation or cloud GPU). macOS is not supported by Isaac Sim. |
| `nvidia-smi failed` | Install/repair the NVIDIA driver; reboot. |
| First launch hangs for minutes | Shader compilation on first run; wait. Later launches are much faster. |
| EULA prompt blocks a script | Answer `Yes` once, or set `OMNI_KIT_ACCEPT_EULA=YES` in `sim/.env`. |
| `Missing .../so101_follower.usd` or `table_with_cube/scene.usd` | `bash sim/setup.sh` again (it only downloads what is missing). |
| Robot or scene loads from the wrong place | `LEISAAC_ASSETS_ROOT` must point at `sim/third_party/leisaac/assets`; the scripts set it. Run Python through the `sim/*.sh` scripts or `source sim/common.sh && load_env`. |
| `Dataset already exists` | `rm -rf sim/data/lerobot/local/<name>` or choose another `SIM_DATASET_NAME`. |
| Expert success rate is low / `Only k/N demos recorded` | Watch it with `bash sim/preview.sh`, then tune `expert:` in the YAML: `grasp_offset` (height of the jaws at the cube), `lift_retract` / `lift_cube_height_above_base` (reach), `target_rpy_deg` or `ik_command: position` (approach angle), `gripper_close_deg`, phase durations. |
| `torchcodec cannot load FFmpeg` | `conda install -c conda-forge ffmpeg=7.1.1` in the `leisaac_sim` env. For a one-off, `inspect_dataset.py --video-backend pyav`. |
| `pip check` conflicts after setup | Isaac Sim's pins often report harmless conflicts. Only act if imports fail; `numpy` must be 1.26.0. |
| `Output directory ... already exists` from training | `RESUME=1 bash sim/train_act.sh <steps>` or a new `OUTPUT_DIR`. |
| W&B asks for a login | `wandb login`, or `WANDB_ENABLE=false` in `sim/.env`. |
| `Checkpoint expects cameras [...]` during eval | The checkpoint was trained on a dataset with different camera names; retrain on the dual-camera dataset. |
| `Env steps at X Hz but --lerobot_dataset_fps=Y` | Keep `DATASET_FPS` equal to 60 / `sim.decimation` in the YAML (30 by default). |

### Fallback: HDF5 recording + conversion

Use this only if the direct LeRobot recorder fails with your LeRobot version:

```bash
source sim/common.sh && load_env && activate_env
python sim/scripts/generate.py --headless --enable_cameras --device cuda --record --num_demos 20 \
    --recorder hdf5 --dataset_file ./sim/datasets/liftcube_dualcam.hdf5
python sim/scripts/convert_hdf5.py --task_name LeIsaac-SO101-LiftCube-DualCam-v0 \
    --repo_id "$DATASET_REPO_ID" --hdf5_root ./sim/datasets --hdf5_files liftcube_dualcam.hdf5 \
    --fps 30 --headless --enable_cameras
```

`convert_hdf5.py` registers our task and then runs LeIsaac's `isaaclab2lerobotv3.py` unchanged
(that converter keeps only successful episodes).

### Toward the real robot (later, not part of this module)

- Match `cameras.front` / `cameras.wrist` in the YAML to the physical camera poses.
- Dataset actions use LeRobot's normalized SO-101 units. The existing MolmoAct client reads the
  arm with `use_degrees=True`, so a real-robot ACT setup must use the same normalization as this
  dataset.
- The default closed gripper target (0°) maps to ≈ 9 in LeRobot units. The physical gripper
  stalls below ~15, so clamp it before deploying.
