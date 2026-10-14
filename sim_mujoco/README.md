# SO-101 Lift in MuJoCo: scripted demos → LeRobot dataset → ACT → eval, all on the HPC

> **SIMULATION ONLY.** Nothing here talks to the real SO-101.

MuJoCo counterpart of [`sim/`](../sim/README.md) (Isaac Sim). Isaac Sim needs an RTX GPU to render
cameras; this cluster has only H100/H200, which MuJoCo renders on with EGL. Same task and dataset
format as `sim/`: front + wrist cameras 640×480 @ 30 fps, `observation.state` (6), `action` (6, absolute
joint targets), `robot_type=so101_follower`, task **"Lift the red cube."** (or the object's task). ACT
training uses the same `sim/train_act_hpc.sh`.

## Setups and objects

A sim config is a physical **setup** (`--profile`: start pose, cameras, table, lighting) plus an **object**
(`--object`). Each pair is its own dataset and its own policy.

| `--profile` | setup |
|---|---|
| `default` | the original sim (yellow robot, checker table) |
| `real` | home setup: white robot, wood table, curtain (fitted 2026-10-12) |
| `office` | office setup: black robot, light wood desk, wall on its right, front webcam on the front-left (fitted 2026-10-14 to [`assets/office/photo_front.png`](assets/office/photo_front.png) + [`photo_wrist.png`](assets/office/photo_wrist.png)) |

| `--object` | object |
|---|---|
| `cube` | red 3 cm cube, "Lift the red cube." |
| `cylinder` | small red tube (Ø 1.45 cm, 2.8 cm tall) on a Ø 2.1 cm flange, "Lift the red cylinder." The jaws touch it at ~9 (gripper units), so the office expert closes to 8: fine on the office arm (`--gripper-floor 8`), too tight for the home arm (`real` + `cylinder` fails: its gripper floor is 15) |

Adding a setup: take photos at the start pose (`local/capture_setup.py` saves both cameras and the joint
values), add a function to `SETUPS` in [`so101_liftcube/config.py`](so101_liftcube/config.py), and line the
sim up with the photos with `python sim_mujoco/tools/overlay_photo.py --profile <new> --photo <front.png>`
(writes photo | sim | blend; `--camera wrist` for the wrist photo). Adding an object: an entry in `OBJECTS`.
Changing a setup changes its data: use a new dataset name / output dir afterwards.

## Commands (repository root, NUS HPC)

```bash
bash sim_mujoco/setup_hpc.sh          # once (after qsub pbs/setup_act_env.pbs)
qsub pbs/mujoco_pipeline.pbs          # generate 100 demos -> train ACT 20k steps -> evaluate 50 episodes
```

Options: `qsub -v NUM_DEMOS=200,STEPS=30000,OUTPUT_DIR=outputs/act_mujoco_try2 pbs/mujoco_pipeline.pbs`.

One policy per setup/object pair, one job each (every pair except `default`/`cube` gets its own dataset
`so101_<profile>_<object>` and checkpoint dir `outputs/act_<profile>_<object>`):

```bash
qsub -v PROFILE=office,OBJECT=cylinder pbs/mujoco_test.pbs     # optional 30-min check; images in sim_mujoco/debug/preview_office_cylinder/
for p in office:cylinder office:cube; do qsub -v PROFILE=${p%%:*},OBJECT=${p##*:},STEPS=60000 pbs/mujoco_pipeline.pbs; done
```

On the robot laptop, copy `outputs/act_<profile>_<object>/checkpoints/last/pretrained_model` from the HPC and
run `python local/run_act_policy.py --profile office --object cylinder --gripper-floor 8 --checkpoint ...`.

Re-submitting is safe: an existing dataset is reused and an existing `OUTPUT_DIR` is resumed.
`qsub pbs/mujoco_test.pbs` is a 30-minute plumbing test on throwaway data.

Log: `logs/mujoco_pipeline/<jobid>.log`. Eval writes
`eval_mujoco_<time>.json` and front|wrist videos of the first episodes next to the checkpoint.
`bash sim/hpc_env.sh python sim_mujoco/scripts/preview.py` (on a GPU node) saves what ACT sees to
`sim_mujoco/debug/preview/`.

## What is in it

| | |
|---|---|
| Robot | MuJoCo Menagerie `robotstudio_so101` (Apache-2.0, pinned commit), loaded unchanged |
| Scene | table, the object (default: red 3 cm cube, spawn x 0.17–0.25 m, y ±0.07 m, yaw ±30°), front camera (±1 cm / ±1.5° jitter), wrist camera from the model; per setup: walls, floor, textures, colour cast per camera, randomization |
| Control | 30 Hz, 6 physics substeps; position actuators of the model |
| Units | arm joints: MJCF joint range → −100..100; gripper: range → 0..100 (0 = closed) |
| Expert | IK on the fingertip site: approach above the object → descend with fingers down and jaws aligned to a cube face (cylinder: the direction needing the least wrist roll from the start pose) → close → lift (position-only IK, ~13 cm) → hold. ~150–180 frames (5–6 s) |
| Success | object centre > 0.10 m above the table for 5 steps (Isaac task: 0.20 m above the base; the SO-101 cannot lift that high near its base while gripping from above) |
| Eval | 12 s time-out; seeds from 100000 (generation uses 0, 1, 2, …) |

All tunables are in [`so101_liftcube/config.py`](so101_liftcube/config.py).

## Rendering inside the Singularity image

The PyTorch image has no EGL libraries. `--nv` binds the host's `libEGL` and NVIDIA EGL driver, but not
`libGLdispatch.so.0` or the EGL vendor file. `setup_hpc.sh` copies both to `/scratch/e1583535/opt/egl_shim`,
and `sim/hpc_env.sh` sets `LD_LIBRARY_PATH`, `__EGL_VENDOR_LIBRARY_FILENAMES` and `MUJOCO_GL=egl`.
Tested on an H200 node: hardware OpenGL (`NVIDIA H200/PCIe/SSE2`), ~190 frames/s at 640×480. OSMesa is
not available in the image.

## Differences from the Isaac version

- Different renderer and scene look, so a policy trained here will not transfer to the Isaac eval or the
  real robot without retraining.
- Units are a linear map of the MJCF joint ranges, not LeRobot's motor calibration.
- Lower success height (above).
