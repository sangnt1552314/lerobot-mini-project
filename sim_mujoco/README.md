# SO-101 Lift in MuJoCo: scripted demos → LeRobot dataset → ACT → eval, all on the HPC

> **SIMULATION ONLY.** Nothing here talks to the real SO-101.

Replaces the Isaac Sim pipeline this repo started with (removed). Isaac Sim needs an RTX GPU to render
cameras; this cluster has only H100/H200, which MuJoCo renders on with EGL. Same task and dataset
format as that pipeline: front + wrist cameras 640×480 @ 30 fps, `observation.state` (6), `action` (6, absolute
joint targets), `robot_type=so101_follower`, task **"Lift the red cube."** (or the object's task). ACT
training: [`sim/train_act_hpc.sh`](../sim/README.md).

## Setups and objects

A sim config is a physical **setup** (`--profile`: start pose, cameras, table, lighting) plus an **object**
(`--object`). Each pair is its own dataset and its own policy.

| `--profile` | setup |
|---|---|
| `default` | the original sim (yellow robot, checker table) |
| `real` | home setup, first rough model: white robot, wood table, curtain (fitted 2026-10-12) |
| `home` | home setup, refitted 2026-10-14 to [`assets/home/photo_front.png`](assets/home/photo_front.png) at the start pose (+ the folded pose) with the joints read from the arm: glossy reddish-brown table, front webcam in front, left of centre; per-camera colour (the front webcam is darker and more saturated); same start pose and expert as `real` |
| `office` | office setup: black robot, light wood desk, wall on its right, front webcam on the front-left (fitted 2026-10-14 to [`assets/office/photo_front.png`](assets/office/photo_front.png) + [`photo_wrist.png`](assets/office/photo_wrist.png)) |
| `office2` | office setup rebuilt 2026-10-15: office arm, light wood desk ending ~31 cm in front of the base, no wall, webcam on a stand 34 cm above the work area looking down. Fitted to photos with the arm moved through 9 poses (overhead camera: robot silhouette; wrist camera: the jaw's silhouette, which showed the real wrist camera is mounted 9° further forward and is wide-angle). Objects of random size (below). Start pose arm frame [0, -57.3, 22.9, 88.8, 93.0], gripper 79 (fingertips 15 cm up; both cameras see the whole square) |

| `--object` | object |
|---|---|
| `cube` | red 3 cm cube, "Lift the red cube." |
| `cylinder` | small red tube (Ø 1.45 cm, 2.8 cm tall) on a Ø 2.1 cm flange, "Lift the red cylinder." The jaws touch it at ~9 (gripper units), so the office expert closes to 8: fine on the office arm (`--gripper-floor 8`), too tight for the home arm (`real` + `cylinder` fails: its gripper floor is 15) |

With `--profile office2` the objects are random per episode instead (`RANDOM_SIZE_OBJECTS` in config.py): a red
cube of side 2-5 cm or a red cylinder of Ø 2-6 cm and 3-7 cm tall, anywhere in a 5 × 5 cm square centred 18 cm in
front of the base (x 0.155-0.205 m, y ±0.025 m, yaw ±45°). Expert success (100 episodes each): cube 100 %;
cylinder 77 % (Ø 2-4 cm 100 %, 4-5 cm 94 %, 5-6 cm 19 %: the moving jaw meets a wide round object at ~33° and it
twists out), so few demos have cylinders wider than 5 cm. On the real desk the square is under the folded arm's
gripper: move the arm to the start pose first, then put the object down (see below).

Adding a setup: take photos at the start pose (`local/capture_setup.py` saves both cameras and the joint
values), add a function to `SETUPS` in [`so101_liftcube/config.py`](so101_liftcube/config.py), and line the
sim up with the photos with `python sim_mujoco/tools/overlay_photo.py --profile <new> --photo <front.png>`
(writes photo | sim | blend; `--camera wrist` for the wrist photo). Adding an object: an entry in `OBJECTS`.
Changing a setup changes its data: use a new dataset name / output dir afterwards.

## Commands (repository root, NUS HPC)

```bash
bash sim_mujoco/setup_hpc.sh          # once (after qsub pbs/setup_act_env.pbs)
qsub pbs/mujoco_pipeline.pbs          # generate 100 demos -> train ACT 20k steps, evaluating 50 episodes every 10k
```

Options: `qsub -v NUM_DEMOS=200,STEPS=30000,OUTPUT_DIR=outputs/act_mujoco_try2 pbs/mujoco_pipeline.pbs`.

One policy per setup/object pair, one job each (every pair except `default`/`cube` gets its own dataset
`so101_<profile>_<object>` and checkpoint dir `outputs/act_<profile>_<object>`):

```bash
qsub -v PROFILE=office,OBJECT=cylinder pbs/mujoco_test.pbs     # optional 30-min check; images in sim_mujoco/debug/preview_office_cylinder/
for p in home:cube office:cube office:cylinder; do qsub -v PROFILE=${p%%:*},OBJECT=${p##*:},STEPS=100000 pbs/mujoco_pipeline.pbs; done
```

The second line continues `outputs/act_home_cube`, `act_office_cube` and `act_office_cylinder` to 100k steps
(add `EVAL_FREQ=5000` to evaluate more often; each 50-episode eval takes ~6 min).

The office2 policies (300 demos, 16 eval videos, 100k steps; datasets `so101_office2_{cube,cylinder}`, runs
`outputs/act_office2_{cube,cylinder}`):

```bash
qsub -v PROFILE=office2,OBJECT=cylinder pbs/mujoco_test.pbs    # optional 30-min check first
for o in cube cylinder; do qsub -v PROFILE=office2,OBJECT=$o,NUM_DEMOS=300,EVAL_VIDEOS=16,STEPS=100000 pbs/mujoco_pipeline.pbs; done
```

On the office arm (copy `outputs/act_office2_cube` to the laptop):

```bash
bash local/run_office.sh act_office2_cube control --wait-for-object
```

It moves the arm to the start pose and holds it there until Enter: put the object in the square (centre 18 cm in
front of the base) meanwhile. (Not two runs: connecting turns the torque off briefly, which drops a raised arm.)

On the robot laptop, copy `outputs/act_<profile>_<object>/checkpoints/last/pretrained_model` from the HPC and
run `python local/run_act_policy.py --profile office --object cylinder --gripper-floor 8 --checkpoint ...`.

Re-submitting is safe: an existing dataset is reused and an existing `OUTPUT_DIR` is resumed.
`qsub pbs/mujoco_test.pbs` is a 30-minute plumbing test on throwaway data.

Log: `logs/mujoco_pipeline/<jobid>.log`. Training pauses every `EVAL_FREQ` (10000) steps for a 50-episode eval:
`OUTPUT_DIR/eval/step_<step>.json` + front|wrist videos of the first 5 episodes in `OUTPUT_DIR/eval/step_<step>/`.
On W&B they go into the training run: `eval/success_rate` against `eval/train_step`, and the `eval/videos` media
panel (one entry per eval, step slider).
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
| Success | object centre > 0.10 m above the table for 5 steps (the Isaac task used 0.20 m above the base; the SO-101 cannot lift that high near its base while gripping from above) |
| Eval | 12 s time-out; seeds from 100000 (generation uses 0, 1, 2, …) |

All tunables are in [`so101_liftcube/config.py`](so101_liftcube/config.py).

## Rendering inside the Singularity image

The PyTorch image has no EGL libraries. `--nv` binds the host's `libEGL` and NVIDIA EGL driver, but not
`libGLdispatch.so.0` or the EGL vendor file. `setup_hpc.sh` copies both to `/scratch/e1583535/opt/egl_shim`,
and `sim/hpc_env.sh` sets `LD_LIBRARY_PATH`, `__EGL_VENDOR_LIBRARY_FILENAMES` and `MUJOCO_GL=egl`.
Tested on an H200 node: hardware OpenGL (`NVIDIA H200/PCIe/SSE2`), ~190 frames/s at 640×480. OSMesa is
not available in the image.

## Notes

- Units are a linear map of the MJCF joint ranges, not LeRobot's motor calibration;
  `local/run_act_policy.py` converts between the two on the real arm.
- Success height 0.10 m (above).
