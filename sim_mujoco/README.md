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
| `office2` | office setup rebuilt 2026-10-15: office arm, light wood desk ending ~31 cm in front of the base, no wall, a RealSense D435 (RGB, OpenCV index 1) on a stand 31 cm above the work area looking down ([`assets/office2/`](assets/office2/): start pose, where the gripper is close and out of focus, and folded). Fitted to photos with the arm moved through 9 poses (overhead camera: robot silhouette; wrist camera: the jaw's silhouette, which showed the real wrist camera is mounted 9° further forward and is wide-angle). Objects of random size (below). Start pose: the folded rest pose with the wrist rolled to -87.6°, arm frame [7.5, -97.5, 93.2, 70.0, -87.6], gripper 0 (read from the arm 2026-10-15): the wrist camera faces away from the arm; with the earlier +93° roll it touched the arm |
| `home2` | the home rig as used for the vision_pick demos of 2026-10-15 (home arm, third-person webcam OpenCV 0 in front looking down, wide-angle wrist camera 1; [`assets/home2/`](assets/home2/)). Front camera and wrist camera mount fitted to 49 frames of those 26 real demos at the joints read from the arm (robot silhouette, the cube's pixel in both cameras vs where the gripper closed on it, fingertips half-way down the cube at the grasp): no joint offsets (the arm's calibrated ranges match the model's), the table 4.2 cm below the bottom of the base (`table_z`). Real 3 cm cube, randomized 2.6-3.4 cm, task "Pick up the red cube and lift it." Start pose: the demos' folded start, arm frame [1.6, -98.5, 96.9, 72.9, -7.3], gripper 15.5. Appearance randomized instead of copied (random tables, background, light, shadows, cube colour, mass, friction). Expert 50/50. Comparison images: `sim_mujoco/debug/home2_alignment/` (regenerate with `tools/overlay_photo.py --profile home2`) |

| `office3` | the office rig of 2026-10-16: office arm, the RealSense D435 (RGB, OpenCV 1) moved to a stand in front of the arm, ~51 cm up, looking down and back at it; the wrist camera as in `office2`. Front camera fitted to the arm's silhouette in 13 photos at poses the arm was moved through by a script (`local/setup_captures/office3/`, IoU 0.78). The real objects: 3 cm cube, Ø 3 × 3 cm cylinder (`OFFICE3_OBJECTS`), 20 % of the training episodes 2.7-4.0 cm. The object spawns anywhere in `OFFICE3_REGION` (from `tools/workspace_scan.py`, below), split into held-out train / val / test cells (`so101_liftcube/placements.py`). Tasks "Pick up the cube and lift it." / "Pick up the cylinder and lift it." Appearance, light, mass and friction randomized as in `home2`. Start pose: the folded rest, arm frame [3.4, -102.9, 96.0, 72.1, -92.0], gripper 8. `office3_plain` = the same with the randomization off (expert smoke tests) |

| `--object` | object |
|---|---|
| `cube` | red 3 cm cube, "Lift the red cube." |
| `cylinder` | small red tube (Ø 1.45 cm, 2.8 cm tall) on a Ø 2.1 cm flange, "Lift the red cylinder." The jaws touch it at ~9 (gripper units), so the office expert closes to 8: fine on the office arm (`--gripper-floor 8`), too tight for the home arm (`real` + `cylinder` fails: its gripper floor is 15) |

With `--profile office2` the objects are random per episode instead (`RANDOM_SIZE_OBJECTS` in config.py): a red
cube of side 2-5 cm or a red cylinder of Ø 2-6 cm and 3-7 cm tall, anywhere in a 5 × 5 cm square centred 18 cm in
front of the base and 9 cm to its left, next to the folded gripper (x 0.155-0.205 m, y 0.065-0.115 m, yaw ±45°).
The expert grasps with the wrist roll nearest the start's (`grasp_near_start_roll`) and drops any plan bringing the
wrist camera mount within 35 mm of the arm (`min_camera_clearance`; none do: >= 47 mm in every demo). Expert success
(80 episodes each): cube 100 %; cylinder 79 % (Ø 2-4 cm 100 %, 4-5 cm 90 %, 5-6 cm 35 %: the moving jaw meets a wide
round object at ~33° and it twists out), so few demos have cylinders wider than 5 cm.

Randomization of the look instead of copying the room (off unless a setup turns it on, so the other setups' data
stay as they were): `Appearance.random_table_textures` builds that many procedural table textures (low-contrast
checkers, speckled flat colours, gradients; random colour and scale, never a saturated red) and gives the table one
of them per episode, or the setup's own texture with `p_own_table`; `background_rgb` gives the floor and walls a
random colour; `ObjectConfig.friction_range` / `density_range` vary the object's friction and mass;
`wrist_cam_jitter_deg` tilts the wrist camera a little per episode. Everything follows from the episode seed, so
`sim_mujoco_generation.json` (config + seed of every saved episode + every dropped attempt and why) reproduces a
dataset.

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

The office2 policies (300 demos, 16 eval videos, 100k steps). `_v2`: the folded start pose of 2026-10-15; the
first `so101_office2_*` / `act_office2_*` were made with a raised start pose that put the wrist camera against the arm,
so they must not be reused (the pipeline would resume them) or run on the arm:

```bash
for o in cube cylinder; do qsub -v PROFILE=office2,OBJECT=$o,NUM_DEMOS=300,EVAL_VIDEOS=16,STEPS=100000,DATASET_NAME=so101_office2_${o}_v2,OUTPUT_DIR=outputs/act_office2_${o}_v2 pbs/mujoco_pipeline.pbs; done
```

On the office arm (copy `outputs/act_office2_cube_v2` to the laptop): put the object in the square (18 cm in front
of the base, 9 cm to its left), then

```bash
bash local/run_office.sh act_office2_cube_v2
```

It homes to the start pose (wrist rolled to -87.6°) and runs the policy. The real arm does not check wrist camera
clearance; only the sim expert does (`min_camera_clearance`).

On the robot laptop, copy `outputs/act_<profile>_<object>` from the HPC into `outputs/` and run
`bash local/run_office.sh act_<profile>_<object>` (or `run_home.sh` on the home arm).

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
