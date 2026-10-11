# SO-101 LiftCube in MuJoCo: scripted demos → LeRobot dataset → ACT → eval, all on the HPC

> **SIMULATION ONLY.** Nothing here talks to the real SO-101.

MuJoCo counterpart of [`sim/`](../sim/README.md) (Isaac Sim). Isaac Sim needs an RTX GPU to render
cameras; this cluster has only H100/H200, which MuJoCo renders on with EGL. Same task and dataset
format as `sim/`: front + wrist cameras 640×480 @ 30 fps, `observation.state` (6), `action` (6, absolute
joint targets), `robot_type=so101_follower`, task **"Lift the red cube."**. ACT training uses the same
`sim/train_act_hpc.sh`.

## Commands (repository root, NUS HPC)

```bash
bash sim_mujoco/setup_hpc.sh          # once (after qsub pbs/setup_act_env.pbs)
qsub pbs/mujoco_pipeline.pbs          # generate 100 demos -> train ACT 20k steps -> evaluate 50 episodes
```

Options: `qsub -v NUM_DEMOS=200,STEPS=30000,OUTPUT_DIR=outputs/act_mujoco_try2 pbs/mujoco_pipeline.pbs`.
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
| Scene | table, red 3 cm cube (spawn x 0.17–0.25 m, y ±0.07 m, yaw ±30°), front camera (±1 cm / ±1.5° jitter), wrist camera from the model |
| Control | 30 Hz, 6 physics substeps; position actuators of the model |
| Units | arm joints: MJCF joint range → −100..100; gripper: range → 0..100 (0 = closed) |
| Expert | IK on the fingertip site: approach above the cube → descend with fingers down and jaws aligned to a cube face → close → lift (position-only IK, ~13 cm) → hold. ~150 frames (5 s) |
| Success | cube centre > 0.10 m above the table for 5 steps (Isaac task: 0.20 m above the base; the SO-101 cannot lift that high near its base while gripping from above) |
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
