# vision_pick: red-cube pickup without learning

Find the red cube in the fixed third-person camera with OpenCV, convert its pixel to a table position in the
robot's base frame with a homography calibrated by touching the cube with the gripper, solve top-down IK, and run
a slow, checked grasp-and-lift on the real SO-101. No ACT, VLA, RL, simulator, server, GPU or training.

**Status:** detection has been checked on a real frame from the home rig. Planning and the control sequence have
been run against a simulated arm. **Nothing has run on the physical arm yet**, so calibration and the first
pickups are still to do.

## Install

Nothing new. It uses the `lerobot` conda env (OpenCV, NumPy, LeRobot's SO-101 driver) and these files in
`local/`: `cameras.py` (cameras), `robot.py` (arm), `safety.py` (`is_safe_action`, `clip_to_arm`, `clamp_action`,
`ARM_LIMITS`), `action_format.py`, and the kinematics in `cube_fallback.py`.

```bash
conda activate lerobot
cd ~/Documents/NUS_PhD/Projects/lerobot-mini-project     # always run from the repo root
python -m local.vision_pick --help
```

## The four commands

```bash
python -m local.vision_pick --mode detect      # camera + detection, sliders for HSV; never moves the arm
python -m local.vision_pick --mode calibrate   # torque OFF, touch the cube at >= 5 spots, writes calibration/home.json
python -m local.vision_pick --mode preview     # detect -> robot x,y -> full plan + checks; never sends a command
python -m local.vision_pick --mode control     # same plan, type 'yes', then grasp, lift, hold, put back
```

- `--arm home` (default) or `--arm office` selects the port, LeRobot id, gripper floor and camera indices from
  `settings.json`.
- `--camera N` overrides the third-person camera index.
- `--image FILE` runs detect mode on a saved frame instead of the live camera.

## Step by step on your arm

### 1. Cameras and detection

```bash
python -m local.vision_pick --mode detect --arm home
```

The window must show the **third-person** view. On the home laptop on 2026-10-15, index 0 was the third-person
camera and index 1 the wrist camera, and `settings.json` uses that. macOS can renumber cameras when they are
replugged; if you see the wrist view, run with `--camera 1`, then fix `third_cam`/`wrist_cam` in
`settings.json`.

Put the cube where it will be picked up. A green box and a white cross should sit on it, with nothing else
boxed. Red boxes show rejected blobs and why they were rejected. If detection is wrong:

- **`m`** switches to the mask view: only the cube should be white.
- **`hue_max`, `sat_min`, `val_min`** need to be tight, because the wooden table is orange. On the home table
  the cube measured hue 1-3 / V ~255 and the table hue 7-14 / V 80-140, so the defaults are hue ±5, S ≥ 150,
  V ≥ 150. `cube_fallback.py`'s looser defaults (hue ≤ 10, S ≥ 120) mark about 62,000 table pixels as red on
  the same frame.
- **`s`** saves the sliders to `settings.json`. **`p`** saves the current frame to `runs/snapshots/`.

Set `cube.size_m` in `settings.json` to your cube's measured edge length (default 0.03).

### 2. Calibrate (about 10 minutes)

The camera must not move after this. If it gets bumped, calibrate again.

```bash
python -m local.vision_pick --mode calibrate --arm home
```

Torque is **off** for the whole calibration, so the arm drops if you let go. Rest it on the table between spots.
Keys go to the camera window. For each spot:

1. **Place the cube.** Keep the arm out of the camera's view of it, then press SPACE. The cube must be seen
   steadily over 5 fresh frames.
2. **Touch it.** Lower the **open** gripper around the cube as if grasping it:
   - fingers vertical;
   - cube centred between the jaws;
   - fingertips about half-way down the cube.

   The overlay shows the fingertip position and the tilt live. Press SPACE (refused if the fingers are more than
   20° off vertical).

**Where to put the spots:** use 6-9 spots, at the corners and middle of the area you want to use, with the jaw
orientation the same each time. Top-down grasps are reachable about **15-28 cm in front of the base and up to
±15 cm sideways** (IK checked over that grid). Further out the fingers can't point straight down.

**Keys:** `u` undoes the last spot, `f` finishes.

**Gripper check:** squeeze the gripper closed on the cube by hand and press SPACE, then open it fully and press
SPACE. This checks the closing direction and gives the closing target: contact minus `squeeze_pct`, never below
the arm's floor or `safety.py`'s `ARM_LIMITS`.

Calibration fits pixel → table (x, y) from the spots: an affine map below 8 spots, a homography from 8 up. (A
homography has 8 unknowns; on 5 clustered spots it extrapolated 6 cm off.) It prints:
- **fit residual** (cm);
- **leave-one-out error** (cm): each spot predicted from the others, which is the honest accuracy number;
- **fingertip heights**, giving the grasp height;
- **table height:** grasp height minus half the cube. The model's base frame has z = 0 at the bottom of the
  base, so this should be about 0. If it isn't, the joint calibration, the cube size or the touches are off.

**Calibration warnings:** preview and control print a warning (control still moves) unless all of these hold:
- ≥ 5 spots;
- leave-one-out ≤ 1.5 cm;
- grasp heights within 2.5 cm of each other;
- table height within 2.5 cm of 0;
- the cube size matches `settings.json`;
- closing the gripper lowers its reading by > 10;
- the camera gives the same image size as during calibration.

Each warning means misses are likely, so recalibrate when you can. Control only refuses when there is no measured
calibration. `calibration/example.json` shows the file format; it is synthetic, marked `"example": true`, and
refused.

### 3. Preview

```bash
python -m local.vision_pick --mode preview --arm home
```

Preview reads the joints without configuring the servos or changing torque, and never sends a command. It
prints:
- the cube's pixel and robot (x, y);
- every waypoint: joint angles, fingertip x/y/z and tilt;
- warnings, and any problem that would stop control mode.

The image shows the calibrated area and a magenta cross where the grasp will go.

### 4. Control

Start with the arm resting and the cube **on one of the calibration spots**.

```bash
python -m local.vision_pick --mode control --arm home
```

1. Connects. Torque goes on and the arm holds where it is.
2. Detects, plans and prints the plan. Any problem stops it here.
3. Asks you to type `yes`.
4. Checks the camera again: if the cube moved more than 10 px or the arm moved since planning, it stops.
5. Moves above the cube (approach height, 6 cm), then opens the gripper.
6. **Pauses: check the jaws are centred over the cube**, then press Enter. If they're off, press Ctrl+C and
   adjust `grasp_offset_x_m` / `grasp_offset_y_m`.
7. Descends straight down in 2 cm steps, closes, and lifts 8 cm.
8. Reports **LIFTED** if the jaws stopped on the cube, or **MISSED** if they closed all the way to the target.
9. Holds. Enter puts the cube back, opens the gripper, goes back up and returns to the start pose, then turns
   torque off.

**Ctrl+C** at any time makes the arm hold where it is. You are then asked to support the arm before torque goes
off.

**What runs automatically:**
- Arm joints move at 25°/s (12°/s near the cube).
- Every command passes `safety.is_safe_action` (wrist roll measured from the calibrated grasp roll), plus
  `clip_to_arm` / the gripper floor, plus `clamp_action`.
- If a joint lags its command by more than 12° for 0.5 s (blocked), the arm holds and the run stops.
- Each target is refused before the arm moves toward it if it fails the safety check.
- A cube outside the calibrated area (plus 2 cm) only gets a warning. Before anything moves, the plan is
  refused if:
  - IK misses by more than 5 mm;
  - the fingers are more than 10° off vertical at the grasp;
  - a joint is out of range;
  - IK changes branch between steps;
  - the fingertips would dip below the cube top plus 1 cm anywhere along the path.

Every preview/control run leaves `runs/<time>_<mode>_<arm>/` with:
- `log.txt`;
- `third.png` (annotated), `third_raw.png`, `wrist.png`;
- `plan.json`;
- for control: `result.json` (success / failure reason) and images after the lift.

## Settings (`settings.json`)

| key | meaning |
|---|---|
| `arms.<name>` | `port`, `robot_id` (LeRobot calibration), `gripper_floor` (home 15: stalls below; office 8), `third_cam`, `wrist_cam` |
| `detection` | `hue_max`, `sat_min`, `val_min`; blob `min_area_px` / `max_area_px`; `min_fill` (area / rotated box, rejects ragged blobs); `max_aspect` |
| `cube.size_m` | cube edge length; must match the calibration |
| `grasp` | `grasp_offset_{x,y,z}_m` (constant corrections), `approach_height_m` (0.06), `lift_height_m` (0.08), `step_m` (vertical step), `open_pct`, `squeeze_pct` (past contact), `min_table_clearance_m`, `workspace_margin_m` |
| `motion` | `speed_deg_s`, `slow_speed_deg_s`, `gripper_speed_pct_s`, `fps`, `track_limit_deg`, `arrive_tol_deg`, `settle_s` |
| `checks` | calibration gate limits, IK tolerances, `stable_frames` / `max_jitter_px` (fresh-frame detection), `max_shift_px` (re-check before moving) |

## If it misses

- **Jaws consistently off to one side at the pause:** set `grasp_offset_x_m` / `grasp_offset_y_m` (robot frame:
  x forward from the base, y to the robot's left).
- **Off in different directions at different spots:** add calibration spots where it's worst and recalibrate.
  Check the leave-one-out numbers.
- **Fingertips hit the table or close above the cube:** adjust `grasp_offset_z_m` (± a few mm).
- **MISSED, but the cube was between the jaws:** raise `squeeze_pct`. Check the gripper check values in the
  calibration file.
- **Cube rotated about 45° to the jaws:** the grasp uses a fixed wrist roll (the one from your touches), so place
  the cube square to the jaws as during calibration.

## How it works

| file | |
|---|---|
| `detection.py` | HSV mask (red wraps around hue 0) → contours → size / fill / aspect filters → largest blob's centroid. `detect_stable` drops buffered frames, requires the cube in N fresh frames within a few px, and refuses a frozen camera. |
| `calibration.py` | Homography pixel → table (x, y) from the touches (`cv2.findHomography`), leave-one-out errors, the gate, JSON load/save. Pixels are the blob centre as seen from the oblique camera; using the real cube for the touches folds that offset into the fit. |
| `kinematics.py` | FK/IK from `cube_fallback.py`. The FK equals MuJoCo on the Menagerie SO-101 model (0.0000 mm over 200 random poses, checked 2026-10-15). Joint convention: MJCF radians = LeRobot degrees. |
| `planner.py` | IK waypoints, fingers down, fixed wrist roll: above → open → vertical descent → grasp → close → vertical lift; place in reverse; all the checks above. |
| `controller.py` | Ramped, checked execution on anything with `get_state()` / `send_action()`. |
| `hardware.py` | Cameras and arm: read-only for preview, torque off for calibrate, torque on for control. |
| `__main__.py` | The four modes. |

Calibrating by touch means no tape measure, camera intrinsics or invented transforms are involved. The table
positions come from the arm's own kinematics, so IK aims at the same coordinates the touches were measured in.

## Recording ACT demonstrations

`python local/collect_act.py --episodes 50` runs this controller once per Enter and records each pickup as a
LeRobot episode (both cameras, joints, the commands sent). See the main README, "ACT demonstrations from
vision_pick".

## Not done yet (lower priority, after the pickup works)

- Cylinder detection.
- Wrist-camera alignment.
- Rotating the jaws to the cube's yaw.

`local/cube_fallback.py` is the earlier single-file version of the same idea. `vision_pick` reuses its kinematics
and replaces its workflow.
