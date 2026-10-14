"""Fallback baseline (not learning-based): find the red cube in the third-person camera with OpenCV,
map it onto the table with a touch-calibrated homography, and grasp + lift it with inverse kinematics.
ACT stays the primary method; this is for comparison or when ACT cannot grasp the real cube.

    cd local
    python cube_fallback.py detect                 # live red-cube detection on the third-person camera (no arm)
    python cube_fallback.py calibrate              # torque OFF: "grasp" the cube by hand at >= 5 spots
    python cube_fallback.py verify                 # torque OFF: 3 new spots; must pass before anything moves
    python cube_fallback.py run                    # plan only: detection -> table xy -> IK -> joint targets
    python cube_fallback.py run --execute          # verified calibration + typed "yes": grasp, lift, put back

Calibration by touch, no tape measure: with torque off, lower the OPEN gripper around the cube as if
grasping it (fingers pointing down, cube centred between the jaws, fingertips about half-way down).
Forward kinematics of that pose gives the cube's position in the robot frame; the camera gives its
pixel; >= 5 such pairs fit the pixel -> table homography, and their heights give the grasp height.
`verify` repeats this at new spots and compares with what the camera + homography predicted; it
only passes if every spot is within VERIFY_XY_CM and the heights agree, which checks the camera
model, the kinematics below and the LeRobot-degrees -> MJCF-radians joint convention together.

Kinematics: MuJoCo Menagerie robotstudio_so101 @ 0059d433 (Apache-2.0, as sim_mujoco/setup_hpc.sh
pins), body offsets copied below; tests/test_cube_fallback.py checks them against MuJoCo. Joint
angles: MJCF radians = LeRobot degrees converted (as local/run_act_policy.py; this repo's sim profiles
were fitted to photos of the real arm with it). Execution reuses local/safety.py (is_safe_action,
clip_to_arm, clamp_action) at low speed, stops if a joint stops tracking its command, and refuses
targets outside the calibrated area, out of reach (IK error) or outside the MJCF joint ranges.
Settings: FOLLOWER_PORT / FOLLOWER_ID as for client.py; camera 1 = third person (client.py's).
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from action_format import JOINT_ORDER
from safety import ARM_LIMITS, SANITY_RANGE, clamp_action, clip_to_arm, is_safe_action

LOCAL_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = LOCAL_DIR / "fallback_calibration"

# Menagerie so101.xml @ 0059d433: (pos, quat w x y z) of each moving body, all hinge joints about local z.
CHAIN = [
    ((0.0388353, 0.0, 0.0624), (0.0, 0.0, -1.0, 0.0)),                                  # shoulder: shoulder_pan
    ((-0.0303992, -0.0182778, -0.0542), (1.0, -1.0, -1.0, -1.0)),                       # upper_arm: shoulder_lift
    ((-0.11257, -0.028, 0.0), (1.0, 0.0, 0.0, 1.0)),                                     # lower_arm: elbow_flex
    ((-0.1349, 0.0052, 0.0), (1.0, 0.0, 0.0, -1.0)),                                     # wrist: wrist_flex
    ((5.55112e-17, -0.0611, 0.0181), (0.0172091, -0.0172091, 0.706897, 0.706897)),      # gripper: wrist_roll
]
# "gripperframe" site: between the fingertips; x points along the fingers, z is the jaw closing axis.
TIP = ((0.012, -0.000218, -0.098127), (1.0, 0.0, 1.0, 0.0))
RANGE_RAD = [(-1.91986, 1.91986), (-1.7453293, 1.7453293), (-1.69, 1.69), (-1.658063, 1.658063),
             (-2.7438473, 2.8412063)]
DOWN = np.array([0.0, 0.0, -1.0])

IK_POS_TOL_M = 0.005
# Fingers must be near vertical at the grasp; above/descend/lift may tilt (high targets near the base
# need more than wrist_flex's 95 deg to point straight down), position always has to be exact.
IK_DOWN_TOL_DEG = {"grasp": 8.0, "descend": 20.0, "above": 30.0, "lift": 30.0}
IK_DOWN_WEIGHT = {"grasp": 0.1, "descend": 0.03, "above": 0.01, "lift": 0.01}
VERIFY_XY_CM = 2.0
VERIFY_Z_CM = 2.5
MIN_POINTS = 5
WORKSPACE_MARGIN_M = 0.03     # how far outside the calibrated spots a cube may be
APPROACH_M = 0.06             # pre-grasp height above the grasp
LIFT_M = 0.10
SPEED_DEG_S = 30.0
FPS = 30.0
TRACK_LIMIT_DEG = 12.0


# ---- kinematics ------------------------------------------------------------------------------

def _rot(quat):
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _tf(pos, rot):
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, pos
    return t


_LINKS = [_tf(p, _rot(q)) for p, q in CHAIN]
_TIP = _tf(TIP[0], _rot(TIP[1]))


def fk(q):
    """q: 5 MJCF joint angles (rad). Returns (fingertip position (3,), site rotation (3, 3)) in the base frame."""
    t = np.eye(4)
    for link, angle in zip(_LINKS, q):
        c, s = math.cos(angle), math.sin(angle)
        t = t @ link @ np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    t = t @ _TIP
    return t[:3, 3], t[:3, :3]


def arm_to_rad(state):
    """Arm frame dict (LeRobot degrees) -> 5 MJCF radians."""
    return np.radians([state[n] for n in JOINT_ORDER[:5]])


def ik(target, seeds, roll, down_weight=0.1, iters=300):
    """Fingertips at target with the fingers pointing down; wrist_roll fixed at roll (rad). Damped
    least squares over pan/lift/elbow/wrist_flex from several seeds. Returns (q (5,), pos_err_m, down_err_deg)."""
    lo, hi = np.array([r[0] for r in RANGE_RAD[:4]]), np.array([r[1] for r in RANGE_RAD[:4]])

    def residual(q4):
        p, r = fk(list(q4) + [roll])
        return np.concatenate([p - target, down_weight * (r[:, 0] - DOWN)])

    best = None
    for seed in seeds:
        q = np.clip(np.asarray(seed[:4], dtype=float), lo, hi)
        for _ in range(iters):
            e = residual(q)
            jac = np.stack([(residual(q + d) - e) / 1e-6 for d in np.eye(4) * 1e-6], axis=1)
            # Active set: a joint at a limit that the step pushes further out is frozen and the step
            # re-solved with the others, so the solution slides along the limit instead of stalling.
            free = np.ones(4, dtype=bool)
            for _ in range(4):
                step = np.zeros(4)
                jf = jac[:, free]
                step[free] = np.linalg.solve(jf.T @ jf + 1e-4 * np.eye(free.sum()), -jf.T @ e)
                blocked = free & (((q <= lo + 1e-9) & (step < 0)) | ((q >= hi - 1e-9) & (step > 0)))
                if not blocked.any():
                    break
                free &= ~blocked
            q = np.clip(q + step, lo, hi)
            if np.linalg.norm(step) < 1e-8:
                break
        p, r = fk(list(q) + [roll])
        pos_err = float(np.linalg.norm(p - target))
        down_err = math.degrees(math.acos(max(-1.0, min(1.0, float(r[:, 0] @ DOWN)))))
        if best is None or pos_err + down_err * 1e-3 < best[1] + best[2] * 1e-3:
            best = (np.concatenate([q, [roll]]), pos_err, down_err)
    return best


def jaw_yaw(q):
    """Yaw (rad) of the jaw closing axis in the table plane."""
    _, r = fk(q)
    return math.atan2(r[1, 2], r[0, 2])


def wrap90(a):
    """Angle difference modulo a cube's 90 deg symmetry, in [-45, 45) deg (radians in, radians out)."""
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


# ---- vision --------------------------------------------------------------------------------------

def detect_red_cube(bgr, min_area=150, s_min=120, v_min=70):
    """Largest red blob: dict(u, v, area, box (4, 2) pixel corners) or None. Red wraps around hue 0."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, s_min, v_min), (10, 255, 255)) | cv2.inRange(hsv, (170, s_min, v_min), (180, 255, 255))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < min_area:
        return None
    m = cv2.moments(c)
    return {"u": m["m10"] / m["m00"], "v": m["m01"] / m["m00"], "area": area,
            "box": cv2.boxPoints(cv2.minAreaRect(c)).tolist()}


def fit_homography(pixels, table_xy):
    """pixel (u, v) -> table (x, y) metres. Returns (H, residuals_m, leave_one_out_errors_m)."""
    import cv2

    px, xy = np.asarray(pixels, dtype=np.float64), np.asarray(table_xy, dtype=np.float64)
    h, _ = cv2.findHomography(px, xy, 0)
    res = np.linalg.norm(apply_h(h, px) - xy, axis=1)
    loo = []
    if len(px) > 4:
        for i in range(len(px)):
            keep = np.arange(len(px)) != i
            hi, _ = cv2.findHomography(px[keep], xy[keep], 0)
            loo.append(float(np.linalg.norm(apply_h(hi, px[i:i + 1])[0] - xy[i])))
    return h, res, np.asarray(loo)


def apply_h(h, pixels):
    p = np.hstack([np.asarray(pixels, dtype=np.float64), np.ones((len(pixels), 1))]) @ np.asarray(h).T
    return p[:, :2] / p[:, 2:3]


def cube_on_table(det, h):
    """(x, y, yaw) of a detection: centre and the yaw of one box edge, both mapped through h."""
    centre = apply_h(h, [[det["u"], det["v"]]])[0]
    box = apply_h(h, det["box"])
    edge = box[1] - box[0]
    return float(centre[0]), float(centre[1]), math.atan2(edge[1], edge[0])


# ---- planning --------------------------------------------------------------------------------

def inside_workspace(xy, points, margin=WORKSPACE_MARGIN_M):
    """Inside the convex hull of the calibration spots, grown by margin."""
    import cv2

    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return cv2.pointPolygonTest(hull, (float(xy[0]), float(xy[1])), True) >= -margin


def check_safe(target, cal, current=None):
    """local/safety.py is_safe_action with wrist_roll measured from this arm's grasp orientation (the median
    wrist_roll of the calibration touches): SANITY_RANGE assumes a wrist reading ~0 there, the home arm reads ~93."""
    relative = list(target)
    relative[4] -= float(cal.get("wrist_roll_grasp", 0.0))
    return is_safe_action(current, relative)


def gripper_floored(cmd, floor):
    """clip_to_arm (ARM_LIMITS), then --gripper-floor if given."""
    out, _ = clip_to_arm(cmd)
    if floor is not None:
        out[5] = max(out[5], floor)
    return out


def plan_grasp(x, y, yaw, cal, current_rad, align_roll=True, max_roll_change_deg=45.0, gripper=50.0):
    """Joint-space waypoints (MJCF rad, 5 joints) for above -> grasp -> lift. Returns (waypoints, problems)."""
    problems = []
    roll_lo, roll_hi = (math.radians(v + float(cal.get("wrist_roll_grasp", 0.0))) for v in SANITY_RANGE["wrist_roll"])
    if not inside_workspace((x, y), cal["table_xy"]):
        problems.append(f"cube at ({x:.3f}, {y:.3f}) m is outside the calibrated area (+{WORKSPACE_MARGIN_M * 100:.0f} cm)")
    z = cal["grasp_z"]
    roll = float(current_rad[4])
    seeds = [current_rad[:4], [math.atan2(y, x), 0.0, 0.5, 1.2], [-math.atan2(y, x), 0.0, 0.5, 1.2]]
    if align_roll:
        q_above, _, _ = ik(np.array([x, y, z]), seeds, roll)
        candidates = np.radians(np.arange(-max_roll_change_deg, max_roll_change_deg + 1, 1.0)) + roll
        candidates = [r for r in candidates if RANGE_RAD[4][0] <= r <= RANGE_RAD[4][1] and roll_lo <= r <= roll_hi]
        if candidates:
            roll = min(candidates, key=lambda r: abs(wrap90(jaw_yaw(list(q_above[:4]) + [r]) - yaw)))
    waypoints = []
    for name, height in (("grasp", z), ("descend", z + APPROACH_M / 2), ("above", z + APPROACH_M),
                         ("lift", z + LIFT_M)):
        q, pos_err, down_err = ik(np.array([x, y, height]), seeds, roll, down_weight=IK_DOWN_WEIGHT[name])
        seeds = [q[:4]] + seeds
        if pos_err > IK_POS_TOL_M or down_err > IK_DOWN_TOL_DEG[name]:
            problems.append(f"{name}: out of reach (IK error {pos_err * 1000:.0f} mm, fingers {down_err:.0f} deg off vertical)")
        safe, reason = check_safe(to_arm(q, gripper), cal)
        if not safe:
            problems.append(f"{name}: {reason}")
        waypoints.append((name, q))
    order = ["above", "descend", "grasp", "lift"]
    return sorted(waypoints, key=lambda w: order.index(w[0])), problems


def to_arm(q, gripper):
    return [math.degrees(v) for v in q] + [float(gripper)]


# ---- calibration files -------------------------------------------------------------------------

def cal_path(robot_id):
    return CALIBRATION_DIR / f"{robot_id}.json"


def load_cal(robot_id):
    p = cal_path(robot_id)
    return json.loads(p.read_text()) if p.is_file() else None


def save_cal(robot_id, cal):
    CALIBRATION_DIR.mkdir(exist_ok=True)
    cal_path(robot_id).write_text(json.dumps(cal, indent=2) + "\n")
    return cal_path(robot_id)


def build_cal(pixels, tips, cube_size, wrist_rolls):
    """pixels: (u, v) per spot; tips: fingertip (x, y, z) from FK in the grasp pose; wrist_rolls: the arm's
    wrist_roll reading (deg) at each touch, i.e. its normal grasp orientation."""
    tips = np.asarray(tips, dtype=float)
    h, res, loo = fit_homography(pixels, tips[:, :2])
    return {"created_at": datetime.datetime.now().isoformat(timespec="seconds"), "cube_size": cube_size,
            "pixels": np.asarray(pixels).round(1).tolist(), "table_xy": tips[:, :2].round(4).tolist(),
            "tip_z": tips[:, 2].round(4).tolist(), "grasp_z": float(np.median(tips[:, 2])),
            "wrist_roll_grasp": float(np.median(wrist_rolls)),
            "homography": np.asarray(h).tolist(), "fit_residual_cm": (res * 100).round(2).tolist(),
            "leave_one_out_cm": (loo * 100).round(2).tolist(), "verified": False}


def verify_errors(cal, predicted_xy, touched_tips):
    """Per spot (xy error cm, z error cm) and whether all pass."""
    errs = []
    for (px, py), tip in zip(predicted_xy, touched_tips):
        errs.append((math.hypot(tip[0] - px, tip[1] - py) * 100, abs(tip[2] - cal["grasp_z"]) * 100))
    ok = bool(errs) and all(e_xy <= VERIFY_XY_CM and e_z <= VERIFY_Z_CM for e_xy, e_z in errs)
    return errs, ok


def execute_gate(cal):
    if cal is None:
        return False, "no calibration: run calibrate, then verify"
    if not cal.get("verified"):
        return False, "calibration not verified: run python cube_fallback.py verify"
    return True, "verified " + cal.get("verified_at", "")


# ---- hardware commands -------------------------------------------------------------------------

def open_hardware(args, need_robot=True):
    from cameras import CameraManager

    cameras = CameraManager(wrist_id=args.wrist_cam, third_id=args.third_cam)
    robot = None
    if need_robot:
        from robot import SO101Robot

        if not args.port:
            sys.exit("set FOLLOWER_PORT or pass --port")
        robot = SO101Robot(args.port, args.robot_id)
        robot.connect()
    return cameras, robot


def detect_once(cameras, args, tries=10):
    for _ in range(tries):
        _, third = cameras.capture()
        det = detect_red_cube(third, args.min_area)
        if det is not None:
            return det, third
    return None, third


def touch_spot(cameras, robot, args, label):
    input(f"\n{label}: place the cube (arm out of the camera's way), then press Enter ")
    det, _ = detect_once(cameras, args)
    if det is None:
        print("  no red cube found in the third-person camera; check `detect`")
        return None
    input("  now lower the OPEN gripper around it as if grasping (fingers down, cube centred between the jaws,\n"
          "  fingertips about half-way down the cube), then press Enter ")
    state = robot.get_state()
    tip, rot = fk(arm_to_rad(state))
    off = math.degrees(math.acos(max(-1.0, min(1.0, float(rot[:, 0] @ DOWN)))))
    print(f"  pixel ({det['u']:.0f}, {det['v']:.0f}) -> fingertips at ({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f}) m, "
          f"fingers {off:.0f} deg off vertical")
    return det, tip, state["wrist_roll"]


def cmd_detect(args):
    import cv2

    cameras, _ = open_hardware(args, need_robot=False)
    print("q quits")
    try:
        while True:
            _, frame = cameras.capture()
            det = detect_red_cube(frame, args.min_area)
            if det:
                cv2.polylines(frame, [np.int32(det["box"])], True, (0, 255, 0), 2)
                cv2.putText(frame, f"({det['u']:.0f},{det['v']:.0f}) area {det['area']:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("red cube (third-person camera)", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        cameras.close()


def cmd_calibrate(args):
    cameras, robot = open_hardware(args)
    robot.robot.bus.disable_torque()
    print(f"torque OFF. Spread >= {MIN_POINTS} spots over where the cube will be. Ctrl+C when done.")
    pixels, tips, rolls = [], [], []
    try:
        while True:
            got = touch_spot(cameras, robot, args, f"spot {len(pixels) + 1}")
            if got:
                pixels.append([got[0]["u"], got[0]["v"]])
                tips.append(got[1].tolist())
                rolls.append(got[2])
    except KeyboardInterrupt:
        print()
    finally:
        robot.disconnect()
        cameras.close()
    if len(pixels) < MIN_POINTS:
        sys.exit(f"only {len(pixels)} spots; need {MIN_POINTS}")
    cal = build_cal(pixels, tips, args.cube_size, rolls)
    print(f"fit residual (cm) {cal['fit_residual_cm']}\nleave-one-out (cm) {cal['leave_one_out_cm']}")
    print(f"grasp height {cal['grasp_z'] * 100:.1f} cm (fingertips; cube {args.cube_size * 100:.1f} cm), "
          f"spread {np.ptp(cal['tip_z']) * 100:.1f} cm")
    if np.ptp(cal["tip_z"]) > 0.03:
        print("WARNING: grasp heights vary by > 3 cm over the table: kinematics / joint convention may be off")
    print(f"saved {save_cal(args.robot_id, cal)} (NOT verified: run python cube_fallback.py verify)")


def cmd_verify(args):
    cal = load_cal(args.robot_id)
    if cal is None:
        sys.exit("no calibration; run calibrate first")
    cameras, robot = open_hardware(args)
    robot.robot.bus.disable_torque()
    predicted, touched = [], []
    try:
        while len(touched) < args.verify_spots:
            got = touch_spot(cameras, robot, args, f"verify spot {len(touched) + 1}/{args.verify_spots} (a NEW spot)")
            if got:
                x, y, _ = cube_on_table(got[0], cal["homography"])
                predicted.append((x, y))
                touched.append(got[1].tolist())
    finally:
        robot.disconnect()
        cameras.close()
    errs, ok = verify_errors(cal, predicted, touched)
    for i, (e_xy, e_z) in enumerate(errs):
        print(f"  spot {i + 1}: camera vs touch {e_xy:.1f} cm (limit {VERIFY_XY_CM}), height {e_z:.1f} cm (limit {VERIFY_Z_CM})")
    cal["verify"] = {"at": datetime.datetime.now().isoformat(timespec="seconds"), "errors_cm": errs, "passed": ok}
    if ok:
        cal["verified"], cal["verified_at"] = True, cal["verify"]["at"]
    save_cal(args.robot_id, cal)
    print("VERIFIED: run may execute." if ok else "FAILED: recalibrate (more/better spread spots) before executing.")


def move_to(robot, start_cmd, target, speed, label, floor, cal):
    """Rate-limited joint-space move through the local safety checks. Returns the last command."""
    cmd = list(start_cmd)
    lag_since = arrived = None
    while True:
        t0 = time.perf_counter()
        current = robot.get_state()
        delta = [t - c for t, c in zip(target, cmd)]
        biggest = max(abs(d) for d in delta[:5])
        scale = min(1.0, speed / FPS / biggest) if biggest > 0 else 1.0
        cmd = [c + d * scale for c, d in zip(cmd[:5], delta[:5])] + [target[5]]
        safe, reason = check_safe(cmd, cal, current)
        if not safe:
            raise RuntimeError(f"{label}: {reason}")
        sent, _ = clamp_action(current, gripper_floored(cmd, floor))  # backstop; the ramp above sets the speed
        robot.send_action(dict(zip(JOINT_ORDER, sent)))
        lag = max(abs(current[n] - c) for n, c in zip(JOINT_ORDER[:5], cmd[:5]))
        if lag > TRACK_LIMIT_DEG:
            lag_since = lag_since or time.time()
            if time.time() - lag_since > 0.5:
                robot.send_action(current)
                raise RuntimeError(f"{label}: a joint is {lag:.0f} deg behind its command (blocked?)")
        else:
            lag_since = None
        if scale >= 1.0:  # command at the target: give the servos up to 2 s (gravity sag) to get there
            arrived = arrived or time.time()
            if lag < 2.0 or time.time() - arrived > 2.0:
                return cmd
        time.sleep(max(0.0, 1.0 / FPS - (time.perf_counter() - t0)))


def settle(robot, cmd, seconds, floor):
    end = time.time() + seconds
    while time.time() < end:
        robot.send_action(dict(zip(JOINT_ORDER, gripper_floored(cmd, floor))))
        time.sleep(1.0 / FPS)


def cmd_run(args):
    cal = load_cal(args.robot_id)
    if cal is None:
        sys.exit("no calibration; run calibrate and verify first")
    gate_ok, gate_reason = execute_gate(cal)
    if args.execute and not gate_ok:
        sys.exit(f"refusing to execute: {gate_reason}")
    floor = args.gripper_floor
    cameras, robot = open_hardware(args)
    try:
        det, _ = detect_once(cameras, args)
        if det is None:
            sys.exit("no red cube in the third-person camera")
        x, y, yaw = cube_on_table(det, cal["homography"])
        state = robot.get_state()
        waypoints, problems = plan_grasp(x, y, yaw, cal, arm_to_rad(state), align_roll=not args.no_align_roll,
                                         gripper=args.open_pct)
        print(f"cube at pixel ({det['u']:.0f}, {det['v']:.0f}) -> table ({x:.3f}, {y:.3f}) m, yaw {math.degrees(yaw):.0f} deg")
        print(f"current {['%.1f' % state[n] for n in JOINT_ORDER]}")
        for name, q in waypoints:
            print(f"  {name:8s} {['%.1f' % v for v in to_arm(q, args.open_pct)]}")
        for p in problems:
            print(f"  PROBLEM: {p}")
        if problems or not args.execute:
            print("plan only (pass --execute to move)" if not problems else "not executable")
            return
        print(f"execute: open {args.open_pct:g} -> close {args.close_pct:g} (floor {floor if floor is not None else ARM_LIMITS['gripper'][0]:g}), "
              f"{SPEED_DEG_S:g} deg/s; calibration {gate_reason}")
        if input("The arm will move. Type 'yes' to continue: ").strip() != "yes":
            sys.exit("not confirmed")
        plan = dict(waypoints)
        cmd = [state[n] for n in JOINT_ORDER]
        cmd = move_to(robot, cmd, to_arm(plan["above"], args.open_pct), SPEED_DEG_S, "approach", floor, cal)
        for name in ("descend", "grasp"):
            cmd = move_to(robot, cmd, to_arm(plan[name], args.open_pct), SPEED_DEG_S / 2, name, floor, cal)
        cmd = cmd[:5] + [args.close_pct]
        settle(robot, cmd, 1.0, floor)
        cmd = move_to(robot, cmd, to_arm(plan["lift"], args.close_pct), SPEED_DEG_S / 2, "lift", floor, cal)
        input("Lifted. Press Enter to put the cube back down ")
        cmd = move_to(robot, cmd, to_arm(plan["grasp"], args.close_pct), SPEED_DEG_S / 2, "lower", floor, cal)
        cmd = cmd[:5] + [args.open_pct]
        settle(robot, cmd, 1.0, floor)
        move_to(robot, cmd, to_arm(plan["above"], args.open_pct), SPEED_DEG_S / 2, "retreat", floor, cal)
        print("done")
    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"\nSTOPPED: {e}")
        robot.send_action(robot.get_state())
    finally:
        if args.execute:
            try:
                input("Support the arm, then press Enter to disable torque... ")
            except (EOFError, KeyboardInterrupt):
                pass
        robot.disconnect()
        cameras.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["detect", "calibrate", "verify", "run"])
    ap.add_argument("--port", default=os.environ.get("FOLLOWER_PORT"))
    ap.add_argument("--robot-id", default=os.environ.get("FOLLOWER_ID", "home_follower"))
    ap.add_argument("--wrist-cam", type=int, default=0)
    ap.add_argument("--third-cam", type=int, default=1)
    ap.add_argument("--min-area", type=float, default=150.0, help="smallest red blob (pixels) taken as the cube")
    ap.add_argument("--cube-size", type=float, default=0.03, help="cube edge, metres (recorded with the calibration)")
    ap.add_argument("--verify-spots", type=int, default=3)
    ap.add_argument("--execute", action="store_true", help="run: actually move (needs a verified calibration)")
    ap.add_argument("--open-pct", type=float, default=90.0)
    ap.add_argument("--close-pct", type=float, default=15.0, help="gripper target when closing (below contact)")
    ap.add_argument("--gripper-floor", type=float, default=None, help="as client.py (default ARM_LIMITS)")
    ap.add_argument("--no-align-roll", action="store_true", help="keep wrist_roll as is (no jaw/cube alignment)")
    args = ap.parse_args()
    {"detect": cmd_detect, "calibrate": cmd_calibrate, "verify": cmd_verify, "run": cmd_run}[args.command](args)


if __name__ == "__main__":
    main()
