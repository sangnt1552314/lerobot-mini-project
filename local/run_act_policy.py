"""Run a MuJoCo-trained ACT checkpoint on the real SO-101 follower.

    cd local
    python run_act_policy.py --no-robot                    # load the model only, no hardware
    python run_act_policy.py                               # preview: robot + cameras, NEVER moves
    python run_act_policy.py --mode control                # home (arm, then wrist roll, then gripper), run ACT
    python run_act_policy.py --mode control --hold-wrist-roll   # never rotate the wrist (camera cable)
    python run_act_policy.py --checkpoint ../outputs/act_office_cylinder --gripper-floor 8   # office arm

--checkpoint takes a run dir (outputs/act_office_cylinder) or a pretrained_model dir. The sim profile and
object it was trained on come from its dataset name (so101_<profile>_<object>, as pbs/mujoco_pipeline.pbs
names them; the older so101_liftcube_mujoco[_real...] datasets are cube); --profile / --object override.
Shortcuts for the two arms: local/run_home.sh and local/run_office.sh.

Settings: --port / $FOLLOWER_PORT (default /dev/tty.usbmodem5C821064861) and --robot-id /
$FOLLOWER_ID (default home_follower, the LeRobot calibration name). .env files are NOT read, so
the follower port in scripts/.env (another arm) can't be picked up by accident.

Units. The checkpoint was trained in sim_mujoco/, whose state/action units are a linear map of the
MuJoCo Menagerie SO-101 joint ranges (arm -100..100, gripper 0..100), not this arm's LeRobot
calibration. That model is derived from TheRobotStudio's so101_new_calib, whose joint zero is meant to
match LeRobot's calibrated mid-range, so the arm is read in degrees (as local/robot.py does) and
converted through the MJCF ranges. Per-joint SIGN / OFFSET_DEG are best guesses: verify them with
--mode preview and by watching the homing move before trusting the policy. A sim-only policy is not expected to
transfer well; this script is for trying it safely.

Safety: preview is the default and never sends commands. In control mode every target is checked
against the checkpoint's own training range (stop if outside), the gripper is clipped by
--gripper-floor (default local/safety.ARM_LIMITS: the home arm stalls below ~15; the office arm's
calibration closes at ~0, and its cylinder policy closes to 8), joints move at most --max-speed deg/s
(--home-speed while homing), and if a joint lags its command by more than --track-limit for 0.5 s
(e.g. blocked) the arm is told to hold and the run stops. It also stops (holding) before any target that would bring
the wrist camera mount within --min-camera-clearance (office2: 35 mm, sim model) of the arm. Ctrl+C stops. Keep a
hand near the arm.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

LOCAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = LOCAL_DIR.parent
sys.path.insert(0, str(LOCAL_DIR))
sys.path.insert(0, str(REPO_ROOT / "sim"))

from action_format import JOINT_ORDER  # noqa: E402
from safety import ARM_LIMITS  # noqa: E402

DEFAULT_CHECKPOINT = REPO_ROOT / "outputs/act_mujoco_real_v2/checkpoints/last/pretrained_model"

# --- sim <-> real joint mapping (edit after checking directions) --------------------------------
# MJCF joint ranges (rad) of MuJoCo Menagerie robotstudio_so101 @ 0059d433 (sim_mujoco/setup_hpc.sh).
MJCF_RANGE = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.7453293, 1.7453293),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.658063, 1.658063),
    "wrist_roll": (-2.7438473, 2.8412063),
    "gripper": (-0.174533, 1.7453292),
}
# real_deg = SIGN * sim_deg + OFFSET_DEG for the five arm joints. The gripper is passed through
# unchanged: sim 0..100 over its MJCF range, real 0..100 over the calibrated range.
SIGN = {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1, "wrist_roll": 1}
OFFSET_DEG = {"shoulder_pan": 0.0, "shoulder_lift": 0.0, "elbow_flex": 0.0, "wrist_flex": 0.0, "wrist_roll": 0.0}


def _load_sim_config_module():
    """sim_mujoco's config.py alone, so MuJoCo isn't needed here."""
    import importlib.util

    path = REPO_ROOT / "sim_mujoco" / "so101_liftcube" / "config.py"
    spec = importlib.util.spec_from_file_location("sim_mujoco_config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses needs the module registered
    spec.loader.exec_module(module)
    return module


SIM_CONFIG = _load_sim_config_module()


def load_sim_config(profile: str, obj: str = "cube"):
    return SIM_CONFIG.make_config(profile, obj)


def resolve_checkpoint(path) -> Path:
    """A pretrained_model dir, or a run dir (outputs/act_x) / checkpoint step dir that contains one."""
    p = Path(path).expanduser().resolve()
    for cand in (p, p / "pretrained_model", p / "checkpoints" / "last" / "pretrained_model"):
        if (cand / "config.json").is_file():
            return cand
    sys.exit(f"No checkpoint in {p} (expected config.json there or in checkpoints/last/pretrained_model)")


def infer_sim_setup(checkpoint: Path):
    """(profile, object) of the sim dataset a checkpoint was trained on, from the dataset name in its
    train_config.json: so101_<profile>_<object>[_suffix], or the older so101_liftcube_mujoco[_real...] (cube).
    None if the name says neither."""
    try:
        name = json.loads((checkpoint / "train_config.json").read_text())["dataset"]["repo_id"].split("/")[-1]
    except (OSError, KeyError, ValueError):
        return None
    if name.startswith("so101_liftcube_mujoco"):
        return ("real" if "_real" in name else "default"), "cube"
    for profile in SIM_CONFIG.PROFILES:
        for obj in SIM_CONFIG.OBJECTS:
            if name == f"so101_{profile}_{obj}" or name.startswith(f"so101_{profile}_{obj}_"):
                return profile, obj
    return None


# Start pose (rad) of the training episodes; set from the checkpoint's profile in main().
SIM_HOME_RAD = load_sim_config("default").home_qpos
# arm-frame target -> distance (m) from the wrist camera mount to the arm; set in main() (None = no check)
CAMERA_GUARD = None


def make_camera_guard(profile: str, obj: str):
    """Distance from the wrist camera mount to the base / shoulder / upper / lower arm at an arm-frame target, in the
    sim_mujoco model (needs MuJoCo, which the lerobot env has). Used to stop before the camera hits the arm."""
    sys.path.insert(0, str(REPO_ROOT / "sim_mujoco"))
    from so101_liftcube import LiftCubeEnv, make_config
    from so101_liftcube.expert import CameraClearance

    clearance = CameraClearance(LiftCubeEnv(make_config(profile, obj), render=False))

    def guard(target):
        q = [math.radians((v - OFFSET_DEG[n]) / SIGN[n]) for n, v in zip(JOINT_ORDER[:5], target[:5])]
        return clearance(np.array(q + [0.0]))

    return guard


def sim_units_to_real(u):
    """sim_mujoco units (6,) -> arm frame: degrees for arm joints, 0..100 for the gripper."""
    real = []
    for name, value in zip(JOINT_ORDER, u):
        lo, hi = MJCF_RANGE[name]
        if name == "gripper":
            real.append(float(value))
            continue
        rad = lo + (float(value) + 100.0) * (hi - lo) / 200.0
        real.append(SIGN[name] * math.degrees(rad) + OFFSET_DEG[name])
    return real


def real_to_sim_units(real):
    """Inverse of sim_units_to_real."""
    u = []
    for name, value in zip(JOINT_ORDER, real):
        lo, hi = MJCF_RANGE[name]
        if name == "gripper":
            u.append(float(value))
            continue
        rad = math.radians((float(value) - OFFSET_DEG[name]) / SIGN[name])
        u.append(200.0 * (rad - lo) / (hi - lo) - 100.0)
    return np.asarray(u, dtype=np.float32)


def sim_rad_to_units(q):
    u = []
    for name, rad in zip(JOINT_ORDER, q):
        lo, hi = MJCF_RANGE[name]
        frac = (rad - lo) / (hi - lo)
        u.append(100.0 * frac if name == "gripper" else 200.0 * frac - 100.0)
    return np.asarray(u, dtype=np.float32)


def training_ranges(checkpoint: Path, pad_deg: float):
    """(state q01..q99 in sim units, padded action min..max in arm frame) from the checkpoint stats."""
    pre = load_file(str(checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"))
    post = load_file(str(checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    state_q = (pre["observation.state.q01"].numpy(), pre["observation.state.q99"].numpy())
    lo, hi = sim_units_to_real(post["action.min"].numpy()), sim_units_to_real(post["action.max"].numpy())
    action_box = {n: (min(a, b) - pad_deg, max(a, b) + pad_deg) for n, a, b in zip(JOINT_ORDER, lo, hi)}
    return state_q, action_box


def fmt(values):
    return "[" + ", ".join(f"{v:7.1f}" for v in values) + "]"


def rate_limit(current, target, max_step):
    """Move every joint toward target by at most max_step, scaling the whole delta uniformly."""
    deltas = [t - c for c, t in zip(current, target)]
    biggest = max(abs(d) for d in deltas)
    if biggest <= max_step or biggest == 0.0:
        return list(target)
    return [c + d * max_step / biggest for c, d in zip(current, deltas)]


def clip_gripper(target, lo):
    out = list(target)
    hi = ARM_LIMITS.get("gripper", (0.0, 100.0))[1]
    out[5] = min(max(out[5], lo), hi)
    return out


class Follower:
    def __init__(self, port: str, robot_id: str, keep_torque: bool):
        from lerobot.robots.so_follower import SOFollower
        from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

        cfg = SOFollowerRobotConfig(
            port=port, id=robot_id, use_degrees=True, disable_torque_on_disconnect=not keep_torque
        )
        self.robot = SOFollower(cfg)

    def connect(self):
        self.robot.connect(calibrate=False)

    def state(self):
        obs = self.robot.get_observation()
        return [float(obs[f"{n}.pos"]) for n in JOINT_ORDER]

    def send(self, target):
        self.robot.send_action({f"{n}.pos": float(v) for n, v in zip(JOINT_ORDER, target)})

    def disconnect(self):
        self.robot.disconnect()


def crop_to_aspect(frame, aspect=4 / 3):
    """Centre-crop to the sim's 4:3 instead of squashing (the wrist camera delivers 16:9)."""
    h, w = frame.shape[:2]
    if w / h > aspect + 1e-3:
        nw = int(round(h * aspect))
        x = (w - nw) // 2
        return frame[:, x : x + nw]
    if w / h < aspect - 1e-3:
        nh = int(round(w / aspect))
        y = (h - nh) // 2
        return frame[y : y + nh]
    return frame


def warm_up(cameras, seconds=1.5):
    """Stream for a moment before the first real frame: the office RealSense's first ~1 s of frames are
    overexposed (almost white) until its auto-exposure settles."""
    t_end = time.time() + seconds
    while time.time() < t_end:
        cameras.capture()


def read_rgb(cameras, size=(640, 480)):
    wrist_bgr, front_bgr = cameras.capture()
    out = {}
    for key, frame in (("front", front_bgr), ("wrist", wrist_bgr)):
        frame = crop_to_aspect(frame, size[0] / size[1])
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        out[key] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return out


class Stuck(RuntimeError):
    pass


class Setpoint:
    """Commanded position that advances toward a target at a fixed speed (units/s, wall clock).

    Commanding a moving setpoint instead of "measured position + small step" lets the servo's
    position loop see the full error, so the arm moves at the requested speed instead of creeping.
    If a joint falls more than track_limit behind its command for 0.5 s (blocked, e.g. the wrist
    camera caught on something), the arm is told to stay where it is and Stuck is raised.
    track_limit None = no check (preview: the arm is not moving, so it always "lags").
    """

    def __init__(self, robot, track_limit: float | None):
        self.robot = robot
        self.track_limit = track_limit
        self.cmd = robot.state() if robot else None
        self.t = time.time()
        self.lag_since = None

    def step(self, target, speed, measured, joints=range(6)):
        now = time.time()
        dt, self.t = min(now - self.t, 0.2), now
        goal = list(self.cmd)
        for i in joints:
            goal[i] = target[i]
        self.cmd = rate_limit(self.cmd, goal, speed * dt)
        lag = max(abs(c - m) for c, m in zip(self.cmd, measured))
        if self.track_limit is not None and lag > self.track_limit:
            self.lag_since = self.lag_since or now
            if now - self.lag_since > 0.5:
                if self.robot:
                    self.robot.send(measured)  # stop pushing
                worst = max(range(6), key=lambda i: abs(self.cmd[i] - measured[i]))
                raise Stuck(f"{JOINT_ORDER[worst]} is {abs(self.cmd[worst] - measured[worst]):.0f} behind its command")
        else:
            self.lag_since = None
        return self.cmd


def go_home(robot, args):
    """Move to the sim start pose in stages so the wrist camera is never rotated while the arm is
    still folded: arm joints first, then wrist_roll (skipped with --hold-wrist-roll), gripper last."""
    target = clip_gripper(sim_units_to_real(sim_rad_to_units(SIM_HOME_RAD)), args.gripper_floor)
    current = robot.state()
    if args.hold_wrist_roll:
        target[4] = current[4]
    if target[1] < -85.0 and abs(target[4] - current[4]) > 20.0:
        # folded start pose (office2): rolling the wrist there sweeps the wrist camera along the arm and the table
        sys.exit(f"The start pose is folded with wrist_roll {target[4]:.0f} deg but the wrist is at {current[4]:.0f}: "
                 f"turn it by hand to ~{target[4]:.0f} (camera facing away from the arm), then run again.")
    print(f"go-home from {fmt(current)}\n          to   {fmt(target)}  at {args.home_speed:.0f} deg/s")
    sp = Setpoint(robot, args.track_limit)
    stages = [("arm", [0, 1, 2, 3]), ("wrist_roll", [4]), ("gripper", [5])]
    for name, joints in stages:
        if name == "wrist_roll" and args.hold_wrist_roll:
            continue
        deadline = None
        while True:
            measured = robot.state()
            cmd = sp.step(target, args.home_speed, measured, joints)
            if CAMERA_GUARD is not None and 1000 * CAMERA_GUARD(cmd) < args.min_camera_clearance:
                robot.send(measured)
                raise Stuck(f"homing would bring the wrist camera within {1000 * CAMERA_GUARD(cmd):.0f} mm of the arm")
            robot.send(cmd)
            if all(abs(cmd[i] - target[i]) < 1e-6 for i in joints):
                deadline = deadline or time.time() + 3.0
                if all(abs(measured[i] - target[i]) < 3.0 for i in joints) or time.time() > deadline:
                    break
            time.sleep(1.0 / args.fps)
        print(f"  {name} done: {fmt(robot.state())}")
    print("at sim start pose")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["preview", "control"], default="preview")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT),
                        help="run dir (outputs/act_office_cube) or pretrained_model dir")
    parser.add_argument("--profile", choices=SIM_CONFIG.PROFILES, default=None,
                        help="sim_mujoco profile the checkpoint was trained with (sets the start pose). "
                             "Default: from the checkpoint's dataset name")
    parser.add_argument("--object", choices=SIM_CONFIG.OBJECTS, default=None,
                        help="object the checkpoint was trained on (sets the task string). Default: from the "
                             "checkpoint's dataset name")
    parser.add_argument("--gripper-floor", type=float, default=ARM_LIMITS.get("gripper", (0.0, 100.0))[0],
                        help="never command the gripper below this (default 15: the home arm stalls below it; "
                             "the office arm closes at ~0: use 8 with the office cylinder policy)")
    parser.add_argument("--port", default=os.environ.get("FOLLOWER_PORT", "/dev/tty.usbmodem5C821064861"))
    parser.add_argument("--robot-id", default=os.environ.get("FOLLOWER_ID", "home_follower"))
    parser.add_argument("--wrist-cam", type=int, default=0, help="OpenCV index (same default as client.py)")
    parser.add_argument("--front-cam", type=int, default=1, help="OpenCV index of the third-person camera")
    parser.add_argument("--seconds", type=float, default=40.0, help="How long to run the policy (x --fps = steps)")
    parser.add_argument("--fps", type=float, default=30.0, help="Control rate; the policy was trained at 30")
    parser.add_argument("--max-speed", type=float, default=90.0, help="Max joint speed under the policy, deg/s")
    parser.add_argument("--sanity-pad-deg", type=float, default=15.0, help="Margin around training actions")
    parser.add_argument("--n-action-steps", type=int, default=None, help="Re-plan every N steps (default: 100)")
    parser.add_argument("--skip-home", action="store_true", help="control mode: don't move to the sim start pose first")
    parser.add_argument("--wait-for-object", action="store_true",
                        help="control mode: after homing, hold the start pose until Enter (put the object down then; "
                             "the office2 square is under the folded gripper)")
    parser.add_argument("--home-speed", type=float, default=45.0, help="Joint speed while homing, deg/s")
    parser.add_argument("--hold-wrist-roll", action="store_true",
                        help="Never rotate wrist_roll (homing and policy); protects the wrist camera cable")
    parser.add_argument("--min-camera-clearance", type=float, default=None,
                        help="stop if a target would bring the wrist camera mount closer than this (mm, in the sim "
                             "model; the real camera is bulkier) to the arm. Default: the profile's "
                             "min_camera_clearance (office2: 35; older profiles: 0 = no check)")
    parser.add_argument("--track-limit", type=float, default=20.0,
                        help="Abort if a joint lags its command by more than this (deg) for 0.5 s")
    parser.add_argument("--keep-torque", action="store_true", help="Leave torque on at exit so the arm holds")
    parser.add_argument("--no-robot", action="store_true", help="No hardware: dummy state and camera images")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    inferred = infer_sim_setup(checkpoint)
    if (args.profile is None or args.object is None) and inferred is None:
        sys.exit(f"Can't tell the sim profile/object of {checkpoint} from its dataset name; pass --profile and --object")
    profile = args.profile or inferred[0]
    obj = args.object or inferred[1]

    global SIM_HOME_RAD, CAMERA_GUARD
    sim_cfg = load_sim_config(profile, obj)
    SIM_HOME_RAD = sim_cfg.home_qpos
    if args.min_camera_clearance is None:
        args.min_camera_clearance = 1000 * sim_cfg.min_camera_clearance
    if args.min_camera_clearance > 0:
        CAMERA_GUARD = make_camera_guard(profile, obj)
        print(f"wrist camera guard: stop before any target within {args.min_camera_clearance:.0f} mm of the arm")
    task = sim_cfg.obj.task
    print(f"profile {profile}, object {obj} ('{task}'): start pose (sim rad) {[round(v, 3) for v in SIM_HOME_RAD]}")

    from leisaac_ext.act_policy import LocalACTPolicy

    print(f"loading {checkpoint} on {args.device} ...")
    policy = LocalACTPolicy(checkpoint, device=args.device, n_action_steps=args.n_action_steps)
    (state_q01, state_q99), action_box = training_ranges(checkpoint, args.sanity_pad_deg)
    print(f"cameras {policy.camera_keys}, re-plan every {policy.policy.config.n_action_steps} steps")
    print("allowed targets (arm frame): " + ", ".join(f"{n} {lo:.0f}..{hi:.0f}" for n, (lo, hi) in action_box.items()))

    robot = cameras = None
    if not args.no_robot:
        from cameras import CameraManager

        print(f"follower port={args.port} id={args.robot_id}")
        cameras = CameraManager(wrist_id=args.wrist_cam, third_id=args.front_cam)
        warm_up(cameras)
        robot = Follower(args.port, args.robot_id, args.keep_torque)
        robot.connect()

    try:
        if robot and args.mode == "control" and not args.skip_home:
            go_home(robot, args)
            if args.wait_for_object:  # the servos hold the start pose meanwhile
                input("at the start pose: put the object down, then press Enter to run the policy (Ctrl+C stops) ")

        current = robot.state() if robot else sim_units_to_real(sim_rad_to_units(SIM_HOME_RAD))
        state_u = real_to_sim_units(current)
        outside = [f"{n} {u:.0f} (train {lo:.0f}..{hi:.0f})" for n, u, lo, hi in zip(JOINT_ORDER, state_u, state_q01, state_q99)
                   if not lo - 5 <= u <= hi + 5]
        print(f"start state arm frame {fmt(current)}\n            sim units {fmt(state_u)}")
        if outside:
            print("WARNING start pose is outside the training data: " + "; ".join(outside))
            print("  In control mode the arm homes first; otherwise check SIGN/OFFSET_DEG if this looks mirrored.")

        if robot:
            frames = read_rgb(cameras)
            debug = LOCAL_DIR / "debug_frames"
            debug.mkdir(exist_ok=True)
            for key, img in frames.items():
                cv2.imwrite(str(debug / f"act_{key}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"saved {debug}/act_front.png and act_wrist.png: check front/wrist are not swapped")

        policy.reset()
        moving = args.mode == "control" and robot is not None
        # preview: no robot handle (never sends) and no lag check (the arm stays where it is)
        sp = Setpoint(robot if moving else None, args.track_limit if moving else None)
        if not moving:
            sp.cmd = list(current)
        hold_roll = current[4]
        n_steps = int(args.seconds * args.fps)
        print(f"{'CONTROL' if moving else 'PREVIEW (no motion)'}: {n_steps} steps, Ctrl+C to stop")
        for step in range(n_steps):
            t0 = time.time()
            if robot:
                current = robot.state()
                frames = read_rgb(cameras)
            else:
                frames = {k: np.random.randint(0, 255, (480, 640, 3), np.uint8) for k in policy.camera_keys}
            obs = {"observation.state": real_to_sim_units(current)}
            obs.update({f"observation.images.{k}": frames[k] for k in policy.camera_keys})
            target = sim_units_to_real(policy(obs, task=task, robot_type="so101_follower"))

            # wrist_roll is overridden below under --hold-wrist-roll, so its target doesn't matter
            bad = [f"{n}={v:.1f} not in {lo:.0f}..{hi:.0f}" for n, v, (lo, hi) in zip(JOINT_ORDER, target, action_box.values())
                   if not lo <= v <= hi and not (args.hold_wrist_roll and n == "wrist_roll")]
            if bad and moving:
                print(f"step {step}: STOP, target outside training range: " + "; ".join(bad))
                break
            if bad:  # preview never moves the arm: report and keep going to show the whole trajectory
                print(f"step {step}: target outside training range (control mode would stop): " + "; ".join(bad))
            goal = clip_gripper(target, args.gripper_floor)
            if args.hold_wrist_roll:
                goal[4] = hold_roll
            command = sp.step(goal, args.max_speed, current)
            gap = 1000 * CAMERA_GUARD(command) if CAMERA_GUARD is not None else np.inf
            if gap < args.min_camera_clearance:
                if moving:
                    robot.send(current)
                    print(f"step {step}: STOP, the wrist camera would come within {gap:.0f} mm of the arm "
                          f"(--min-camera-clearance {args.min_camera_clearance:.0f}); holding")
                    break
                print(f"step {step}: wrist camera {gap:.0f} mm from the arm (control mode would stop)")

            if step % 15 == 0:
                print(f"step {step:4d} state {fmt(current)}  target {fmt(target)}  send {fmt(command)}")
            if moving:
                robot.send(command)
            elif not robot:
                current = command  # simulate perfect tracking for --no-robot
            time.sleep(max(0.0, 1.0 / args.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped by user")
    except Stuck as e:
        print(f"\nSTOPPED: {e}. The arm was told to hold its current position.")
    finally:
        if robot:
            robot.disconnect()
        if cameras:
            cameras.close()


if __name__ == "__main__":
    main()
