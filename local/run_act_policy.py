"""Run a MuJoCo-trained ACT checkpoint on the real SO-101 follower. Start it through the arm's script:

    bash local/run_home.sh act_office2_cube             # control: go to the sim start pose, then run the policy
    bash local/run_office.sh act_office2_cube preview   # cameras + model only, the arm never moves

The sim profile and object come from the checkpoint's dataset name (so101_<profile>_<object>; the older
so101_liftcube_mujoco[_real...] datasets are cube). They set the start pose and the task string.

Units. The checkpoint was trained in sim_mujoco/, whose state/action units are a linear map of the
MuJoCo Menagerie SO-101 joint ranges (arm -100..100, gripper 0..100), not this arm's LeRobot
calibration. That model is derived from TheRobotStudio's so101_new_calib, whose joint zero is meant to
match LeRobot's calibrated mid-range, so the arm is read in degrees (as local/robot.py does) and
converted through the MJCF ranges (SIGN / OFFSET_DEG below).

Safety: preview never sends commands. In control mode the run stops if a target leaves the checkpoint's training
range (+PAD_DEG), the gripper never goes below --gripper-floor, joints move at most MAX_SPEED (HOME_SPEED while
homing), and if a joint lags its command by TRACK_LIMIT for 0.5 s (blocked) the arm holds and the run stops.
Ctrl+C stops. Keep a hand near the arm.
"""

import argparse
import json
import math
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

FPS = 30.0  # control rate; the policy was trained at 30
SECONDS = 40.0  # how long the policy runs
MAX_SPEED = 90.0  # deg/s under the policy
HOME_SPEED = 45.0  # deg/s while homing
TRACK_LIMIT = 20.0  # deg a joint may lag its command (for 0.5 s) before the run stops
PAD_DEG = 15.0  # margin around the training actions
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

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

# Per arm (LeRobot calibration name): OpenCV camera indices and the arm's OFFSET_DEG. macOS can renumber cameras
# when they are replugged: preview saves the first frames to local/debug_frames/ to check which is which.
ARMS = {
    "home_follower": {"wrist_cam": 1, "front_cam": 0, "offset_deg": {}},
    "my_follower": {"wrist_cam": 0, "front_cam": 1, "offset_deg": {}},
}


def use_arm(robot_id):
    """Set OFFSET_DEG for this arm; returns its (wrist, front) camera indices (unknown arms: 0, 1, no offsets)."""
    arm = ARMS.get(robot_id, {})
    OFFSET_DEG.update(arm.get("offset_deg", {}))
    return arm.get("wrist_cam", 0), arm.get("front_cam", 1)


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


def training_action_box(checkpoint: Path):
    """Padded action min..max (arm frame) per joint, from the checkpoint stats."""
    post = load_file(str(checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
    lo, hi = sim_units_to_real(post["action.min"].numpy()), sim_units_to_real(post["action.max"].numpy())
    return {n: (min(a, b) - PAD_DEG, max(a, b) + PAD_DEG) for n, a, b in zip(JOINT_ORDER, lo, hi)}


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


def go_home(robot, home):
    """Move to the sim start pose: arm joints first, then wrist_roll, gripper last."""
    print(f"go-home from {fmt(robot.state())}\n          to   {fmt(home)}  at {HOME_SPEED:.0f} deg/s")
    sp = Setpoint(robot, TRACK_LIMIT)
    for name, joints in [("arm", [0, 1, 2, 3]), ("wrist_roll", [4]), ("gripper", [5])]:
        deadline = None
        while True:
            measured = robot.state()
            cmd = sp.step(home, HOME_SPEED, measured, joints)
            robot.send(cmd)
            if all(abs(cmd[i] - home[i]) < 1e-6 for i in joints):
                deadline = deadline or time.time() + 3.0
                if all(abs(measured[i] - home[i]) < 3.0 for i in joints) or time.time() > deadline:
                    break
            time.sleep(1.0 / FPS)
        print(f"  {name} done: {fmt(robot.state())}")
    print("at sim start pose")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["preview", "control"], default="preview")
    parser.add_argument("--checkpoint", required=True, help="run dir (outputs/act_office_cube) or pretrained_model dir")
    parser.add_argument("--gripper-floor", type=float, default=ARM_LIMITS.get("gripper", (0.0, 100.0))[0],
                        help="never command the gripper below this (home arm 15, office arm 8)")
    parser.add_argument("--port", required=True)
    parser.add_argument("--robot-id", required=True, help="LeRobot calibration name (home_follower, my_follower)")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    wrist_cam, front_cam = use_arm(args.robot_id)
    setup = infer_sim_setup(checkpoint)
    if setup is None:
        sys.exit(f"Can't tell the sim profile/object of {checkpoint} from its dataset name")
    profile, obj = setup
    sim_cfg = SIM_CONFIG.make_config(profile, obj)
    home = clip_gripper(sim_units_to_real(sim_rad_to_units(sim_cfg.home_qpos)), args.gripper_floor)
    task = sim_cfg.obj.task
    print(f"profile {profile}, object {obj} ('{task}'), start pose (arm frame) {fmt(home)}")

    from cameras import CameraManager
    from leisaac_ext.act_policy import LocalACTPolicy

    print(f"loading {checkpoint} on {DEVICE} ...")
    policy = LocalACTPolicy(checkpoint, device=DEVICE)
    action_box = training_action_box(checkpoint)
    print("allowed targets (arm frame): " + ", ".join(f"{n} {lo:.0f}..{hi:.0f}" for n, (lo, hi) in action_box.items()))

    offsets = {n: v for n, v in OFFSET_DEG.items() if v}
    print(f"follower port={args.port} id={args.robot_id}, cameras wrist={wrist_cam} front={front_cam}, "
          f"joint offsets (real = sim + offset) {offsets or 'none'}")
    cameras = CameraManager(wrist_id=wrist_cam, third_id=front_cam)
    warm_up(cameras)
    debug_dir = LOCAL_DIR / "debug_frames"
    debug_dir.mkdir(exist_ok=True)
    for key, frame in read_rgb(cameras).items():  # what the policy sees: check the cameras aren't swapped
        cv2.imwrite(str(debug_dir / f"act_{key}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"first frames: {debug_dir}/act_front.png (third-person) and act_wrist.png")
    robot = Follower(args.port, args.robot_id, keep_torque=False)
    robot.connect()
    moving = args.mode == "control"
    try:
        if moving:
            input(f"CONTROL: the arm will move to the start pose {fmt(home)} and then run the policy.\n"
                  "Clear the workspace, keep a hand near the arm. Enter = start, Ctrl+C = abort ")
            go_home(robot, home)
        policy.reset()
        # preview: no robot handle (never sends) and no lag check (the arm stays where it is)
        sp = Setpoint(robot if moving else None, TRACK_LIMIT if moving else None)
        if not moving:
            sp.cmd = robot.state()
        n_steps = int(SECONDS * FPS)
        print(f"{'CONTROL' if moving else 'PREVIEW (no motion)'}: {n_steps} steps, Ctrl+C to stop")
        for step in range(n_steps):
            t0 = time.time()
            current = robot.state()
            frames = read_rgb(cameras)
            obs = {"observation.state": real_to_sim_units(current)}
            obs.update({f"observation.images.{k}": frames[k] for k in policy.camera_keys})
            target = sim_units_to_real(policy(obs, task=task, robot_type="so101_follower"))

            bad = [f"{n}={v:.1f} not in {lo:.0f}..{hi:.0f}" for n, v, (lo, hi) in zip(JOINT_ORDER, target, action_box.values())
                   if not lo <= v <= hi]
            if bad and moving:
                print(f"step {step}: STOP, target outside training range: " + "; ".join(bad))
                break
            if bad:  # preview never moves the arm: report and keep going to show the whole trajectory
                print(f"step {step}: target outside training range (control mode would stop): " + "; ".join(bad))
            command = sp.step(clip_gripper(target, args.gripper_floor), MAX_SPEED, current)
            if step % 15 == 0:
                print(f"step {step:4d} state {fmt(current)}  target {fmt(target)}  send {fmt(command)}")
            if moving:
                robot.send(command)
            time.sleep(max(0.0, 1.0 / FPS - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped by user")
    except Stuck as e:
        print(f"\nSTOPPED: {e}. The arm was told to hold its current position.")
    finally:
        robot.disconnect()
        cameras.close()


if __name__ == "__main__":
    main()
