"""Run a Hugging Face ACT checkpoint on the real SO-101, locally (no server). The model is downloaded on the
first run (cached in ~/.cache/huggingface) and runs on this laptop (MPS ~70 ms per chunk, CPU ~180 ms).

    cd local
    python run_act.py                                       # preview: arm + cameras + model, NEVER moves
    python run_act.py --mode control --start-pose=-9,-74,88,54,0,95
    python run_act.py --checkpoint ../outputs/act_my_run    # your fine-tuned run (local dir or HF repo id)
    python run_act.py --no-robot                            # model only, dummy inputs

Default checkpoint: omiya239532/so101_act_cotrain (pick a 4 cm cube, place it on a white sheet; cameras
"world" = top-down third-person camera, "wrist"). It was trained on a rig with a top-down camera, see
docs/act_reference_rig.jpg; a front/side webcam is a different input for it.

Units. That checkpoint works in radians, this arm in LeRobot degrees (gripper 0..100):
  arm joints   rad = radians(deg), checked against the checkpoint's own statistics (its real half was recorded
               in LeRobot degrees). wrist_roll first subtracts --wrist-roll-offset, your arm's wrist_roll reading
               with the gripper in the grasp orientation of docs/act_reference_rig.jpg (the author's reads ~0; the
               home arm read -1.0 on 2026-10-14, so 0 is fine there).
  gripper      NOT radians(%): a line fitted from the checkpoint's stats, 0 % -> -0.0314 rad, 100 % -> 1.2006 rad
               (the author's arm: open 96.6 %, on the cube 15.6 %). For another arm: --gripper-open <reading with
               the jaw fully open> --gripper-on-cube <reading closed on the cube> (mapped to 1.159 / 0.158 rad).
               The home arm read 97.1 / 15.6 on 2026-10-14, i.e. the default fits it.
A checkpoint trained on your own lerobot-record data (--robot.use_degrees=true) is already in this arm's units;
that is detected from its statistics and nothing is converted.

Safety in control mode: typed "yes" first; refuses to start with a joint far outside the checkpoint's training
states; every target must lie inside the checkpoint's training action range (+ --pad) or the run stops; gripper
floor (local/safety.py ARM_LIMITS, --gripper-floor 8 for the office arm); joints move at most --max-speed deg/s;
if a joint lags its command for 0.5 s (blocked) the arm holds and the run stops; Ctrl+C holds the pose.
The action queue is cleared at the start (policy.reset()); each chunk comes from one observation.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.numpy import load_file

LOCAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_DIR))

from action_format import JOINT_ORDER  # noqa: E402
from run_act_policy import Follower, Setpoint, Stuck, clip_gripper, crop_to_aspect, fmt  # noqa: E402
from safety import ARM_LIMITS  # noqa: E402

from leisaac_ext.act_policy import LocalACTPolicy  # noqa: E402  (sim/ is put on sys.path by run_act_policy)

DEFAULT_CHECKPOINT = "omiya239532/so101_act_cotrain"
AUTHOR_GRIPPER = ((0.0, -0.0314), (100.0, 1.2006))   # (gripper %, jaw rad), fitted from the checkpoint's stats
ON_CUBE_RAD, OPEN_RAD = 0.158, 1.159                  # the author's jaw on the 4 cm cube / fully open
FAR_OUTSIDE_DEG = 15.0                                # control refuses to start this far outside the training states


def resolve_checkpoint(name):
    p = Path(name).expanduser()
    if p.exists():
        for cand in (p, p / "pretrained_model", p / "checkpoints" / "last" / "pretrained_model"):
            if (cand / "config.json").is_file():
                return cand.resolve()
        sys.exit(f"No checkpoint in {p} (expected config.json there or in checkpoints/last/pretrained_model)")
    from huggingface_hub import snapshot_download

    print(f"loading {name} from Hugging Face (downloaded on the first run, cached after) ...")
    return Path(snapshot_download(name, allow_patterns=["config.json", "model.safetensors", "policy_*"]))


class Units:
    """This arm's frame (LeRobot degrees, gripper 0..100) <-> the checkpoint's frame."""

    def __init__(self, checkpoint, wrist_roll_offset, gripper_open, gripper_on_cube):
        self.stats = {}
        for f in sorted(checkpoint.glob("policy_*processor*.safetensors")):
            self.stats.update(load_file(str(f)))
        peak = max(np.abs(self.stats["action.min"][:5]).max(), np.abs(self.stats["action.max"][:5]).max())
        self.rad = peak < 3.5  # radian SO-101 data stays within the joint limits; degree data spans tens of degrees
        self.wrist_offset = wrist_roll_offset if self.rad else 0.0
        if gripper_open is not None and gripper_on_cube is not None:
            self.grip = ((gripper_on_cube, ON_CUBE_RAD), (gripper_open, OPEN_RAD))
        else:
            self.grip = AUTHOR_GRIPPER

    def to_model(self, arm):
        if not self.rad:
            return np.asarray(arm, dtype=np.float32)
        (p0, m0), (p1, m1) = self.grip
        joints = [math.radians(v) for v in arm[:5]]
        joints[4] = math.radians(arm[4] - self.wrist_offset)
        return np.asarray(joints + [m0 + (arm[5] - p0) * (m1 - m0) / (p1 - p0)], dtype=np.float32)

    def to_arm(self, model):
        if not self.rad:
            return [float(v) for v in model]
        (p0, m0), (p1, m1) = self.grip
        arm = [math.degrees(float(v)) for v in model[:5]] + [p0 + (float(model[5]) - m0) * (p1 - p0) / (m1 - m0)]
        arm[4] += self.wrist_offset
        return arm

    def box(self, lo_key, hi_key, pad=0.0):
        lo, hi = self.to_arm(self.stats[lo_key]), self.to_arm(self.stats[hi_key])
        return [(min(a, b) - pad, max(a, b) + pad) for a, b in zip(lo, hi)]


def outside(values, box, skip=()):
    return [f"{n}={v:.1f} not in {lo:.0f}..{hi:.0f}" for i, (n, v, (lo, hi)) in enumerate(zip(JOINT_ORDER, values, box))
            if i not in skip and not lo <= v <= hi]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["preview", "control"], default="preview")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="HF repo id or local run / pretrained_model dir")
    ap.add_argument("--wrist-roll-offset", type=float, default=0.0,
                    help="your wrist_roll reading in the grasp orientation (radian checkpoints only)")
    ap.add_argument("--gripper-open", type=float, default=None, help="gripper reading, jaw fully open")
    ap.add_argument("--gripper-on-cube", type=float, default=None, help="gripper reading, jaw closed on the cube")
    ap.add_argument("--gripper-floor", type=float, default=ARM_LIMITS.get("gripper", (0.0, 100.0))[0],
                    help="never command the gripper below this (home arm 15, office arm 8)")
    ap.add_argument("--start-pose", default=None, help="control: move here first, 6 comma-separated arm-frame values")
    ap.add_argument("--port", default=os.environ.get("FOLLOWER_PORT", "/dev/tty.usbmodem5C821064861"))
    ap.add_argument("--robot-id", default=os.environ.get("FOLLOWER_ID", "home_follower"))
    ap.add_argument("--wrist-cam", type=int, default=0)
    ap.add_argument("--front-cam", type=int, default=1, help="third-person camera (the checkpoint's 'world')")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=30.0, help="control rate; so101_act_cotrain was trained at 30")
    ap.add_argument("--max-speed", type=float, default=45.0, help="max joint speed, deg/s")
    ap.add_argument("--n-action-steps", type=int, default=30, help="re-plan every N steps (chunk is 100)")
    ap.add_argument("--pad", type=float, default=10.0, help="margin around the training action range, deg")
    ap.add_argument("--track-limit", type=float, default=20.0, help="stop if a joint lags its command this much")
    ap.add_argument("--hold-wrist-roll", action="store_true", help="never rotate wrist_roll (wrist camera cable)")
    ap.add_argument("--keep-torque", action="store_true", help="leave torque on at exit so the arm holds")
    ap.add_argument("--no-robot", action="store_true", help="no hardware: dummy state and images")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    policy = LocalACTPolicy(checkpoint, device=args.device, n_action_steps=args.n_action_steps)
    units = Units(checkpoint, args.wrist_roll_offset, args.gripper_open, args.gripper_on_cube)
    action_box = units.box("action.min", "action.max", args.pad)
    state_box = units.box("observation.state.min", "observation.state.max")
    image_shapes = {k: policy.policy.config.image_features[f"observation.images.{k}"].shape for k in policy.camera_keys}
    print(f"checkpoint {checkpoint}\n  units {'radians (converted)' if units.rad else 'arm frame (no conversion)'}, "
          f"cameras {policy.camera_keys}, chunk {policy.policy.config.chunk_size}, re-plan every {args.n_action_steps}")
    if units.rad:
        print(f"  wrist_roll offset {units.wrist_offset:g}, gripper map {units.grip} (% -> rad)")
    print("  allowed targets: " + ", ".join(f"{n} {lo:.0f}..{hi:.0f}" for n, (lo, hi) in zip(JOINT_ORDER, action_box)))

    start = None
    if args.start_pose:
        start = [float(v) for v in args.start_pose.split(",")]
        if len(start) != 6 or outside(start, action_box):
            sys.exit(f"--start-pose must be 6 values inside the allowed targets: {outside(start, action_box)}")
        start = clip_gripper(start, args.gripper_floor)

    robot = cameras = None
    if not args.no_robot:
        from cameras import CameraManager

        cameras = CameraManager(wrist_id=args.wrist_cam, third_id=args.front_cam)
        robot = Follower(args.port, args.robot_id, args.keep_torque)
        print(f"connecting {args.robot_id} on {args.port} ...")
        robot.connect()

    def frames():
        if robot is None:
            return {k: np.random.randint(0, 255, (s[1], s[2], 3), np.uint8) for k, s in image_shapes.items()}
        wrist_bgr, front_bgr = cameras.capture()
        out = {}
        for key, (_, h, w) in image_shapes.items():
            frame = crop_to_aspect(wrist_bgr if "wrist" in key else front_bgr, w / h)
            out[key] = cv2.cvtColor(cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        return out

    moving = args.mode == "control" and robot is not None
    try:
        current = robot.state() if robot else (start or [sum(b) / 2 for b in state_box])
        far = outside(current, [(lo - FAR_OUTSIDE_DEG, hi + FAR_OUTSIDE_DEG) for lo, hi in state_box],
                      skip=(4,) if args.hold_wrist_roll else ())
        print(f"arm now {fmt(current)}  (model frame {fmt(units.to_model(current))})")
        if far:
            print("WARNING arm is far outside the checkpoint's training states: " + "; ".join(far))
        if moving:
            if far and start is None:
                sys.exit("refusing control: use --start-pose (and check --wrist-roll-offset)")
            if input("The arm will move. Type 'yes' to continue: ").strip() != "yes":
                sys.exit("not confirmed")
            if start is not None:
                if args.hold_wrist_roll:
                    start[4] = current[4]
                print(f"moving to start pose {fmt(start)}")
                sp = Setpoint(robot, args.track_limit)
                deadline = time.time() + 15.0
                while time.time() < deadline:
                    measured = robot.state()
                    robot.send(sp.step(start, 30.0, measured))
                    if max(abs(m - s) for m, s in zip(measured, start)) < 3.0:
                        break
                    time.sleep(1.0 / args.fps)
                current = robot.state()

        policy.reset()
        sp = Setpoint(robot if moving else None, args.track_limit if moving else None)
        sp.cmd = list(current)
        hold_roll = current[4]
        print(f"{'CONTROL' if moving else 'PREVIEW (the arm does not move)'}: {args.seconds:g} s, Ctrl+C to stop")
        for step in range(int(args.seconds * args.fps)):
            t0 = time.time()
            if robot:
                current = robot.state()
            obs = {"observation.state": units.to_model(current)}
            obs.update({f"observation.images.{k}": img for k, img in frames().items()})
            target = units.to_arm(policy(obs, task="pick up the red cube", robot_type="so101_follower"))
            if args.hold_wrist_roll:
                target[4] = hold_roll
            bad = outside(target, action_box)
            if bad and moving:
                print(f"step {step}: STOP, target outside the training range: " + "; ".join(bad))
                break
            if bad:
                print(f"step {step}: target outside the training range (control would stop): " + "; ".join(bad))
            command = sp.step(clip_gripper(target, args.gripper_floor), args.max_speed, current)
            if step % 15 == 0:
                print(f"step {step:4d} arm {fmt(current)}  target {fmt(target)}  send {fmt(command)}")
            if moving:
                robot.send(command)
            elif robot is None:
                current = command  # --no-robot: pretend the arm follows
            time.sleep(max(0.0, 1.0 / args.fps - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped by user")
    except Stuck as e:
        print(f"\nSTOPPED: {e}. The arm was told to hold its position.")
    finally:
        if moving:
            robot.send(robot.state())  # hold where it is
        if robot:
            robot.disconnect()
        if cameras:
            cameras.close()


if __name__ == "__main__":
    main()
