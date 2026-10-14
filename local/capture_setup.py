"""Capture what a new sim_mujoco profile needs from a physical setup: the start pose and camera photos.

    cd local
    python capture_setup.py --name office --robot-id office_follower --port /dev/tty.usbmodemXXXX

Torque is OFF the whole time: pose the arm by hand into the start pose the policy should begin from
(gripper open, wrist camera looking at the table), put the red cube in the middle of where it should
spawn, and press Enter. Each capture saves to local/setup_captures/<name>/<n>_*:
  front.png / wrist.png       640x480 RGB as the policy sees them (centre-cropped to 4:3)
  front_raw.png / wrist_raw.png   full camera resolution, for cutting table/background textures
  pose.json                   arm frame (LeRobot degrees, gripper 0..100) and sim radians (home_qpos)
Capture a few: the start pose with the cube, and an empty table for textures. Ctrl+C to finish.
The arm frame -> sim mapping is run_act_policy.py's (SIGN / OFFSET_DEG / MJCF_RANGE).
"""

import argparse
import json
import math
import os
import select
import sys
import time

import cv2

from run_act_policy import JOINT_ORDER, LOCAL_DIR, MJCF_RANGE, Follower, crop_to_aspect, fmt, real_to_sim_units

GRIPPER_FLOOR = 15.0  # the follower stalls below this (local/safety.py ARM_LIMITS)


def sim_units_to_rad(u):
    rad = []
    for name, value in zip(JOINT_ORDER, u):
        lo, hi = MJCF_RANGE[name]
        frac = value / 100.0 if name == "gripper" else (value + 100.0) / 200.0
        rad.append(lo + frac * (hi - lo))
    return rad


def enter_pressed():
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
    return bool(ready)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="Setup name, e.g. office (becomes the output folder)")
    ap.add_argument("--port", default=os.environ.get("FOLLOWER_PORT", "/dev/tty.usbmodem5C821064861"))
    ap.add_argument("--robot-id", default=os.environ.get("FOLLOWER_ID", "home_follower"))
    ap.add_argument("--wrist-cam", type=int, default=0)
    ap.add_argument("--front-cam", type=int, default=1)
    args = ap.parse_args()

    from cameras import CameraManager

    out = LOCAL_DIR / "setup_captures" / args.name
    out.mkdir(parents=True, exist_ok=True)
    n = len(list(out.glob("*_pose.json")))

    cameras = CameraManager(wrist_id=args.wrist_cam, third_id=args.front_cam)
    robot = Follower(args.port, args.robot_id, keep_torque=False)
    robot.connect()
    robot.robot.bus.disable_torque()
    print(f"follower port={args.port} id={args.robot_id}, torque OFF: pose the arm by hand")
    print(f"saving to {out}. Enter = capture, Ctrl+C = quit\n")
    try:
        last_print = 0.0
        while True:
            wrist_bgr, front_bgr = cameras.capture()  # keep reading so the frames are fresh
            if time.time() - last_print > 0.5:
                print(f"\rarm frame {fmt(robot.state())}", end="", flush=True)
                last_print = time.time()
            if not enter_pressed():
                continue
            state = robot.state()
            rad = sim_units_to_rad(real_to_sim_units(state))
            prefix = out / f"{n}"
            for key, frame in (("front", front_bgr), ("wrist", wrist_bgr)):
                cv2.imwrite(f"{prefix}_{key}_raw.png", frame)
                cv2.imwrite(f"{prefix}_{key}.png", cv2.resize(crop_to_aspect(frame), (640, 480), interpolation=cv2.INTER_AREA))
            pose = {
                "robot_id": args.robot_id,
                "joints": list(JOINT_ORDER),
                "arm_frame": [round(v, 2) for v in state],
                "home_qpos_rad": [round(v, 4) for v in rad],
                "front_raw_size": [front_bgr.shape[1], front_bgr.shape[0]],
                "wrist_raw_size": [wrist_bgr.shape[1], wrist_bgr.shape[0]],
            }
            (out / f"{n}_pose.json").write_text(json.dumps(pose, indent=2))
            print(f"\ncapture {n}: arm frame {fmt(state)}")
            print(f"           sim rad   [{', '.join(f'{v:.4f}' for v in rad)}]  (deg {[round(math.degrees(v), 1) for v in rad]})")
            if state[5] < GRIPPER_FLOOR:
                print(f"           NOTE gripper {state[5]:.0f} < {GRIPPER_FLOOR:.0f}: open it for the start pose")
            print(f"           saved {prefix}_{{front,wrist}}{{,_raw}}.png, {prefix.name}_pose.json\n")
            n += 1
    except KeyboardInterrupt:
        print(f"\ndone, {n} capture(s) in {out}")
    finally:
        robot.disconnect()
        cameras.close()


if __name__ == "__main__":
    main()
