#!/usr/bin/env python3
"""Simple movement sequence for the SO-101 follower arm.

Run with:
    conda run -n lerobot python move_follower.py
"""

import time

from lerobot.robots.so_follower import SOFollower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

PORT = "/dev/tty.usbmodem5C821064861"
ROBOT_ID = "home_follower"  # matches ~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_follower.json

STEP_DELAY = 2.0  # seconds to wait between moves


def build_sequence(current: dict[str, float]) -> list[dict[str, float]]:
    """Return a list of target joint positions relative to the arm's current pose.

    Body joints are in degrees; gripper is a percentage (0 = closed, 100 = open).
    All moves are small offsets so the arm won't slam into limits.
    """
    pan = current["shoulder_pan"]
    lift = current["shoulder_lift"]
    gripper = current["gripper"]

    return [
        # 1. open gripper
        {**current, "gripper": min(gripper + 30.0, 100.0)},
        # 2. pan left
        {**current, "gripper": min(gripper + 30.0, 100.0), "shoulder_pan": pan - 20.0},
        # 3. pan right
        {**current, "gripper": min(gripper + 30.0, 100.0), "shoulder_pan": pan + 20.0},
        # 4. return to center with gripper open
        {**current, "gripper": min(gripper + 30.0, 100.0)},
        # 5. close gripper and return to start
        current,
    ]


def main() -> None:
    config = SOFollowerRobotConfig(port=PORT, id=ROBOT_ID)
    robot = SOFollower(config)

    print(f"Connecting to follower arm on {PORT} ...")
    robot.connect(calibrate=False)

    obs = robot.get_observation()
    current = {k.removesuffix(".pos"): v for k, v in obs.items() if k.endswith(".pos")}
    print(f"Current positions: {current}")

    sequence = build_sequence(current)

    for i, target in enumerate(sequence, start=1):
        action = {f"{k}.pos": v for k, v in target.items()}
        print(f"\nStep {i}/{len(sequence)}: {target}")
        robot.send_action(action)
        time.sleep(STEP_DELAY)

    print("\nSequence complete. Disconnecting ...")
    try:
        robot.disconnect()
    except RuntimeError as e:
        # Disabling torque can trip a servo overload alarm if that joint (often
        # shoulder_lift, id 2) is still bearing the arm's weight unsupported.
        # Support the arm by hand before/while torque is released, then retry.
        print(f"Warning: disconnect reported a motor error and may not have fully disabled torque: {e}")


if __name__ == "__main__":
    main()
