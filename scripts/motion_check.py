#!/usr/bin/env python3
"""Isolates the motor path from the model path.

client.py --mode control does: get_state -> ask server -> safety -> clamp ->
send_action. This script does ONLY the get_state/send_action half, through the
exact same local/robot.py wrapper and the same use_degrees=True config, so a
failure here is the arm/driver and a success here puts the problem upstream in
the server response, safety gate or clamp.

Moves one joint by a small amount and reports whether the arm actually got
there. Run:  python scripts/motion_check.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "local"))

from dotenv import load_dotenv

from robot import SO101Robot

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

JOINT = "shoulder_pan"
DELTA_DEG = 8.0
SETTLE_S = 1.5
# Anything under this and the joint effectively did not move.
MOVED_THRESHOLD_DEG = 1.0


def main():
    port = os.environ["FOLLOWER_PORT"]
    robot_id = os.environ.get("FOLLOWER_ID", "home_follower")

    robot = SO101Robot(port, robot_id)
    print(f"config: {getattr(robot.robot, 'config', '<not exposed>')}")
    print(f"Connecting on {port} ...")
    robot.connect()

    try:
        start = robot.get_state()
        print(f"\nstate before: { {k: round(v, 1) for k, v in start.items()} }")

        target = dict(start)
        target[JOINT] = start[JOINT] + DELTA_DEG
        print(f"sending:      {JOINT} {start[JOINT]:.1f} -> {target[JOINT]:.1f} ({DELTA_DEG:+.1f})")
        robot.send_action(target)

        time.sleep(SETTLE_S)
        end = robot.get_state()
        print(f"state after:  { {k: round(v, 1) for k, v in end.items()} }")

        moved = end[JOINT] - start[JOINT]
        print(f"\n{JOINT} moved {moved:+.1f} deg (asked for {DELTA_DEG:+.1f})")

        if abs(moved) < MOVED_THRESHOLD_DEG:
            print(
                "RESULT: the arm did NOT move.\n"
                "  The motor path itself is the problem, not the model or safety layer.\n"
                "  Check: torque enabled, correct FOLLOWER_PORT, calibration id matches,\n"
                "  and whether scripts/move_follower.py (no use_degrees) moves it."
            )
        else:
            print(
                "RESULT: the arm moved.\n"
                "  The motor path works, so client.py --mode control is failing earlier:\n"
                "  the safety gate, an empty/near-identity action chunk, or a request error."
            )

        print("\nReturning to the starting pose ...")
        robot.send_action(start)
        time.sleep(SETTLE_S)
    finally:
        try:
            robot.disconnect()
        except RuntimeError as e:
            # Releasing torque can trip an overload alarm if a joint is bearing
            # the arm's weight - support it by hand and retry.
            print(f"Warning: disconnect reported a motor error: {e}")


if __name__ == "__main__":
    main()
