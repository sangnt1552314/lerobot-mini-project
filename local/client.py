"""Local SO101 client: captures cameras, sends observation to server, prints fake action.

Does NOT move the robot. This is just a communication prototype.
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

from cameras import CameraManager
from robot import SO101Robot
from action_format import JOINT_ORDER, validate_action

load_dotenv()

# Set via `export SERVER_URL=...` before running, e.g.:
# export SERVER_URL="https://xxxx.ngrok-free.dev"
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")

INSTRUCTION = "pick up the red cube"

# Real SO101 follower connection settings (same values used in
# scripts/move_follower.py / scripts/teleoperating.sh).
FOLLOWER_PORT = os.environ["FOLLOWER_PORT"]
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "home_follower")

# preview = never sent to motors (this stage always runs in preview mode)
MODE = "preview"

REQUEST_TIMEOUT_S = 10


def main():
    cameras = CameraManager(0, 1)  # (wrist_id, third_person_id) - change if needed
    robot = SO101Robot(FOLLOWER_PORT, FOLLOWER_ID)

    print(f"Connecting to follower arm on {FOLLOWER_PORT} ...")
    robot.connect()

    observation_id = 0

    try:
        while True:
            wrist_frame, third_frame = cameras.capture()
            wrist_jpeg = cameras.encode_jpeg(wrist_frame)
            third_jpeg = cameras.encode_jpeg(third_frame)
            robot_state = robot.get_state()

            files = {
                "wrist": ("wrist.jpg", wrist_jpeg, "image/jpeg"),
                "third": ("third.jpg", third_jpeg, "image/jpeg"),
            }
            data = {
                "observation_id": observation_id,
                "instruction": INSTRUCTION,
                "joint_positions": json.dumps(robot_state),
            }

            start = time.perf_counter()
            try:
                response = requests.post(
                    f"{SERVER_URL}/infer",
                    files=files,
                    data=data,
                    timeout=REQUEST_TIMEOUT_S,
                )
                latency_ms = (time.perf_counter() - start) * 1000
                response.raise_for_status()
                result = response.json()
                actions = result["actions"]
                print(
                    f"observation={observation_id} robot_state={robot_state} "
                    f"latency={latency_ms:.1f} ms actions={actions} mode={MODE}"
                )

                # Stage 3: inspect the action format only - never execute it yet.
                valid, reason = validate_action(actions[0]) if actions else (False, "empty actions")
                print(f"mode={MODE}")
                print(f"action shape: ({len(actions)}, {len(actions[0]) if actions else 0})")
                print(f"action type: {type(actions).__name__}")
                print(f"joint order: {JOINT_ORDER}")
                print("gripper representation: SO101 arm-frame degrees (converted server-side, see server/policy.py)")
                print(f"valid: {valid} ({reason})")
            except requests.RequestException as e:
                print(f"observation={observation_id} request failed: {e}")

            observation_id += 1
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C received)...")
    finally:
        cameras.close()
        robot.disconnect()


if __name__ == "__main__":
    main()

