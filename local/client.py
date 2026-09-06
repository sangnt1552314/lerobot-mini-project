"""Local SO101 client: captures cameras, sends observation to server, prints the
predicted action.

--mode preview (default): continuous loop, never touches the motors.
--mode control: repeatedly executes the first EXECUTE_STEPS validated actions
of each returned chunk, then requests a fresh chunk. Runs until Ctrl+C or any
safety rejection/error.
"""

import argparse
import json
import os
import time

import requests
from dotenv import load_dotenv

from cameras import CameraManager
from robot import SO101Robot
from action_format import JOINT_ORDER, validate_action
from safety import is_safe_action

load_dotenv()

# Set via `export SERVER_URL=...` before running, e.g.:
# export SERVER_URL="https://xxxx.ngrok-free.dev"
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")

INSTRUCTION = "pick up the block"

# Real SO101 follower connection settings (same values used in
# scripts/move_follower.py / scripts/teleoperating.sh).
FOLLOWER_PORT = os.environ["FOLLOWER_PORT"]
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "home_follower")

REQUEST_TIMEOUT_S = 10

# Stage 7: execute only the first N actions of each returned chunk (not the
# whole ~30-step chunk) before requesting a fresh observation/chunk.
EXECUTE_STEPS = 4
# Local pacing between executed actions within a chunk (seconds).
STEP_DELAY_S = 0.1

# Debug: set SAVE_FRAMES=1 to overwrite local/debug_frames/wrist.jpg and
# third.jpg each request - lets you see exactly what the model receives.
SAVE_FRAMES = os.environ.get("SAVE_FRAMES") == "1"
DEBUG_FRAMES_DIR = os.path.join(os.path.dirname(__file__), "debug_frames")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "control"], default="preview")
    return parser.parse_args()


def capture_and_request(cameras, robot, observation_id):
    """Captures one observation and calls /infer. Returns (robot_state, result, latency_ms)."""
    wrist_frame, third_frame = cameras.capture()
    wrist_jpeg = cameras.encode_jpeg(wrist_frame)
    third_jpeg = cameras.encode_jpeg(third_frame)
    robot_state = robot.get_state()

    if SAVE_FRAMES:
        os.makedirs(DEBUG_FRAMES_DIR, exist_ok=True)
        with open(os.path.join(DEBUG_FRAMES_DIR, "wrist.jpg"), "wb") as f:
            f.write(wrist_jpeg)
        with open(os.path.join(DEBUG_FRAMES_DIR, "third.jpg"), "wb") as f:
            f.write(third_jpeg)

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
    response = requests.post(
        f"{SERVER_URL}/infer", files=files, data=data, timeout=REQUEST_TIMEOUT_S
    )
    latency_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    return robot_state, response.json(), latency_ms


def print_diagnostics(observation_id, robot_state, result, latency_ms, mode):
    actions = result["actions"]
    print(
        f"observation={observation_id} robot_state={robot_state} "
        f"latency={latency_ms:.1f} ms actions={actions} mode={mode}"
    )
    valid, reason = validate_action(actions[0]) if actions else (False, "empty actions")
    print(f"mode={mode}")
    print(f"action shape: ({len(actions)}, {len(actions[0]) if actions else 0})")
    print(f"action type: {type(actions).__name__}")
    print(f"joint order: {JOINT_ORDER}")
    print("gripper representation: SO101 arm-frame degrees (converted server-side, see server/policy.py)")
    print(f"valid: {valid} ({reason})")
    return actions


def run_preview_loop(cameras, robot):
    """Continuous, print-only loop. Runs until Ctrl+C. Never executes anything."""
    observation_id = 0
    while True:
        try:
            robot_state, result, latency_ms = capture_and_request(cameras, robot, observation_id)
            actions = print_diagnostics(observation_id, robot_state, result, latency_ms, "preview")
            safe, safety_reason = is_safe_action(
                robot_state,
                actions[0],
                expected_observation_id=observation_id,
                response_observation_id=result.get("observation_id"),
            )
            print(f"safety: {'OK' if safe else 'REJECTED -'} {safety_reason}")
        except requests.RequestException as e:
            print(f"observation={observation_id} request failed: {e}")

        observation_id += 1
        time.sleep(1)


def run_chunk_control_loop(cameras, robot):
    """Stage 7: capture observation -> get chunk -> execute first EXECUTE_STEPS
    validated actions -> capture a fresh observation -> repeat. Stops on
    Ctrl+C, a safety rejection, or any request/execution error."""
    observation_id = 0
    while True:
        try:
            robot_state, result, latency_ms = capture_and_request(cameras, robot, observation_id)
        except requests.RequestException as e:
            print(f"observation={observation_id} request failed: {e}")
            return

        actions = print_diagnostics(observation_id, robot_state, result, latency_ms, "control")
        steps = actions[:EXECUTE_STEPS]

        for i, action_values in enumerate(steps):
            # Re-check against the arm's live state - it moved since the chunk was requested.
            current_state = robot.get_state()
            safe, safety_reason = is_safe_action(
                current_state,
                action_values,
                expected_observation_id=observation_id,
                response_observation_id=result.get("observation_id"),
            )
            if not safe:
                print(f"safety: REJECTED at step {i} - {safety_reason}")
                print("Stopping chunk execution.")
                return

            action = dict(zip(JOINT_ORDER, action_values))
            print(f"safety: OK - executing step {i + 1}/{len(steps)}: {action}")
            try:
                robot.send_action(action)
            except Exception as e:
                print(f"Execution error: {e}")
                return
            time.sleep(STEP_DELAY_S)

        observation_id += 1


def main():
    args = parse_args()
    mode = args.mode

    if mode == "preview":
        print("MODE: PREVIEW")
        print("Robot actions will NOT be executed.")
    else:
        print("MODE: CONTROL")
        print(f"Robot actions may be executed - {EXECUTE_STEPS} steps per chunk, until Ctrl+C.")

    cameras = CameraManager(1, 0)  # (wrist_id, third_person_id) - confirmed via scripts/camera_id_check.py
    robot = SO101Robot(FOLLOWER_PORT, FOLLOWER_ID)

    print(f"Connecting to follower arm on {FOLLOWER_PORT} ...")
    robot.connect()

    try:
        if mode == "control":
            run_chunk_control_loop(cameras, robot)
        else:
            run_preview_loop(cameras, robot)
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C received)...")
    finally:
        cameras.close()
        robot.disconnect()


if __name__ == "__main__":
    main()

