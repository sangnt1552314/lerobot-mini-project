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
from safety import clamp_action, clip_to_arm, is_safe_action
import pose_check

load_dotenv()

# Set via `export SERVER_URL=...` before running, e.g.:
# export SERVER_URL="https://xxxx.ngrok-free.dev"
SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8000")

INSTRUCTION = "grab the white bowl"

# Which server-side model to run. Override without touching code, e.g.:
# export MODEL_REPO_ID="tsangb34/molmoact2-so101-soccer-red_bowl-40episodes"
# Empty/unset -> the server's default (allenai/MolmoAct2-SO100_101).
MODEL_REPO_ID = os.environ.get("MODEL_REPO_ID", "")

# Real SO101 follower connection settings (same values used in
# scripts/move_follower.py / scripts/teleoperating.sh).
FOLLOWER_PORT = os.environ["FOLLOWER_PORT"]
FOLLOWER_ID = os.environ.get("FOLLOWER_ID", "home_follower")

REQUEST_TIMEOUT_S = 300

# MolmoAct2 chunks are trained at 30 fps: chunk[k] is meant to be sent at
# t_start + k/30 s. Pacing them any slower just plays the trajectory in slow
# motion (ACTION_FPS in the reference implementation).
ACTION_FPS = 30.0
STEP_DELAY_S = 1.0 / ACTION_FPS

# Execute the first N actions of each chunk before re-observing. Round-trip to
# the server measures ~1.3 s, so a chunk is already stale on arrival and this
# is a straight trade: more steps = more of the planned motion actually runs,
# but more of it runs open-loop on an older observation. 15 of 30 steps is
# 0.5 s of motion per ~1.8 s cycle. Every step is still re-checked and scaled
# against the arm's live state before it is sent.
EXECUTE_STEPS = 15

# MolmoAct2-SO100_101 was trained on two third-person views, not a wrist view
# (see server/policy.py). Set SCENE_ONLY=1 to send the third-person frame in
# both slots and skip the wrist frame. The server has its own SCENE_ONLY, but
# this one needs no server restart. Measured A/B from the ready pose: the wrist
# frame steers the arm toward its folded pose, the scene-only pair keeps it
# inside the training distribution.
SCENE_ONLY = os.environ.get("SCENE_ONLY") == "1"

# Debug: set SAVE_FRAMES=1 to overwrite local/debug_frames/wrist.jpg and
# third.jpg each request - lets you see exactly what the model receives.
SAVE_FRAMES = os.environ.get("SAVE_FRAMES") == "1"
DEBUG_FRAMES_DIR = os.path.join(os.path.dirname(__file__), "debug_frames")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["preview", "control"], default="preview")
    parser.add_argument(
        "--home",
        action="store_true",
        help="Ramp the arm into the model's training-median pose before starting. "
             "MolmoAct2 predicts 'hold still' from the folded rest pose, so without "
             "this the arm has to already be somewhere sensible.",
    )
    return parser.parse_args()


HOME_RAMP_STEPS = 8
HOME_DWELL_S = 0.5
# If a joint lags its ramp target by more than this, something is in the way.
HOME_STALL_TOL_DEG = 6.0


def move_to_ready(robot):
    """Ramps the arm from wherever it is into pose_check.READY_POSE.

    Interpolates in HOME_RAMP_STEPS hops and checks each one landed, so an
    obstruction stops the ramp instead of the servos grinding against it."""
    start = robot.get_state()
    goal = pose_check.READY_POSE
    print(f"homing: {[round(start[k], 1) for k in JOINT_ORDER]} -> {[round(goal[k], 1) for k in JOINT_ORDER]}")

    for n in range(1, HOME_RAMP_STEPS + 1):
        frac = n / HOME_RAMP_STEPS
        target = {k: start[k] + (goal[k] - start[k]) * frac for k in JOINT_ORDER}
        robot.send_action(target)
        time.sleep(HOME_DWELL_S)

        now = robot.get_state()
        worst = max(JOINT_ORDER, key=lambda k: abs(now[k] - target[k]))
        error = now[worst] - target[worst]
        if abs(error) > HOME_STALL_TOL_DEG:
            print(f"homing: STALLED on {worst} ({error:+.1f} deg off target) - stopping ramp.")
            print("  Check for an obstruction, then re-run.")
            return False

    print(f"homing: done, at {[round(robot.get_state()[k], 1) for k in JOINT_ORDER]}")
    return True


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
        "wrist": ("wrist.jpg", third_jpeg if SCENE_ONLY else wrist_jpeg, "image/jpeg"),
        "third": ("third.jpg", third_jpeg, "image/jpeg"),
    }
    data = {
        "observation_id": observation_id,
        "instruction": INSTRUCTION,
        "joint_positions": json.dumps(robot_state),
    }
    if MODEL_REPO_ID:
        data["model_repo_id"] = MODEL_REPO_ID

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
            if safe:
                clipped, clip_notes = clip_to_arm(actions[0])
                if clip_notes:
                    print(f"clip: {'; '.join(clip_notes)}")
                _, clamp_notes = clamp_action(robot_state, clipped)
                print(f"clamp: {'; '.join(clamp_notes) if clamp_notes else 'none needed'}")
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

            # The model targets an arm whose gripper closes further than ours
            # does; pull the target into reach before anything else.
            reachable, clip_notes = clip_to_arm(action_values)
            if clip_notes:
                print(f"clip: {'; '.join(clip_notes)}")

            # Absolute targets from the model are normally further than one
            # step away - scale the delta down rather than jumping.
            clamped_values, clamp_notes = clamp_action(current_state, reachable)
            if clamp_notes:
                print(f"clamp: {'; '.join(clamp_notes)}")

            action = dict(zip(JOINT_ORDER, clamped_values))
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

    # Verified against local/debug_frames/: index 0 is the close-up wrist view,
    # index 1 is the wide third-person view. Re-check with scripts/camera_id_check.py
    # if the USB cameras get re-enumerated.
    cameras = CameraManager(wrist_id=0, third_id=1)
    robot = SO101Robot(FOLLOWER_PORT, FOLLOWER_ID)

    print(f"Connecting to follower arm on {FOLLOWER_PORT} ...")
    robot.connect()

    if args.home:
        move_to_ready(robot)

    # A pose outside MolmoAct2's training distribution makes it predict the
    # current pose back, which looks identical to a client that never sends
    # anything. Surface it before the run rather than after.
    pose_check.report(robot.get_state())

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

