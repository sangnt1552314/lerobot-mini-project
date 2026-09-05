"""Local SO101 client: captures cameras, sends observation to server, prints fake action.

Does NOT move the robot. This is just a communication prototype.
"""

import json
import time

import requests

from cameras import CameraManager

# Later replace with your ngrok URL, e.g.:
# SERVER_URL = "https://xxxx.ngrok.app"
SERVER_URL = "http://localhost:8000"

INSTRUCTION = "pick up the red cube"

# Temporary fake SO101 joint state. SO101 has 6 joints (5 arm joints + gripper).
# Replace this with the real robot joint positions later.
FAKE_JOINT_POSITIONS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

REQUEST_TIMEOUT_S = 10


def main():
    cameras = CameraManager(0, 1)  # (wrist_id, third_person_id) - change if needed
    observation_id = 0

    try:
        while True:
            wrist_frame, third_frame = cameras.capture()
            wrist_jpeg = cameras.encode_jpeg(wrist_frame)
            third_jpeg = cameras.encode_jpeg(third_frame)

            files = {
                "wrist": ("wrist.jpg", wrist_jpeg, "image/jpeg"),
                "third": ("third.jpg", third_jpeg, "image/jpeg"),
            }
            data = {
                "observation_id": observation_id,
                "instruction": INSTRUCTION,
                "joint_positions": json.dumps(FAKE_JOINT_POSITIONS),
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
                print(
                    f"observation={observation_id} latency={latency_ms:.1f} ms "
                    f"actions={result['actions']}"
                )
            except requests.RequestException as e:
                print(f"observation={observation_id} request failed: {e}")

            observation_id += 1
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C received)...")
    finally:
        cameras.close()


if __name__ == "__main__":
    main()
