"""Minimal FastAPI server: receives an observation and returns MolmoAct2's
predicted action chunk.

Stage 2: real model, real actions returned - but the local client only PRINTS
them. The server never decides whether the robot moves; execution/safety
stay entirely on the local computer (see local/client.py).
"""

import io
import json
import time

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

from policy import MolmoAct2Policy

app = FastAPI()

# Loaded on first /infer request (not at import time) so `python server.py`
# and `/health` stay fast even before the model/GPU is touched.
policy = None


@app.get("/health")
def health():
    return {"status": "ready"}


@app.post("/infer")
async def infer(
    wrist: UploadFile = File(...),
    third: UploadFile = File(...),
    joint_positions: str = Form(...),
    instruction: str = Form(...),
    observation_id: int = Form(...),
):
    global policy
    if policy is None:
        policy = MolmoAct2Policy()

    start = time.perf_counter()

    wrist_bytes = await wrist.read()
    third_bytes = await third.read()
    joints = json.loads(joint_positions)

    print(
        f"request={observation_id} wrist={len(wrist_bytes)} bytes "
        f"third={len(third_bytes)} bytes"
    )
    print(f"joints={joints}")
    print(f"instruction={instruction}")

    wrist_image = Image.open(io.BytesIO(wrist_bytes)).convert("RGB")
    third_image = Image.open(io.BytesIO(third_bytes)).convert("RGB")
    # joint_positions arrives as a dict (see local/robot.py) - keep insertion order.
    robot_state = list(joints.values()) if isinstance(joints, dict) else joints

    actions = policy.predict(wrist_image, third_image, robot_state, instruction)

    server_ms = (time.perf_counter() - start) * 1000
    print(f"server_ms={server_ms:.1f} action_shape=({len(actions)}, {len(actions[0]) if actions else 0})")

    return {
        "observation_id": observation_id,
        "actions": actions,
        "server_ms": server_ms,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
