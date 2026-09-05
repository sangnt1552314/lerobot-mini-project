"""Minimal FastAPI server: receives an observation and returns a fake action chunk.

No model inference yet - this only tests the communication path.
"""

import json
import time

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI()


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

    # Fake action chunk. Replace with real model inference later.
    fake_actions = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]

    server_ms = (time.perf_counter() - start) * 1000

    return {
        "observation_id": observation_id,
        "actions": fake_actions,
        "server_ms": server_ms,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
