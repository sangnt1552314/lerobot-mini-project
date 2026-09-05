"""Minimal MolmoAct2 policy wrapper for allenai/MolmoAct2-SO100_101.

Loading + predict_action call pattern follows the published SO-101 reference
implementation:
https://github.com/irenegracekp/molmoact2-so101/blob/main/molmoact_so101/model/policy.py

NOTE (do not execute these actions on the robot yet):
This does NOT yet convert between SO101's raw "arm frame" joint state and
MolmoAct2's "model frame" state/action convention (that reference
implementation applies a per-arm signs/offsets calibration before/after
inference). Verifying joint order, units, and whether that conversion is
needed is Stage 3 - see server/server.py comments. This stage only checks
that we can load the real model and get a real-shaped action chunk back.
"""

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"


class MolmoAct2Policy:
    def __init__(self, device="cuda", dtype=torch.bfloat16, num_steps=10):
        local_dir = snapshot_download(REPO_ID)
        print(f"[MolmoAct2] Loading {REPO_ID} from {local_dir} (dtype={dtype}, device={device}) ...")
        self.processor = AutoProcessor.from_pretrained(local_dir, trust_remote_code=True)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir, trust_remote_code=True, dtype=dtype
            )
            .to(device)
            .eval()
        )
        self.num_steps = num_steps
        print("[MolmoAct2] Model ready.")

    def predict(self, wrist_image, third_image, robot_state, instruction):
        """wrist_image/third_image: PIL.Image. robot_state: list of floats (raw
        SO101 joint values, order as read from robot.get_state()). Returns a
        (T, JOINT_COUNT) list of lists - raw model output, unconverted."""
        state = np.asarray(robot_state, dtype=np.float32)

        with torch.inference_mode():
            # MolmoAct2 expects images in [scene, wrist] order.
            out = self.model.predict_action(
                processor=self.processor,
                images=[third_image, wrist_image],
                task=instruction,
                state=state,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=self.num_steps,
                normalize_language=True,
                enable_cuda_graph=False,
            )

        actions = np.asarray(out.actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions.tolist()
