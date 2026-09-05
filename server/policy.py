"""Minimal MolmoAct2 policy wrapper for allenai/MolmoAct2-SO100_101.

Loading + predict_action call pattern follows the published SO-101 reference
implementation:
https://github.com/irenegracekp/molmoact2-so101/blob/main/molmoact_so101/model/policy.py

STAGE 3: arm<->model frame conversion. MolmoAct2 was trained on the old
LeRobot v2.1 SO-100/101 joint convention (degrees, zero = arm fully
extended), while local/robot.py reads the arm with use_degrees=True to match
that unit convention. The zero-point/sign difference still needs the official
conversion documented at https://huggingface.co/docs/lerobot/backwardcomp and
used as the inference.py defaults in the reference repo above
(--joint-offsets, --joint-signs). predict() applies this both ways, so
callers only ever see SO101 arm-frame values in and out.
"""

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ID = "allenai/MolmoAct2-SO100_101"
NORM_TAG = "so100_so101_molmoact2"

# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
JOINT_OFFSETS = np.array([0.0, 90.0, 90.0, 0.0, 0.0, 0.0], dtype=np.float32)
JOINT_SIGNS = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def arm_to_model(state):
    return JOINT_SIGNS * np.asarray(state, dtype=np.float32) + JOINT_OFFSETS


def model_to_arm(actions):
    return (np.asarray(actions, dtype=np.float32) - JOINT_OFFSETS) * JOINT_SIGNS


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
        """wrist_image/third_image: PIL.Image. robot_state: list of floats, in
        SO101 arm frame (degrees, use_degrees=True - see local/robot.py).
        Returns a (T, JOINT_COUNT) list of lists, also in SO101 arm frame."""
        model_state = arm_to_model(robot_state)

        with torch.inference_mode():
            # MolmoAct2 expects images in [scene, wrist] order.
            out = self.model.predict_action(
                processor=self.processor,
                images=[third_image, wrist_image],
                task=instruction,
                state=model_state,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=self.num_steps,
                normalize_language=True,
                enable_cuda_graph=False,
            )

        actions = out.actions
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().to("cpu", torch.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return model_to_arm(actions).tolist()
