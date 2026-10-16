"""Run squint's train_squint.py (third_party/squint, unmodified) with the office3 envs registered.

    python sim_squint/train.py --env_id=SO101LiftCubeOffice3-v1 [any train_squint.py flag]
    python sim_squint/train.py --env_id=SO101LiftCubeOffice3Both-v1      # front + wrist camera

Runs write to runs/<exp_name>/ under the current directory (pbs/train_squint_office3.pbs: outputs/squint/).

Two cameras (…Both-v1): ManiSkill stacks their RGB along the channels (6) and squint's CNN takes any channel count,
but its colour jitter (torchvision ColorJitter) needs 3-channel images: it is replaced here by one that jitters each
camera's 3 channels on its own (an independent jitter per camera, as their real white balance differs).
"""

import runpy
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SQUINT = HERE.parent / "third_party" / "squint"
sys.path.insert(0, str(SQUINT))
sys.path.append(str(HERE))

import utils as squint_utils  # noqa: E402  (squint's utils.py)

import office3_env  # noqa: E402,F401  (registers the office3 envs)


class PerCameraColorJitterWrapper(squint_utils.ColorJitterWrapper):
    def observation(self, obs):
        rgb = obs["rgb"]
        if rgb.shape[-1] == 3:
            return super().observation(obs)
        parts = []
        for c in range(0, rgb.shape[-1], 3):
            obs["rgb"] = rgb[..., c : c + 3]
            parts.append(super().observation(obs)["rgb"])
        obs["rgb"] = torch.cat(parts, dim=-1)
        return obs


squint_utils.ColorJitterWrapper = PerCameraColorJitterWrapper

if __name__ == "__main__":
    sys.argv[0] = str(SQUINT / "train_squint.py")
    runpy.run_path(str(SQUINT / "train_squint.py"), run_name="__main__")
