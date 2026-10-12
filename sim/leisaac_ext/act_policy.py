"""Run a trained LeRobot ACT checkpoint in-process (no Isaac imports, so it is testable alone).

Same steps as LeRobot 0.4.2's lerobot.utils.control_utils.predict_action (used by lerobot-record):
ACTPolicy.from_pretrained, the pre/post-processors saved next to the weights (normalisation stats from
the dataset), and policy.select_action, which manages ACT's action-chunk queue. predict_action itself
was removed in later LeRobot releases, so the steps are spelled out here; this works with LeRobot
0.4.2 (sim eval) and 0.6.x (the real-robot machine).
"""

from pathlib import Path

import numpy as np
import torch


class LocalACTPolicy:
    def __init__(self, checkpoint: str | Path, device: str = "cuda", n_action_steps: int | None = None):
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        checkpoint = Path(checkpoint).expanduser().resolve()
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"No config.json in {checkpoint}; pass .../checkpoints/<step>/pretrained_model")
        self.checkpoint = checkpoint
        self.device = torch.device(device)

        self.policy = ACTPolicy.from_pretrained(str(checkpoint))
        if n_action_steps is not None:
            self.policy.config.n_action_steps = n_action_steps
        self.policy.to(self.device)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    @property
    def camera_keys(self) -> list[str]:
        """Camera names the checkpoint was trained on, e.g. ['front', 'wrist']."""
        return sorted(k.removeprefix("observation.images.") for k in self.policy.config.image_features)

    @property
    def state_dim(self) -> int:
        return self.policy.config.robot_state_feature.shape[0]

    def reset(self) -> None:
        """Clear the queued action chunk; call at the start of every episode."""
        self.policy.reset()

    def __call__(self, observation: dict[str, np.ndarray], task: str, robot_type: str) -> np.ndarray:
        """observation: 'observation.state' (D,) float32 in LeRobot units and
        'observation.images.<cam>' (H, W, 3) uint8. Returns the next action, shape (A,)."""
        from lerobot.policies.utils import prepare_observation_for_inference

        with torch.inference_mode():
            batch = prepare_observation_for_inference(dict(observation), self.device, task, robot_type)
            batch = self.preprocessor(batch)
            action = self.postprocessor(self.policy.select_action(batch))
        return action.reshape(-1).float().cpu().numpy()
