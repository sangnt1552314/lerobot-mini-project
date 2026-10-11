"""Run a trained LeRobot ACT checkpoint in-process (no Isaac imports, so it is testable alone).

Uses the same LeRobot 0.4.2 code path as lerobot-record: ACTPolicy.from_pretrained, the
pre/post-processors saved next to the weights (normalisation stats from the dataset), and
lerobot.utils.control_utils.predict_action, which also manages ACT's action-chunk queue.
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
        from lerobot.utils.control_utils import predict_action

        action = predict_action(
            observation,
            self.policy,
            self.device,
            self.preprocessor,
            self.postprocessor,
            use_amp=False,
            task=task,
            robot_type=robot_type,
        )
        return action.reshape(-1).cpu().numpy()
