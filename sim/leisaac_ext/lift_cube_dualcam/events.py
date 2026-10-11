"""Reset events for the dual-camera LiftCube task."""

from typing import Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera


def randomize_camera_around_initial_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pose_range: dict[str, tuple[float, float]],
    convention: Literal["opengl", "ros", "world"] = "ros",
):
    """Jitter a camera around the pose it had on the first reset.

    Same sampling as leisaac.enhance.envs.mdp.randomize_camera_uniform, but that function
    offsets from the camera's *current* pose, so the camera random-walks a little further
    on every reset. Over hundreds of demos that drift is centimetres. Here the offset is
    always applied to the pose recorded on the first call.
    """
    camera: Camera = env.scene[asset_cfg.name]
    if not hasattr(camera, "_sim_initial_pose"):
        quat_w = {
            "ros": camera.data.quat_w_ros,
            "opengl": camera.data.quat_w_opengl,
            "world": camera.data.quat_w_world,
        }[convention]
        camera._sim_initial_pose = (camera.data.pos_w.clone(), quat_w.clone())
    pos0, quat0 = camera._sim_initial_pose

    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=camera.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=camera.device)

    positions = pos0[env_ids] + samples[:, 0:3]
    delta = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    orientations = math_utils.quat_mul(quat0[env_ids], delta)
    camera.set_world_poses(positions, orientations, env_ids, convention)
