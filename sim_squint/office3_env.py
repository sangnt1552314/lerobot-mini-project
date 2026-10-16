"""Squint's Lift task (third_party/squint/envs/lift.py) in the office3 setup (sim_mujoco/so101_liftcube/config.py
_office3): the RealSense D435 in front of the arm and/or the wide-angle wrist camera, the folded start pose, the real
3 cm cube / cylinder, office3's placements and a photo of the desk as squint's background overlay.

Squint's SO-101 URDF and the MuJoCo Menagerie model office3 was fitted with have the same kinematics (same base frame
and joint zeros: < 0.01 mm / 0.01 deg apart over random poses), so office3's camera poses, start pose and placements
are used as they are. squint itself is not modified: these envs subclass its classes and are registered on import.

    SO101Lift{Cube,Cylinder}Office3{,Wrist,Both}{,Val,Test}-v1
      cameras:  ""     the front D435 only        sensor "base_camera"
                Wrist  the wrist camera only      sensor "wrist_camera"
                Both   both; the policy gets their RGB stacked along the channels (base_camera first, 6 channels)
      split:    ""     office3's 80 training cells; Val / Test its held-out validation / test cells

What differs from squint's own Lift:
- front camera: office3's fitted D435 pose (incl. its roll), fovy 42.5; the 128 x 128 render is the centre square of
  the D435's 640 x 480 frame, which is what deploy.py crops. Per-step jitter sized like office3's per-episode one
  (+-1.5 cm, ~1 deg, fovy +-2 deg).
- wrist camera: office3's (Menagerie camera mount + the 9 deg fitted on the real one), pinhole fovy 94-108 for the
  wide-angle lens (fitted 101; squint's camera: 71), +-2 mm / +-2 deg per step.
- start: office3's folded rest instead of squint's camera-up pose. The pose to lift the object to (success, reward) is
  still squint's camera-up pose ("start" keyframe; fingertips ~8 cm above the table, in the D435's view).
- object: red, 2.7-3.5 cm (real: 3 cm), at a random office3 point (kept inside the cropped front image); full yaw.
- background: the front camera gets sim_squint/assets/office3_front_overlay.png (a D435 photo of the desk), the wrist
  camera office3_wrist_overlay.png (the desk's colour: it moves, so no photo fits); tools/make_overlay.py makes both.
  Robot painted black like the real arm.
- episodes 60 steps (6 s at 10 Hz) instead of 50: the folded start is further from the object than squint's.
Physics (friction, density, gripper) and the reward are squint's.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import numpy as np
import sapien
import torch
from sapien.render import RenderBodyComponent

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion, quaternion_multiply
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

from envs.base_random_env import ThirdCameraEnv, WristCameraEnv
from envs.lift import Lift, LiftRandomizationConfig

ASSETS = Path(__file__).resolve().parent / "assets"
CAMERAS = {"third": ("base_camera",), "wrist": ("wrist_camera",), "both": ("base_camera", "wrist_camera")}

# office3's front camera (config.py _office3): world = robot base frame, +x forward, z up from the table
FRONT_POS = (0.3972, -0.0534, 0.5081)
FRONT_LOOKAT = (0.1840, -0.0685, 0.1700)
FRONT_FOVY_DEG = 42.5
# office3 rolls its MuJoCo camera by -1.71 deg about the camera's own z, which points backwards: +1.71 deg about the
# viewing direction, the axis squint/ManiSkill roll about
FRONT_ROLL_RAD = np.deg2rad(1.71)
CROP_PX = (80, 560)            # the 480 x 480 centre square of the 640 x 480 frame (deploy.py's crop)

# office3's wrist camera in gripper_link's frame, SAPIEN camera axes (tools/wrist_camera_pose.py)
WRIST_POS = (-0.00296, 0.05436, -0.0531)
WRIST_QUAT = (-0.3732, 0.59509, -0.59155, -0.3958)   # wxyz
WRIST_FOVY_DEG = 101.0

# office3's folded rest (arm frame [3.4, -102.9, 96.0, 72.1, -92.0] deg, shoulder_lift kept in the model's +-100),
# gripper at the office arm's floor of 8 (LeRobot 0..100 over the model's -10..100 deg)
HOME_QPOS = (0.0592, -1.7453, 1.6763, 1.2581, -1.6056, -0.174533 + 0.08 * (1.7453292 + 0.174533))
OBJECT_RGB = (0.85, 0.08, 0.08)
ROBOT_RGB = (0.05, 0.05, 0.05)
GRID = 0.005                   # spacing of the exported placement points (placements.py)
SPLITS = {"train": 0, "val": 1, "test": 2}


@dataclass
class Office3LiftRandomizationConfig(LiftRandomizationConfig):
    rgb_overlay_path: Optional[str] = str(ASSETS / "office3_front_overlay.png")
    wrist_overlay_path: Optional[str] = str(ASSETS / "office3_wrist_overlay.png")
    robot_color: Optional[Union[str, Sequence[float]]] = ROBOT_RGB
    initial_qpos_noise_scale: float = 0.02
    # office3: +-1.5 cm, +-2 deg about a random axis, fovy +-2 deg (per episode); here per step, as squint does
    third_camera_pos_noise: Sequence[float] = (0.03, 0.03, 0.03)   # full width of the box: +-1.5 cm
    third_camera_target_noise: float = 0.007                        # std, m: ~1 deg at the target's 0.40 m
    third_camera_rot_noise: float = np.deg2rad(1)
    third_camera_fov_noise: float = np.deg2rad(2)
    # office3: +-2 deg about a random axis per episode, fovy 94-108 (wide-angle lens, pinhole fit 101)
    wrist_camera_pos_noise: Sequence[float] = (0.002, 0.002, 0.002)
    wrist_camera_rot_noise: Sequence[float] = (np.deg2rad(2), np.deg2rad(2), np.deg2rad(2))
    wrist_camera_fov_noise: float = np.deg2rad(7)
    # the real objects are 3 cm (office3 trains 20 % of its episodes at 2.7-4.0 cm)
    cube_half_size_range: Sequence[float] = (0.0135, 0.0175)
    can_radius_range: Sequence[float] = (0.0135, 0.0175)
    can_half_height_range: Sequence[float] = (0.0135, 0.0175)


def front_camera_pixels(xyz: np.ndarray) -> np.ndarray:
    """(u, v) in the D435's 640 x 480 frame of world points (N, 3), from office3's camera (MuJoCo convention)."""
    pos = np.asarray(FRONT_POS)
    f = np.asarray(FRONT_LOOKAT) - pos
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    c, s = np.cos(-FRONT_ROLL_RAD), np.sin(-FRONT_ROLL_RAD)
    R = np.stack([x, y, -f], axis=1) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pc = (np.asarray(xyz) - pos) @ R
    fpx = 240 / np.tan(np.deg2rad(FRONT_FOVY_DEG) / 2)
    return np.stack([320 + fpx * pc[:, 0] / -pc[:, 2], 240 - fpx * pc[:, 1] / -pc[:, 2]], axis=1)


def load_points(item_type: str, split: str, max_half_size: float) -> np.ndarray:
    """office3's placement points of a split that keep the whole object in the cropped (square) front image."""
    data = np.load(ASSETS / f"office3_points_{'cube' if item_type == 'cube' else 'cylinder'}.npz")
    pts = data["points"][data["split"] == SPLITS[split]].astype(np.float64)
    r = max_half_size * np.sqrt(2) + GRID / 2   # the object's top corners: its widest extent at any yaw
    ok = np.ones(len(pts), bool)
    for dx in (-r, r):
        for dy in (-r, r):
            u, v = front_camera_pixels(np.c_[pts[:, 0] + dx, pts[:, 1] + dy, np.full(len(pts), 2 * max_half_size)]).T
            ok &= (u >= CROP_PX[0]) & (u <= CROP_PX[1]) & (v >= 0) & (v <= 480)
    if not ok.any():
        raise RuntimeError(f"no office3 {split} point of {item_type} is inside the cropped camera image")
    return pts[ok]


class Office3Lift(ThirdCameraEnv, Lift):
    """squint's Lift with office3's cameras. MRO: ThirdCameraEnv (front camera mount) over Lift over WristCameraEnv
    (wrist camera mount); the sensors, mount updates and overlays are chosen here by `cameras`."""

    DEFAULT_CAMERA_POS = list(FRONT_POS)
    DEFAULT_CAMERA_TARGET = list(FRONT_LOOKAT)
    DEFAULT_CAMERA_FOV = np.deg2rad(FRONT_FOVY_DEG)

    def __init__(self, *args, cameras="third", split="train", domain_randomization_config=None, **kwargs):
        if cameras not in CAMERAS:
            raise ValueError(f"cameras={cameras!r}; use one of {tuple(CAMERAS)}")
        self.cameras, self.split = cameras, split
        self.use_third, self.use_wrist = cameras in ("third", "both"), cameras in ("wrist", "both")
        self._overlays: dict[str, torch.Tensor] = {}
        super().__init__(*args, domain_randomization_config=domain_randomization_config or Office3LiftRandomizationConfig(),
                         **kwargs)

    # ---- scene -------------------------------------------------------------------------------------------------
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        cfg = self.domain_randomization_config
        hi = cfg.cube_half_size_range[1] if self.item_type == "cube" else max(cfg.can_radius_range[1], cfg.can_half_height_range[1])
        self.spawn_points = common.to_tensor(load_points(self.item_type, self.split, hi), device=self.device).float()
        self.home_qpos = common.to_tensor(HOME_QPOS, device=self.device).float()
        for obj in self.item._objs:
            body = obj.find_component_by_type(RenderBodyComponent)
            if body is None:   # render_backend="none"
                continue
            for shape in body.render_shapes:
                for part in getattr(shape, "parts", None) or [shape]:
                    part.material.set_base_color(list(OBJECT_RGB) + [1])

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)   # squint: camera, table, robot at its start, object in its box
        with torch.device(self.device):
            b = len(env_idx)
            noise = torch.randn(size=(b, self.home_qpos.shape[-1])) * self.domain_randomization_config.initial_qpos_noise_scale
            self.agent.robot.set_qpos(self.home_qpos + noise)
            self.agent.controller.reset()   # targets of the delta controller from the new qpos
            i = torch.randint(len(self.spawn_points), (b,))
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = self.spawn_points[i] + (torch.rand((b, 2)) - 0.5) * GRID
            xyz[:, 2] = self.item_half_sizes[env_idx]
            self.item.set_pose(Pose.create_from_pq(xyz, randomization.random_quaternions(b, lock_x=True, lock_y=True)))

    # ---- cameras -----------------------------------------------------------------------------------------------
    @property
    def _default_sensor_configs(self):
        cfg = self.domain_randomization_config

        def fov(base, noise):
            if self.domain_randomization and noise > 0:
                return base + noise * (2 * self._batched_episode_rng.rand() - 1)
            return base

        sensors = []
        if self.use_third:
            sensors.append(CameraConfig("base_camera", pose=sapien.Pose(), width=128, height=128,
                                        fov=fov(self.DEFAULT_CAMERA_FOV, cfg.third_camera_fov_noise), near=0.01, far=100,
                                        mount=self.camera_mount))
        if self.use_wrist:
            sensors.append(CameraConfig("wrist_camera", pose=sapien.Pose(), width=128, height=128,
                                        fov=fov(np.deg2rad(WRIST_FOVY_DEG), cfg.wrist_camera_fov_noise), near=0.005, far=100,
                                        mount=self.wrist_camera_mount))
        return sensors

    def sample_camera_poses(self, n: int):
        """squint's ThirdCameraEnv sampling around office3's camera, plus its fixed roll about the viewing direction."""
        cfg = self.domain_randomization_config
        eye = common.to_tensor(self.base_camera_settings["pos"], device=self.device).float()
        target = common.to_tensor(self.base_camera_settings["target"], device=self.device).float()
        if self.domain_randomization:
            eyes = randomization.camera.make_camera_rectangular_prism(
                n, scale=common.to_tensor(cfg.third_camera_pos_noise, device=self.device), center=eye, theta=0, device=self.device)
            targets = torch.normal(mean=target.view(1, 3).repeat(n, 1), std=cfg.third_camera_target_noise)
            roll = FRONT_ROLL_RAD + torch.normal(torch.zeros(n, device=self.device), std=cfg.third_camera_rot_noise)
        else:
            eyes, targets = eye.view(1, 3).repeat(n, 1), target.view(1, 3).repeat(n, 1)
            roll = torch.full((n,), FRONT_ROLL_RAD, device=self.device)
        poses = sapien_utils.look_at(eye=eyes, target=targets, device=self.device)
        axes = targets - eyes
        axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True)
        q = quaternion_multiply(axis_angle_to_quaternion(axes * roll.view(-1, 1)), poses.q)
        return Pose.create_from_pq(poses.p, q=q)

    def _update_wrist_camera_pose(self):
        """Wrist camera mount = gripper_link * office3's camera pose * per-step noise (replaces squint's euler pose)."""
        cfg = self.domain_randomization_config
        n = self.num_envs
        p = torch.tensor(WRIST_POS, device=self.device).float().repeat(n, 1)
        q = torch.tensor(WRIST_QUAT, device=self.device).float().repeat(n, 1)
        if self.domain_randomization:
            r = 2 * torch.rand(n, 6, device=self.device) - 1
            p = p + r[:, :3] * torch.tensor(cfg.wrist_camera_pos_noise, device=self.device).float()
            dq = axis_angle_to_quaternion(r[:, 3:] * torch.tensor(cfg.wrist_camera_rot_noise, device=self.device).float())
            q = quaternion_multiply(q, dq)
        gripper_pose = self.agent.robot.links_map["gripper_link"].pose
        self.wrist_camera_mount.set_pose(gripper_pose * Pose.create_from_pq(p=p, q=q))

    def reset(self, *args, **kwargs):
        return (WristCameraEnv.reset if self.use_wrist else BaseEnv.reset)(self, *args, **kwargs)

    def _before_control_step(self):
        if self.use_third:
            ThirdCameraEnv._before_control_step(self)

    def _after_control_step(self):
        if self.use_wrist:
            WristCameraEnv._after_control_step(self)

    # ---- background overlay, per camera ---------------------------------------------------------------------------
    def _overlay(self, name: str, rgb: torch.Tensor) -> Optional[torch.Tensor]:
        if name not in self._overlays:
            cfg = self.domain_randomization_config
            path = cfg.rgb_overlay_path if name == "base_camera" else cfg.wrist_overlay_path
            h, w = rgb.shape[-3:-1]
            img = np.zeros((h, w, 3), np.uint8) if path is None else cv2.resize(
                cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
            self._overlays[name] = common.to_tensor(img, device=rgb.device)
        return self._overlays[name]

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True):
        obs = BaseEnv._get_obs_sensor_data(self, apply_texture_transforms)
        if not (self.domain_randomization_config.apply_overlay and self.obs_mode_struct.visual.rgb
                and self.obs_mode_struct.visual.segmentation):
            return obs
        for name, cam in obs.items():
            if name in CAMERAS["both"] and isinstance(cam, dict) and "rgb" in cam and "segmentation" in cam:
                cam["rgb"] = self._green_screen_rgb(cam["rgb"], cam["segmentation"], self._overlay(name, cam["rgb"]))
        return obs


def _register(name, item_type, cameras, split):
    @register_env(name, max_episode_steps=60)
    class _Env(Office3Lift):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("cameras", cameras)
            kwargs.setdefault("split", split)
            super().__init__(*args, item_type=item_type, **kwargs)

    _Env.__name__ = name.replace("-", "_")
    return _Env


for _item, _label in (("cube", "Cube"), ("can", "Cylinder")):
    for _cams, _csuffix in (("third", ""), ("wrist", "Wrist"), ("both", "Both")):
        for _split, _ssuffix in (("train", ""), ("val", "Val"), ("test", "Test")):
            _register(f"SO101Lift{_label}Office3{_csuffix}{_ssuffix}-v1", _item, _cams, _split)
