"""LeIsaac-SO101-LiftCube-DualCam-v0: upstream LiftCube with the wrist camera kept.

Upstream leisaac.tasks.lift_cube.LiftCubeEnvCfg deletes the template's wrist camera and
wrist observation. This config is built from the same pieces (SingleArm template, Table
with Cube scene, cube_height_above_base success term) but keeps both cameras, so the
LeRobot recorder writes observation.images.front and observation.images.wrist.

Everything tunable comes from sim/configs/lift_cube_dualcam.yaml.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.assets.scenes.simple import TABLE_WITH_CUBE_CFG, TABLE_WITH_CUBE_USD_PATH
from leisaac.tasks.lift_cube import mdp
from leisaac.tasks.template import (
    SingleArmObservationsCfg,
    SingleArmTaskEnvCfg,
    SingleArmTaskSceneCfg,
    SingleArmTerminationsCfg,
)
from leisaac.utils.domain_randomization import domain_randomization, randomize_object_uniform
from leisaac.utils.general_assets import parse_usd_and_create_subassets

from ..config import load_config
from .events import randomize_camera_around_initial_pose

CFG = load_config()


def _deg_range(values):
    return (math.radians(values[0]), math.radians(values[1]))


def _camera_cfg(name: str) -> TiledCameraCfg:
    cam = CFG["cameras"][name]
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{cam['parent']}/{name}_camera",
        offset=TiledCameraCfg.OffsetCfg(pos=tuple(cam["pos"]), rot=tuple(cam["rot"]), convention=cam["convention"]),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=cam["focal_length"],
            focus_distance=400.0,
            horizontal_aperture=cam["horizontal_aperture"],
            clipping_range=(0.01, 50.0),
            lock_camera=True,
        ),
        width=cam["width"],
        height=cam["height"],
        update_period=1.0 / cam["fps"],
    )


@configclass
class LiftCubeDualCamSceneCfg(SingleArmTaskSceneCfg):
    """Table with Cube + SO-101 + front and wrist cameras."""

    scene: AssetBaseCfg = TABLE_WITH_CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    front: TiledCameraCfg = _camera_cfg("front")
    wrist: TiledCameraCfg = _camera_cfg("wrist")

    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )


@configclass
class LiftCubeDualCamTerminationsCfg(SingleArmTerminationsCfg):

    success = DoneTerm(
        func=mdp.cube_height_above_base,
        params={
            "cube_cfg": SceneEntityCfg("cube"),
            "robot_cfg": SceneEntityCfg("robot"),
            "robot_base_name": "base",
            "height_threshold": CFG["task"]["success_height_above_base"],
        },
    )


@configclass
class LiftCubeDualCamEnvCfg(SingleArmTaskEnvCfg):
    """Dual-camera LiftCube. Policy observations: joint state, front, wrist (template defaults)."""

    scene: LiftCubeDualCamSceneCfg = LiftCubeDualCamSceneCfg(env_spacing=8.0)

    observations: SingleArmObservationsCfg = SingleArmObservationsCfg()

    terminations: LiftCubeDualCamTerminationsCfg = LiftCubeDualCamTerminationsCfg()

    task_description: str = CFG["task"]["description"]

    def __post_init__(self) -> None:
        super().__post_init__()

        # 60 Hz physics, 30 Hz env step: one env step == one dataset frame == one camera frame.
        self.sim.dt = CFG["sim"]["physics_dt"]
        self.decimation = CFG["sim"]["decimation"]
        self.sim.render_interval = self.decimation
        self.episode_length_s = CFG["task"]["episode_length_s"]

        self.viewer.eye = (-0.4, -0.6, 0.5)
        self.viewer.lookat = (0.9, 0.0, -0.3)

        robot = CFG["robot"]
        self.scene.robot.init_state.pos = tuple(robot["base_pos"])
        self.scene.robot.init_state.joint_pos = {
            name: math.radians(deg) for name, deg in robot["init_joint_pos_deg"].items()
        }
        self.scene.robot.spawn.rigid_props.disable_gravity = robot["disable_gravity"]
        for actuator_name, part in (("sts3215-arm", "arm"), ("sts3215-gripper", "gripper")):
            actuator = self.scene.robot.actuators[actuator_name]
            actuator.stiffness = robot[part]["stiffness"]
            actuator.damping = robot[part]["damping"]
            actuator.effort_limit_sim = robot[part]["effort_limit"]
        self.dynamic_reset_gripper_effort_limit = robot["dynamic_gripper_effort_limit"]

        # Creates self.scene.cube (RigidObjectCfg) from the scene USD, as upstream does.
        parse_usd_and_create_subassets(TABLE_WITH_CUBE_USD_PATH, self)

        rand = CFG["randomization"]
        options = [
            randomize_object_uniform(
                "cube",
                pose_range={
                    "x": tuple(rand["cube"]["x"]),
                    "y": tuple(rand["cube"]["y"]),
                    "z": (0.0, 0.0),
                    "yaw": _deg_range(rand["cube"]["yaw_deg"]),
                },
            )
        ]
        xyz, rpy = tuple(rand["front_camera"]["xyz"]), _deg_range(rand["front_camera"]["rpy_deg"])
        if any(xyz) or any(rpy):
            options.append(
                EventTerm(
                    func=randomize_camera_around_initial_pose,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg("front"),
                        "pose_range": {"x": xyz, "y": xyz, "z": xyz, "roll": rpy, "pitch": rpy, "yaw": rpy},
                        "convention": CFG["cameras"]["front"]["convention"],
                    },
                )
            )
        domain_randomization(self, random_options=options)

    def use_teleop_device(self, teleop_device) -> None:
        super().use_teleop_device(teleop_device)
        # Isaac Lab's JointPositionAction adds the default joint pose to every action by
        # default. Turn that off so a recorded action is always the absolute joint target,
        # even if robot.init_joint_pos_deg is changed away from zeros.
        for term in (self.actions.arm_action, self.actions.gripper_action):
            if isinstance(term, mdp.JointPositionActionCfg):
                term.use_default_offset = False
