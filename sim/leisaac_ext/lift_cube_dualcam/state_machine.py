"""Scripted LiftCube expert for LeIsaac-SO101-LiftCube-DualCam-v0.

Upstream LeIsaac (as of commit 24d3bcd) ships only PickOrangeStateMachine, so this class
follows the same StateMachineBase contract and PickOrange's recipe: target the
`gripper` body with damped-least-squares IK, keep a world-fixed orientation, and use the
same grasp offsets.

One deliberate difference: PickOrange feeds an 8-D end-effector pose to an IK *action
term*, so the dataset would record poses. Here the same DLS IK step runs inside the state
machine and the env receives 6-D absolute joint targets (the `so101leader` joint-position
action space). Recorded actions are therefore joint positions in LeRobot units, which is
what ACT is evaluated with in sim and what the physical SO-101 consumes.
"""

import math

import torch
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_from_euler_xyz, quat_inv, subtract_frame_transforms
from leisaac.datagen.state_machine.base import StateMachineBase
from leisaac.tasks.lift_cube.mdp import cube_height_above_base
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES

from ..config import PHASES, load_config

ARM_JOINT_NAMES = SINGLE_ARM_JOINT_NAMES[:5]


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class LiftCubeStateMachine(StateMachineBase):
    """approach -> hover -> descend -> grasp -> lift -> hold, ending early once lifted."""

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or load_config()
        self._expert = cfg["expert"]
        self._success_height = cfg["task"]["success_height_above_base"]
        self._phase_steps: dict[str, int] = {}
        self._ik: DifferentialIKController | None = None
        self.reset()

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def setup(self, env) -> None:
        robot = env.scene["robot"]
        if list(robot.data.joint_names) != SINGLE_ARM_JOINT_NAMES:
            raise RuntimeError(
                f"Robot joint order {robot.data.joint_names} != {SINGLE_ARM_JOINT_NAMES}; "
                "LeIsaac's LeRobot conversions assume this order."
            )
        if env.action_manager.total_action_dim != len(SINGLE_ARM_JOINT_NAMES):
            raise RuntimeError(
                "LiftCubeStateMachine emits 6-D joint targets; configure the env with "
                "env_cfg.use_teleop_device('so101leader')."
            )

        body_ids, _ = robot.find_bodies("gripper")
        self._body_idx = body_ids[0]
        self._arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        # Fixed-base articulations have no Jacobian row for the root body.
        if robot.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._arm_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._arm_ids]

        self._ik_command = self._expert["ik_command"]
        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type=self._ik_command, ik_method="dls", ik_params={"lambda_val": self._expert["ik_lambda"]}
            ),
            num_envs=env.num_envs,
            device=env.device,
        )
        roll, pitch, yaw = (torch.tensor(math.radians(a), device=env.device) for a in self._expert["target_rpy_deg"])
        self._target_quat_w = quat_from_euler_xyz(roll, pitch, yaw).reshape(1, 4).repeat(env.num_envs, 1)

        self._phase_steps = {p: max(1, round(self._expert["phase_s"][p] / env.step_dt)) for p in PHASES}
        self._success_hold_steps = max(1, round(self._expert["success_hold_s"] / env.step_dt))
        self._device = env.device
        self._grasp_offset = torch.tensor(self._expert["grasp_offset"], device=env.device)
        self._gripper_open = math.radians(self._expert["gripper_open_deg"])
        self._gripper_close = math.radians(self._expert["gripper_close_deg"])

    def check_success(self, env) -> bool:
        return bool(self._success_now(env).all().item())

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        cube_pos_w = env.scene["cube"].data.root_pos_w
        phase, t = self._phase_at(self._step_count)

        if self._step_count == 0:
            self._start_ee_w = robot.data.body_pos_w[:, self._body_idx].clone()
        # Track the cube until the descent starts, then commit to that grasp point.
        if phase in ("approach", "hover"):
            self._cube_at_grasp_w = cube_pos_w.clone()

        grasp_w = self._cube_at_grasp_w + self._grasp_offset
        hover_w = grasp_w.clone()
        hover_w[:, 2] += self._expert["hover_height"]
        lift_w = self._lift_target_w(robot) + self._grasp_offset

        s = _smoothstep(t)
        if phase == "approach":
            target_w, grip = self._start_ee_w + (hover_w - self._start_ee_w) * s, self._gripper_open
        elif phase == "hover":
            target_w, grip = hover_w, self._gripper_open
        elif phase == "descend":
            target_w, grip = hover_w + (grasp_w - hover_w) * s, self._gripper_open
        elif phase == "grasp":
            target_w = grasp_w
            grip = self._gripper_open + (self._gripper_close - self._gripper_open) * min(t / 0.6, 1.0)
        elif phase == "lift":
            target_w, grip = grasp_w + (lift_w - grasp_w) * s, self._gripper_close
        else:  # hold
            target_w, grip = lift_w, self._gripper_close

        if phase in ("lift", "hold"):
            self._success_streak = self._success_streak + 1 if self.check_success(env) else 0

        arm_target = self._solve_ik(robot, target_w)
        gripper_target = torch.full((env.num_envs, 1), grip, device=self._device)
        return torch.cat([arm_target, gripper_target], dim=-1)

    def advance(self) -> None:
        self._step_count += 1
        if self._success_streak >= self._success_hold_steps or self._step_count >= sum(self._phase_steps.values()):
            self._episode_done = True

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._success_streak = 0
        self._start_ee_w = None
        self._cube_at_grasp_w = None

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _phase_at(self, step: int) -> tuple[str, float]:
        for phase in PHASES:
            n = self._phase_steps[phase]
            if step < n:
                return phase, step / n
            step -= n
        return PHASES[-1], 1.0

    def _success_now(self, env) -> torch.Tensor:
        return cube_height_above_base(
            env,
            cube_cfg=SceneEntityCfg("cube"),
            robot_cfg=SceneEntityCfg("robot"),
            robot_base_name="base",
            height_threshold=self._success_height,
        )

    def _lift_target_w(self, robot) -> torch.Tensor:
        """Cube-centre target: lifted above the base and pulled back toward it to stay in reach."""
        base_w = robot.data.root_pos_w
        target = self._cube_at_grasp_w.clone()
        to_base = base_w[:, :2] - target[:, :2]
        target[:, :2] += self._expert["lift_retract"] * to_base / to_base.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        target[:, 2] = base_w[:, 2] + self._expert["lift_cube_height_above_base"]
        return target

    def _solve_ik(self, robot, target_pos_w: torch.Tensor) -> torch.Tensor:
        """One DLS step, same maths as Isaac Lab's DifferentialInverseKinematicsAction."""
        root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, robot.data.body_pos_w[:, self._body_idx], robot.data.body_quat_w[:, self._body_idx]
        )
        # World-fixed orientation (identity by default, as in PickOrangeStateMachine).
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, self._target_quat_w
        )

        jacobian = robot.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        base_rot = matrix_from_quat(quat_inv(root_quat_w))
        jacobian[:, :3, :] = torch.bmm(base_rot, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot, jacobian[:, 3:, :])

        if self._ik_command == "pose":
            self._ik.set_command(torch.cat([target_pos_b, target_quat_b], dim=-1))
        else:
            self._ik.set_command(target_pos_b, ee_quat=ee_quat_b)
        return self._ik.compute(ee_pos_b, ee_quat_b, jacobian, robot.data.joint_pos[:, self._arm_ids])
