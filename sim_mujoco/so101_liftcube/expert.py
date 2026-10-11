"""Scripted LiftCube expert for the MuJoCo SO-101: approach -> descend -> close -> lift -> hold.

Plans the whole episode at reset from the cube pose, then replays it as 6-D absolute joint targets
(LeRobot units) at 30 Hz. Inverse kinematics is damped least squares on the Menagerie model's
`gripperframe` site (at the fingertips; its x-axis points along the fingers and the jaws close along
its z-axis): fingers pointing straight down, jaw axis aligned with a face of the cube. The descent and
lift follow straight vertical lines (IK per frame); the approach interpolates in joint space. Near the
base the wrist_flex limit makes "straight down" unreachable above ~9 cm, so the lift solves for position
only and lets the held cube tilt (up to ~30 deg).
"""

import mujoco
import numpy as np

from .config import LiftCubeConfig
from .env import LiftCubeEnv

ARM = slice(0, 5)  # pan, lift, elbow, wrist_flex, wrist_roll; the gripper is commanded separately


def smoothstep(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return s * s * (3 - 2 * s)


class IK:
    def __init__(self, env: LiftCubeEnv):
        self.m = env.model
        self.d = mujoco.MjData(env.model)
        self.qadr = env.qadr
        self.dofadr = np.array([self.m.jnt_dofadr[j] for j in env.joint_ids])
        self.site = self.m.site("gripperframe").id
        self.lo, self.hi = env.jrange[:, 0], env.jrange[:, 1]

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, q0: np.ndarray, pos: np.ndarray, rot: np.ndarray, iters: int = 200, w_rot: float = 0.3):
        q = q0.copy()
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, R = self.fk(q)
            e_pos = pos - p
            e_rot = 0.5 * sum(np.cross(R[:, i], rot[:, i]) for i in range(3))
            if np.linalg.norm(e_pos) < 2e-4 and np.linalg.norm(e_rot) < 2e-3:
                break
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dofadr[ARM]], w_rot * jacr[:, self.dofadr[ARM]]])
            e = np.concatenate([e_pos, w_rot * e_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q[ARM] = np.clip(q[ARM] + np.clip(dq, -0.2, 0.2), self.lo[ARM], self.hi[ARM])
        p, R = self.fk(q)
        err = (float(np.linalg.norm(pos - p)), float(np.linalg.norm(0.5 * sum(np.cross(R[:, i], rot[:, i]) for i in range(3)))))
        return q, err


class ScriptedLiftExpert:
    def __init__(self, env: LiftCubeEnv):
        self.env = env
        self.cfg: LiftCubeConfig = env.cfg
        self.ik = IK(env)
        self.plan_q: np.ndarray | None = None
        self.ik_error = (0.0, 0.0)

    def grasp_rotation(self, cube_xy: np.ndarray, cube_yaw: float) -> np.ndarray:
        """Fingers down (site x = -world z); jaw axis (site z) along a cube face, picking the face
        direction closest to the arm's natural wrist roll (jaw axis ~ perpendicular to the reach)."""
        natural = np.arctan2(cube_xy[1], cube_xy[0]) + np.pi / 2
        cands = cube_yaw + np.arange(4) * np.pi / 2
        theta = cands[np.argmin(np.abs(np.angle(np.exp(1j * (cands - natural)))))]
        x = np.array([0.0, 0.0, -1.0])
        z = np.array([np.cos(theta), np.sin(theta), 0.0])
        y = np.cross(z, x)
        return np.stack([x, y, z], axis=1)

    def plan(self) -> None:
        c, env, r = self.cfg, self.env, self.env.rng
        cube, yaw = env.cube_pos, env.cube_yaw
        rot = self.grasp_rotation(cube[:2], yaw)
        # the cube centre sits on the jaw axis between the jaws: shift the fingertips along it
        tip_xy = cube[:2] - c.cube_in_site_z * rot[:2, 2]
        grasp_z = cube[2] - c.grasp_depth
        hover_z = cube[2] + c.hover_height

        def dur(t):
            return max(2, int(round(t * (1 + r.uniform(-c.timing_noise, c.timing_noise)) * c.fps)))

        n_app, n_desc, n_close, n_lift, n_hold = map(dur, (c.t_approach, c.t_descend, c.t_close, c.t_lift, c.t_hold))
        q_start = env.qpos
        # good IK seed: the model's own top-down pickup pose
        seed = np.array([np.arctan2(tip_xy[1], tip_xy[0]), 0.0, 0.47, 1.18, 1.58, c.gripper_open])
        errs = []

        def ik_line(q, z0, z1, n, w_rot):
            out = []
            for s in smoothstep(np.linspace(0, 1, n + 1)[1:]):
                q, e = self.ik.solve(q, np.array([*tip_xy, z0 + (z1 - z0) * s]), rot, w_rot=w_rot)
                errs.append(e)
                out.append(q.copy())
            return q, out

        q_hover, e = self.ik.solve(seed, np.array([*tip_xy, hover_z]), rot, iters=500)
        errs.append(e)
        traj = []
        for s in smoothstep(np.linspace(0, 1, n_app + 1)[1:]):  # approach in joint space, open gripper
            q = q_start + (q_hover - q_start) * s
            q[5] = q_start[5] + (c.gripper_open - q_start[5]) * min(1.0, 2 * s)
            traj.append(q)
        q, seg = ik_line(q_hover, hover_z, grasp_z, n_desc, w_rot=0.3)
        traj += [np.r_[qq[ARM], c.gripper_open] for qq in seg]
        q_grasp = traj[-1].copy()
        for s in smoothstep(np.linspace(0, 1, n_close + 1)[1:]):
            traj.append(np.r_[q_grasp[ARM], c.gripper_open + (c.gripper_closed - c.gripper_open) * s])
        q, seg = ik_line(q, grasp_z, c.lift_height, n_lift, w_rot=0.0)
        traj += [np.r_[qq[ARM], c.gripper_closed] for qq in seg]
        traj += [traj[-1].copy() for _ in range(n_hold)]
        self.plan_q = np.array(traj)
        self.ik_error = (max(e[0] for e in errs), max(e[1] for e in errs))

    def __len__(self) -> int:
        return 0 if self.plan_q is None else len(self.plan_q)

    def action(self, t: int) -> np.ndarray:
        """Joint target for env step t, in LeRobot units."""
        q = self.plan_q[min(t, len(self.plan_q) - 1)]
        return self.env.rad_to_units(q).astype(np.float32)
