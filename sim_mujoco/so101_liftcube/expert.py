"""Scripted Lift expert for the MuJoCo SO-101: approach -> descend -> close -> lift -> hold.

Plans the whole episode at reset from the object pose, then replays it as 6-D absolute joint targets
(LeRobot units) at 30 Hz. Inverse kinematics is damped least squares on the Menagerie model's
`gripperframe` site (at the fingertips; its x-axis points along the fingers and the jaws close along
its z-axis): fingers pointing straight down, jaw axis aligned with a face of a box, or, for a cylinder
(any direction grips it), the direction that needs the least wrist roll from the start pose. The descent
and lift follow straight vertical lines (IK per frame); the approach interpolates in joint space. Near the
base the wrist_flex limit makes "straight down" unreachable above ~9 cm, so the lift solves for position
only and lets the held object tilt (up to ~30 deg).
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


class CameraClearance:
    """Distance between the wrist camera mount's collision boxes and the arm links it can hit (base, shoulder, upper
    and lower arm), from the links' collision-mesh vertices (exact enough here, and unlike contact margins it does not
    miss mesh-box pairs)."""

    LINKS = ("base", "shoulder", "upper_arm", "lower_arm")

    def __init__(self, env: LiftCubeEnv):
        m = self.m = env.model
        self.d = mujoco.MjData(m)
        self.qadr = env.qadr
        body = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        collides = [bool(m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)]
        self.cam = [g for g in range(m.ngeom) if body[g] == "camera_mount" and collides[g]]
        self.verts = {}
        for g in range(m.ngeom):
            if body[g] in self.LINKS and collides[g]:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    a, n = m.mesh_vertadr[m.geom_dataid[g]], m.mesh_vertnum[m.geom_dataid[g]]
                    self.verts[g] = m.mesh_vert[a : a + n].copy()
                else:  # primitives: corners and face centres of the bounding box
                    h = m.geom_size[g]
                    self.verts[g] = np.array([[x, y, z] for x in (-h[0], 0, h[0]) for y in (-h[1], 0, h[1]) for z in (-h[2], 0, h[2])])

    def __call__(self, q: np.ndarray) -> float:
        m, d = self.m, self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        best = np.inf
        for g, v in self.verts.items():
            pts = d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T
            for c in self.cam:  # signed distance of the points to the camera box
                local = np.abs((pts - d.geom_xpos[c]) @ d.geom_xmat[c].reshape(3, 3)) - m.geom_size[c]
                dist = np.linalg.norm(np.maximum(local, 0), axis=1) + np.minimum(local.max(axis=1), 0)
                best = min(best, float(dist.min()))
        return best


class ScriptedLiftExpert:
    def __init__(self, env: LiftCubeEnv):
        self.env = env
        self.cfg: LiftCubeConfig = env.cfg
        self.ik = IK(env)
        self.plan_q: np.ndarray | None = None
        self.ik_error = (0.0, 0.0)
        self.camera_clearance = CameraClearance(env) if self.cfg.min_camera_clearance > 0 else None
        self.min_clearance = np.inf     # of the last plan (inf when not checked)
        self.safe = True                # False: the last plan brings the wrist camera too close to the arm

    @staticmethod
    def down_rotation(theta: float) -> np.ndarray:
        """Fingers down (site x = -world z), jaw axis (site z) at world yaw theta."""
        x = np.array([0.0, 0.0, -1.0])
        z = np.array([np.cos(theta), np.sin(theta), 0.0])
        return np.stack([x, np.cross(z, x), z], axis=1)

    def grasp_rotations(self, obj_xy: np.ndarray, obj_yaw: float) -> list[np.ndarray]:
        """Box: jaw axis along one of the 4 face directions, ordered by closeness to the arm's natural
        wrist roll (jaw axis ~ perpendicular to the reach)."""
        natural = np.arctan2(obj_xy[1], obj_xy[0]) + np.pi / 2
        cands = obj_yaw + np.arange(4) * np.pi / 2
        order = np.argsort(np.abs(np.angle(np.exp(1j * (cands - natural)))), kind="stable")
        return [self.down_rotation(theta) for theta in cands[order]]

    def grasp_rotation(self, obj_xy: np.ndarray, obj_yaw: float) -> np.ndarray:
        return self.grasp_rotations(obj_xy, obj_yaw)[0]

    def self_collides(self, q: np.ndarray) -> bool:
        """True if the robot touches itself at joint config q (contacts with the table/object ignored)."""
        m, d = self.ik.m, self.ik.d
        d.qpos[self.ik.qadr] = q
        mujoco.mj_fwdPosition(m, d)
        skip = {0, self.env.object_body}
        return any(m.geom_bodyid[d.contact[i].geom1] not in skip and m.geom_bodyid[d.contact[i].geom2] not in skip
                   for i in range(d.ncon))

    def plan(self) -> None:
        c, env, r = self.cfg, self.env, self.env.rng
        obj, yaw = env.object_pos, env.object_yaw
        grasp_z = obj[2] - env.grasp_depth
        hover_z = obj[2] + c.hover_height
        high_z = hover_z + c.approach_clearance

        def dur(t):
            return max(2, int(round(t * (1 + r.uniform(-c.timing_noise, c.timing_noise)) * c.fps)))

        n_app, n_desc, n_close, n_lift, n_hold = map(dur, (c.t_approach, c.t_descend, c.t_close, c.t_lift, c.t_hold))
        n_lower = dur(c.t_lower) if c.approach_clearance > 0 else 0
        q_start = env.qpos
        errs = []

        def ik_line(q, tip_xy, rot, z0, z1, n, w_rot, log=True):
            out = []
            for s in smoothstep(np.linspace(0, 1, n + 1)[1:]):
                q, e = self.ik.solve(q, np.array([*tip_xy, z0 + (z1 - z0) * s]), rot, w_rot=w_rot)
                if log:
                    errs.append(e)
                out.append(q.copy())
            return q, out

        seed_roll = q_start[4] if c.grasp_near_start_roll else 1.58

        def grasp_path(rot):
            # the object centre sits on the jaw axis between the jaws: shift the fingertips along it
            tip_xy = obj[:2] - env.in_site_z * rot[:2, 2]
            # good IK seed: the model's own top-down pickup pose (wrist roll: see grasp_near_start_roll)
            seed = np.array([np.arctan2(tip_xy[1], tip_xy[0]), 0.0, 0.47, 1.18, seed_roll, c.gripper_open])
            q_high, e_high = self.ik.solve(seed, np.array([*tip_xy, high_z]), rot, iters=500)
            return tip_xy, q_high, e_high

        if c.obj.shape == "cylinder" or c.grasp_near_start_roll:
            # cylinder: jaw yaws every 10 deg over a full turn; box: its 4 faces. Reachable ones (fingertips
            # within 5 mm, fingers down), least wrist roll change from the start first
            if c.obj.shape == "cylinder":
                natural = np.arctan2(obj[1], obj[0]) + np.pi / 2
                rots = map(self.down_rotation, natural + np.deg2rad(np.arange(-180, 180, 10)))
            else:
                rots = map(self.down_rotation, yaw + np.arange(4) * np.pi / 2)
            paths = [(rot, *grasp_path(rot)) for rot in rots]
            paths = [p for p in paths if p[3][0] < 0.005 and p[3][1] < 0.05] or paths
            paths.sort(key=lambda p: abs(p[2][4] - q_start[4]))
        else:
            paths = [(rot, *grasp_path(rot)) for rot in self.grasp_rotations(obj[:2], yaw)[:1]]
        rot, tip_xy, q_high, e_high = paths[0]
        if c.collision_aware_grasp:
            cands = [p[0] for p in paths] if c.obj.shape == "cylinder" or c.grasp_near_start_roll else self.grasp_rotations(obj[:2], yaw)
            for cand in cands:
                txy, qh, eh = grasp_path(cand)
                _, low = ik_line(qh, txy, cand, high_z, grasp_z, 8, w_rot=0.3, log=False)
                if not any(self.self_collides(q) for q in [qh, *low]):
                    rot, tip_xy, q_high, e_high = cand, txy, qh, eh
                    break
        errs.append(e_high)
        traj = []
        for s in smoothstep(np.linspace(0, 1, n_app + 1)[1:]):  # approach in joint space, open gripper
            q = q_start + (q_high - q_start) * s
            q[5] = q_start[5] + (c.gripper_open - q_start[5]) * min(1.0, 2 * s)
            traj.append(q)
        q = q_high
        if n_lower:
            q, seg = ik_line(q, tip_xy, rot, high_z, hover_z, n_lower, w_rot=0.3)
            traj += [np.r_[qq[ARM], c.gripper_open] for qq in seg]
        q, seg = ik_line(q, tip_xy, rot, hover_z, grasp_z, n_desc, w_rot=0.3)
        traj += [np.r_[qq[ARM], c.gripper_open] for qq in seg]
        q_grasp = traj[-1].copy()
        for s in smoothstep(np.linspace(0, 1, n_close + 1)[1:]):
            traj.append(np.r_[q_grasp[ARM], c.gripper_open + (c.gripper_closed - c.gripper_open) * s])
        q, seg = ik_line(q, tip_xy, rot, grasp_z, c.lift_height, n_lift, w_rot=0.0)
        traj += [np.r_[qq[ARM], c.gripper_closed] for qq in seg]
        traj += [traj[-1].copy() for _ in range(n_hold)]
        self.plan_q = np.array(traj)
        self.ik_error = (max(e[0] for e in errs), max(e[1] for e in errs))
        if self.camera_clearance is not None:
            self.min_clearance = min(self.camera_clearance(q) for q in self.plan_q[::3])
            self.safe = self.min_clearance >= c.min_camera_clearance

    def __len__(self) -> int:
        return 0 if self.plan_q is None else len(self.plan_q)

    def action(self, t: int) -> np.ndarray:
        """Joint target for env step t, in LeRobot units."""
        q = self.plan_q[min(t, len(self.plan_q) - 1)]
        return self.env.rad_to_units(q).astype(np.float32)
