"""MuJoCo SO-101 LiftCube with a front and a wrist camera (counterpart of the Isaac
LeIsaac-SO101-LiftCube-DualCam-v0 task in sim/). SIMULATION ONLY.

Robot: MuJoCo Menagerie `robotstudio_so101` (Apache-2.0), loaded unchanged; the table, cube, front
camera and lights are added with MjSpec. One env step = one 30 Hz camera/dataset frame.

Units. Observations and actions use LeRobot-style SO-101 units, linear in joint angle: the five arm
joints map their MJCF joint range to -100..100 and the gripper maps its range to 0..100 (0 = closed).
These are the units of the dataset, of ACT's inputs/outputs, and of eval.
"""

import os
from pathlib import Path

import mujoco
import numpy as np

from .config import JOINTS, LiftCubeConfig

SIM_MUJOCO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MJCF = SIM_MUJOCO_DIR / "third_party" / "mujoco_menagerie" / "robotstudio_so101" / "so101.xml"


def look_at_quat(pos, lookat) -> np.ndarray:
    """Quaternion of a MuJoCo camera at `pos` looking at `lookat` with world +z up."""
    f = np.asarray(lookat, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    mat = np.stack([x, y, -f], axis=1)  # camera looks along its -z
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


def rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


class LiftCubeEnv:
    def __init__(self, cfg: LiftCubeConfig | None = None, render: bool = True, mjcf: str | Path | None = None):
        self.cfg = cfg or LiftCubeConfig()
        mjcf = Path(mjcf or os.environ.get("SO101_MJCF", DEFAULT_MJCF))
        if not mjcf.is_file():
            raise FileNotFoundError(f"SO-101 MJCF not found: {mjcf}. Run: bash sim_mujoco/setup_hpc.sh")
        self.model = self._build(mjcf)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = [self.model.joint(j).id for j in JOINTS]
        self.qadr = np.array([self.model.jnt_qposadr[j] for j in self.joint_ids])
        self.jrange = self.model.jnt_range[self.joint_ids].copy()
        self.act_ids = [self.model.actuator(j).id for j in JOINTS]
        self.ctrlrange = self.model.actuator_ctrlrange[self.act_ids].copy()
        self.cube_body = self.model.body("cube").id
        self.cube_qadr = self.model.jnt_qposadr[self.model.joint("cube_joint").id]
        self.front_cam = self.model.camera("front").id
        self.front_pos0 = self.model.cam_pos[self.front_cam].copy()
        self.front_quat0 = self.model.cam_quat[self.front_cam].copy()
        self.renderer = mujoco.Renderer(self.model, height=self.cfg.height, width=self.cfg.width) if render else None
        self.rng = np.random.default_rng()
        self.steps = 0
        self._above = 0

    # ---- scene ---------------------------------------------------------------------------------
    def _build(self, mjcf: Path) -> mujoco.MjModel:
        c = self.cfg
        spec = mujoco.MjSpec.from_file(str(mjcf))
        spec.option.timestep = 1.0 / (c.fps * c.substeps)
        spec.visual.global_.offwidth = max(c.width, 640)
        spec.visual.global_.offheight = max(c.height, 480)
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]

        tex = spec.add_texture(name="table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                               builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                               rgb1=[0.62, 0.55, 0.47], rgb2=[0.56, 0.49, 0.41])
        mat = spec.add_material(name="table_mat", texrepeat=[8, 8], texuniform=True, reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=256, height=1536,
                         rgb1=[0.75, 0.78, 0.82], rgb2=[0.35, 0.38, 0.42])

        wb = spec.worldbody
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1.0, 1.0, 0.05], material="table_mat")
        wb.add_light(name="key", pos=[0.4, -0.4, 1.2], dir=[-0.3, 0.3, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     diffuse=[0.6, 0.6, 0.6], castshadow=True)
        wb.add_light(name="fill", pos=[-0.3, 0.5, 1.0], dir=[0.3, -0.5, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     diffuse=[0.25, 0.25, 0.25], castshadow=False)

        h = c.cube_half
        cube = wb.add_body(name="cube", pos=[0.2, 0.0, h])
        cube.add_freejoint(name="cube_joint")
        # contact settings as in Menagerie's scene_box.xml for this robot
        cube.add_geom(name="cube", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h, h, h], mass=c.cube_mass,
                      rgba=[0.85, 0.08, 0.08, 1.0], friction=[c.cube_friction, 0.03, 0.003], condim=3,
                      contype=2, conaffinity=1, solref=[0.01, 1.0])

        wb.add_camera(name="front", pos=list(c.front.pos), quat=list(look_at_quat(c.front.pos, c.front.lookat)),
                      fovy=c.front.fovy_deg)
        return spec.compile()

    # ---- unit conversion -----------------------------------------------------------------------
    def rad_to_units(self, q: np.ndarray) -> np.ndarray:
        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        u = 200.0 * (q - lo) / (hi - lo) - 100.0
        u[5] = 100.0 * (q[5] - lo[5]) / (hi[5] - lo[5])
        return u

    def units_to_rad(self, u: np.ndarray) -> np.ndarray:
        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        q = lo + (np.asarray(u, float) + 100.0) * (hi - lo) / 200.0
        q[5] = lo[5] + u[5] * (hi[5] - lo[5]) / 100.0
        return q

    # ---- episode -------------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict:
        c = self.cfg
        self.rng = np.random.default_rng(seed)
        r = self.rng
        mujoco.mj_resetData(self.model, self.data)
        q0 = np.array(c.home_qpos) + r.uniform(-c.home_noise, c.home_noise, 6)
        q0 = np.clip(q0, self.jrange[:, 0], self.jrange[:, 1])
        self.data.qpos[self.qadr] = q0
        self.data.ctrl[self.act_ids] = q0
        a = self.cube_qadr
        self.data.qpos[a : a + 3] = [r.uniform(*c.cube_x), r.uniform(*c.cube_y), c.cube_half]
        self.data.qpos[a + 3 : a + 7] = rot_z(r.uniform(-c.cube_yaw_deg, c.cube_yaw_deg))
        # front camera jitter (position + small rotation)
        self.model.cam_pos[self.front_cam] = self.front_pos0 + r.uniform(-c.camera_jitter_pos, c.camera_jitter_pos, 3)
        dq = np.zeros(4)
        axis = r.normal(size=3)
        mujoco.mju_axisAngle2Quat(dq, axis / np.linalg.norm(axis), np.deg2rad(r.uniform(-c.camera_jitter_deg, c.camera_jitter_deg)))
        qq = np.zeros(4)
        mujoco.mju_mulQuat(qq, self.front_quat0, dq)
        self.model.cam_quat[self.front_cam] = qq
        mujoco.mj_forward(self.model, self.data)
        for _ in range(20):  # let the cube settle on the table
            mujoco.mj_step(self.model, self.data)
        self.steps = 0
        self._above = 0
        return self.observation()

    def step(self, action_units: np.ndarray) -> dict:
        q = np.clip(self.units_to_rad(action_units), self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        self.data.ctrl[self.act_ids] = q
        for _ in range(self.cfg.substeps):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self._above = self._above + 1 if self.cube_pos[2] > self.cfg.success_height else 0
        return self.observation()

    # ---- observations / status -----------------------------------------------------------------
    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    @property
    def cube_yaw(self) -> float:
        w, x, y, z = self.data.xquat[self.cube_body]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def render(self, camera: str) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def observation(self) -> dict:
        obs = {"observation.state": self.rad_to_units(self.qpos).astype(np.float32)}
        if self.renderer is not None:
            obs["observation.images.front"] = self.render("front")
            obs["observation.images.wrist"] = self.render("wrist_cam")
        return obs

    def is_success(self) -> bool:
        return self._above >= self.cfg.success_hold_steps

    def timed_out(self) -> bool:
        return self.steps >= int(self.cfg.episode_length_s * self.cfg.fps)
