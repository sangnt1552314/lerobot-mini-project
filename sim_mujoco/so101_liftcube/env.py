"""MuJoCo SO-101 Lift (a cube or a cylinder) with a front and a wrist camera (the dataset format of the
original Isaac LeIsaac-SO101-LiftCube-DualCam-v0 task). SIMULATION ONLY.

Robot: MuJoCo Menagerie `robotstudio_so101` (Apache-2.0), loaded unchanged; the table, object, walls,
front camera and lights are added with MjSpec. One env step = one 30 Hz camera/dataset frame.

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
ASSETS_DIR = SIM_MUJOCO_DIR / "assets"
_MENAGERIE_YELLOW = np.array([1.0, 0.82, 0.12])


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
        self.object_body = self.model.body("object").id
        self.object_qadr = self.model.jnt_qposadr[self.model.joint("object_joint").id]
        self.front_cam = self.model.camera("front").id
        self.front_pos0 = self.model.cam_pos[self.front_cam].copy()
        self.front_quat0 = self.model.cam_quat[self.front_cam].copy()
        self.wrist_cam = self.model.camera("wrist_cam").id
        self._fovy0 = self.model.cam_fovy.copy()
        ap = self.cfg.appearance
        tint_names = (["table_mat", *self._painted] + (["floor_mat"] if ap.floor_texture or ap.floor_rgba else [])
                      + [f"wall{i}_mat" for i in range(len(ap.walls))])
        self._tint0 = {self.model.material(n).id: self.model.mat_rgba[self.model.material(n).id].copy() for n in tint_names}
        self._light_diffuse0 = self.model.light_diffuse.copy()
        self._light_dir0 = self.model.light_dir.copy()
        self._headlight0 = (self.model.vis.headlight.ambient.copy(), self.model.vis.headlight.diffuse.copy())
        self.renderer = mujoco.Renderer(self.model, height=self.cfg.height, width=self.cfg.width) if render else None
        self.rng = np.random.default_rng()
        self._gain = {}
        self.steps = 0
        self._above = 0

    # ---- scene ---------------------------------------------------------------------------------
    def _build(self, mjcf: Path) -> mujoco.MjModel:
        c = self.cfg
        spec = mujoco.MjSpec.from_file(str(mjcf))
        spec.option.timestep = 1.0 / (c.fps * c.substeps)
        spec.visual.global_.offwidth = max(c.width, 640)
        spec.visual.global_.offheight = max(c.height, 480)
        amb = c.appearance.headlight_ambient if c.appearance.headlight_ambient is not None else 0.35
        spec.visual.headlight.ambient = [amb, amb, amb]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]

        ap = c.appearance
        self._painted = []
        if ap.robot_rgba is not None:
            for m in spec.materials:
                if np.allclose(np.asarray(m.rgba[:3]), _MENAGERIE_YELLOW, atol=0.02):
                    m.rgba = list(ap.robot_rgba)
                    self._painted.append(m.name)

        if ap.table_texture:
            tex = spec.add_texture(name="table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   file=str(ASSETS_DIR / ap.table_texture))
        else:
            tex = spec.add_texture(name="table_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512,
                                   rgb1=[0.62, 0.55, 0.47], rgb2=[0.56, 0.49, 0.41])
        mat = spec.add_material(name="table_mat", texrepeat=list(ap.texture_repeat), texuniform=True, reflectance=0.0)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        if ap.floor_texture:
            ftex = spec.add_texture(name="floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                    file=str(ASSETS_DIR / ap.floor_texture))
            fmat = spec.add_material(name="floor_mat", texrepeat=[2, 2], texuniform=True, reflectance=0.0)
            fmat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = ftex.name
        elif ap.floor_rgba:
            spec.add_material(name="floor_mat", rgba=list(ap.floor_rgba), reflectance=0.0)
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=256, height=1536,
                         rgb1=[0.75, 0.78, 0.82], rgb2=[0.35, 0.38, 0.42])

        wb = spec.worldbody
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[*ap.table_center, 0.0],
                    size=[*ap.table_half, 0.05], material="table_mat")
        if ap.floor_texture or ap.floor_rgba:  # visual only, below the table; the object rests on the table plane
            wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[0.0, 0.0, ap.floor_z], size=[2.0, 2.0, 0.05],
                        material="floor_mat", contype=0, conaffinity=0)
        for i, wall in enumerate(ap.walls):  # visual only, 4 x 4 m centred at table height
            spec.add_material(name=f"wall{i}_mat", rgba=list(wall.rgba), reflectance=0.0)
            pos = [wall.offset, 0.0, 0.0] if wall.axis == "x" else [0.0, wall.offset, 0.0]
            normal = [-np.sign(wall.offset), 0.0, 0.0] if wall.axis == "x" else [0.0, -np.sign(wall.offset), 0.0]
            quat = np.zeros(4)
            mujoco.mju_quatZ2Vec(quat, np.array(normal))   # plane normal = its local z
            wb.add_geom(name=f"wall{i}", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=pos, quat=list(quat),
                        size=[2.0, 2.0, 0.05], material=f"wall{i}_mat", contype=0, conaffinity=0)
        key = c.appearance.key_light_diffuse if c.appearance.key_light_diffuse is not None else 0.6
        key_dir = list(c.appearance.key_light_dir or (-0.3, 0.3, -1.0))
        wb.add_light(name="key", pos=[0.4, -0.4, 1.2], dir=key_dir, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     diffuse=[key, key, key], castshadow=True)
        wb.add_light(name="fill", pos=[-0.3, 0.5, 1.0], dir=[0.3, -0.5, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                     diffuse=[0.25, 0.25, 0.25], castshadow=False)

        o = c.obj
        rgba = list(ap.object_rgba or o.rgba)
        if o.shape == "box":
            gtype, size = mujoco.mjtGeom.mjGEOM_BOX, [o.half_width, o.half_width, o.half_height]
        elif o.shape == "cylinder":
            gtype, size = mujoco.mjtGeom.mjGEOM_CYLINDER, [o.half_width, o.half_height, 0.0]
        else:
            raise ValueError(f"unknown object shape {o.shape!r}")
        body = wb.add_body(name="object", pos=[0.2, 0.0, o.half_height])
        body.add_freejoint(name="object_joint")
        # contact settings as in Menagerie's scene_box.xml for this robot
        body.add_geom(name="object", type=gtype, size=size, mass=o.mass, rgba=rgba,
                      friction=[o.friction, 0.03, 0.003], condim=3, contype=2, conaffinity=1, solref=[0.01, 1.0])
        if o.base_half_width > 0:  # flange at the foot of a cylinder
            body.add_geom(name="object_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          pos=[0.0, 0.0, o.base_half_height - o.half_height], size=[o.base_half_width, o.base_half_height, 0.0],
                          mass=0.0, rgba=rgba, friction=[o.friction, 0.03, 0.003], condim=3, contype=2,
                          conaffinity=1, solref=[0.01, 1.0])

        quat = look_at_quat(c.front.pos, c.front.lookat)
        if c.front.roll_deg:
            roll = np.deg2rad(c.front.roll_deg)
            mujoco.mju_mulQuat(quat, quat.copy(), np.array([np.cos(roll / 2), 0.0, 0.0, np.sin(roll / 2)]))
        wb.add_camera(name="front", pos=list(c.front.pos), quat=list(quat), fovy=c.front.fovy_deg)
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
        a = self.object_qadr
        self.data.qpos[a : a + 3] = [r.uniform(*c.object_x), r.uniform(*c.object_y), c.obj.half_height]
        self.data.qpos[a + 3 : a + 7] = rot_z(r.uniform(-c.object_yaw_deg, c.object_yaw_deg))
        # front camera jitter (position + small rotation)
        self.model.cam_pos[self.front_cam] = self.front_pos0 + r.uniform(-c.camera_jitter_pos, c.camera_jitter_pos, 3)
        dq = np.zeros(4)
        axis = r.normal(size=3)
        mujoco.mju_axisAngle2Quat(dq, axis / np.linalg.norm(axis), np.deg2rad(r.uniform(-c.camera_jitter_deg, c.camera_jitter_deg)))
        qq = np.zeros(4)
        mujoco.mju_mulQuat(qq, self.front_quat0, dq)
        self.model.cam_quat[self.front_cam] = qq
        self._randomize_appearance(r)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(20):  # let the object settle on the table
            mujoco.mj_step(self.model, self.data)
        self.steps = 0
        self._above = 0
        return self.observation()

    def _randomize_appearance(self, r: np.random.Generator) -> None:
        """Per-episode look randomization of the real-setup profiles. Each draw happens only when its
        setting is on, so the default profile keeps exactly its original random sequence."""
        ap, m = self.cfg.appearance, self.model
        if ap.tint_jitter:
            for mid, rgba0 in self._tint0.items():
                m.mat_rgba[mid, :3] = np.clip(rgba0[:3] * r.uniform(1 - ap.tint_jitter, 1 + ap.tint_jitter, 3), 0, 1)
        if ap.light_jitter:
            scale = r.uniform(1 - ap.light_jitter, 1 + ap.light_jitter, (m.nlight, 1))
            m.light_diffuse[:] = np.clip(self._light_diffuse0 * scale, 0, 1)
            k = r.uniform(1 - ap.light_jitter, 1 + ap.light_jitter)
            m.vis.headlight.ambient[:] = np.clip(self._headlight0[0] * k, 0, 1)
            m.vis.headlight.diffuse[:] = np.clip(self._headlight0[1] * k, 0, 1)
        if ap.light_dir_jitter:
            d = self._light_dir0 + r.uniform(-ap.light_dir_jitter, ap.light_dir_jitter, self._light_dir0.shape)
            m.light_dir[:] = d / np.linalg.norm(d, axis=1, keepdims=True)
        if ap.front_fovy_jitter_deg:
            m.cam_fovy[self.front_cam] = self._fovy0[self.front_cam] + r.uniform(-ap.front_fovy_jitter_deg, ap.front_fovy_jitter_deg)
        if ap.wrist_fovy_range:
            m.cam_fovy[self.wrist_cam] = r.uniform(*ap.wrist_fovy_range)
        self._gain = {}
        for cam in ("front", "wrist_cam"):
            gain = np.array(ap.camera_gain.get(cam.removesuffix("_cam"), (1.0, 1.0, 1.0)))
            if ap.camera_gain_jitter:
                gain = gain * r.uniform(1 - ap.camera_gain_jitter, 1 + ap.camera_gain_jitter, 3)
            if not np.allclose(gain, 1.0):
                self._gain[cam] = gain.astype(np.float32)

    def step(self, action_units: np.ndarray) -> dict:
        q = np.clip(self.units_to_rad(action_units), self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        self.data.ctrl[self.act_ids] = q
        for _ in range(self.cfg.substeps):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self._above = self._above + 1 if self.object_pos[2] > self.cfg.success_height else 0
        return self.observation()

    # ---- observations / status -----------------------------------------------------------------
    @property
    def qpos(self) -> np.ndarray:
        return self.data.qpos[self.qadr].copy()

    @property
    def object_pos(self) -> np.ndarray:
        return self.data.xpos[self.object_body].copy()

    @property
    def object_yaw(self) -> float:
        w, x, y, z = self.data.xquat[self.object_body]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def render(self, camera: str) -> np.ndarray:
        """RGB image of a camera, with that camera's colour (Appearance.camera_saturation, then camera_gain; the
        per-episode gain jitter comes after the saturation, so it is not amplified by it)."""
        self.renderer.update_scene(self.data, camera=camera)
        img = self.renderer.render().copy()
        gain = self._gain.get(camera)
        sat = self.cfg.appearance.camera_saturation.get(camera.removesuffix("_cam"))
        if gain is not None or sat is not None:
            out = img.astype(np.float32)
            if sat is not None:
                luma = out @ np.array([0.299, 0.587, 0.114], np.float32)
                out = luma[..., None] + sat * (out - luma[..., None])
            if gain is not None:
                out = out * gain
            img = np.clip(out, 0, 255).astype(np.uint8)
        return img

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
