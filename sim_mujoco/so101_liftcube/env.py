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
# grasp geometry of random-size objects (gripper site frame): the fixed jaw's inner face is at z = -0.02 and the
# object's side is placed this far from it; the jaws reach at most this far down around the object
FIXED_JAW_Z = -0.02
JAW_CLEARANCE = 0.002
MAX_GRASP_DEPTH = 0.045
MIN_TIP_HEIGHT = 0.006                      # fingertips above the table when grasping


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


def _r(a, n: int = 4):
    """Rounded plain Python floats (lists for arrays) for JSON metadata."""
    a = np.asarray(a, dtype=float)
    return round(float(a), n) if a.ndim == 0 else [round(float(v), n) for v in a.ravel()]


def point_in_polygon(p, poly) -> bool:
    """Even-odd rule; poly is (n, 2) vertices."""
    x, y = p
    inside = False
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


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
        self.object_geom = self.model.geom("object").id
        self._object_rgba0 = self.model.geom_rgba[self.object_geom].copy()
        o = self.cfg.obj
        self.random_size = o.half_width_range is not None
        if self.random_size and o.shape == "cylinder" and o.half_height_range is None:
            raise ValueError("a random-size cylinder needs half_height_range")
        if self.random_size:
            self.set_object_size(o.half_width, o.half_height)
        else:
            self._fixed_grasp()
        self.object_qadr = self.model.jnt_qposadr[self.model.joint("object_joint").id]
        self.front_cam = self.model.camera("front").id
        self.front_pos0 = self.model.cam_pos[self.front_cam].copy()
        self.front_quat0 = self.model.cam_quat[self.front_cam].copy()
        self.wrist_cam = self.model.camera("wrist_cam").id
        if any(self.cfg.wrist_cam_rot_deg):
            q = self.model.cam_quat[self.wrist_cam].copy()
            for axis, deg in enumerate(self.cfg.wrist_cam_rot_deg):   # x, then y, then z, each in the camera's frame
                half = np.deg2rad(deg) / 2
                dq = np.array([np.cos(half), 0.0, 0.0, 0.0])
                dq[1 + axis] = np.sin(half)
                mujoco.mju_mulQuat(q, q.copy(), dq)
            self.model.cam_quat[self.wrist_cam] = q
        self.model.cam_pos[self.wrist_cam] += self.cfg.wrist_cam_pos_offset
        self.wrist_quat0 = self.model.cam_quat[self.wrist_cam].copy()
        self._fovy0 = self.model.cam_fovy.copy()
        ap = self.cfg.appearance
        table_mats = ["table_mat", *(f"table_rand{i}_mat" for i in range(ap.random_table_textures))]
        background = ((["floor_mat"] if ap.floor_texture or ap.floor_rgba else []) + [f"wall{i}_mat" for i in range(len(ap.walls))]
                      + (["plate_mat"] if self.cfg.table_z < 0 else []))
        tint_names = [*table_mats, *self._painted, *background]
        self._tint0 = {self.model.material(n).id: self.model.mat_rgba[self.model.material(n).id].copy() for n in tint_names}
        self._table_mats = [self.model.material(n).id for n in table_mats]
        self._background_mats = [self.model.material(n).id for n in background]
        self._props = [(self.model.geom(f"prop_{pr.name}").id, pr) for pr in ap.props]
        self.table_geom = self.model.geom("table").id
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
        self._add_random_table_textures(spec, ap.random_table_textures)
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
        if c.table_z < 0:  # visual only; its colour is randomized with the background (Appearance.background_rgb)
            spec.add_material(name="plate_mat", rgba=[0.05, 0.05, 0.05, 1.0], reflectance=0.0)
            wb.add_geom(name="base_plate", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.065, -c.table_z / 2, 0.0],
                        pos=[0.0, 0.0, c.table_z / 2], material="plate_mat", contype=0, conaffinity=0)
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[*ap.table_center, c.table_z],
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
        for pr in ap.props:  # visual only
            gtype = mujoco.mjtGeom.mjGEOM_BOX if pr.shape == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER
            size = list(pr.size) if pr.shape == "box" else [pr.size[0], pr.size[1], 0.0]
            pitch = np.radians(pr.pitch_deg) / 2
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, rot_z(pr.yaw_deg), np.array([np.cos(pitch), 0.0, np.sin(pitch), 0.0]))
            wb.add_geom(name=f"prop_{pr.name}", type=gtype, size=size, pos=list(pr.pos), quat=list(quat),
                        rgba=list(pr.rgba), contype=0, conaffinity=0)
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
        # contact settings as in Menagerie's scene_box.xml for this robot. Random-size objects add torsional friction
        # (condim 4; the jaws touch a round object over a small patch, not a point), without which a wide cylinder
        # pinched at an angle spins out of the jaws
        if o.half_width_range is None:
            friction, condim = [o.friction, 0.03, 0.003], 3
        else:
            friction, condim = [o.friction, 0.01, 0.003], 4
        body.add_geom(name="object", type=gtype, size=size, mass=o.mass, rgba=rgba,
                      friction=friction, condim=condim, contype=2, conaffinity=1, solref=[0.01, 1.0])
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

    @staticmethod
    def _add_random_table_textures(spec: mujoco.MjSpec, n: int) -> None:
        """n procedural table materials table_rand<i>_mat (Appearance.random_table_textures), from a fixed seed so the
        model is the same every time: low-contrast checkers, speckled flat colours and gradients in random colours
        and scales, none of them a saturated red like the object."""
        import colorsys

        r = np.random.default_rng(1234)

        def colour():
            while True:
                hue, sat, val = r.uniform(0, 1), r.uniform(0.0, 0.7), r.uniform(0.25, 0.95)
                if sat < 0.35 or min(hue, 1 - hue) > 0.05:   # not within 18 deg of red unless greyish
                    return np.array(colorsys.hsv_to_rgb(hue, sat, val))

        for i in range(n):
            base = colour()
            other = np.clip(base * r.uniform(0.75, 1.25, 3), 0, 1)
            kind = (mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mujoco.mjtBuiltin.mjBUILTIN_FLAT, mujoco.mjtBuiltin.mjBUILTIN_GRADIENT)[i % 3]
            mark = mujoco.mjtMark.mjMARK_RANDOM if kind != mujoco.mjtBuiltin.mjBUILTIN_GRADIENT else mujoco.mjtMark.mjMARK_NONE
            spec.add_texture(name=f"table_rand{i}", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=kind, width=256, height=256,
                             rgb1=list(base), rgb2=list(other), mark=mark, random=float(r.uniform(0.02, 0.3)),
                             markrgb=list(np.clip(base * r.uniform(0.6, 1.4, 3), 0, 1)))
            rep = float(r.uniform(2.0, 14.0))
            m = spec.add_material(name=f"table_rand{i}_mat", texrepeat=[rep, rep], texuniform=True, reflectance=0.0)
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = f"table_rand{i}"

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
    def _fixed_grasp(self) -> None:
        o = self.cfg.obj
        self.object_half_width, self.object_half_height = o.half_width, o.half_height
        self.grasp_depth, self.in_site_z = o.grasp_depth, o.in_site_z

    def set_object_size(self, half_width: float, half_height: float, density: float | None = None) -> None:
        """Resize the object geom in place (collision, rendering, mass, inertia) and derive the expert's grasp:
        one side of the object JAW_CLEARANCE from the fixed jaw, fingertips MIN_TIP_HEIGHT above the table or
        MAX_GRASP_DEPTH below the top, whichever is higher."""
        m, o, g = self.model, self.cfg.obj, self.object_geom
        hw, hh = half_width, half_height
        density = density or o.density
        if o.shape == "box":
            m.geom_size[g] = [hw, hw, hh]
            m.geom_aabb[g] = [0, 0, 0, hw, hw, hh]
            m.geom_rbound[g] = np.sqrt(2 * hw**2 + hh**2)
            mass = density * 8 * hw * hw * hh
            inertia = mass / 3 * np.array([hw**2 + hh**2, hw**2 + hh**2, 2 * hw**2])
        else:
            m.geom_size[g] = [hw, hh, 0.0]
            m.geom_aabb[g] = [0, 0, 0, hw, hw, hh]
            m.geom_rbound[g] = np.sqrt(hw**2 + hh**2)
            mass = density * np.pi * hw**2 * 2 * hh
            inertia = mass * np.array([(3 * hw**2 + 4 * hh**2) / 12, (3 * hw**2 + 4 * hh**2) / 12, hw**2 / 2])
        b = self.object_body
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.body_iquat[b])
        rot = rot.reshape(3, 3)
        m.body_mass[b] = mass
        m.body_inertia[b] = np.diag(rot.T @ np.diag(inertia) @ rot)   # in the body's inertial frame
        qpos, qvel = self.data.qpos.copy(), self.data.qvel.copy()
        st = m.stat
        stat = (st.extent, st.center.copy(), st.meaninertia, st.meanmass, st.meansize)
        mujoco.mj_setConst(m, self.data)                    # derived masses; it moves the data to qpos0
        # ...and recomputes the scene statistics, which would count props parked out of sight (z = -10) and push
        # the renderer's near clipping plane (znear * extent) past the jaws: keep the compiled ones
        st.extent, st.center[:], st.meaninertia, st.meanmass, st.meansize = stat
        self.data.qpos[:], self.data.qvel[:] = qpos, qvel
        mujoco.mj_forward(m, self.data)
        self.object_half_width, self.object_half_height = hw, hh
        self.in_site_z = FIXED_JAW_Z + hw + JAW_CLEARANCE
        self.grasp_depth = hh - max(MIN_TIP_HEIGHT, 2 * hh - MAX_GRASP_DEPTH)

    def reset(self, seed: int | None = None, placement=None) -> dict:
        """New episode. placement (placements.Placement): the object's x, y, yaw and size instead of random ones (the
        held-out train / val / test cells of profiles with an object_region); everything else is still drawn from seed."""
        c = self.cfg
        self.rng = np.random.default_rng(seed)
        r = self.rng
        if self.random_size:  # before mj_resetData: mj_setConst uses the data
            if placement is not None:
                hw, hh = placement.half_width, placement.half_height
            elif c.p_random_size >= 1.0 or r.random() < c.p_random_size:
                hw = r.uniform(*c.obj.half_width_range)
                hh = hw if c.obj.shape == "box" else r.uniform(*c.obj.half_height_range)
            else:
                hw, hh = c.obj.half_width, c.obj.half_height
            self.set_object_size(hw, hh, r.uniform(*c.obj.density_range) if c.obj.density_range else None)
        if c.obj.friction_range:
            self.model.geom_friction[self.object_geom, 0] = r.uniform(*c.obj.friction_range)
        mujoco.mj_resetData(self.model, self.data)
        q0 = np.array(c.home_qpos) + r.uniform(-c.home_noise, c.home_noise, 6)
        q0 = np.clip(q0, self.jrange[:, 0], self.jrange[:, 1])
        self.data.qpos[self.qadr] = q0
        self.data.ctrl[self.act_ids] = q0
        a = self.object_qadr
        if placement is not None:
            self.data.qpos[a : a + 3] = [placement.x, placement.y, c.table_z + self.object_half_height]
            self.data.qpos[a + 3 : a + 7] = rot_z(placement.yaw_deg)
        else:
            for _ in range(100 if c.object_clearance > 0 else 1):
                self.data.qpos[a : a + 2] = self._draw_xy(r)
                self.data.qpos[a + 2] = c.table_z + self.object_half_height
                self.data.qpos[a + 3 : a + 7] = rot_z(r.uniform(-c.object_yaw_deg, c.object_yaw_deg))
                if c.object_clearance <= 0 or self._object_robot_distance() >= c.object_clearance:
                    break
        b, g = self.object_body, self.object_geom
        self.episode_info = {   # what this episode drew (dataset metadata, eval results)
            "seed": seed, "object_xy_m": _r(self.data.qpos[a : a + 2]),
            "object_yaw_deg": _r(np.degrees(2 * np.arctan2(self.data.qpos[a + 6], self.data.qpos[a + 3])), 2),
            "object_size_m": _r([2 * self.object_half_width, 2 * self.object_half_height]),
            "object_mass_kg": _r(self.model.body_mass[b], 5), "object_friction": _r(self.model.geom_friction[g, 0], 3),
            "start_qpos_rad": _r(q0),
        }
        # front camera jitter (position + small rotation)
        self.model.cam_pos[self.front_cam] = self.front_pos0 + r.uniform(-c.camera_jitter_pos, c.camera_jitter_pos, 3)
        dq = np.zeros(4)
        axis = r.normal(size=3)
        mujoco.mju_axisAngle2Quat(dq, axis / np.linalg.norm(axis), np.deg2rad(r.uniform(-c.camera_jitter_deg, c.camera_jitter_deg)))
        qq = np.zeros(4)
        mujoco.mju_mulQuat(qq, self.front_quat0, dq)
        self.model.cam_quat[self.front_cam] = qq
        if c.wrist_cam_jitter_deg:
            axis = r.normal(size=3)
            mujoco.mju_axisAngle2Quat(dq, axis / np.linalg.norm(axis), np.deg2rad(r.uniform(-c.wrist_cam_jitter_deg, c.wrist_cam_jitter_deg)))
            mujoco.mju_mulQuat(qq, self.wrist_quat0, dq)
            self.model.cam_quat[self.wrist_cam] = qq
        self._randomize_appearance(r)
        m = self.model
        self.episode_info.update({
            "front_cam_pos_m": _r(m.cam_pos[self.front_cam]), "front_cam_quat": _r(m.cam_quat[self.front_cam]),
            "wrist_cam_quat": _r(m.cam_quat[self.wrist_cam]),
            "fovy_deg": {"front": _r(m.cam_fovy[self.front_cam], 2), "wrist": _r(m.cam_fovy[self.wrist_cam], 2)},
            "table_material": m.material(int(m.geom_matid[self.table_geom])).name, "object_rgb": _r(m.geom_rgba[self.object_geom, :3], 3),
            "light_diffuse": _r(m.light_diffuse[:, 0], 3), "shadows": bool(m.light_castshadow[m.light("key").id]),
            "camera_gain": {k: _r(v, 3) for k, v in self._gain.items()},
        })
        mujoco.mj_forward(self.model, self.data)
        for _ in range(20):  # let the object settle on the table
            mujoco.mj_step(self.model, self.data)
        self.steps = 0
        self._above = 0
        return self.observation()

    def _draw_xy(self, r: np.random.Generator) -> list[float]:
        """Uniform x, y in the object_x / object_y rectangle, or in the object_region polygon."""
        c = self.cfg
        if c.object_region is None:
            return [r.uniform(*c.object_x), r.uniform(*c.object_y)]
        poly = np.array(c.object_region)
        while True:
            p = r.uniform(poly.min(axis=0), poly.max(axis=0))
            if point_in_polygon(p, poly):
                return p.tolist()

    def _object_robot_distance(self) -> float:
        """Smallest distance (m, up to 5 cm) between the object and the robot's collision geoms at the current qpos."""
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)   # also updates the geoms' bounding volumes
        m, best = self.model, 0.05
        for g in range(m.ngeom):
            if m.geom_bodyid[g] not in (0, self.object_body) and (m.geom_contype[g] or m.geom_conaffinity[g]):
                best = min(best, mujoco.mj_geomDistance(m, self.data, self.object_geom, g, 0.05, None))
        return best

    def _randomize_appearance(self, r: np.random.Generator) -> None:
        """Per-episode look randomization of the real-setup profiles. Each draw happens only when its
        setting is on, so the default profile keeps exactly its original random sequence."""
        ap, m = self.cfg.appearance, self.model
        if ap.tint_jitter:
            for mid, rgba0 in self._tint0.items():
                m.mat_rgba[mid, :3] = np.clip(rgba0[:3] * r.uniform(1 - ap.tint_jitter, 1 + ap.tint_jitter, 3), 0, 1)
        if ap.random_table_textures:
            own = r.random() < ap.p_own_table
            m.geom_matid[self.table_geom] = self._table_mats[0 if own else 1 + r.integers(len(self._table_mats) - 1)]
        if ap.background_rgb:
            for mid in self._background_mats:
                m.mat_rgba[mid, :3] = r.uniform(*ap.background_rgb)
        far = 0.0
        if ap.table_far_edge:  # the plane collides everywhere; its size is only what is drawn
            near = ap.table_center[0] - ap.table_half[0]
            far = r.uniform(*ap.table_far_edge)
            self.model.geom_pos[self.table_geom, 0] = (near + far) / 2
            self.model.geom_size[self.table_geom, 0] = (far - near) / 2
        for gid, pr in self._props:
            present = pr.p_present >= 1.0 or r.random() < pr.p_present
            pos = np.array(pr.pos) + r.uniform(-1.0, 1.0, 3) * np.array(pr.pos_jitter) + [far if pr.at_far_edge else 0.0, 0, 0]
            self.model.geom_pos[gid] = pos if present else [0.0, 0.0, -10.0]
        if ap.object_tint_jitter:
            j = ap.object_tint_jitter
            self.model.geom_rgba[self.object_geom, :3] = np.clip(self._object_rgba0[:3] * r.uniform(1 - j, 1 + j, 3), 0, 1)
        if ap.light_jitter:
            scale = r.uniform(1 - ap.light_jitter, 1 + ap.light_jitter, (m.nlight, 1))
            m.light_diffuse[:] = np.clip(self._light_diffuse0 * scale, 0, 1)
            k = r.uniform(1 - ap.light_jitter, 1 + ap.light_jitter)
            m.vis.headlight.ambient[:] = np.clip(self._headlight0[0] * k, 0, 1)
            m.vis.headlight.diffuse[:] = np.clip(self._headlight0[1] * k, 0, 1)
        if ap.shadow_prob < 1.0:
            self.model.light_castshadow[self.model.light("key").id] = r.random() < ap.shadow_prob
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
        self._above = self._above + 1 if self.object_pos[2] - self.cfg.table_z > self.cfg.success_height else 0
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
