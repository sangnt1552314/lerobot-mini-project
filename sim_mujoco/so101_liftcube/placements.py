"""Held-out object placements for profiles with an object_region (office3).

The region (where the expert can pick the object up and the front camera sees it, from tools/workspace_scan.py) minus a
margin around the robot's start pose is split into CELLS equal-area cells (k-means of a 5 mm grid). 10 cells spread over
the region (farthest-point order from its centre, so the centre, its edges and corners) are validation cells, 10 more
test cells, the other 80 training cells: no training episode is placed in a validation or test cell, and validation /
test points keep HELD_OUT_MARGIN from the training cells (the rim of a held-out cell is not used).

Trial k of a split is a point drawn uniformly in cell k % n of that split (n = its number of cells); the next pass over
the cells (k >= n) draws new points in the same cells. So the 80 training episodes cover the region evenly (one per
training cell), and the 20 validation trials (2 per validation cell) are the same at every checkpoint evaluation. A
training placement whose demo fails is redrawn in its cell (retry). Each trial has its own env seed (appearance, start
pose noise), distinct across splits. Sizes: validation and test use the object's real size; 20 % of the training cells
(drawn once) a random size in the object's half_width_range / half_height_range.

ACT never sees any of this: the expert plans from the object's simulated pose, the policy gets only images and joints.
"""

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .env import LiftCubeEnv, point_in_polygon, rot_z

SPLITS = ("train", "val", "test")
CELLS = {"train": 80, "val": 10, "test": 10}
SEED_OFFSET = {"train": 0, "val": 10_000_000, "test": 20_000_000}   # env seed = offset + 100 * trial + retry
GRID = 0.005           # m, candidate points of the layout
HELD_OUT_MARGIN = 0.0075   # m, validation / test points at least this far from every training cell's points
LAYOUT_SEED = 0


@dataclass
class Placement:
    split: str
    trial: int
    retry: int
    cell: int           # index into Layout.centers
    seed: int           # env.reset seed
    x: float
    y: float
    yaw_deg: float
    half_width: float
    half_height: float
    random_size: bool

    def as_dict(self) -> dict:
        return asdict(self)


class Layout:
    def __init__(self, env: LiftCubeEnv):
        c = env.cfg
        if c.object_region is None:
            raise ValueError("placements need a profile with an object_region (e.g. office3)")
        self.cfg = c
        self.points = self._candidates(env)
        self.centers, self.label = self._kmeans(self.points, sum(CELLS.values()))
        self.cells = self._assign_splits()
        train_pts = self.points[np.isin(self.label, self.cells["train"])]
        self._held_out_points = {}   # held-out cells without their rim, so they never touch a training placement
        for cell in self.cells["val"] + self.cells["test"]:
            mine = self.points[self.label == cell]
            far = np.linalg.norm(mine[:, None] - train_pts[None], axis=2).min(axis=1) >= HELD_OUT_MARGIN
            self._held_out_points[cell] = mine[far] if far.any() else mine
        r = np.random.default_rng(LAYOUT_SEED)
        n_random = int(round(c.p_random_size * CELLS["train"])) if c.obj.half_width_range else 0
        self.random_size_cells = set(r.choice(self.cells["train"], n_random, replace=False).tolist()) if n_random else set()

    def _candidates(self, env: LiftCubeEnv) -> np.ndarray:
        """Grid points in the region at least object_clearance from the robot at the start pose, for the largest object
        at its widest yaw."""
        c, m, d = self.cfg, env.model, env.data
        poly = np.array(c.object_region)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        xs, ys = np.arange(lo[0], hi[0] + 1e-9, GRID), np.arange(lo[1], hi[1] + 1e-9, GRID)
        pts = np.array([(x, y) for x in xs for y in ys if point_in_polygon((x, y), poly)])
        o = c.obj
        if o.half_width_range:
            env.set_object_size(o.half_width_range[1], (o.half_height_range or o.half_width_range)[1])
        mujoco.mj_resetData(m, d)
        d.qpos[env.qadr] = c.home_qpos
        a = env.object_qadr
        keep = []
        for x, y in pts:
            d.qpos[a : a + 3] = [x, y, c.table_z + env.object_half_height]
            d.qpos[a + 3 : a + 7] = rot_z(45.0)
            keep.append(c.object_clearance <= 0 or env._object_robot_distance() >= c.object_clearance)
        if o.half_width_range:
            env.set_object_size(o.half_width, o.half_height)
        return pts[np.array(keep)]

    @staticmethod
    def _kmeans(pts: np.ndarray, k: int, iters: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Lloyd's k-means from a farthest-point start (deterministic): roughly equal-area cells."""
        centers = [pts[np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1))]]
        dist = np.linalg.norm(pts - centers[0], axis=1)
        for _ in range(k - 1):
            centers.append(pts[np.argmax(dist)])
            dist = np.minimum(dist, np.linalg.norm(pts - centers[-1], axis=1))
        centers = np.array(centers)
        for _ in range(iters):
            label = np.argmin(np.linalg.norm(pts[:, None] - centers[None], axis=2), axis=1)
            new = np.array([pts[label == i].mean(axis=0) if (label == i).any() else centers[i] for i in range(k)])
            if np.allclose(new, centers):
                break
            centers = new
        return centers, label

    def _assign_splits(self) -> dict[str, list[int]]:
        """Farthest-point order of the cell centres from the centre-most one: the first 20 alternate val / test."""
        ctr = self.centers
        order = [int(np.argmin(np.linalg.norm(ctr - ctr.mean(axis=0), axis=1)))]
        dist = np.linalg.norm(ctr - ctr[order[0]], axis=1)
        while len(order) < CELLS["val"] + CELLS["test"]:
            order.append(int(np.argmax(dist)))
            dist = np.minimum(dist, np.linalg.norm(ctr - ctr[order[-1]], axis=1))
        val, test = order[0::2], order[1::2]
        held = set(order)
        return {"train": [i for i in range(len(ctr)) if i not in held], "val": val, "test": test}

    def placement(self, split: str, trial: int, retry: int = 0) -> Placement:
        cells = self.cells[split]
        cell = cells[trial % len(cells)]
        seed = SEED_OFFSET[split] + 100 * trial + retry
        r = np.random.default_rng([LAYOUT_SEED, SPLITS.index(split), trial, retry])
        mine = self._held_out_points[cell] if split != "train" else self.points[self.label == cell]
        x, y = mine[r.integers(len(mine))] + r.uniform(-GRID / 2, GRID / 2, 2)
        yaw = r.uniform(-self.cfg.object_yaw_deg, self.cfg.object_yaw_deg)
        o = self.cfg.obj
        hw, hh, rand = o.half_width, o.half_height, False
        if split == "train" and cell in self.random_size_cells:
            hw = r.uniform(*o.half_width_range)
            hh = hw if o.shape == "box" else r.uniform(*o.half_height_range)
            rand = True
        return Placement(split, trial, retry, cell, seed, round(float(x), 4), round(float(y), 4), round(float(yaw), 2),
                         round(float(hw), 5), round(float(hh), 5), rand)

    def summary(self) -> dict:
        """The layout, for the dataset / eval metadata."""
        return {"grid_m": GRID, "held_out_margin_m": HELD_OUT_MARGIN, "region": [list(p) for p in self.cfg.object_region], "clearance_m": self.cfg.object_clearance,
                "cell_centers": np.round(self.centers, 4).tolist(), "cells": self.cells,
                "random_size_cells": sorted(self.random_size_cells), "seed_offsets": SEED_OFFSET}
