"""Overlap: how much two instances share space, in [0, 1]. Exactly one method is active.

    voxel   shared occupied voxels / voxels of the smaller instance
    point   fraction of the smaller cloud's points within max_dist of the larger cloud

Each method answers two questions:
    candidates()    which pairs are worth judging (broadphase), in the order of a double loop
                    i < j over the instances. A pair left out must be one the method would
                    score 0, so leaving it out never loses a fusion.
    measure(a, b)   the overlap of one pair.

    overlap:
      method: voxel                           # voxel | point
      voxel: {size: 0.05, origin: [0, 0, 0]}
      point: {max_dist: 0.1}
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import Literal

import numpy as np
import torch

from ..voxel import VoxelGrid, VoxelSet, overlapping_pairs

Pair = tuple[int, int]


@dataclass(frozen=True)
class VoxelCfg:
    size: float = 0.05
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PointCfg:
    max_dist: float = 0.1


@dataclass(frozen=True)
class OverlapCfg:
    method: Literal["voxel", "point"] = "voxel"
    voxel: VoxelCfg = field(default_factory=VoxelCfg)
    point: PointCfg = field(default_factory=PointCfg)


class VoxelOverlap:
    """The broadphase is exact and already yields every overlap, so `measure` is a lookup."""

    def __init__(self, cfg: VoxelCfg):
        self.grid = VoxelGrid(cfg.size, cfg.origin)

    def prepare(self, order: list[int], pcds: dict[int, torch.Tensor]) -> None:
        sets = {i: VoxelSet(pcds[i], self.grid) for i in order}
        self._overlaps = {(p.first, p.second): p.overlap for p in overlapping_pairs(sets)}

    def candidates(self) -> list[Pair]:
        return list(self._overlaps)

    def measure(self, a: int, b: int) -> float:
        return self._overlaps.get((a, b), 0.0)


class PointOverlap:
    """No exact broadphase: every pair is a candidate."""

    def __init__(self, cfg: PointCfg):
        self.max_dist = cfg.max_dist

    def prepare(self, order: list[int], pcds: dict[int, torch.Tensor]) -> None:
        self._order, self._pcds = order, pcds

    def candidates(self) -> list[Pair]:
        return list(combinations(self._order, 2))

    def measure(self, a: int, b: int) -> float:
        import open3d as o3d  # only needed by this method

        p1, p2 = self._pcds[a], self._pcds[b]
        small, large = (p1, p2) if len(p1) <= len(p2) else (p2, p1)
        if len(small) == 0:
            return 0.0
        to_pcd = lambda p: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p.detach().cpu().double().numpy()))
        dists = np.asarray(to_pcd(small).compute_point_cloud_distance(to_pcd(large)))
        return float((dists < self.max_dist).mean())


def build_overlap(cfg: OverlapCfg):
    if cfg.method == "voxel":
        return VoxelOverlap(cfg.voxel)
    if cfg.method == "point":
        return PointOverlap(cfg.point)
    raise ValueError(f"instance_fusion.overlap.method must be voxel or point, got {cfg.method!r}")
