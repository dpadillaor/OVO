"""Overlap methods (ovo.entities.fusion.overlap): broadphase candidates and per-pair measure."""

from itertools import combinations

import pytest
import torch

from fakes import cells, line, random_scene
from ovo.entities.fusion.overlap import (OverlapCfg, PointCfg, PointOverlap, VoxelCfg, VoxelOverlap,
                                         build_overlap)
from ovo.entities.voxel import VoxelGrid, VoxelSet


def _prepared(method, pcds):
    method.prepare(list(pcds), pcds)
    return method


# ---------------------------------------------------------------- voxel

@pytest.mark.parametrize("seed", range(10))
def test_voxel_candidates_are_pairs_sharing_cells_in_double_loop_order(seed):
    _, pcds = random_scene(seed)
    grid = VoxelGrid(0.05)
    sets = {i: VoxelSet(p, grid) for i, p in pcds.items()}
    order = list(pcds)
    expected = [(a, b) for a, b in combinations(order, 2) if sets[a].shared(sets[b]) > 0]

    voxel = _prepared(VoxelOverlap(VoxelCfg()), pcds)
    assert voxel.candidates() == expected
    for a, b in combinations(order, 2):
        assert voxel.measure(a, b) == sets[a].containment(sets[b])


def test_voxel_uses_the_configured_grid():
    pcds = {1: line(0, 2), 2: line(1, 3)}          # share one 5 cm cell of two
    assert _prepared(VoxelOverlap(VoxelCfg()), pcds).measure(1, 2) == 0.5
    coarse = _prepared(VoxelOverlap(VoxelCfg(size=1.0)), pcds)
    assert coarse.measure(1, 2) == 1.0             # everything falls in one 1 m cell


def test_voxel_pair_without_shared_cells_measures_zero():
    voxel = _prepared(VoxelOverlap(VoxelCfg()), {1: line(0, 2), 2: line(5, 7)})
    assert voxel.candidates() == []
    assert voxel.measure(1, 2) == 0.0


# ---------------------------------------------------------------- point

def _point(pcds, max_dist=0.1):
    return _prepared(PointOverlap(PointCfg(max_dist)), pcds)


def test_point_candidates_are_all_pairs_in_order():
    order = [9, 2, 5, 7]
    point = PointOverlap(PointCfg())
    point.prepare(order, {i: line(0, 1) for i in order})
    assert point.candidates() == [(9, 2), (9, 5), (9, 7), (2, 5), (2, 7), (5, 7)]


def test_point_identical_clouds_score_one():
    assert _point({1: line(0, 10), 2: line(0, 10)}).measure(1, 2) == 1.0


def test_point_far_clouds_score_zero():
    assert _point({1: line(0, 10), 2: line(0, 10) + 5.0}).measure(1, 2) == 0.0


def test_point_empty_cloud_scores_zero():
    empty = torch.empty(0, 3, dtype=torch.float64)
    point = _point({1: empty, 2: line(0, 10)})
    assert point.measure(1, 2) == 0.0
    assert point.measure(2, 1) == 0.0


def test_point_measures_the_smaller_cloud_against_the_larger():
    # small: 4 points, 2 of them on the large cloud and 2 far away
    large = line(0, 20)
    small = torch.cat([line(0, 2), line(0, 2) + torch.tensor([0.0, 3.0, 0.0], dtype=torch.float64)])
    point = _point({1: large, 2: small})
    assert point.measure(1, 2) == 0.5
    assert point.measure(2, 1) == 0.5               # symmetric: the order of the pair does not matter


def test_point_max_dist_is_strict():
    a, b = cells((0, 0, 0)), cells((2, 0, 0))     # 0.1 m apart (2 cells of 5 cm)
    assert _point({1: a, 2: b}, max_dist=0.11).measure(1, 2) == 1.0
    assert _point({1: a, 2: b}, max_dist=0.09).measure(1, 2) == 0.0


# ---------------------------------------------------------------- build

def test_build_overlap():
    assert isinstance(build_overlap(OverlapCfg()), VoxelOverlap)
    assert isinstance(build_overlap(OverlapCfg(method="point")), PointOverlap)


def test_build_overlap_bad_method():
    with pytest.raises(ValueError, match=r"instance_fusion\.overlap\.method.*'mesh'"):
        build_overlap(OverlapCfg(method="mesh"))
