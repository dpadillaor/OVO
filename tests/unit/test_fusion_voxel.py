"""Tests for the voxel overlap criterion and its sparse cell index."""

from types import SimpleNamespace

import torch

from ovo.entities.fusion import VoxelIndex, VoxelOverlapCriterion


def _pts(coords):
    return torch.tensor(coords, dtype=torch.float32)


class TestVoxelIndex:
    def test_points_in_the_same_cube_collapse_to_one_cell(self):
        # three points within the same 0.1 m cube
        idx = VoxelIndex.build(_pts([[0.01, 0.01, 0.01], [0.05, 0.02, 0.09], [0.09, 0.09, 0.01]]), size=0.1)
        assert len(idx) == 1

    def test_shared_grid_makes_the_same_cube_the_same_key(self):
        a = VoxelIndex.build(_pts([[1.02, 0.51, -0.21]]), size=0.05)
        b = VoxelIndex.build(_pts([[1.04, 0.53, -0.23]]), size=0.05)  # 2 cm away, same cube
        assert a.shared(b) == 1

    def test_adjacent_cubes_do_not_share(self):
        a = VoxelIndex.build(_pts([[0.02, 0.0, 0.0]]), size=0.05)
        b = VoxelIndex.build(_pts([[0.08, 0.0, 0.0]]), size=0.05)  # next cube over
        assert a.shared(b) == 0

    def test_negative_coordinates_pack_without_collision(self):
        a = VoxelIndex.build(_pts([[-1.23, -0.5, -2.0]]), size=0.05)
        b = VoxelIndex.build(_pts([[5.0, 3.0, 1.5]]), size=0.05)
        assert a.shared(b) == 0
        assert len(a) == 1 and len(b) == 1

    def test_empty_cloud_is_empty(self):
        idx = VoxelIndex.build(torch.empty(0, 3), size=0.05)
        assert len(idx) == 0
        assert idx.shared(VoxelIndex.build(_pts([[0, 0, 0]]), size=0.05)) == 0


def _instances(*ids):
    return [SimpleNamespace(id=i) for i in ids]


def _prepared(criterion, clouds):
    """Build obj_pcds ({id: points}) and run the per-pass precompute."""
    objs = _instances(*clouds)
    criterion.prepare(objs, dict(clouds))
    return {i: SimpleNamespace(id=i) for i in clouds}


class TestVoxelOverlapCriterion:
    def test_strong_overlap_accepts_on_geometry_alone(self):
        crit = VoxelOverlapCriterion(voxel_size=0.05)
        block = _pts([[x / 100, 0.0, 0.0] for x in range(0, 20)])  # 20 cells along x
        ins = _prepared(crit, {1: block, 2: block.clone()})
        verdict, dec = crit.check(ins[1], ins[2], None, None, {"cos_sim": 0.0})
        assert verdict is True and dec["accept_mode"] == "A"

    def test_no_overlap_rejects(self):
        crit = VoxelOverlapCriterion(voxel_size=0.05)
        a = _pts([[0.0, 0.0, 0.0]])
        b = _pts([[2.0, 2.0, 2.0]])  # far away
        ins = _prepared(crit, {1: a, 2: b})
        verdict, dec = crit.check(ins[1], ins[2], None, None, {"cos_sim": 0.99})
        assert verdict is False and dec["p_dist"] == 0.0

    def test_weak_overlap_needs_semantics(self):
        crit = VoxelOverlapCriterion(voxel_size=0.05, th_geom=0.5, th_sem=0.2)
        big = _pts([[x / 100, 0.0, 0.0] for x in range(0, 50)])    # x in [0.00, 0.49] -> cells 0..9
        small = _pts([[x / 100, 0.0, 0.0] for x in range(40, 70)])  # x in [0.40, 0.69] -> cells 8..13
        # shared cells {8, 9} = 2, min(10, 6) = 6, ratio = 0.33: above th_sem, below th_geom
        ins = _prepared(crit, {1: big, 2: small})
        low = crit.check(ins[1], ins[2], None, None, {"cos_sim": 0.0})
        high = crit.check(ins[1], ins[2], None, None, {"cos_sim": 0.95})
        assert low[0] is False              # weak geometry, no semantics -> reject
        assert high[0] is True and high[1]["accept_mode"] == "B"
