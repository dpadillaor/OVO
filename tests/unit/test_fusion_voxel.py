"""Tests for the voxel overlap criterion and its sparse cell index."""

from types import SimpleNamespace

import pytest
import torch

from ovo.entities.fusion import VoxelGrid, VoxelSet, VoxelOverlapCriterion


def _pts(coords):
    return torch.tensor(coords, dtype=torch.float32)


class TestVoxelSet:
    def test_points_in_the_same_cube_collapse_to_one_cell(self):
        # three points within the same 0.1 m cube
        idx = VoxelSet(_pts([[0.01, 0.01, 0.01], [0.05, 0.02, 0.09], [0.09, 0.09, 0.01]]), VoxelGrid(0.1))
        assert len(idx) == 1

    def test_shared_grid_makes_the_same_cube_the_same_key(self):
        a = VoxelSet(_pts([[1.02, 0.51, -0.21]]), VoxelGrid(0.05))
        b = VoxelSet(_pts([[1.04, 0.53, -0.23]]), VoxelGrid(0.05))  # 2 cm away, same cube
        assert a.shared(b) == 1

    def test_adjacent_cubes_do_not_share(self):
        a = VoxelSet(_pts([[0.02, 0.0, 0.0]]), VoxelGrid(0.05))
        b = VoxelSet(_pts([[0.08, 0.0, 0.0]]), VoxelGrid(0.05))  # next cube over
        assert a.shared(b) == 0

    def test_negative_coordinates_pack_without_collision(self):
        a = VoxelSet(_pts([[-1.23, -0.5, -2.0]]), VoxelGrid(0.05))
        b = VoxelSet(_pts([[5.0, 3.0, 1.5]]), VoxelGrid(0.05))
        assert a.shared(b) == 0
        assert len(a) == 1 and len(b) == 1

    def test_empty_cloud_is_empty(self):
        idx = VoxelSet(torch.empty(0, 3), VoxelGrid(0.05))
        assert len(idx) == 0
        assert idx.shared(VoxelSet(_pts([[0, 0, 0]]), VoxelGrid(0.05))) == 0

    def test_containment_is_the_larger_one_and_order_independent(self):
        grid = VoxelGrid(0.05)
        big = VoxelSet(_pts([[x / 100, 0.0, 0.0] for x in range(0, 50)]), grid)       # cells 0..9
        fragment = VoxelSet(_pts([[x / 100, 0.0, 0.0] for x in range(10, 20)]), grid)  # cells 2..3
        assert big.containment(fragment) == fragment.containment(big) == 1.0
        assert big.containment(VoxelSet(torch.empty(0, 3), grid)) == 0.0

    def test_sets_on_different_grids_cannot_be_compared(self):
        a = VoxelSet(_pts([[0.0, 0.0, 0.0]]), VoxelGrid(0.05))
        b = VoxelSet(_pts([[0.0, 0.0, 0.0]]), VoxelGrid(0.10))
        with pytest.raises(ValueError):
            a.shared(b)


def _instances(*ids):
    return [SimpleNamespace(id=i) for i in ids]


def _prepared(criterion, clouds):
    """Build obj_pcds ({id: points}) and run the per-pass precompute."""
    objs = _instances(*clouds)
    criterion.prepare(objs, dict(clouds))
    return {i: SimpleNamespace(id=i) for i in clouds}


class TestVoxelOverlapCriterion:
    def test_strong_overlap_accepts_on_geometry_alone(self):
        crit = VoxelOverlapCriterion(VoxelGrid(0.05))
        block = _pts([[x / 100, 0.0, 0.0] for x in range(0, 20)])  # 20 cells along x
        ins = _prepared(crit, {1: block, 2: block.clone()})
        verdict, dec = crit.check(ins[1], ins[2], None, None, {"cos_sim": 0.0})
        assert verdict is True and dec["accept_mode"] == "A"

    def test_no_overlap_rejects(self):
        crit = VoxelOverlapCriterion(VoxelGrid(0.05))
        a = _pts([[0.0, 0.0, 0.0]])
        b = _pts([[2.0, 2.0, 2.0]])  # far away
        ins = _prepared(crit, {1: a, 2: b})
        verdict, dec = crit.check(ins[1], ins[2], None, None, {"cos_sim": 0.99})
        assert verdict is False and dec["p_dist"] == 0.0

    def test_weak_overlap_needs_semantics(self):
        crit = VoxelOverlapCriterion(VoxelGrid(0.05), th_geom=0.5, th_sem=0.2)
        big = _pts([[x / 100, 0.0, 0.0] for x in range(0, 50)])    # x in [0.00, 0.49] -> cells 0..9
        small = _pts([[x / 100, 0.0, 0.0] for x in range(40, 70)])  # x in [0.40, 0.69] -> cells 8..13
        # shared cells {8, 9} = 2, min(10, 6) = 6, ratio = 0.33: above th_sem, below th_geom
        ins = _prepared(crit, {1: big, 2: small})
        low = crit.check(ins[1], ins[2], None, None, {"cos_sim": 0.0})
        high = crit.check(ins[1], ins[2], None, None, {"cos_sim": 0.95})
        assert low[0] is False              # weak geometry, no semantics -> reject
        assert high[0] is True and high[1]["accept_mode"] == "B"
