"""Unit tests for ovo.entities.voxel: grids, voxel sets, containment and the broadphase."""

import random

import pytest
import torch

from ovo.entities.voxel import (
    _AXIS_OFFSET,
    OverlapPair,
    VoxelGrid,
    VoxelSet,
    _containment,
    overlapping_pairs,
)

GRID = VoxelGrid(0.05)


def _set(points, grid=GRID):
    return VoxelSet(torch.tensor(points, dtype=torch.float64), grid)


# ---------------------------------------------------------------- VoxelGrid

@pytest.mark.parametrize("size", [0.0, -0.05])
def test_grid_rejects_non_positive_size(size):
    with pytest.raises(ValueError):
        VoxelGrid(size)


@pytest.mark.parametrize("origin", [(0.0, 0.0), (0.0, 0.0, 0.0, 0.0)])
def test_grid_rejects_origin_without_three_values(origin):
    with pytest.raises(ValueError):
        VoxelGrid(0.05, origin)


def test_grid_from_cfg_defaults():
    grid = VoxelGrid.from_cfg({})
    assert grid.size == 0.05
    assert grid.origin == (0.0, 0.0, 0.0)


def test_grid_from_cfg_reads_values_as_floats():
    grid = VoxelGrid.from_cfg({"voxel_size": 1, "voxel_origin": [1, 2, 3]})
    assert grid == VoxelGrid(1.0, (1.0, 2.0, 3.0))
    assert all(isinstance(v, float) for v in (grid.size, *grid.origin))


# ---------------------------------------------------------------- VoxelSet

def test_points_in_one_cell_collapse():
    voxels = _set([[0.01, 0.01, 0.01], [0.04, 0.02, 0.03], [0.0, 0.049, 0.0]])
    assert len(voxels) == 1


def test_adjacent_cells_do_not_share():
    a = _set([[0.01, 0.01, 0.01]])
    for neighbour in ([0.06, 0.01, 0.01], [0.01, 0.06, 0.01], [0.01, 0.01, 0.06]):
        b = _set([neighbour])
        assert a.shared(b) == 0
        assert a.containment(b) == 0.0


def test_cells_around_origin_do_not_collide():
    # cell centres for i, j, k in -2..2: 125 distinct cells, negatives included
    size = 1.0
    coords = [[i + 0.5, j + 0.5, k + 0.5] for i in range(-2, 3) for j in range(-2, 3) for k in range(-2, 3)]
    voxels = _set(coords, VoxelGrid(size))
    assert len(voxels) == 125
    assert int(voxels.cells.min()) >= 0  # sign bit never set


def test_negative_side_of_origin_is_its_own_cell():
    assert _set([[-0.01, 0.0, 0.0]]).shared(_set([[0.01, 0.0, 0.0]])) == 0


def test_origin_shifts_cell_boundaries():
    points = [[0.01, 0.0, 0.0], [0.04, 0.0, 0.0]]
    assert len(_set(points)) == 1
    assert len(_set(points, VoxelGrid(0.05, (0.025, 0.0, 0.0)))) == 2


def test_empty_cloud():
    empty = VoxelSet(torch.empty(0, 3), GRID)
    full = _set([[0.0, 0.0, 0.0]])
    assert len(empty) == 0
    assert empty.cells.dtype == torch.int64
    assert empty.shared(full) == 0 and full.shared(empty) == 0
    assert empty.containment(full) == 0.0
    assert empty.containment(empty) == 0.0


def test_out_of_range_cells_clamp_to_edge():
    # Documented behaviour: coords beyond +/- _AXIS_OFFSET cells are clamped, so far points collide.
    grid = VoxelGrid(1.0)
    far_a = _set([[3.0 * _AXIS_OFFSET, 0.0, 0.0]], grid)
    far_b = _set([[5.0 * _AXIS_OFFSET, 0.0, 0.0]], grid)
    edge = _set([[_AXIS_OFFSET - 0.5, 0.0, 0.0]], grid)  # last in-range cell
    assert far_a.shared(far_b) == 1
    assert far_a.shared(edge) == 1
    neg_a = _set([[-3.0 * _AXIS_OFFSET, 0.0, 0.0]], grid)
    neg_b = _set([[-5.0 * _AXIS_OFFSET, 0.0, 0.0]], grid)
    assert neg_a.shared(neg_b) == 1
    assert neg_a.shared(far_a) == 0


def test_in_range_extremes_do_not_clamp():
    grid = VoxelGrid(1.0)
    last = _set([[_AXIS_OFFSET - 0.5, 0.0, 0.0]], grid)
    before_last = _set([[_AXIS_OFFSET - 1.5, 0.0, 0.0]], grid)
    first = _set([[-_AXIS_OFFSET + 0.5, 0.0, 0.0]], grid)
    assert last.shared(before_last) == 0
    assert first.shared(last) == 0


def test_mismatched_grids_raise():
    a = _set([[0.0, 0.0, 0.0]], VoxelGrid(0.05))
    for other_grid in (VoxelGrid(0.1), VoxelGrid(0.05, (0.01, 0.0, 0.0))):
        b = _set([[0.0, 0.0, 0.0]], other_grid)
        with pytest.raises(ValueError):
            a.shared(b)
        with pytest.raises(ValueError):
            a.containment(b)


# ---------------------------------------------------------------- containment

def test_containment_helper():
    assert _containment(2, 4, 8) == 0.5
    assert _containment(3, 3, 10) == 1.0
    assert _containment(0, 0, 5) == 0.0
    assert _containment(0, 5, 0) == 0.0


def test_containment_is_symmetric():
    a = _set([[0.01, 0.0, 0.0], [0.06, 0.0, 0.0], [0.11, 0.0, 0.0]])
    b = _set([[0.06, 0.0, 0.0], [0.16, 0.0, 0.0]])
    assert a.containment(b) == b.containment(a) == 0.5
    assert a.shared(b) == b.shared(a) == 1


def test_fragment_inside_object_scores_one():
    obj = _set([[0.05 * i + 0.01, 0.0, 0.0] for i in range(10)])
    fragment = _set([[0.05 * i + 0.02, 0.0, 0.0] for i in range(3, 6)])
    assert obj.containment(fragment) == 1.0
    assert fragment.containment(obj) == 1.0


# ---------------------------------------------------------------- overlapping_pairs

def _brute_force_pairs(sets):
    """Reference: double loop i < j over dict order, containment on every pair, keep shared > 0."""
    keys = list(sets)
    pairs = []
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            if sets[k1].shared(sets[k2]) > 0:
                pairs.append((k1, k2, sets[k1].containment(sets[k2])))
    return pairs


def _random_scene(seed, n_sets, extent):
    """Random sets on a small cell lattice (so many sets share cells), with shuffled
    non-sequential ids and a few empty sets."""
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    ids = rng.sample(range(1000), n_sets)
    sets = {}
    for ins_id in ids:
        n_points = rng.choice([0, 1, 5, 30, 200])
        centre = torch.randint(0, extent, (3,), generator=gen).double()
        spread = rng.uniform(0.5, extent)
        points = centre + (torch.rand(n_points, 3, generator=gen, dtype=torch.float64) - 0.5) * spread
        sets[ins_id] = VoxelSet(points, VoxelGrid(1.0))
    return sets


@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("n_sets, extent", [(2, 3), (8, 4), (30, 6), (60, 3)])
def test_overlapping_pairs_matches_brute_force(seed, n_sets, extent):
    sets = _random_scene(seed, n_sets, extent)
    expected = _brute_force_pairs(sets)
    got = overlapping_pairs(sets)
    assert all(isinstance(p, OverlapPair) for p in got)
    # same pairs, same order, exact same overlap values
    assert [tuple(p) for p in got] == expected


def test_overlapping_pairs_all_sets_share_one_cell():
    sets = {ins_id: _set([[0.01, 0.01, 0.01], [0.05 * ins_id + 1.0, 0.0, 0.0]]) for ins_id in (7, 3, 9, 1, 5)}
    got = overlapping_pairs(sets)
    assert [tuple(p) for p in got] == _brute_force_pairs(sets)
    assert len(got) == 10
    assert all(p.overlap == 0.5 for p in got)


def test_overlapping_pairs_order_follows_dict_not_ids():
    a = _set([[0.0, 0.0, 0.0]])
    sets = {10: a, 2: _set([[0.0, 0.0, 0.0]]), 5: _set([[0.0, 0.0, 0.0]])}
    assert [(p.first, p.second) for p in overlapping_pairs(sets)] == [(10, 2), (10, 5), (2, 5)]


def test_overlapping_pairs_edge_cases():
    assert overlapping_pairs({}) == []
    assert overlapping_pairs({1: _set([[0.0, 0.0, 0.0]])}) == []
    empty = VoxelSet(torch.empty(0, 3), GRID)
    assert overlapping_pairs({1: empty, 2: empty}) == []
    disjoint = {1: _set([[0.0, 0.0, 0.0]]), 2: _set([[1.0, 0.0, 0.0]])}
    assert overlapping_pairs(disjoint) == []
