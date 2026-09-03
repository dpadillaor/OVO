"""Voxel candidate broadphase: lossless replacement of the O(N^2) pair loop."""

from types import SimpleNamespace

import torch

from ovo.entities.fusion import (
    FusionStrategy,
    VoxelOverlapCriterion,
    CentroidDistanceCriterion,
    PointOverlapOldCriterion,
    collect_merge_pairs,
)


def _pts(coords):
    return torch.tensor(coords, dtype=torch.float32)


def _block(cx, n=20):
    """A dense little block of cells starting at x=cx (each point its own 5 cm cell)."""
    return _pts([[cx + i * 0.05, 0.0, 0.0] for i in range(n)])


def _brute_accept(strategy, objects_list, obj_pcds):
    """Every pair judged, as the old planner did — the reference set."""
    accepted = set()
    for i, a in enumerate(objects_list):
        for b in objects_list[i + 1:]:
            if strategy.same_instance(a, b, obj_pcds[a.id], obj_pcds[b.id]):
                accepted.add(frozenset((a.id, b.id)))
    return accepted


class TestCandidatePairs:
    def test_only_cell_sharing_instances_are_candidates(self):
        crit = VoxelOverlapCriterion(voxel_size=0.05)
        objs = [SimpleNamespace(id=i) for i in (1, 2, 3)]
        pcds = {1: _block(0.0), 2: _block(0.5), 3: _block(10.0)}  # 1&2 overlap, 3 far
        crit.prepare(objs, pcds)
        cands = crit.candidate_pairs()
        assert cands == {(1, 2)}  # 3 shares no cell with anyone


class TestBroadphaseEqualsBruteForce:
    def _setup(self):
        objs = [SimpleNamespace(id=i) for i in (1, 2, 3, 4, 5)]
        pcds = {
            1: _block(0.00),
            2: _block(0.50),   # overlaps 1
            3: _block(0.60),   # overlaps 1 and 2
            4: _block(20.0),   # far, isolated
            5: _block(40.0),   # far, isolated
        }
        return objs, pcds

    def test_same_accepted_pairs_as_brute_force(self):
        objs, pcds = self._setup()
        # Broadphase path: voxel is the sole accepting criterion -> candidate_pairs drives it.
        strat_bp = FusionStrategy([VoxelOverlapCriterion(0.05)])
        strat_bp.prepare(objs, pcds)
        bp = {frozenset(p) for p in collect_merge_pairs(objs, pcds, strat_bp)}

        strat_bf = FusionStrategy([VoxelOverlapCriterion(0.05)])
        strat_bf.prepare(objs, pcds)
        bf = _brute_accept(strat_bf, objs, pcds)

        assert bp == bf

    def test_broadphase_skips_the_far_pairs(self):
        objs, pcds = self._setup()
        strat = FusionStrategy([VoxelOverlapCriterion(0.05)])
        strat.prepare(objs, pcds)
        # candidate_pairs must never include a pair with an isolated instance (4 or 5)
        cands = strat.candidate_pairs()
        assert all(4 not in p and 5 not in p for p in cands)


class TestGuardFallsBackToBruteForce:
    def test_no_voxel_no_candidates(self):
        strat = FusionStrategy([CentroidDistanceCriterion(1.5)])
        assert strat.candidate_pairs() is None

    def test_voxel_plus_overlap_old_is_not_lossless_so_no_candidates(self):
        # overlap_old accepts on point proximity, not cell co-occupancy -> broadphase unsafe.
        strat = FusionStrategy([PointOverlapOldCriterion(0.1), VoxelOverlapCriterion(0.05)])
        assert strat.candidate_pairs() is None
