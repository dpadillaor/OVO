"""Voxel candidate broadphase: lossless replacement of the O(N^2) pair loop."""

from types import SimpleNamespace

import torch

from ovo.entities.fusion import (
    FusionStrategy,
    VoxelGrid,
    VoxelOverlapCriterion,
    CentroidDistanceCriterion,
    PointOverlapOldCriterion,
    collect_merge_pairs,
    eager_partners,
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
        crit = VoxelOverlapCriterion(VoxelGrid(0.05))
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
        strat_bp = FusionStrategy([VoxelOverlapCriterion(VoxelGrid(0.05))])
        strat_bp.prepare(objs, pcds)
        bp = {frozenset(p) for p in collect_merge_pairs(objs, pcds, strat_bp)}

        strat_bf = FusionStrategy([VoxelOverlapCriterion(VoxelGrid(0.05))])
        strat_bf.prepare(objs, pcds)
        bf = _brute_accept(strat_bf, objs, pcds)

        assert bp == bf

    def test_broadphase_skips_the_far_pairs(self):
        objs, pcds = self._setup()
        strat = FusionStrategy([VoxelOverlapCriterion(VoxelGrid(0.05))])
        strat.prepare(objs, pcds)
        # candidate_pairs must never include a pair with an isolated instance (4 or 5)
        cands = strat.candidate_pairs()
        assert all(4 not in p and 5 not in p for p in cands)


def _eager_fused(strategy, objects_list, obj_pcds, candidates):
    """The `_fuse_eager` loop without the map side effects: {absorbed_id: survivor_id}."""
    fused = {}
    for a, partners in eager_partners(objects_list, candidates):
        if a.id in fused:
            continue
        for b in partners:
            if b.id not in fused and strategy.same_instance(a, b, obj_pcds[a.id], obj_pcds[b.id]):
                fused[b.id] = a.id
    return fused


class TestEagerBroadphase:
    def test_same_merges_as_brute_force(self):
        objs = [SimpleNamespace(id=i) for i in (1, 2, 3, 4, 5)]
        pcds = {1: _block(0.00), 2: _block(0.50), 3: _block(0.60), 4: _block(20.0), 5: _block(40.0)}
        strat = FusionStrategy([VoxelOverlapCriterion(VoxelGrid(0.05))])
        strat.prepare(objs, pcds)
        bp = _eager_fused(strat, objs, pcds, strat.candidate_pairs())
        bf = _eager_fused(strat, objs, pcds, None)
        assert bp == bf and bp  # identical and non-trivial

    def test_partners_keep_list_order_and_skip_far_instances(self):
        objs = [SimpleNamespace(id=i) for i in (7, 3, 9)]
        partners = {a.id: [b.id for b in bs] for a, bs in eager_partners(objs, {(3, 7), (7, 9)})}
        assert partners == {7: [3, 9], 3: [], 9: []}

    def test_no_candidates_means_every_later_instance(self):
        objs = [SimpleNamespace(id=i) for i in (1, 2, 3)]
        partners = {a.id: [b.id for b in bs] for a, bs in eager_partners(objs, None)}
        assert partners == {1: [2, 3], 2: [3], 3: []}


class TestGuardFallsBackToBruteForce:
    def test_no_voxel_no_candidates(self):
        strat = FusionStrategy([CentroidDistanceCriterion(1.5)])
        assert strat.candidate_pairs() is None

    def test_voxel_plus_overlap_old_is_not_lossless_so_no_candidates(self):
        # overlap_old accepts on point proximity, not cell co-occupancy -> broadphase unsafe.
        strat = FusionStrategy([PointOverlapOldCriterion(0.1), VoxelOverlapCriterion(VoxelGrid(0.05))])
        assert strat.candidate_pairs() is None
