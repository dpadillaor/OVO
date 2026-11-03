"""Fusion pipeline (ovo.entities.fusion.pipeline): end to end on fake instances.

The default config must reproduce the historical fusion exactly (same merges, same order), so the
reference below is written independently of the fusion package, from the voxel broadphase.
"""

from collections import Counter

import pytest
import torch

from fakes import FakeInstance, feature, line, random_scene
from ovo.entities.fusion import FusionCfg, FusionPipeline
from ovo.entities.voxel import VoxelGrid, VoxelSet, overlapping_pairs

SEEDS = range(200)


def _cos(inst_a, inst_b):
    return torch.nn.functional.cosine_similarity(inst_a.clip_feature[0], inst_b.clip_feature[0], dim=0).item()


def _accepted_pairs(instances, pcds):
    """Every pair sharing a voxel that the historical rule accepts, in double-loop order."""
    grid = VoxelGrid(0.05)
    by_id = {inst.id: inst for inst in instances}
    sets = {inst.id: VoxelSet(pcds[inst.id], grid) for inst in instances}
    accepted = []
    for a, b, overlap in overlapping_pairs(sets):
        cos = _cos(by_id[a], by_id[b])
        if cos >= 0.81 and (overlap > 0.5 or (cos > 0.9 and overlap > 0.2)):
            accepted.append((a, b))
    return accepted


def historical_merges(instances, pcds):
    """The fusion loop before the pipeline: greedy over shared-voxel pairs, skip absorbed."""
    grid = VoxelGrid(0.05)
    by_id = {inst.id: inst for inst in instances}
    sets = {inst.id: VoxelSet(pcds[inst.id], grid) for inst in instances}
    absorbed, merges = set(), []
    for a, b, overlap in overlapping_pairs(sets):
        if a in absorbed or b in absorbed:
            continue
        cos = _cos(by_id[a], by_id[b])
        if cos < 0.81:
            continue
        if overlap > 0.5 or (cos > 0.9 and overlap > 0.2):
            absorbed.add(b)
            merges.append((a, b))
    return merges


def union_find_merges(instances, pcds):
    """Groups formed by all accepted pairs; each member merged into the group's earliest instance."""
    order = [inst.id for inst in instances]
    rank = {i: k for k, i in enumerate(order)}
    group = {i: {i} for i in order}
    for a, b in _accepted_pairs(instances, pcds):
        if group[a] is not group[b]:
            joined = group[a] | group[b]
            for i in joined:
                group[i] = joined
    merges = []
    for i in order:
        survivor = min(group[i], key=rank.get)
        if survivor != i:
            merges.append((survivor, i))
    return sorted(merges, key=lambda m: (rank[m[0]], rank[m[1]]))


def run(instances, pcds, **cfg):
    return FusionPipeline(FusionCfg.from_dict(cfg)).run(instances, pcds)


# ---------------------------------------------------------------- equivalence on random scenes

def test_default_config_reproduces_the_historical_fusion():
    total, modes, reasons, chains = 0, Counter(), Counter(), 0
    for seed in SEEDS:
        instances, pcds = random_scene(seed)
        report = run(instances, pcds)
        expected = historical_merges(instances, pcds)
        assert report.merges == expected, f"seed {seed}"
        total += len(expected)
        modes.update(d.mode for d in report.decisions if d.accepted)
        reasons.update(d.reason for d in report.decisions if not d.accepted)
        chains += report.merges != union_find_merges(instances, pcds)
    # the scenes must exercise the rule, not just agree on empty results
    assert total > 500
    assert min(modes["strong"], modes["weak"], modes["both"]) > 20
    assert reasons["cos_sim"] > 100 and reasons["overlap"] > 100
    assert chains > 20  # scenes where greedy and chained differ


def test_chained_equals_union_find_over_all_accepted_pairs():
    for seed in SEEDS:
        instances, pcds = random_scene(seed)
        assert run(instances, pcds, apply="chained").merges == union_find_merges(instances, pcds), f"seed {seed}"


def test_merges_are_applicable_in_order():
    """Every absorbed instance is absorbed once, never survives later, and survivors are not absorbed."""
    for seed in SEEDS:
        instances, pcds = random_scene(seed)
        for apply in ("greedy", "chained"):
            merges = run(instances, pcds, apply=apply).merges
            absorbed = [b for _, b in merges]
            assert len(absorbed) == len(set(absorbed))
            assert not set(absorbed) & {a for a, _ in merges}


# ---------------------------------------------------------------- decisions

def _scene():
    """1~2 strong (same feature, 2 inside 1), 1 vs 3 different feature, 1~4 tiny overlap,
    2 vs 3 overlapping (but 2 is absorbed first), 5 far from everything."""
    same, other = feature(1.0, 0.0), feature(0.0, 1.0)
    instances = [FakeInstance(1, same), FakeInstance(2, same), FakeInstance(3, other),
                 FakeInstance(4, same), FakeInstance(5, same)]
    pcds = {1: line(0, 10), 2: line(2, 6), 3: line(5, 9), 4: line(9, 30), 5: line(100, 110)}
    return instances, pcds


def test_decisions_record_reason_and_measured_values():
    report = run(*_scene())
    by_pair = {(d.first, d.second): d for d in report.decisions}
    assert list(by_pair) == [(1, 2), (1, 3), (1, 4)]   # double-loop order, judged pairs only

    strong = by_pair[(1, 2)]
    assert strong.accepted and strong.reason == "overlap" and strong.mode == "both"
    assert strong.cos_sim == pytest.approx(1.0) and strong.overlap == 1.0

    cos = by_pair[(1, 3)]
    assert not cos.accepted and cos.reason == "cos_sim"
    assert cos.cos_sim == pytest.approx(0.0)
    assert cos.overlap is None and cos.mode is None  # overlap not measured once semantics reject

    small = by_pair[(1, 4)]
    assert not small.accepted and small.reason == "overlap" and small.mode is None
    assert small.overlap == 0.1                       # 1 shared cell / 10

    assert all(d.gate_values == {} for d in report.decisions)
    assert report.merges == [(1, 2)]
    assert report.seconds >= 0.0


def test_pairs_skipped_by_the_policy_are_absent():
    instances, pcds = _scene()
    judged = {(d.first, d.second) for d in run(instances, pcds).decisions}
    assert (2, 3) not in judged and (2, 4) not in judged   # 2 was absorbed by 1
    assert (1, 5) not in judged                            # no shared voxel: never a candidate


def test_point_overlap_is_measured_only_for_pairs_that_reach_it():
    instances, pcds = _scene()
    pipe = FusionPipeline(FusionCfg.from_dict({"overlap": {"method": "point"}}))
    measured = []
    original = pipe.overlap.measure
    pipe.overlap.measure = lambda a, b: measured.append((a, b)) or original(a, b)

    report = pipe.run(instances, pcds)
    reached = [(d.first, d.second) for d in report.decisions if d.overlap is not None]
    assert measured == reached
    judged = {(d.first, d.second) for d in report.decisions}
    assert (2, 3) not in judged and (2, 3) not in measured  # skipped: 2 already absorbed
    assert (1, 3) in judged and (1, 3) not in measured      # rejected by cos_sim first
    assert (1, 5) in measured                                # point has no broadphase


def test_point_overlap_skips_measure_on_gate_veto():
    instances, pcds = _scene()
    pipe = FusionPipeline(FusionCfg.from_dict({"overlap": {"method": "point"},
                                               "gates": [{"centroid": {"max_dist": 1.0}}]}))
    measured = []
    original = pipe.overlap.measure
    pipe.overlap.measure = lambda a, b: measured.append((a, b)) or original(a, b)

    report = pipe.run(instances, pcds)
    vetoed = [(d.first, d.second) for d in report.decisions if d.reason == "centroid"]
    assert (1, 5) in vetoed
    assert not set(vetoed) & set(measured)


def test_centroid_gate_vetoes_and_records_the_distance():
    # 2 lies inside 1 (overlap 1), but 1 is a long object whose centroid is far from 2
    same = feature(1.0, 0.0)
    instances = [FakeInstance(1, same), FakeInstance(2, same)]
    pcds = {1: line(0, 100), 2: line(0, 2)}
    assert run(instances, pcds).merges == [(1, 2)]

    report = run(instances, pcds, gates=[{"centroid": {"max_dist": 1.0}}])
    (d,) = report.decisions
    assert report.merges == []
    assert not d.accepted and d.reason == "centroid"
    assert d.gate_values["centroid"] == pytest.approx(2.45)   # centroids at x = 2.525 and 0.075
    assert d.cos_sim is None and d.overlap is None

    passed = run(instances, pcds, gates=[{"centroid": {"max_dist": 3.0}}])
    (d,) = passed.decisions
    assert d.accepted and d.gate_values["centroid"] == pytest.approx(2.45)


def test_feature_attr_is_respected():
    instances = [FakeInstance(1, feature(1.0, 0.0)), FakeInstance(2, feature(0.0, 1.0))]
    for inst in instances:
        inst.dino_feature = feature(1.0, 1.0)
    pcds = {1: line(0, 4), 2: line(0, 4)}
    assert FusionPipeline(FusionCfg()).run(instances, pcds).merges == []
    assert FusionPipeline(FusionCfg(), feature_attr="dino_feature").run(instances, pcds).merges == [(1, 2)]


def test_bad_apply_policy():
    with pytest.raises(ValueError, match=r"instance_fusion\.apply.*'eager'"):
        FusionPipeline(FusionCfg(apply="eager"))
