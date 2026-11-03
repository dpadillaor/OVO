"""OVO.update_map and OVO._fuse_instances (ovo.entities.ovo) on a small hand-made map.

The OVO is built without its constructor (which loads CLIP and SAM): only the attributes these
methods read are set. The fusion pipeline and instance_utils.fuse_instances are the real ones.

Scene, one point per voxel of the default 5 cm grid, plus two background points (id -1):
    10  cells 0..9    class x   survivor
    20  cells 3..9    class x   inside 10: strong overlap -> absorbed by 10
    30  cells 5..9    class y   inside 10, other class -> rejected on cos_sim
    40  cells 9..28   class x   shares one cell with 10 (overlap 0.1) -> rejected on overlap
    50  no points             in the instances, not in the map -> removed
"""

from collections import deque

import pytest
import torch

from fakes import cells, line
from ovo.entities.fusion import FusionCfg, FusionPipeline
from ovo.entities.instance3d import Instance3D
from ovo.entities.ovo import OVO

X = torch.tensor([[1.0, 0.0, 0.0]])
Y = torch.tensor([[0.0, 1.0, 0.0]])
SPANS = {10: (0, 10), 20: (3, 10), 30: (5, 10), 40: (9, 29)}
FEATURES = {10: X, 20: X, 30: Y, 40: X, 50: X}
KF_OF = {10: 0, 20: 1, 30: 0, 40: 2, 50: 3}  # keyframe where each instance was seen


def build_scene():
    """Returns (ovo, map_data, descriptors): descriptors[ins_id] is what its keyframe stores."""
    clouds, ins_ids, objects = [cells((-20, 0, 0), (-21, 0, 0))], [-1, -1], {}
    n_points = 2
    for ins_id, (start, stop) in SPANS.items():
        clouds.append(line(start, stop))
        ins_ids += [ins_id] * (stop - start)
        objects[ins_id] = Instance3D(ins_id, KF_OF[ins_id], list(range(n_points, n_points + stop - start)), 100)
        n_points += stop - start
    objects[50] = Instance3D(50, KF_OF[50], [], 100)
    for ins_id, instance in objects.items():
        instance.clip_feature = FEATURES[ins_id]

    points_3d = torch.cat(clouds)
    map_data = (points_3d, torch.arange(len(points_3d)), torch.tensor(ins_ids))

    descriptors = {ins_id: torch.full((3,), float(ins_id)) for ins_id in objects}
    ins_descriptors = {}
    for ins_id, kf in KF_OF.items():
        ins_descriptors.setdefault(kf, {})[ins_id] = descriptors[ins_id]

    ovo = OVO.__new__(OVO)
    ovo.fusion = FusionPipeline(FusionCfg())
    ovo.objects = objects
    ovo.keyframes = {"ins_descriptors": ins_descriptors, "frame_id": [0, 1, 2, 3], "ins_maps": []}
    ovo.keyframes_queue = deque()  # nothing pending: complete_semantic_info runs no CLIP
    return ovo, map_data, descriptors


@pytest.fixture
def scene(monkeypatch):
    ovo, map_data, descriptors = build_scene()
    calls = []
    # Recomputing descriptors (mv_fusion over keyframe descriptors) is not under test here.
    monkeypatch.setattr(ovo, "update_objects_clip", lambda: calls.append(sorted(ovo.objects)))
    return ovo, map_data, descriptors, calls


def decided(report):
    return {(d.first, d.second): d for d in report.decisions}


def test_update_map_returns_ids_report_and_time(scene):
    ovo, map_data, _, _ = scene
    points_ins_ids, report, seconds = ovo.update_map(map_data, kfs=[0, 1, 2, 3])
    assert isinstance(points_ins_ids, torch.Tensor)
    assert points_ins_ids.shape == map_data[2].shape
    assert seconds >= report.seconds >= 0


def test_update_map_merges_one_pair_and_rejects_the_others(scene):
    ovo, map_data, _, _ = scene
    _, report, _ = ovo.update_map(map_data, kfs=[0, 1, 2, 3])

    assert report.merges == [(10, 20)]
    decisions = decided(report)
    assert {pair for pair, d in decisions.items() if d.accepted} == {(10, 20)}
    assert decisions[(10, 20)].mode in ("strong", "both")
    assert decisions[(10, 30)].reason == "cos_sim" and not decisions[(10, 30)].accepted
    assert decisions[(10, 40)].reason == "overlap" and decisions[(10, 40)].mode is None
    assert decisions[(10, 40)].overlap == pytest.approx(0.1)
    assert all(20 not in pair for pair in decisions if pair != (10, 20))  # absorbed: never judged again
    assert all(50 not in pair for pair in decisions)  # not in the map: never judged


def test_update_map_drops_absorbed_and_removed_instances(scene):
    ovo, map_data, _, calls = scene
    before = map_data[2].clone()
    points_ins_ids, _, _ = ovo.update_map(map_data, kfs=[0, 1, 2, 3])

    assert sorted(ovo.objects) == [10, 30, 40]
    assert 20 not in points_ins_ids
    assert torch.equal(points_ins_ids[before == 20], torch.full(((before == 20).sum(),), 10))
    unchanged = before != 20
    assert torch.equal(points_ins_ids[unchanged], before[unchanged])
    assert calls == [[10, 30, 40]]  # descriptors refreshed once, on the surviving instances


def test_update_map_survivor_absorbs_points_and_keyframes(scene):
    ovo, map_data, _, _ = scene
    absorbed_points = list(ovo.objects[20].points_ids)
    ovo.update_map(map_data, kfs=[0, 1, 2, 3])

    survivor = ovo.objects[10]
    assert set(absorbed_points) <= set(survivor.points_ids)
    assert set(survivor.kfs_ids) == {0, 1}


def test_update_map_moves_descriptors_of_absorbed_to_survivor(scene):
    ovo, map_data, descriptors, _ = scene
    ovo.update_map(map_data, kfs=[0, 1, 2, 3])

    ins_descriptors = ovo.keyframes["ins_descriptors"]
    assert all(20 not in by_ins for by_ins in ins_descriptors.values())
    assert ins_descriptors[1] == {10: descriptors[20]}  # keyframe 1 only saw 20
    assert ins_descriptors[0] == {10: descriptors[10], 30: descriptors[30]}  # untouched
    assert ins_descriptors[2] == {40: descriptors[40]}


def test_update_map_forgets_deleted_keyframes(scene):
    ovo, map_data, _, _ = scene
    ovo.update_map(map_data, kfs=[0, 1, 2])

    assert 3 not in ovo.keyframes["ins_descriptors"]
    assert ovo.keyframes["frame_id"] == [0, 1, 2, "Deleted"]


def test_fuse_instances_applies_the_pipeline_merges(scene):
    ovo, map_data, _, _ = scene
    in_map = [ovo.objects[i] for i in (10, 20, 30, 40)]
    objects, fused, points_ins_ids, report, seconds = ovo._fuse_instances(in_map, map_data)

    assert fused == {20: 10}
    assert fused == {absorbed: survivor for survivor, absorbed in report.merges}
    assert sorted(objects) == [10, 30, 40]
    assert all(objects[i] is ovo.objects[i] for i in objects)
    assert 20 not in points_ins_ids
    assert seconds >= report.seconds >= 0


def test_fuse_instances_without_merges_changes_nothing(scene):
    ovo, map_data, _, _ = scene
    in_map = [ovo.objects[i] for i in (10, 30, 40)]
    before = map_data[2].clone()
    objects, fused, points_ins_ids, report, _ = ovo._fuse_instances(in_map, map_data)

    assert fused == {} and report.merges == []
    assert len(report.decisions) > 0
    assert sorted(objects) == [10, 30, 40]
    assert torch.equal(points_ins_ids, before)
