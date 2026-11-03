"""ovo.replay.state: capture and restore of what a fusion reads and writes."""

import io

import numpy as np
import torch

from ovo.entities.ovo import OVO
from ovo.replay import state
from replay_fakes import config, scene

FIELDS = ("clip_feature", "clip_feature_kf", "kfs_ids", "points_ids", "top_kf", "to_update")


def _equal(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    return a == b


def _through_disk(captured: dict) -> dict:
    buffer = io.BytesIO()
    torch.save(captured, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def _assert_same_state(ovo: OVO, other: OVO) -> None:
    assert list(ovo.objects) == list(other.objects)
    for ins_id, obj in ovo.objects.items():
        restored = other.objects[ins_id]
        assert restored.id == obj.id
        for field in FIELDS:
            assert _equal(getattr(restored, field), getattr(obj, field)), (ins_id, field)
    assert ovo.keyframes["frame_id"] == other.keyframes["frame_id"]
    descriptors, restored = ovo.keyframes["ins_descriptors"], other.keyframes["ins_descriptors"]
    assert {kf: set(d) for kf, d in descriptors.items()} == {kf: set(d) for kf, d in restored.items()}
    for kf, by_ins in descriptors.items():
        for ins_id, d in by_ins.items():
            assert torch.equal(restored[kf][ins_id], d)


def _restored(captured: dict) -> OVO:
    other = OVO.for_fusion(config()["semantic"], device="cpu")
    state.restore(other, captured)
    return other


def _varied_scene():
    """The scene with every kind of value a field takes while OVO runs."""
    ovo, map_data, _ = scene()
    ovo.objects[10].clip_feature_kf = torch.tensor(1)  # medoid index, from l1_medoid
    ovo.objects[20].clip_feature_kf = 0                # single view
    ovo.objects[30].clip_feature = None                # never described yet
    ovo.objects[30].to_update = True
    ovo.objects[40].top_kf = [(120, 2), (80, 5)]
    ovo.keyframes["frame_id"] = [0, 10, 20, 30]
    return ovo, map_data


def test_round_trip_through_disk_restores_every_field_and_descriptor():
    ovo, _ = _varied_scene()
    _assert_same_state(ovo, _restored(_through_disk(state.capture(ovo))))


def test_restored_state_is_independent_of_the_captured_ovo():
    ovo, _ = _varied_scene()
    other = _restored(state.capture(ovo))
    other.objects[10].points_ids.append(999)
    other.keyframes["ins_descriptors"][0].pop(10)
    other.keyframes["frame_id"].append(40)
    assert 999 not in ovo.objects[10].points_ids
    assert 10 in ovo.keyframes["ins_descriptors"][0]
    assert ovo.keyframes["frame_id"] == [0, 10, 20, 30]


def test_capture_is_plain_cpu_data():
    ovo, _ = _varied_scene()
    captured = state.capture(ovo)
    assert set(captured) == {"objects", "descriptors", "frame_ids"}
    assert set(captured["objects"][10]) == set(FIELDS)
    tensors = [o["clip_feature"] for o in captured["objects"].values() if o["clip_feature"] is not None]
    tensors += [d for by_ins in captured["descriptors"].values() for d in by_ins.values()]
    assert tensors and all(t.device.type == "cpu" for t in tensors)


def test_descriptors_are_restored_on_cpu_whatever_the_device():
    # While OVO runs its descriptors are on CPU; a replay on cuda must not move them.
    ovo, _ = _varied_scene()
    other = OVO.for_fusion(config()["semantic"], device="cuda")
    state.restore(other, state.capture(ovo))
    assert all(d.device.type == "cpu" for by_ins in other.keyframes["ins_descriptors"].values()
               for d in by_ins.values())


def test_without_semantic_log_frame_ids_are_empty_and_the_fusion_still_runs():
    # `semantic.log: False` leaves keyframes["frame_id"] empty; a fusion does not read it.
    ovo, map_data, _ = scene()
    ovo.keyframes["frame_id"] = []
    other = _restored(_through_disk(state.capture(ovo)))
    assert other.keyframes["frame_id"] == []
    ids, report, _ = other.update_map(tuple(t.clone() for t in map_data))
    assert report.merges == [(10, 20)]


def test_numpy_ids_from_a_restored_map_round_trip_and_fuse_the_same():
    # restore_dict (run_eval, restore_map) gives numpy instance ids.
    ovo, map_data, _ = scene()
    ovo.objects = {np.int64(i): obj for i, obj in ovo.objects.items()}
    for obj in ovo.objects.values():
        obj.id = np.int64(obj.id)
    ovo.keyframes["ins_descriptors"] = {kf: {np.int64(i): d for i, d in by_ins.items()}
                                        for kf, by_ins in ovo.keyframes["ins_descriptors"].items()}
    other = _restored(_through_disk(state.capture(ovo)))
    _assert_same_state(ovo, other)

    reference, _, _ = scene()
    expected_ids, expected, _ = reference.update_map(tuple(t.clone() for t in map_data))
    ids, report, _ = other.update_map(tuple(t.clone() for t in map_data))
    assert torch.equal(ids, expected_ids)
    assert [(int(a), int(b)) for a, b in report.merges] == expected.merges
    assert sorted(int(i) for i in other.objects) == sorted(reference.objects)
