"""Instance3D.created_at_frame: set by OVO when its segmentation creates an instance, kept by
export/restore (ovo_map.ckpt).

The OVO is OVO.for_fusion (no CLIP, no SAM); the camera geometry of _match_and_track_instances is
faked, so the tracking runs on CPU over a hand-made match.
"""

import numpy as np
import pytest
import torch

from ovo.entities import ovo as ovo_module
from ovo.entities.instance3d import Instance3D
from ovo.entities.ovo import OVO


def _ovo() -> OVO:
    ovo = OVO.for_fusion({"clip": {}, "sam": {}, "verbose": False, "track_th": 2, "match_distance_th": 0.1},
                         device="cpu")
    ovo.objects[0] = Instance3D(0, kf_id=0, points_ids=[0, 1, 2, 3, 4], mask_area=5, created_at_frame=3)
    ovo.next_ins_id = 1
    return ovo


def _frame():
    """10 points in view: 0..4 already belong to instance 0, 5..9 to none. Mask 0 covers the
    first five, mask 1 the other five."""
    seg_map = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    binary_maps = torch.stack([seg_map == 0, seg_map == 1])
    points_ins_ids = torch.tensor([0] * 5 + [-1] * 5)
    return seg_map, binary_maps, points_ins_ids


def test_created_at_frame_defaults_to_none():
    assert Instance3D(1).created_at_frame is None
    assert Instance3D(1, 0, [1, 2], 10).created_at_frame is None


def test_created_at_frame_is_keyword_only():
    with pytest.raises(TypeError):
        Instance3D(1, 0, [1, 2], 10, 42)
    assert Instance3D(1, 0, [1, 2], 10, created_at_frame=42).created_at_frame == 42


@pytest.mark.parametrize("debug_info", [False, True])
@pytest.mark.parametrize("frame", [0, 42, None])
def test_export_restore_round_trip(debug_info, frame):
    obj = Instance3D(7, 1, [3, 4], 20, created_at_frame=frame)
    obj.clip_feature = torch.ones(3)
    exported = obj.export(debug_info)
    assert exported["ins3d_7_created_at_frame"] == frame
    restored = Instance3D(7)
    restored.restore(exported, debug_info)
    assert restored.created_at_frame == frame


def test_restore_of_an_old_dict_without_the_key_gives_none():
    restored = Instance3D(7, created_at_frame=99)
    restored.restore({"ins3d_7_clip_feature": torch.ones(3), "ins3d_7_clip_feature_kf": 0}, debug_info=False)
    assert restored.created_at_frame is None


def test_capture_and_restore_dict_keep_it():
    ovo = _ovo()
    ovo.objects[0].clip_feature = torch.ones(3)
    ovo.objects[5] = Instance3D(5, created_at_frame=None)
    ovo.objects[5].clip_feature = torch.zeros(3)
    other = OVO.for_fusion({"clip": {}, "verbose": False}, device="cpu")
    other.restore_dict(ovo.capture_dict(debug_info=False), debug_info=False)
    assert {int(i): o.created_at_frame for i, o in other.objects.items()} == {0: 3, 5: None}


def test_track_objects_stamps_new_instances_with_the_frame():
    ovo = _ovo()
    seg_map, _, points_ins_ids = _frame()
    ids, info = ovo._track_objects(torch.arange(10), points_ins_ids, torch.arange(10),
                                   seg_map.reshape(-1), seg_map, track_th=2, kf_id=4, frame_id=42)
    assert sorted(ovo.objects) == [0, 1]
    assert ovo.objects[1].created_at_frame == 42
    assert ovo.objects[0].created_at_frame == 3  # tracked again, not created: unchanged
    assert ids.tolist() == [0] * 5 + [1] * 5
    assert ovo.objects[1].points_ids == [5, 6, 7, 8, 9]


def test_match_and_track_passes_the_frame_to_the_instances_it_creates(monkeypatch):
    ovo = _ovo()
    ovo.cam_intrinsics = torch.eye(3)
    seg_map, binary_maps, points_ins_ids = _frame()
    pixels = torch.tensor([[x, y] for y in range(2) for x in range(5)])  # point i -> pixel (x, y)
    monkeypatch.setattr(ovo_module.geometry_utils, "compute_camera_frustum_corners", lambda *a: None)
    monkeypatch.setattr(ovo_module.geometry_utils, "compute_frustum_point_ids",
                        lambda points, corners, device: torch.ones(len(points), dtype=torch.bool))
    monkeypatch.setattr(ovo_module.geometry_utils, "match_3d_points_to_2d_pixels",
                        lambda depth, w2c, points, K, th: (torch.arange(len(points)), pixels.clone()))
    map_data = (torch.zeros(10, 3), torch.arange(10), points_ins_ids)
    frame_data = (np.zeros((2, 5, 3)), np.ones((2, 5), dtype=np.float32), ())

    ins_ids, _, n_matched, updated = ovo._match_and_track_instances(frame_data, map_data, torch.eye(4), seg_map,
                                                                    binary_maps, 77)

    assert (ins_ids, n_matched) == ([0, 1], 10)
    assert updated.tolist() == [0] * 5 + [1] * 5
    assert ovo.objects[1].created_at_frame == 77


def test_detect_and_track_hands_its_frame_id_down(monkeypatch):
    ovo = _ovo()
    seg_map, binary_maps, _ = _frame()

    class _Masks:
        def get_masks(self, image, frame_id):
            return seg_map, binary_maps

    seen = []

    def match(frame_data, map_data, c2w, seg, maps, frame_id):
        seen.append(frame_id)
        return [], maps, 0, map_data[2]

    ovo.mask_generator = _Masks()
    monkeypatch.setattr(ovo, "_match_and_track_instances", match)
    ovo.detect_and_track_objects((31, np.zeros((2, 5, 3)), np.ones((2, 5)), ()), (None, None, torch.zeros(1)),
                                 torch.eye(4))
    assert seen == [31]
