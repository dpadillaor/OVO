"""fuse_instances (ovo.utils.instance_utils): how one merge from the fusion pipeline is applied."""

import pytest
import torch

from ovo.utils.instance_utils import fuse_instances


class _RecordingInstance:
    """Records what fuse_instances hands to the survivor (Instance3D needs torchvision, not in CI)."""

    def __init__(self, ins_id, points_ids=(), kfs_ids=(), top_kf=(), created_at_frame=None):
        self.id = ins_id
        self.created_at_frame = created_at_frame
        self.points_ids = list(points_ids)
        self.kfs_ids = list(kfs_ids)
        self.top_kf = list(top_kf)
        self.calls = []

    def add_points_ids(self, points_ids):
        self.calls.append(("points", list(points_ids)))

    def add_keyframes(self, kf_id):
        self.calls.append(("kf", kf_id))

    def add_top_kf(self, kf_id, area):
        self.calls.append(("top_kf", kf_id, area))


def test_survivor_takes_the_absorbed_points_keyframes_and_top_keyframes():
    survivor = _RecordingInstance(1)
    absorbed = _RecordingInstance(2, points_ids=[4, 5], kfs_ids=[10, 20], top_kf=[(300, 10), (50, 20)])
    map_data = (torch.zeros(6, 3), torch.arange(6), torch.tensor([1, 1, 3, 2, 2, 2]))

    result, points_ins_ids = fuse_instances(survivor, absorbed, map_data)

    assert result is survivor
    assert survivor.calls == [("points", [4, 5]), ("kf", 10), ("kf", 20), ("top_kf", 10, 300), ("top_kf", 20, 50)]
    assert points_ins_ids.tolist() == [1, 1, 3, 1, 1, 1]   # absorbed points relabelled, others kept
    assert points_ins_ids is map_data[2]                    # relabelled in place


@pytest.mark.parametrize("survivor_frame, absorbed_frame, expected", [
    (40, 10, 10),       # the absorbed instance is older: the survivor takes its frame
    (10, 40, 10),       # the survivor is older: it keeps its own
    (25, 25, 25),
    (None, 30, 30),     # unknown (an old ovo_map.ckpt) is ignored
    (30, None, 30),
    (None, None, None),
])
def test_survivor_keeps_the_earliest_known_creation_frame(survivor_frame, absorbed_frame, expected):
    survivor = _RecordingInstance(1, created_at_frame=survivor_frame)
    absorbed = _RecordingInstance(2, points_ids=[1], created_at_frame=absorbed_frame)
    fuse_instances(survivor, absorbed, (torch.zeros(2, 3), torch.arange(2), torch.tensor([1, 2])))
    assert survivor.created_at_frame == expected
    assert absorbed.created_at_frame == absorbed_frame  # the absorbed instance is left as it was
