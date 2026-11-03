"""OVO.for_fusion: an OVO without models whose fusion works, and the same state as OVO()."""

from collections import deque

import pytest
import torch

from ovo.entities import ovo as ovo_module
from ovo.entities.fusion import FusionCfg
from ovo.entities.instance3d import Instance3D, l1_medoid
from ovo.entities.ovo import OVO


def _semantic(**extra) -> dict:
    return {"clip": {"embed_type": "vanilla"}, "sam": {}, "verbose": False, **extra}


class _FakeModel:
    def __init__(self, *args, **kwargs):
        self.args = args


@pytest.fixture
def full_ovo(monkeypatch):
    """OVO() with fake CLIP and SAM, to compare with OVO.for_fusion."""
    monkeypatch.setattr(ovo_module, "CLIPGenerator", _FakeModel)
    monkeypatch.setattr(ovo_module, "MaskGenerator", _FakeModel)

    def build(config):
        return OVO(config, logger="logger", scene_name="office0", cam_intrinsics=torch.eye(3), device="cpu")
    return build


def test_for_fusion_has_no_models_and_empty_state():
    ovo = OVO.for_fusion(_semantic(), device="cpu")
    assert ovo.clip_generator is None and ovo.mask_generator is None
    assert ovo.logger is None and ovo.cam_intrinsics is None
    assert ovo.device == "cpu"
    assert ovo.objects == {} and ovo.keyframes_queue == deque()
    assert ovo.keyframes == {"ins_descriptors": {}, "frame_id": [], "ins_maps": []}
    assert ovo.next_ins_id == 0 and ovo.kf_id == 0


def test_for_fusion_reads_instance_fusion_and_clip_settings():
    ovo = OVO.for_fusion(_semantic(instance_fusion={"accept": {"min_cos": 0.5}},
                                   clip={"k_top_views": 3, "fusion": "l1_medoid"}), device="cpu")
    assert ovo.fusion.cfg == FusionCfg.from_dict({"accept": {"min_cos": 0.5}})
    assert ovo.n_top_views == 3 and Instance3D.n_top_kf == 3
    assert Instance3D.mv_fusion is l1_medoid
    OVO.for_fusion(_semantic(), device="cpu")  # restore the class defaults for other tests
    assert Instance3D.n_top_kf == 0


def test_for_fusion_rejects_an_invalid_instance_fusion():
    with pytest.raises(ValueError, match="unknown key"):
        OVO.for_fusion(_semantic(instance_fusion={"nope": 1}), device="cpu")


def test_for_fusion_runs_update_map():
    # Two instances on the same points with the same descriptor fuse; descriptors are recomputed.
    ovo = OVO.for_fusion(_semantic(), device="cpu")
    feature = torch.tensor([1.0, 0.0, 0.0])
    for ins_id in (0, 1):
        ovo.objects[ins_id] = Instance3D(ins_id, kf_id=ins_id, points_ids=[0, 1, 2], mask_area=10)
        ovo.objects[ins_id].clip_feature = feature[None]
        ovo.keyframes["ins_descriptors"][ins_id] = {ins_id: feature}
    points = torch.tensor([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.1, 0.0, 0.0]] * 2)
    map_data = (points, torch.arange(6), torch.tensor([0, 0, 0, 1, 1, 1]))

    ids, report, _ = ovo.update_map(map_data)

    assert report.merges == [(0, 1)]
    assert torch.equal(ids, torch.zeros(6, dtype=torch.long))
    assert list(ovo.objects) == [0]
    assert torch.equal(ovo.objects[0].clip_feature.flatten(), feature)


def test_constructor_builds_the_same_state_plus_the_models(full_ovo):
    full = full_ovo(_semantic(instance_fusion={"accept": {"min_cos": 0.7}}, clip={"embed_type": "vanilla", "k_top_views": 2}))
    bare = OVO.for_fusion(full.config, device="cpu")

    assert isinstance(full.clip_generator, _FakeModel) and isinstance(full.mask_generator, _FakeModel)
    assert full.logger == "logger" and torch.equal(full.cam_intrinsics, torch.eye(3))
    assert full.config["sam"]["multi_crop"] is False  # set by the constructor only
    skip = {"clip_generator", "mask_generator", "logger", "cam_intrinsics", "fusion"}
    assert {k: v for k, v in vars(full).items() if k not in skip} == {k: v for k, v in vars(bare).items() if k not in skip}
    assert full.fusion.cfg == bare.fusion.cfg
    OVO.for_fusion(_semantic(), device="cpu")


def test_constructor_in_eval_mode_loads_no_sam(full_ovo):
    ovo = OVO(_semantic(), logger=None, eval=True, device="cpu")
    assert isinstance(ovo.clip_generator, _FakeModel) and ovo.mask_generator is None
