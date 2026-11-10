"""The part of OVO's state a fusion reads and writes, as plain CPU data.

A fusion (`OVO.update_map`) works on the 3D instances and on the per-keyframe descriptors it
recomputes them from. It needs no model, no images and no instance maps, so this is much
smaller than `OVO.capture_dict(debug_info=True)`.

    {"objects":     {ins_id: {"clip_feature", "clip_feature_kf", "kfs_ids", "points_ids",
                              "top_kf", "to_update", "created_at_frame"}},
     "descriptors": {kf_index: {ins_id: tensor}},
     "frame_ids":   [frame of semantic keyframe 0, 1, ...]}
"""

from typing import Any

import torch

from ..entities.instance3d import Instance3D
from ..entities.ovo import OVO


def capture(ovo: OVO) -> dict[str, Any]:
    return {
        "objects": {ins_id: _export(obj) for ins_id, obj in ovo.objects.items()},
        "descriptors": {kf: {ins_id: d.detach().cpu() for ins_id, d in by_ins.items()}
                        for kf, by_ins in ovo.keyframes["ins_descriptors"].items()},
        "frame_ids": list(ovo.keyframes["frame_id"]),
    }


def restore(ovo: OVO, state: dict[str, Any]) -> None:
    ovo.objects = {ins_id: _import(ins_id, fields) for ins_id, fields in state["objects"].items()}
    # On CPU, as OVO keeps them while it runs (CLIP embeddings are moved to CPU when stored).
    ovo.keyframes["ins_descriptors"] = {kf: dict(by_ins) for kf, by_ins in state["descriptors"].items()}
    ovo.keyframes["frame_id"] = list(state["frame_ids"])


def _export(obj: Instance3D) -> dict[str, Any]:
    return {
        "clip_feature": _cpu(obj.clip_feature),
        "clip_feature_kf": obj.clip_feature_kf,
        "kfs_ids": list(obj.kfs_ids),
        "points_ids": list(obj.points_ids),
        "top_kf": list(obj.top_kf),
        "to_update": obj.to_update,
        "created_at_frame": obj.created_at_frame,
    }


def _import(ins_id: int, fields: dict[str, Any]) -> Instance3D:
    obj = Instance3D(ins_id, created_at_frame=fields["created_at_frame"])
    obj.clip_feature = fields["clip_feature"]
    obj.clip_feature_kf = fields["clip_feature_kf"]
    obj.kfs_ids = list(fields["kfs_ids"])
    obj.points_ids = list(fields["points_ids"])
    obj.top_kf = list(fields["top_kf"])
    obj.to_update = fields["to_update"]
    return obj


def _cpu(value):
    return value.detach().cpu() if isinstance(value, torch.Tensor) else value
