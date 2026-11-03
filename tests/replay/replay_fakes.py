"""Shared by the replay tests: the small fusion scene of tests/fusion, on an OVO built for fusion.

The scene (tests/fusion/test_ovo_update_map.py): 10 absorbs 20, 30 is rejected on cos_sim, 40 on
overlap, 50 has no points and is removed. The OVO is a real `OVO.for_fusion`, so `update_map`
runs the whole step, descriptor recomputation included.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fusion"))
from test_ovo_update_map import build_scene  # noqa: E402

from ovo.entities.ovo import OVO  # noqa: E402

SCENE_NAME = "office0"


def config(instance_fusion: dict | None = None) -> dict:
    """The part of a run's config a replay reads."""
    return {
        "dataset_name": "Replica",
        "data": {"scene_name": SCENE_NAME},
        "semantic": {"clip": {}, "instance_fusion": dict(instance_fusion or {}), "verbose": False},
        "replay": {"checkpoints": "all"},
    }


def scene(cfg: dict | None = None):
    """(ovo, map_data, slam): an OVO with the scene's instances and descriptors, its map and a
    SLAM whose `get_map_dict` saves that map."""
    cfg = cfg or config()
    built, map_data, _ = build_scene()
    ovo = OVO.for_fusion(cfg["semantic"], device="cpu")
    ovo.objects = built.objects
    ovo.keyframes = built.keyframes
    return ovo, map_data, FakeSLAM(map_data)


class FakeSLAM:
    """`get_map_dict` of a vanilla map (copies, obj_ids as (N, 1)) over live map tensors."""

    def __init__(self, map_data):
        self.map_data = map_data

    def get_map_dict(self) -> dict:
        xyz, ids, obj_ids = self.map_data
        return {"xyz": xyz.clone(), "ids": ids.clone(), "obj_ids": obj_ids.clone().unsqueeze(-1),
                "max_id": len(ids), "color": torch.zeros((len(ids), 3), dtype=torch.uint8)}
