"""Dataclasses for the covisibility module.

KeyFrameData wraps the data already flowing through OVO so the graph can
reason about keyframes without depending on OVO internals.
"""
from dataclasses import dataclass

import torch


@dataclass
class KeyFrameData:
    kf_id: int
    frame_id: int
    c2w: torch.Tensor
    depth: torch.Tensor
    cam_intrinsics: torch.Tensor


@dataclass
class CovisibilityEdge:
    kf_id_a: int
    kf_id_b: int
    overlap_ratio: float


@dataclass
class CovisiblePair:
    instance_id_a: int
    instance_id_b: int
    kf_id_a: int
    kf_id_b: int
    overlap_ratio: float
