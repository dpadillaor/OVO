"""Keyframe selection: a mapped frame becomes a keyframe (and adds points to the map) when the
camera has moved or turned enough since the last keyframe.

YAML `slam.simulated.keyframes`:
    dist: 0.1      # metres
    rot_deg: 5.0   # degrees
"""

from dataclasses import dataclass
from typing import Optional

import torch

from . import se3


@dataclass(frozen=True)
class KeyframeCfg:
    dist: float = 0.1
    rot_deg: float = 5.0

    def __post_init__(self):
        if self.dist < 0 or self.rot_deg < 0:
            raise ValueError(f"keyframes: thresholds must be >= 0, got {self}")


def is_new_keyframe(cfg: KeyframeCfg, pose: torch.Tensor, last_kf_pose: Optional[torch.Tensor]) -> bool:
    if last_kf_pose is None:
        return True
    dist = torch.linalg.norm(pose[:3, 3] - last_kf_pose[:3, 3]).item()
    rot = se3.angle_deg(pose[:3, :3] @ last_kf_pose[:3, :3].T)
    return dist > cfg.dist or rot > cfg.rot_deg
