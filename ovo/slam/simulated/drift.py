"""Drift: a small random error added to every tracked frame, which accumulates like the
odometry drift of a real SLAM.

YAML `slam.simulated.drift` (absent = no drift):
    translation_std: 0.01   # metres per frame, per axis
    rotation_std_deg: 0.5   # degrees per frame, per axis
    seed: 42
"""

import math
from dataclasses import dataclass

import torch

from . import se3


@dataclass(frozen=True)
class DriftCfg:
    translation_std: float = 0.01
    rotation_std_deg: float = 0.5
    seed: int = 42

    def __post_init__(self):
        if self.translation_std < 0 or self.rotation_std_deg < 0:
            raise ValueError(f"drift: std must be >= 0, got {self}")


class Drift:
    """Samples the per-frame error, in camera axes. Seeded on CPU, so the same seed gives the
    same trajectory on any device."""

    def __init__(self, cfg: DriftCfg) -> None:
        self.cfg = cfg
        self.generator = torch.Generator().manual_seed(cfg.seed)

    def step(self) -> torch.Tensor:
        """4x4 error of one frame: a random rotation vector and a random translation."""
        rotvec = torch.randn(3, generator=self.generator, dtype=torch.float64) * math.radians(self.cfg.rotation_std_deg)
        translation = torch.randn(3, generator=self.generator, dtype=torch.float64) * self.cfg.translation_std
        return se3.transform(se3.axis_angle(rotvec), translation)
