"""Jumps: a sudden pose error at a given frame, like a SLAM relocalising in the wrong place.
The surfaces seen after the jump are mapped displaced, as a copy of what was already there.

YAML `slam.simulated.jumps`, one entry per jump:
    - at_frame: 300              # first tracked frame >= 300
      axes: camera               # camera (default) | world: axes of `translation`
      translation: [0, 0, -0.1]  # metres the pose moves; in camera axes z is where it looks, so
                                 # this moves it back and the copy lands 10 cm closer to the camera
      rotation: [5, 0, 0]        # yaw, pitch, roll in degrees, about the camera centre
`translation` or `rotation` given as a single number means that magnitude in a random direction
(drawn with `slam.simulated.jump_seed`). Both default to no change.
"""

import math
from dataclasses import dataclass
from typing import Union

import torch

from . import se3

Vec3 = Union[float, int, list, tuple]
AXES = ("camera", "world")


@dataclass(frozen=True)
class JumpCfg:
    at_frame: int
    translation: Vec3 = 0.0
    rotation: Vec3 = 0.0
    axes: str = "camera"

    def __post_init__(self):
        if self.axes not in AXES:
            raise ValueError(f"jumps: axes must be one of {AXES}, got {self.axes!r}")
        if self.at_frame < 0:
            raise ValueError(f"jumps: at_frame must be >= 0, got {self.at_frame}")
        for name in ("translation", "rotation"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) and not (
                    isinstance(value, (list, tuple)) and len(value) == 3
                    and all(isinstance(v, (int, float)) for v in value)):
                raise ValueError(f"jumps: {name} must be a number or 3 numbers, got {value!r}")


@dataclass(frozen=True)
class Jump:
    """A jump with its random parts already drawn."""
    at_frame: int
    axes: str
    translation: torch.Tensor  # (3,) metres, in `axes`
    rotation: torch.Tensor     # (3, 3), camera axes

    def apply(self, pose: torch.Tensor) -> torch.Tensor:
        """Pose after the jump. The rotation turns the camera in place; the translation moves it
        along `axes`."""
        R = pose[:3, :3]
        t = R @ self.translation if self.axes == "camera" else self.translation
        return se3.transform(R @ self.rotation, pose[:3, 3] + t)

    def to_json(self) -> dict:
        return {"at_frame": self.at_frame, "axes": self.axes,
                "translation": self.translation.tolist(),
                "rotation_deg": round(se3.angle_deg(self.rotation), 6)}


def resolve(cfgs: list[JumpCfg], seed: int) -> list[Jump]:
    """Draw the random parts of every jump, in config order, so the same seed gives the same jumps."""
    generator = torch.Generator().manual_seed(seed)
    jumps = []
    for cfg in cfgs:
        if isinstance(cfg.translation, (int, float)):
            translation = se3.random_unit(generator) * cfg.translation
        else:
            translation = torch.tensor(cfg.translation, dtype=torch.float64)
        if isinstance(cfg.rotation, (int, float)):
            rotation = se3.axis_angle(se3.random_unit(generator) * math.radians(cfg.rotation))
        else:
            rotation = se3.yaw_pitch_roll(*cfg.rotation)
        jumps.append(Jump(cfg.at_frame, cfg.axes, translation, rotation))
    return sorted(jumps, key=lambda j: j.at_frame)
