"""Rigid-transform helpers for the simulator. 4x4 float64 tensors on CPU.

Camera axes follow the datasets (OpenCV): x right, y down, z forward (where the camera looks).
"""

import math

import torch


def transform(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """4x4 rigid transform from a 3x3 rotation and a 3-vector."""
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def axis_angle(rotvec: torch.Tensor) -> torch.Tensor:
    """Rotation matrix of a rotation vector (axis * angle in radians), Rodrigues' formula."""
    rotvec = rotvec.to(torch.float64)
    angle = torch.linalg.norm(rotvec)
    if angle == 0:
        return torch.eye(3, dtype=torch.float64)
    x, y, z = (rotvec / angle).tolist()
    K = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)


def yaw_pitch_roll(yaw_deg: float, pitch_deg: float, roll_deg: float) -> torch.Tensor:
    """Rotation in camera axes: yaw turns the head left/right (about y), pitch nods (about x),
    roll tilts (about z). Applied yaw, then pitch, then roll."""
    yaw, pitch, roll = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    f64 = torch.float64  # float32 here would cost ~1e-7 deg
    return (axis_angle(torch.tensor([0.0, yaw, 0.0], dtype=f64))
            @ axis_angle(torch.tensor([pitch, 0.0, 0.0], dtype=f64))
            @ axis_angle(torch.tensor([0.0, 0.0, roll], dtype=f64)))


def angle_deg(rotation: torch.Tensor) -> float:
    """Rotation angle of a 3x3 rotation matrix, in degrees."""
    cos = ((torch.trace(rotation) - 1) / 2).clamp(-1.0, 1.0)
    return math.degrees(torch.acos(cos).item())


def random_unit(generator: torch.Generator) -> torch.Tensor:
    """Uniformly random 3D unit vector."""
    v = torch.randn(3, generator=generator, dtype=torch.float64)
    return v / torch.linalg.norm(v)
