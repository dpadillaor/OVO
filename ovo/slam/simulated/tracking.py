from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING
import torch

from ...utils import geometry_utils

if TYPE_CHECKING:
    from .jump_drift import JumpDriftController


class TrackingStrategy:
    """Interface: compute the camera-to-world pose for a frame.

    `fallback_pose` is a zero-arg callable returning a pose to use when the
    frame is out of trajectory bounds.
    """

    def compute_pose(self, frame_id: int, fallback_pose) -> torch.Tensor:
        raise NotImplementedError


class GroundTruthTracking(TrackingStrategy):
    """Returns the ground-truth pose for the frame, unmodified."""

    def __init__(self, trajectory: List[torch.Tensor], device: str) -> None:
        self.trajectory = trajectory
        self.device = device

    def compute_pose(self, frame_id: int, fallback_pose) -> torch.Tensor:
        if frame_id < len(self.trajectory):
            return self.trajectory[frame_id].to(self.device)
        print(f"Warning: Frame ID {frame_id} is out of bounds for the trajectory of length {len(self.trajectory)}.")
        return fallback_pose()


class JumpTracking(TrackingStrategy):
    """Returns the ground-truth pose left-multiplied by the accumulated jump offset."""

    def __init__(self, trajectory: List[torch.Tensor], device: str, jump_controller: "JumpDriftController") -> None:
        self.trajectory = trajectory
        self.device = device
        self.jump_controller = jump_controller

    def compute_pose(self, frame_id: int, fallback_pose) -> torch.Tensor:
        if frame_id < len(self.trajectory):
            gt_pose = self.trajectory[frame_id].to(self.device)
            return self.jump_controller.offset @ gt_pose
        print(f"Warning: Frame ID {frame_id} is out of bounds for the trajectory of length {len(self.trajectory)}.")
        return fallback_pose()


class NoisyTracking(TrackingStrategy):
    """Accumulates seeded noise on top of the GT relative motion to simulate realistic drift."""

    def __init__(self, trajectory: List[torch.Tensor], device: str,
                 translation_noise_std: float, rotation_noise_std_rad: float, seed: int) -> None:
        self.trajectory = trajectory
        self.device = device
        self.translation_noise_std = translation_noise_std
        self.rotation_noise_std_rad = rotation_noise_std_rad
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.last_noisy_c2w: Optional[torch.Tensor] = None
        self.last_processed_frame_id = -1

    def compute_pose(self, frame_id: int, fallback_pose) -> torch.Tensor:
        if frame_id >= len(self.trajectory):
            print(f"Warning: Frame ID {frame_id} is out of bounds for the trajectory of length {len(self.trajectory)}.")
            return fallback_pose()

        current_gt_c2w = self.trajectory[frame_id].to(self.device)

        if self.last_processed_frame_id < 0:
            self.last_noisy_c2w = current_gt_c2w
            self.last_processed_frame_id = frame_id
            return current_gt_c2w

        # Ideal relative motion between the last processed and current GT poses.
        prev_gt_c2w = self.trajectory[self.last_processed_frame_id].to(self.device)
        real_relative_motion = geometry_utils.get_relative_pose(prev_gt_c2w, current_gt_c2w)

        # Seeded perturbation (axis-angle rotation + translation).
        translation_noise = torch.randn(3, device=self.device, generator=self.rng) * self.translation_noise_std
        rotation_noise = torch.randn(3, device=self.device, generator=self.rng) * self.rotation_noise_std_rad
        rotation_matrix = geometry_utils.rodrigues_rotation_matrix(rotation_noise)
        perturbation = geometry_utils.create_transformation_matrix(rotation_matrix, translation_noise)

        noisy_relative_motion = real_relative_motion @ perturbation
        c2w = self.last_noisy_c2w @ noisy_relative_motion
        self.last_noisy_c2w = c2w
        self.last_processed_frame_id = frame_id
        return c2w
