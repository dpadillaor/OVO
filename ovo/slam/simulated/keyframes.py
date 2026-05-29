from typing import Optional
import torch


class KeyframeSelector:
    """Decides whether the current frame should become a new keyframe based on
    translational/rotational distance from the last keyframe."""

    def __init__(self, dist_thresh: float, rot_thresh_deg: float) -> None:
        self.dist_thresh = dist_thresh
        self.rot_thresh_deg = rot_thresh_deg

    def is_new_keyframe(self, current_c2w: torch.Tensor, last_kf_pose: Optional[torch.Tensor]) -> bool:
        if last_kf_pose is None:
            return True

        dist = torch.norm(current_c2w[:3, 3] - last_kf_pose[:3, 3])

        rot_diff = current_c2w[:3, :3] @ last_kf_pose[:3, :3].T
        trace = torch.trace(rot_diff)
        angle_rad = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
        angle_deg = angle_rad * (180 / torch.pi)

        return dist > self.dist_thresh or angle_deg > self.rot_thresh_deg
