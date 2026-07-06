from typing import Any, Dict, List, Optional
import torch

from ...utils import geometry_utils


class JumpDriftController:
    """Owns jump-drift configuration and the accumulated jump offset.

    Jump drift applies one or more large, discrete pose jumps at configurable
    keyframe indices, creating sharp duplicated instances rather than noisy point
    clouds.
    """

    def __init__(self, noise_config: Dict[str, Any], device: str) -> None:
        self.device = device
        self.enabled = noise_config.get("jump_drift_enabled", False)
        self._offset = torch.eye(4, device=device)
        self._applied_triggers: set = set()
        self.jump_configs: List[Dict[str, Any]] = []
        self.fired_events: List[Dict[str, Any]] = []

        if not self.enabled:
            return

        self.seed = noise_config.get("jump_seed", 42)
        self.generator = torch.Generator(device=device).manual_seed(self.seed)
        for jump_entry in noise_config.get("jumps", []):
            self.jump_configs.append(self._parse_jump(jump_entry))

        print(f"Jump drift enabled: {len(self.jump_configs)} jumps configured, seed={self.seed}")

    def _parse_jump(self, jump_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a config entry into an explicit translation vector and rotation matrix."""
        kf_index = int(jump_entry.get("kf_index", -1))
        frame_id = jump_entry.get("frame_id", None)
        forward = jump_entry.get("forward", False)
        forward_mag = 0.0

        if "translation" in jump_entry:
            translation = torch.tensor(jump_entry["translation"], dtype=torch.float32, device=self.device)
        elif "translation_magnitude" in jump_entry:
            mag = float(jump_entry["translation_magnitude"])
            if forward:
                translation = torch.zeros(3, device=self.device)
                forward_mag = mag
            else:
                rand_vec = torch.randn(3, generator=self.generator, device=self.device)
                rand_vec = rand_vec / torch.norm(rand_vec)
                translation = rand_vec * mag
        else:
            translation = torch.zeros(3, device=self.device)

        is_local_rotation = False
        if "rotation_jump" in jump_entry:
            is_local_rotation = True
            ypr_cfg = jump_entry["rotation_jump"]
            deg2rad = torch.pi / 180.0
            yaw_deg = self._sample_axis_deg(ypr_cfg.get("yaw"))
            pitch_deg = self._sample_axis_deg(ypr_cfg.get("pitch"))
            roll_deg = self._sample_axis_deg(ypr_cfg.get("roll"))
            rotation_matrix = geometry_utils.rotation_matrix_from_ypr(
                yaw_deg * deg2rad, pitch_deg * deg2rad, roll_deg * deg2rad
            ).to(self.device)
        elif "rotation" in jump_entry:
            euler_rad = torch.tensor(jump_entry["rotation"], dtype=torch.float32, device=self.device) * (torch.pi / 180.0)
            rotation_matrix = geometry_utils.rodrigues_rotation_matrix(euler_rad)
        elif "rotation_magnitude" in jump_entry:
            mag_rad = float(jump_entry["rotation_magnitude"]) * (torch.pi / 180.0)
            rand_axis = torch.randn(3, generator=self.generator, device=self.device)
            rand_axis = rand_axis / torch.norm(rand_axis)
            rotation_matrix = geometry_utils.rodrigues_rotation_matrix(rand_axis * mag_rad)
        else:
            rotation_matrix = torch.eye(3, device=self.device)

        result = {
            "kf_index": kf_index,
            "translation": translation,
            "rotation_matrix": rotation_matrix,
            "is_local_rotation": is_local_rotation,
            "forward": forward,
            "forward_mag": forward_mag,
        }
        if frame_id is not None:
            result["frame_id"] = frame_id
        return result

    def _sample_axis_deg(self, axis_cfg: Optional[Dict[str, Any]]) -> float:
        """Sample one yaw/pitch/roll axis in degrees; 0 if missing or disabled."""
        if not axis_cfg or not axis_cfg.get("enabled", False):
            return 0.0
        std_deg = float(axis_cfg["std_deg"])
        max_deg = float(axis_cfg.get("max_deg", std_deg * 4))
        return geometry_utils.sample_truncated_normal_deg(std_deg, max_deg, self.generator, self.device)

    @property
    def offset(self) -> torch.Tensor:
        return self._offset

    def maybe_trigger(self, kf_count: int, frame_id: int, gt_pose: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]:
        """Apply any jump whose trigger condition matches the current frame to the
        accumulated offset (once). Returns a telemetry event per jump applied (empty if none).

        A jump can be triggered by ``frame_id`` (preferred) or ``kf_index``
        (backward-compatible fallback). When ``forward: true``, the translation
        is applied along the camera's forward (-Z) direction at the jump frame
        instead of a fixed world-space vector; ``gt_pose`` must be provided.
        """
        events: List[Dict[str, Any]] = []
        if not self.enabled:
            return events

        for cfg in self.jump_configs:
            if "frame_id" in cfg:
                trigger_id = cfg["frame_id"]
                current_id = frame_id
                match = current_id >= trigger_id  # first KF at or after the target frame
            else:
                trigger_id = cfg["kf_index"]
                current_id = kf_count
                match = current_id == trigger_id

            if match and trigger_id not in self._applied_triggers:
                translation = cfg["translation"]
                needs_current_pose = cfg.get("forward", False) or cfg.get("is_local_rotation", False)
                current_pose = self._offset @ gt_pose if (needs_current_pose and gt_pose is not None) else None

                if cfg.get("forward", False) and current_pose is not None:
                    forward_dir = current_pose[:3, :3] @ torch.tensor([0, 0, -1.0], device=self.device)
                    translation = forward_dir * cfg["forward_mag"]

                rotation_matrix = cfg["rotation_matrix"]
                if cfg.get("is_local_rotation", False) and current_pose is not None:
                    R_cur = current_pose[:3, :3]
                    rotation_matrix = R_cur @ rotation_matrix @ R_cur.T

                T_jump = geometry_utils.create_transformation_matrix(rotation_matrix, translation)
                self._offset = T_jump @ self._offset
                self._applied_triggers.add(trigger_id)

                t_mag = torch.norm(translation).item()
                R = cfg["rotation_matrix"]
                angle_rad = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1.0, 1.0))
                event = {
                    "kf_index": kf_count,
                    "frame_id": frame_id,
                    "translation_magnitude": t_mag,
                    "rotation_magnitude": (angle_rad * 180 / torch.pi).item(),
                }
                self.fired_events.append(event)
                events.append(event)
        return events

    def reset(self) -> None:
        self._offset = torch.eye(4, device=self.device)
        self._applied_triggers.clear()
