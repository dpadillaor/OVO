from typing import Any, Dict, List
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
        self._applied_kf_indices: set = set()
        self.jump_configs: List[Dict[str, Any]] = []

        if not self.enabled:
            return

        self.seed = noise_config.get("jump_seed", 42)
        self.generator = torch.Generator(device=device).manual_seed(self.seed)
        for jump_entry in noise_config.get("jumps", []):
            self.jump_configs.append(self._parse_jump(jump_entry))

        print(f"Jump drift enabled: {len(self.jump_configs)} jumps configured, seed={self.seed}")

    def _parse_jump(self, jump_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a config entry into an explicit translation vector and rotation matrix."""
        kf_index = int(jump_entry["kf_index"])

        if "translation" in jump_entry:
            translation = torch.tensor(jump_entry["translation"], dtype=torch.float32, device=self.device)
        elif "translation_magnitude" in jump_entry:
            mag = float(jump_entry["translation_magnitude"])
            rand_vec = torch.randn(3, generator=self.generator, device=self.device)
            rand_vec = rand_vec / torch.norm(rand_vec)
            translation = rand_vec * mag
        else:
            translation = torch.zeros(3, device=self.device)

        if "rotation" in jump_entry:
            euler_rad = torch.tensor(jump_entry["rotation"], dtype=torch.float32, device=self.device) * (torch.pi / 180.0)
            rotation_matrix = geometry_utils.rodrigues_rotation_matrix(euler_rad)
        elif "rotation_magnitude" in jump_entry:
            mag_rad = float(jump_entry["rotation_magnitude"]) * (torch.pi / 180.0)
            rand_axis = torch.randn(3, generator=self.generator, device=self.device)
            rand_axis = rand_axis / torch.norm(rand_axis)
            rotation_matrix = geometry_utils.rodrigues_rotation_matrix(rand_axis * mag_rad)
        else:
            rotation_matrix = torch.eye(3, device=self.device)

        return {"kf_index": kf_index, "translation": translation, "rotation_matrix": rotation_matrix}

    @property
    def offset(self) -> torch.Tensor:
        return self._offset

    def maybe_trigger(self, kf_count: int) -> List[Dict[str, Any]]:
        """Apply any jump configured for keyframe index `kf_count` to the accumulated
        offset (once). Returns a telemetry event per jump applied (empty if none)."""
        events: List[Dict[str, Any]] = []
        if not self.enabled:
            return events

        for cfg in self.jump_configs:
            if cfg["kf_index"] == kf_count and kf_count not in self._applied_kf_indices:
                T_jump = geometry_utils.create_transformation_matrix(cfg["rotation_matrix"], cfg["translation"])
                self._offset = T_jump @ self._offset
                self._applied_kf_indices.add(kf_count)

                t_mag = torch.norm(cfg["translation"]).item()
                R = cfg["rotation_matrix"]
                angle_rad = torch.acos(torch.clamp((torch.trace(R) - 1) / 2, -1.0, 1.0))
                events.append({
                    "kf_index": kf_count,
                    "translation_magnitude": t_mag,
                    "rotation_magnitude": (angle_rad * 180 / torch.pi).item(),
                })
        return events

    def reset(self) -> None:
        self._offset = torch.eye(4, device=self.device)
