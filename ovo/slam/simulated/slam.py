"""SimulatedSLAM: a SLAM backbone driven by the dataset's GT poses, with controlled errors.

It maps exactly like VanillaMapper (unprojects depth, skips already-mapped surface), but the
pose it uses comes from PoseSimulator (GT + drift + jumps), and it simulates loop closures:
a correction moves every keyframe back to its GT pose, carrying its points along, and raises
`map_updated` so OVO re-fuses the map.

YAML: `slam.simulated` (see config.py) and `slam.close_loops`. Mapping parameters are Vanilla's.
What actually happened (jumps with their drawn values, corrections) is written to
`{scene}/simulation.json` (ovo.layout.simulation).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ... import layout
from ..vanilla_mapper import VanillaMapper
from .config import SimulatedCfg
from .drift import Drift
from .jumps import resolve
from .keyframes import is_new_keyframe
from .poses import PoseSimulator


class SimulatedSLAM(VanillaMapper):
    def __init__(self, config: Dict[str, Any], cam_intrinsics: torch.Tensor, n_frames: int) -> None:
        super().__init__(config, cam_intrinsics)
        self.cfg = SimulatedCfg.from_dict(config["slam"].get("simulated"))
        drift = Drift(self.cfg.drift) if self.cfg.drift is not None else None
        self.poses = PoseSimulator(drift, resolve(list(self.cfg.jumps), self.cfg.jump_seed))
        self._gt: Dict[int, torch.Tensor] = {}  # GT of every tracked frame, for corrections

        # Frames at which a correction is due: the configured ones plus the final one (the last
        # mapped frame falls in the last `map_every` window). After the final one, poses are GT.
        self._corrections_due: List[int] = []
        self._final_correction: Optional[int] = None
        if config["slam"].get("close_loops", True):
            self._final_correction = max(n_frames - config["mapping"].get("map_every", 10), 0)
            self._corrections_due = sorted({f for f in self.cfg.correction.at_frames
                                            if f < self._final_correction} | {self._final_correction})

        self.events: Dict[str, list] = {"jumps": [], "corrections": []}
        out = config.get("output_path")
        self._events_path: Optional[Path] = layout.simulation(Path(out)) if out else None
        self._write_events()

    def track_camera(self, frame_data: List[Any]) -> None:
        frame_id, gt_np = frame_data[0], np.asarray(frame_data[3])
        if not np.isfinite(gt_np).all():
            return
        gt = torch.from_numpy(gt_np.astype(np.float64))
        estimated, fired = self.poses.track(frame_id, gt)
        self._gt[frame_id] = gt
        self.estimated_c2ws[frame_id] = estimated.to(device=self.device, dtype=torch.float32)
        for jump in fired:
            self.events["jumps"].append({"frame": frame_id, **jump.to_json()})
        if fired:
            self._write_events()

    def map(self, frame_data: List[Any], c2w: torch.Tensor) -> None:
        frame_id = frame_data[0]
        last_kf_pose = self.kfs[next(reversed(self.kfs))]["pose"] if self.kfs else None
        if is_new_keyframe(self.cfg.keyframes, c2w, last_kf_pose):
            start = self.pcd.shape[0]
            super().map(frame_data, c2w)
            self.kfs[frame_id] = {"id": frame_id, "pcd_idxs": (start, self.pcd.shape[0]), "pose": c2w}
        self._maybe_correct(frame_id)

    def _maybe_correct(self, frame_id: int) -> None:
        """Run the correction if one is due and there is some error to correct."""
        due = final = False
        while self._corrections_due and self._corrections_due[0] <= frame_id:
            final = self._corrections_due.pop(0) == self._final_correction
            due = True
        if not due:
            return
        has_error = self.poses.has_error
        if final:
            self.poses.stop()
        if not has_error:
            return

        moved = self._snap_keyframes_to_gt()
        # In place: the caller still holds this frame's pose and uses it after map() (semantics).
        for f, c2w in self.estimated_c2ws.items():
            c2w.copy_(self._gt[f])
        if not final:
            self.poses.correct()
        self.map_updated = moved > 0
        self.events["corrections"].append({"frame": frame_id, "keyframes_moved": moved})
        self._write_events()

    def _snap_keyframes_to_gt(self) -> int:
        """Move every keyframe that is off its GT pose back onto it, with its points.
        Returns how many keyframes moved."""
        moved = 0
        for kf_id, kf in self.kfs.items():
            estimated = kf["pose"].detach().cpu().to(torch.float64)
            gt = self._gt[kf_id]
            if torch.equal(estimated, gt.to(torch.float32).to(torch.float64)):
                continue
            T = (gt @ torch.linalg.inv(estimated)).to(device=self.device, dtype=self.pcd.dtype)
            start, end = kf["pcd_idxs"]
            self.pcd[start:end] = self.pcd[start:end] @ T[:3, :3].T + T[:3, 3]
            kf["pose"] = gt.to(device=self.device, dtype=torch.float32)
            moved += 1
        return moved

    def _write_events(self) -> None:
        if self._events_path is None:
            return
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        self._events_path.write_text(json.dumps(self.events, indent=2))
