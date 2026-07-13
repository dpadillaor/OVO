"""The shared 3D map scene: instance cloud, camera, trajectory and map events."""

from typing import Sequence

import numpy as np
import rerun as rr

from .rerun_frame import Frame
from .rerun_sink import RerunSink, log_all
from .rerun_utils import get_instance_cmap

POINT_RADIUS = 0.008
BACKGROUND_COLOR = (60, 60, 60)
TRAJECTORY_COLOR = (0, 255, 255)
UPDATE_MAP_COLOR = (255, 200, 0)
JUMP_COLOR = (255, 50, 50)


class InstanceCloudPainter:
    """One entity per instance so each stays toggleable; instances that vanish get cleared."""

    def __init__(self, cmap):
        self.cmap = cmap
        self._known: dict[int, set[int]] = {}

    def paint(self, sinks: Sequence[RerunSink], points: np.ndarray, instance_ids: np.ndarray) -> None:
        for sink in sinks:
            idx = sink.budget_indices(len(points))
            if idx is None:
                self._paint_one(sink, points, instance_ids)
            else:
                self._paint_one(sink, points[idx], instance_ids[idx])

    def _paint_one(self, sink: RerunSink, points: np.ndarray, instance_ids: np.ndarray) -> None:
        unique = {int(uid) for uid in np.unique(instance_ids)}

        for uid in self._known.get(id(sink), set()) - unique:
            if uid >= 0:
                sink.clear(f"world/instances/obj_{uid}")

        for uid in unique:
            pts = points[instance_ids == uid]
            if uid < 0:
                path, color = "world/background", BACKGROUND_COLOR
            else:
                path, color = f"world/instances/obj_{uid}", self.cmap[uid % len(self.cmap)]
            sink.log(
                path,
                rr.Points3D(
                    pts,
                    colors=np.tile(color, (len(pts), 1)).astype(np.uint8),
                    radii=np.full(len(pts), POINT_RADIUS, dtype=np.float32),
                ),
            )

        self._known[id(sink)] = unique


class MapScene:
    """Everything both renderers show under `world/`. Owns the trajectory state."""

    def __init__(self, cam_intrinsic):
        self.cam_intrinsic = cam_intrinsic
        self.cmap = get_instance_cmap()
        self.cloud = InstanceCloudPainter(self.cmap)
        self.trajectory: list = []

    def setup(self, sinks: Sequence[RerunSink]) -> None:
        pinhole = rr.Pinhole(
            resolution=[self.cam_intrinsic["width"], self.cam_intrinsic["height"]],
            image_from_camera=self.cam_intrinsic["intrinsic"],
        )
        log_all(sinks, "world/camera", pinhole, static=True)

    def update(self, sinks: Sequence[RerunSink], frame: Frame) -> None:
        if frame.corrected_trajectory is not None:
            self.trajectory = frame.corrected_trajectory
        else:
            self.trajectory.append(frame.c2w[:3, 3].tolist())

        self.cloud.paint(sinks, frame.points, frame.instance_ids)

        log_all(
            sinks,
            "world/camera",
            rr.Transform3D(translation=frame.c2w[:3, 3], mat3x3=frame.c2w[:3, :3]),
        )
        if len(self.trajectory) >= 2:
            log_all(
                sinks,
                "world/trajectory",
                rr.LineStrips3D([self.trajectory], colors=[TRAJECTORY_COLOR], radii=[0.005]),
            )

    def log_update_map_event(self, sinks, cam_pos, frame_id: int, n_fused: int) -> None:
        log_all(
            sinks,
            "world/update_map_events",
            rr.Points3D(
                [cam_pos],
                colors=[UPDATE_MAP_COLOR],
                radii=[0.04],
                labels=[f"update_map #{frame_id} ({n_fused} fused)"],
            ),
        )

    def log_jump_event(self, sinks, cam_pos, kf_index: int, t_mag: float, r_mag: float) -> None:
        log_all(
            sinks,
            "world/jump_events",
            rr.Points3D(
                [cam_pos],
                colors=[JUMP_COLOR],
                radii=[0.06],
                labels=[f"jump KF#{kf_index} (t={t_mag:.3f}m, r={r_mag:.2f}°)"],
            ),
        )

