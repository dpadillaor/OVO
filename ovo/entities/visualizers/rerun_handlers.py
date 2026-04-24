from datetime import timedelta
from pathlib import Path
from typing import Any

RERUN_PORT_LIVE = 9877  # live experiment streaming port

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .rerun_utils import (
    ceiling_mask,
    get_instance_cmap,
    resolve_instance_ids,
)
from .rerun_contracts import (
    UpdateMapMessage,
    is_stream_frame_message,
    is_stream_message,
    is_update_map_message,
)


class BaseRerunRenderer:
    name = "RerunRenderer"
    app_suffix = ""
    idle_sleep_s = 0.05
    error_sleep_s = 1.0

    def __init__(
        self,
        cam_intrinsic,
        scene_name,
        output_path,
        show,
        save_rrd=False,
        visual_mode="spawn",
    ):
        self.cam_intrinsic = cam_intrinsic
        self.scene_name = scene_name
        self.output_path = output_path
        self.show = show
        self.save_rrd = save_rrd
        self.visual_mode = str(visual_mode).lower()

    def setup(self):
        rr.init(f"OVO_{self.scene_name}{self.app_suffix}")

        if self.visual_mode == "serve":
            self.live_rec = rr.RecordingStream(
                f"OVO_{self.scene_name}{self.app_suffix}_live",
                batcher_config=rr.ChunkBatcherConfig(
                    flush_tick=timedelta(milliseconds=33),
                    flush_num_bytes=512_000,
                    flush_num_rows=256,
                ),
            )
            rr.serve_grpc(
                grpc_port=RERUN_PORT_LIVE,
                server_memory_limit="200MB",
                recording=self.live_rec,
            )
        else:
            self.live_rec = None

        if self.save_rrd:
            self.file_rec = rr.RecordingStream(
                f"OVO_{self.scene_name}{self.app_suffix}_file",
            )
            rr.save(
                str(Path(self.output_path) / "rerun.rrd"),
                recording=self.file_rec,
            )
        else:
            self.file_rec = None

        if self.visual_mode == "spawn" and self.show:
            rr.spawn()

        self._send_blueprint()
        self.post_setup()

    def _send_blueprint(self):
        bp = self.build_blueprint()
        if self.live_rec is not None:
            rr.send_blueprint(bp, recording=self.live_rec)
        if self.file_rec is not None:
            rr.send_blueprint(bp, recording=self.file_rec)
        if self.live_rec is None and self.file_rec is None:
            rr.send_blueprint(bp)

    def _log(self, path: str, entity, *, file_static: bool = False):
        """Log to live stream (never static) and file stream (with file_static)."""
        if self.live_rec is not None:
            rr.log(path, entity, recording=self.live_rec)
        if self.file_rec is not None:
            rr.log(path, entity, static=file_static, recording=self.file_rec)
        if self.live_rec is None and self.file_rec is None:
            rr.log(path, entity)

    def _set_time(self, timeline: str, **kwargs):
        if self.live_rec is not None:
            rr.set_time(timeline, recording=self.live_rec, **kwargs)
        if self.file_rec is not None:
            rr.set_time(timeline, recording=self.file_rec, **kwargs)
        if self.live_rec is None and self.file_rec is None:
            rr.set_time(timeline, **kwargs)

    def build_blueprint(self):
        raise NotImplementedError

    def post_setup(self):
        pass

    def handle_message(self, data: Any):
        raise NotImplementedError


class StreamRenderer(BaseRerunRenderer):
    name = "RerunVis"
    idle_sleep_s = 0.01

    MAX_LIVE_POINTS = 80_000
    MAX_FILE_POINTS = 300_000

    def post_setup(self):
        width = self.cam_intrinsic["width"]
        height = self.cam_intrinsic["height"]
        K = self.cam_intrinsic["intrinsic"]

        pinhole = rr.Pinhole(resolution=[width, height], image_from_camera=K)
        for rec in [r for r in [self.live_rec, self.file_rec] if r is not None]:
            rr.log("world/camera", pinhole, static=True, recording=rec)
        if self.live_rec is None and self.file_rec is None:
            rr.log("world/camera", pinhole, static=True)

        self.cmap = get_instance_cmap()
        self.step = 0
        self.trajectory = []
        self._known_instance_ids: dict[int, set[int]] = {}  # keyed by id(recording) or -1 for default

    def _log_instances_3d(self, points: np.ndarray, instance_ids: np.ndarray, *, recording=None, static: bool = False):
        radii = np.full(len(points), 0.008, dtype=np.float32)
        kwargs = {"recording": recording} if recording is not None else {}
        rec_key = id(recording) if recording is not None else -1
        unique_ids = set(int(uid) for uid in np.unique(instance_ids))

        for uid in self._known_instance_ids.get(rec_key, set()) - unique_ids:
            if uid >= 0:
                rr.log(f"world/instances/obj_{uid}", rr.Clear(recursive=False), **kwargs)

        for uid in unique_ids:
            mask = instance_ids == uid
            pts = points[mask]
            if uid < 0:
                path = "world/background"
                colors = np.full((len(pts), 3), 60, dtype=np.uint8)
            else:
                path = f"world/instances/obj_{uid}"
                colors = np.tile(self.cmap[uid % len(self.cmap)], (len(pts), 1)).astype(np.uint8)
            rr.log(path, rr.Points3D(pts, colors=colors, radii=radii[:len(pts)]), static=static, **kwargs)

        self._known_instance_ids[rec_key] = unique_ids

    def build_blueprint(self):
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(name="Instances3D", contents="world/**"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="RGB", contents="frame/rgb"),
                    rrb.Spatial2DView(name="SAM Masks", contents="frame/sam_map"),
                    rrb.Spatial2DView(name="Instances2D", contents="frame/ins_map"),
                ),
            ),
            collapse_panels=False,
        )

    def handle_message(self, data: Any):
        if is_update_map_message(data):
            self._handle_update_map(data)
            return

        rgb = None
        ins_map = None
        time_step = self.step

        if is_stream_frame_message(data):
            points = data["points"]
            obj_ids = data["obj_ids"]
            colors = data["colors"]
            c2w = data["c2w"]
            rgb = data["rgb"]
            ins_map = data["ins_map"]
            sam_map = data["sam_map"]
            time_step = int(data["frame_id"])
        elif is_stream_message(data):
            points, obj_ids, colors, c2w = data
            sam_map = None
        else:
            return

        points = points.astype(np.float32)
        c2w = c2w.astype(np.float32)

        instance_ids = resolve_instance_ids(obj_ids, points.shape[0])

        mask = ceiling_mask(points)
        points_full = points[mask]
        instance_ids_full = instance_ids[mask]

        self.trajectory.append(c2w[:3, 3].tolist())

        if len(points_full) > self.MAX_LIVE_POINTS and self.live_rec is not None:
            idx = np.random.choice(len(points_full), self.MAX_LIVE_POINTS, replace=False)
            idx.sort()
            points_live = points_full[idx]
            instance_ids_live = instance_ids_full[idx]
        else:
            points_live = points_full
            instance_ids_live = instance_ids_full

        if len(points_full) > self.MAX_FILE_POINTS:
            idx = np.random.choice(len(points_full), self.MAX_FILE_POINTS, replace=False)
            idx.sort()
            points_file = points_full[idx]
            instance_ids_file = instance_ids_full[idx]
        else:
            points_file = points_full
            instance_ids_file = instance_ids_full

        if self.live_rec is not None:
            rr.set_time("step", sequence=time_step, recording=self.live_rec)
            self._log_instances_3d(points_live, instance_ids_live, recording=self.live_rec)
            rr.log(
                "world/camera",
                rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]),
                recording=self.live_rec,
            )
            if len(self.trajectory) >= 2:
                rr.log(
                    "world/trajectory",
                    rr.LineStrips3D([self.trajectory], colors=[[0, 255, 255]], radii=[0.005]),
                    recording=self.live_rec,
                )

            if rgb is not None:
                rr.log("frame/rgb", rr.Image(rgb), recording=self.live_rec)
            if ins_map is not None:
                ins_map = ins_map.astype(np.int32)
                ins_vis = np.full(ins_map.shape + (3,), 40, dtype=np.uint8)
                valid = ins_map >= 0
                if valid.any():
                    ins_vis[valid] = self.cmap[(ins_map[valid] % len(self.cmap)).astype(np.int32)]
                rr.log("frame/ins_map", rr.Image(ins_vis), recording=self.live_rec)
            if sam_map is not None:
                sam_map_i32 = sam_map.astype(np.int32)
                sam_vis = np.full(sam_map_i32.shape + (3,), 40, dtype=np.uint8)
                valid_sam = sam_map_i32 >= 0
                if valid_sam.any():
                    sam_vis[valid_sam] = self.cmap[(sam_map_i32[valid_sam] % len(self.cmap)).astype(np.int32)]
                rr.log("frame/sam_map", rr.Image(sam_vis), recording=self.live_rec)

        if self.file_rec is not None:
            rr.set_time("step", sequence=time_step, recording=self.file_rec)
            self._log_instances_3d(points_file, instance_ids_file, recording=self.file_rec)
            rr.log(
                "world/camera",
                rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]),
                recording=self.file_rec,
            )
            if len(self.trajectory) >= 2:
                rr.log(
                    "world/trajectory",
                    rr.LineStrips3D([self.trajectory], colors=[[0, 255, 255]], radii=[0.005]),
                    recording=self.file_rec,
                )

            if rgb is not None:
                rr.log("frame/rgb", rr.Image(rgb), recording=self.file_rec)
            if ins_map is not None:
                ins_map = ins_map.astype(np.int32)
                ins_vis = np.full(ins_map.shape + (3,), 40, dtype=np.uint8)
                valid = ins_map >= 0
                if valid.any():
                    ins_vis[valid] = self.cmap[(ins_map[valid] % len(self.cmap)).astype(np.int32)]
                rr.log("frame/ins_map", rr.Image(ins_vis), recording=self.file_rec)
            if sam_map is not None:
                sam_map_i32 = sam_map.astype(np.int32)
                sam_vis = np.full(sam_map_i32.shape + (3,), 40, dtype=np.uint8)
                valid_sam = sam_map_i32 >= 0
                if valid_sam.any():
                    sam_vis[valid_sam] = self.cmap[(sam_map_i32[valid_sam] % len(self.cmap)).astype(np.int32)]
                rr.log("frame/sam_map", rr.Image(sam_vis), recording=self.file_rec)

        if self.live_rec is None and self.file_rec is None:
            rr.set_time("step", sequence=time_step)
            rr.log(
                "world/camera",
                rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]),
            )
            if len(self.trajectory) >= 2:
                rr.log(
                    "world/trajectory",
                    rr.LineStrips3D([self.trajectory], colors=[[0, 255, 255]], radii=[0.005]),
                )
            self._log_instances_3d(points_full, instance_ids_full)

            if rgb is not None:
                rr.log("frame/rgb", rr.Image(rgb))
            if ins_map is not None:
                ins_map = ins_map.astype(np.int32)
                ins_vis = np.full(ins_map.shape + (3,), 40, dtype=np.uint8)
                valid = ins_map >= 0
                if valid.any():
                    ins_vis[valid] = self.cmap[(ins_map[valid] % len(self.cmap)).astype(np.int32)]
                rr.log("frame/ins_map", rr.Image(ins_vis))
            if sam_map is not None:
                sam_map_i32 = sam_map.astype(np.int32)
                sam_vis = np.full(sam_map_i32.shape + (3,), 40, dtype=np.uint8)
                valid_sam = sam_map_i32 >= 0
                if valid_sam.any():
                    sam_vis[valid_sam] = self.cmap[(sam_map_i32[valid_sam] % len(self.cmap)).astype(np.int32)]
                rr.log("frame/sam_map", rr.Image(sam_vis))

        self.step = max(self.step + 1, time_step + 1)

    def _handle_update_map(self, data: UpdateMapMessage):
        frame_id = data["frame_id"]
        cam_pos = data["c2w"][:3, 3].astype(np.float32)
        n_fused = data["n_fused"]
        decisions = data["decisions"]

        self._set_time("step", sequence=frame_id)

        # Spatial marker at camera position
        label = f"update_map #{frame_id} ({n_fused} fused)"
        self._log(
            "world/update_map_events",
            rr.Points3D([cam_pos], colors=[[255, 200, 0]], radii=[0.04], labels=[label]),
        )

