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
    get_instance_colors,
    log_instances,
    log_loop_closure_pair,
    resolve_instance_ids,
    subsample_consistent,
)
from .rerun_contracts import (
    FusionMessage,
    LoopClosureMessage,
    UpdateMapMessage,
    is_fusion_message,
    is_loop_closure_message,
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



class FusionRenderer(BaseRerunRenderer):
    name = "FusionVis"
    app_suffix = "_fusion"

    def post_setup(self):
        self.cmap = get_instance_cmap()
        self.step = 0
        self.max_fusion_points = 400_000

    def build_blueprint(self):
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        name="Before Fusion",
                        contents=["before/instances/**", "diff/**"],
                        origin="/",
                    ),
                    rrb.Spatial3DView(
                        name="After Fusion",
                        contents=["after/instances/**", "diff/**"],
                        origin="/",
                    ),
                ),
                rrb.Spatial3DView(
                    name="Loop Closure",
                    contents=["lc/**"],
                    origin="/",
                ),
                rrb.TextDocumentView(
                    name="Fusion Stats",
                    contents="stats/**",
                ),
            ),
            collapse_panels=False,
            auto_layout=True,
        )

    def _handle_loop_closure(self, data: LoopClosureMessage):
        pcd_before = data["pcd_before"].astype(np.float32)
        pcd_after = data["pcd_after"].astype(np.float32)
        ids = data["ids"].astype(np.int32)
        traj_before = data["traj_before"]
        traj_after = data["traj_after"]

        if pcd_after.size == 0:
            return

        ceiling_z = pcd_after[:, -1].max() - 0.2
        mask_before = pcd_before[:, -1] < ceiling_z if pcd_before.size > 0 else np.zeros(0, dtype=bool)
        mask_after = pcd_after[:, -1] < ceiling_z

        pts_b = pcd_before[mask_before]
        ids_b = ids[mask_before]
        pts_a = pcd_after[mask_after]
        ids_a = ids[mask_after]

        pts_b, ids_b = subsample_consistent((pts_b, ids_b), self.max_fusion_points)
        pts_a, ids_a = subsample_consistent((pts_a, ids_a), self.max_fusion_points)

        log_loop_closure_pair(
            path_prefix="lc",
            pts_before=pts_b,
            ids_before=ids_b,
            pts_after=pts_a,
            ids_after=ids_a,
            traj_before=traj_before,
            traj_after=traj_after,
            cmap=self.cmap,
            step=0,
        )

    def _handle_fusion(self, data: FusionMessage):
        points = data["points"].astype(np.float32)
        before_ids = resolve_instance_ids(data["before_ids"], points.shape[0])
        after_ids = resolve_instance_ids(data["after_ids"], points.shape[0])
        frame_id = data["frame_id"]

        mask = ceiling_mask(points)
        pts = points[mask]
        b_ids = before_ids[mask]
        a_ids = after_ids[mask]

        if np.array_equal(b_ids, a_ids):
            return

        pts, b_ids, a_ids = subsample_consistent((pts, b_ids, a_ids), self.max_fusion_points)

        before_colors = get_instance_colors(b_ids, self.cmap)
        after_colors = get_instance_colors(a_ids, self.cmap)

        unique_before = np.unique(b_ids[b_ids >= 0])
        unique_after = np.unique(a_ids[a_ids >= 0])
        n_before = len(unique_before)
        n_after = len(unique_after)
        fused = n_before - n_after

        changed_mask = b_ids != a_ids
        disappeared_ids = set(unique_before) - set(unique_after)

        fusion_map = {}
        fusion_details = []
        for after_id in unique_after:
            mask_after = a_ids == after_id
            before_ids_in_region = np.unique(b_ids[mask_after])
            before_ids_in_region = before_ids_in_region[before_ids_in_region >= 0]

            if len(before_ids_in_region) <= 1:
                continue

            contributors = []
            for bid in before_ids_in_region:
                if bid != after_id:
                    fusion_map[bid] = after_id
                contributors.append(int(bid))
            fusion_details.append((int(after_id), contributors))

        arrow_origins = []
        arrow_vectors = []
        for deleted_id, survivor_id in fusion_map.items():
            mask_deleted = b_ids == deleted_id
            mask_survivor = a_ids == survivor_id

            if mask_deleted.sum() == 0 or mask_survivor.sum() == 0:
                continue

            centroid_deleted = pts[mask_deleted].mean(axis=0)
            centroid_survivor = pts[mask_survivor].mean(axis=0)
            arrow_origins.append(centroid_deleted)
            arrow_vectors.append(centroid_survivor - centroid_deleted)

        self._set_time("fusion_event", sequence=self.step)

        log_instances("before", pts, b_ids, before_colors)
        self._log("before/info", rr.TextLog(f"Frame {frame_id} | {n_before} instances"), file_static=True)

        log_instances("after", pts, a_ids, after_colors)
        self._log("after/info", rr.TextLog(f"Frame {frame_id} | {n_after} instances ({fused} fused)"), file_static=True)

        if changed_mask.sum() > 0:
            self._log(
                "diff/changed_points",
                rr.Points3D(
                    pts[changed_mask],
                    colors=[255, 255, 0],
                    radii=np.full(changed_mask.sum(), 0.012, dtype=np.float32),
                ),
                file_static=True,
            )

        for dis_id in disappeared_ids:
            mask_dis = b_ids == dis_id
            if mask_dis.sum() == 0:
                continue

            self._log(
                f"diff/disappeared/ins_{dis_id}",
                rr.Points3D(
                    pts[mask_dis],
                    colors=[255, 0, 0],
                    radii=np.full(mask_dis.sum(), 0.01, dtype=np.float32),
                ),
                file_static=True,
            )

            pts_ins = pts[mask_dis]
            bbox_min = pts_ins.min(axis=0)
            bbox_max = pts_ins.max(axis=0)
            center = (bbox_min + bbox_max) / 2
            half_sizes = (bbox_max - bbox_min) / 2
            self._log(
                f"diff/boxes/ins_{dis_id}",
                rr.Boxes3D(
                    centers=[center],
                    half_sizes=[half_sizes],
                    labels=[f"Fused: {dis_id}"],
                    colors=[[255, 0, 0]],
                ),
                file_static=True,
            )

        if len(arrow_origins) > 0:
            self._log(
                "diff/merge_arrows",
                rr.Arrows3D(
                    origins=np.array(arrow_origins),
                    vectors=np.array(arrow_vectors),
                    colors=[255, 0, 255],
                    radii=[0.02],
                ),
                file_static=True,
            )

        stats_text = f"""# Frame {frame_id} Fusion Event
## Summary
- **Before:** {n_before} instances
- **After:** {n_after} instances
- **Fused:** {fused} instances

## Fusion Details
"""
        if len(fusion_details) > 0:
            for survivor_id, contributors in fusion_details:
                contributors_str = ", ".join(str(c) for c in contributors)
                stats_text += f"\n- Instance **{survivor_id}** <- merged from [{contributors_str}]"
        else:
            stats_text += "\n*No multi-instance fusions (only deletions)*"

        stats_text += "\n\n## Disappeared Instances\n"
        if len(disappeared_ids) > 0:
            stats_text += ", ".join(str(d) for d in sorted(disappeared_ids))
        else:
            stats_text += "*None*"

        self._log(
            "stats/fusion_info",
            rr.TextDocument(stats_text, media_type=rr.MediaType.MARKDOWN),
            file_static=True,
        )

        self.step += 1

    def handle_message(self, data: Any):
        if is_loop_closure_message(data):
            self._handle_loop_closure(data)
            return

        if is_fusion_message(data):
            self._handle_fusion(data)


class LoopClosureRenderer(BaseRerunRenderer):
    name = "LCVis"
    app_suffix = "_loopclosure"

    def post_setup(self):
        self.cmap = get_instance_cmap()
        self.max_points = 400_000
        self.step = 0

    def build_blueprint(self):
        return rrb.Blueprint(
            rrb.Spatial3DView(name="Loop Closure", contents="world/**"),
            collapse_panels=True,
        )

    def handle_message(self, data: Any):
        if not is_loop_closure_message(data):
            return

        pcd_before = data["pcd_before"].astype(np.float32)
        pcd_after = data["pcd_after"].astype(np.float32)
        ids = data["ids"].astype(np.int32)
        traj_before = data["traj_before"]
        traj_after = data["traj_after"]

        if pcd_after.size == 0:
            return

        ceiling_z = pcd_after[:, -1].max() - 0.2
        mask_before = pcd_before[:, -1] < ceiling_z if pcd_before.size > 0 else np.zeros(0, dtype=bool)
        mask_after = pcd_after[:, -1] < ceiling_z

        pts_before = pcd_before[mask_before]
        pts_after = pcd_after[mask_after]
        ids_before = ids[mask_before]
        ids_after = ids[mask_after]

        pts_before, ids_before = subsample_consistent((pts_before, ids_before), self.max_points)
        pts_after, ids_after = subsample_consistent((pts_after, ids_after), self.max_points)

        log_loop_closure_pair(
            path_prefix="world",
            pts_before=pts_before,
            ids_before=ids_before,
            pts_after=pts_after,
            ids_after=ids_after,
            traj_before=traj_before,
            traj_after=traj_after,
            cmap=self.cmap,
            step=self.step,
            traj_before_path="world/trajectory",
            traj_after_path="world/trajectory",
        )

        self.step += 1
