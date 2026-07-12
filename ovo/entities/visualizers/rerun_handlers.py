from datetime import timedelta
from pathlib import Path
from typing import Any

RERUN_PORT_LIVE = 9877  # live experiment streaming port

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .rerun_contracts import (
    JumpEventMessage,
    UpdateMapMessage,
    is_jump_event_message,
    is_update_map_message,
)
from .rerun_frame import Frame, decode
from .rerun_scene import MapScene
from .rerun_sink import RerunSink, log_all, set_time_all
from .rerun_tracking import SignalsPainter, TrackingPainter, split_layers

MAX_LIVE_POINTS = 80_000
MAX_FILE_POINTS = 300_000
TIMELINE = "step"


class BaseRerunRenderer:
    """Rerun lifecycle: recordings, sinks, blueprint, and the message dispatch skeleton."""

    name = "RerunRenderer"
    app_suffix = ""
    rrd_filename = "stream.rrd"
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
        self.sinks: list[RerunSink] = []
        self.step = 0

    def setup(self):
        rr.init(f"OVO_{self.scene_name}{self.app_suffix}")

        if self.visual_mode == "serve":
            live_rec = rr.RecordingStream(
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
                recording=live_rec,
            )
            # The live viewer never gets static data: it must stay on the timeline.
            self.sinks.append(RerunSink(live_rec, max_points=MAX_LIVE_POINTS, allow_static=False))

        if self.save_rrd:
            file_rec = rr.RecordingStream(f"OVO_{self.scene_name}{self.app_suffix}_file")
            rr.save(str(Path(self.output_path) / self.rrd_filename), recording=file_rec)
            self.sinks.append(RerunSink(file_rec, max_points=MAX_FILE_POINTS))

        if not self.sinks:
            self.sinks.append(RerunSink())

        if self.visual_mode == "spawn" and self.show:
            rr.spawn()

        blueprint = self.build_blueprint()
        for sink in self.sinks:
            sink.send_blueprint(blueprint)

        self.post_setup()

    def handle_message(self, data: Any):
        if is_update_map_message(data):
            set_time_all(self.sinks, TIMELINE, data["frame_id"])
            self.on_update_map(data)
            return
        if is_jump_event_message(data):
            set_time_all(self.sinks, TIMELINE, data["frame_id"])
            self.on_jump_event(data)
            return

        frame = decode(data, self.step)
        if frame is None:
            return

        set_time_all(self.sinks, TIMELINE, frame.frame_id)
        self.on_frame(frame)
        self.step = max(self.step + 1, frame.frame_id + 1)

    def build_blueprint(self):
        raise NotImplementedError

    def post_setup(self):
        pass

    def on_frame(self, frame: Frame):
        raise NotImplementedError

    def on_update_map(self, data: UpdateMapMessage):
        pass

    def on_jump_event(self, data: JumpEventMessage):
        pass


class SceneRenderer(BaseRerunRenderer):
    """A renderer that shows the map scene under `world/`. Owns the scene and its events."""

    def post_setup(self):
        self.scene = MapScene(self.cam_intrinsic)
        self.scene.setup(self.sinks)

    def on_update_map(self, data: UpdateMapMessage):
        self.scene.log_update_map_event(
            self.sinks, data["c2w"][:3, 3].astype(np.float32), data["frame_id"], data["n_fused"]
        )

    def on_jump_event(self, data: JumpEventMessage):
        self.scene.log_jump_event(
            self.sinks,
            data["c2w"][:3, 3].astype(np.float32),
            data["kf_index"],
            data["translation_magnitude"],
            data["rotation_magnitude"],
        )

    def finalize(self):
        """Called once when the stream closes (sentinel received)."""
        self.scene.finalize(self.sinks)


class StreamRenderer(SceneRenderer):
    """Live instance map plus the per-frame 2D panels (RGB, SAM masks, assigned/top-KF instances)."""

    name = "RerunVis"
    idle_sleep_s = 0.01

    def build_blueprint(self):
        return rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(
                        name="Instances3D",
                        contents="world/**",
                        # Per-instance normals (world/normals/obj_<id>) are saved but
                        # hidden by default; toggling the parent reveals them. The user
                        # then toggles each object under it in the entity tree.
                        overrides={"world/normals": rrb.EntityBehavior(visible=False)},
                    ),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="RGB", contents="frame/rgb"),
                    rrb.Spatial2DView(name="SAM Masks", contents="frame/sam_map"),
                    rrb.Spatial2DView(name="Assigned Instances", contents="frame/assigned_map"),
                    rrb.Spatial2DView(name="Top-KF Instances", contents="frame/ins_map"),
                ),
            ),
            collapse_panels=False,
        )

    def on_frame(self, frame: Frame):
        self.scene.update(self.sinks, frame)

        if frame.rgb is not None:
            log_all(self.sinks, "frame/rgb", rr.Image(frame.rgb))
        for path, id_map in (
            ("frame/sam_map", frame.sam_map),
            ("frame/ins_map", frame.ins_map),
            # All assigned instances (pre top-kf filter) — what co-occurrence actually counts.
            ("frame/assigned_map", frame.assigned_map),
        ):
            if id_map is not None:
                log_all(self.sinks, path, rr.Image(self._colorize(id_map)))

        # kf_id of this frame — lets you map co-occurrence kf indices back to rrd steps.
        if frame.kf_id is not None:
            log_all(self.sinks, "frame/kf_id", rr.TextLog(f"kf_id={frame.kf_id}"))

    def _colorize(self, id_map):
        """Map int instance ids to RGB; -1 (background) stays dark gray."""
        id_map = np.asarray(id_map, dtype=np.int32)
        cmap = self.scene.cmap
        vis = np.full(id_map.shape + (3,), 40, dtype=np.uint8)
        valid = id_map >= 0
        if valid.any():
            vis[valid] = cmap[(id_map[valid] % len(cmap)).astype(np.int32)]
        return vis


class TrackingRenderer(SceneRenderer):
    """Diagnostics for mask tracking. Reuses the map scene and adds, per tracking KF:
      - which reprojected points already had an instance (green) and which did not (orange),
        in 3D over a grey map and in 2D over the frame, with robbed/born subsets hidden by default;
      - SAM / top-KF / assigned masks as translucent overlays on the frame (hidden by default);
      - counts, their RAW derivatives (no smoothing, so a single-frame spike survives) and camera speed.
    All on the `step` (== frame_id) timeline.
    """

    name = "RerunTrackingVis"
    rrd_filename = "tracking.rrd"
    idle_sleep_s = 0.01

    def post_setup(self):
        super().post_setup()
        self.tracking = TrackingPainter(self.scene.cmap)
        self.tracking.setup(self.sinks)
        self.signals = SignalsPainter()
        self.signals.setup(self.sinks)

    def build_blueprint(self):
        hidden = {
            "tracking/robbed": rrb.EntityBehavior(visible=False),
            "tracking/births": rrb.EntityBehavior(visible=False),
        }
        hidden_2d = {
            "frame/pts2d/robbed": rrb.EntityBehavior(visible=False),
            "frame/pts2d/births": rrb.EntityBehavior(visible=False),
            "frame/seg": rrb.EntityBehavior(visible=False),
        }
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(
                        name="SLAM",
                        contents="world/**",
                        overrides={"world/normals": rrb.EntityBehavior(visible=False)},
                    ),
                    rrb.Horizontal(
                        rrb.Spatial3DView(
                            name="Tracking KF",
                            contents="tracking/**",
                            overrides=hidden,
                        ),
                        rrb.Spatial2DView(
                            name="Tracking KF (2D)",
                            contents=["frame/rgb", "frame/pts2d/**", "frame/seg/**"],
                            overrides=hidden_2d,
                        ),
                    ),
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(name="Points per KF", origin="signals/count"),
                    rrb.TimeSeriesView(name="Points Δ/frame", origin="signals/deriv"),
                    rrb.TimeSeriesView(name="Camera speed", origin="signals/cam"),
                    rrb.TimeSeriesView(name="Camera jerk", origin="signals/accel"),
                ),
            ),
            collapse_panels=False,
        )

    def on_frame(self, frame: Frame):
        # None when the signals belong to an older KF (update_map re-sends the frame): both
        # painters then know to clear / skip instead of showing stale data.
        layers = split_layers(frame)
        self.scene.update(self.sinks, frame)
        self.tracking.paint(self.sinks, frame, layers)
        self.signals.paint(self.sinks, frame, layers)
