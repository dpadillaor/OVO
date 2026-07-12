"""Painters for the tracking diagnostics: the per-KF point partition (3D + 2D) and its signals."""

from typing import Sequence

import numpy as np
import rerun as rr

from .rerun_frame import Frame
from .rerun_sink import RerunSink, clear_all, log_all

GREY = (90, 90, 90)
ASSIGNED_COLOR = (60, 200, 90)
UNASSIGNED_COLOR = (255, 230, 0)
ROBBED_COLOR = (255, 40, 40)
BIRTH_COLOR = (40, 200, 255)

GREY_RADIUS = 0.006
LAYER_RADIUS = 0.008
SUBSET_RADIUS = 0.02  # robbed/births sit on top of the base layers, so they need to be fatter
PX_RADIUS = 1.5
PX_SUBSET_RADIUS = 3.0

# (layer, colour) — assigned/unassigned partition the matched points; robbed/births are subsets.
LAYERS = (
    ("assigned", ASSIGNED_COLOR),
    ("unassigned", UNASSIGNED_COLOR),
    ("robbed", ROBBED_COLOR),
    ("births", BIRTH_COLOR),
)

# (axis, colour, base unit) — the derivative of each is that unit per frame².
CAM_SERIES = (
    ("lin", (120, 180, 255), "m"),
    ("ang", (220, 120, 255), "deg"),
)

SEG_OVERLAYS = (("sam", "sam_map", 1.0), ("ins", "ins_map", 1.1), ("assigned", "assigned_map", 1.2))
SEG_OPACITY = 0.45
BACKGROUND_CLASS = 0


def split_layers(frame: Frame) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    """Matched points -> the four layers, as (permanent ids, pixels) pairs.

    None when the signals do not belong to this frame: update_map re-sends the frame with a newer
    frame_id but the previous KF's signals, and painting those would resurrect stale highlights.
    """
    signals = frame.signals
    if signals is None or signals.get("frame_id") != frame.frame_id:
        return None

    ids = np.asarray(signals["matched_ids"])
    px = np.asarray(signals["matched_px"])
    pre = np.asarray(signals["matched_pre"], dtype=bool)

    masks = {
        "assigned": pre,
        "unassigned": ~pre,
        "robbed": np.isin(ids, np.asarray(signals.get("robbed_ids", []), dtype=ids.dtype)),
        "births": np.isin(ids, np.asarray(signals.get("birth_ids", []), dtype=ids.dtype)),
    }
    return {name: (ids[mask], px[mask]) for name, mask in masks.items()}


def seg_annotation_context(cmap) -> rr.AnnotationContext:
    """Class 0 = background, fully transparent; class i+1 = colour i of the instance cmap."""
    infos = [rr.AnnotationInfo(id=BACKGROUND_CLASS, label="bg", color=(0, 0, 0, 0))]
    infos += [
        rr.AnnotationInfo(id=i + 1, label=f"c{i}", color=tuple(int(c) for c in cmap[i]))
        for i in range(len(cmap))
    ]
    return rr.AnnotationContext(infos)


def _to_classes(id_map: np.ndarray, n_colors: int) -> np.ndarray:
    """Instance id map -> segmentation classes: -1 becomes the transparent background class."""
    ids = np.asarray(id_map, dtype=np.int32)
    classes = np.zeros(ids.shape, dtype=np.uint16)
    valid = ids >= 0
    classes[valid] = (ids[valid] % n_colors) + 1
    return classes


class TrackingPainter:
    """Paints, per tracking KF, which reprojected points already had an instance and which did not."""

    def __init__(self, cmap):
        self.cmap = cmap

    def setup(self, sinks: Sequence[RerunSink]) -> None:
        log_all(sinks, "frame/seg", seg_annotation_context(self.cmap), static=True)

    def paint(self, sinks: Sequence[RerunSink], frame: Frame, layers: dict | None) -> None:
        self._paint_greymap(sinks, frame)

        if frame.rgb is not None:
            log_all(sinks, "frame/rgb", rr.Image(frame.rgb, draw_order=0.0))
        self._paint_seg_overlays(sinks, frame)

        if layers is None:
            self._clear_layers(sinks)
            return

        self._paint_3d(sinks, frame, layers)
        self._paint_2d(sinks, layers)

    def _paint_greymap(self, sinks: Sequence[RerunSink], frame: Frame) -> None:
        for sink in sinks:
            idx = sink.budget_indices(len(frame.points))
            pts = frame.points if idx is None else frame.points[idx]
            sink.log(
                "tracking/greymap",
                rr.Points3D(
                    pts,
                    colors=np.tile(GREY, (len(pts), 1)).astype(np.uint8),
                    radii=np.full(len(pts), GREY_RADIUS, dtype=np.float32),
                ),
            )

    def _paint_seg_overlays(self, sinks: Sequence[RerunSink], frame: Frame) -> None:
        for name, attr, draw_order in SEG_OVERLAYS:
            id_map = getattr(frame, attr)
            if id_map is None:
                continue
            log_all(
                sinks,
                f"frame/seg/{name}",
                rr.SegmentationImage(
                    _to_classes(id_map, len(self.cmap)),
                    opacity=SEG_OPACITY,
                    draw_order=draw_order,
                ),
            )

    def _paint_3d(self, sinks, frame: Frame, layers: dict) -> None:
        if frame.point_ids is None:
            return
        for name, color in LAYERS:
            ids, _ = layers[name]
            pts = frame.points[np.isin(frame.point_ids, ids)] if len(ids) else frame.points[:0]
            path = f"tracking/{name}"
            if len(pts) == 0:
                clear_all(sinks, path)
                continue
            radius = SUBSET_RADIUS if name in ("robbed", "births") else LAYER_RADIUS
            log_all(
                sinks,
                path,
                rr.Points3D(
                    pts,
                    colors=np.tile(color, (len(pts), 1)).astype(np.uint8),
                    radii=np.full(len(pts), radius, dtype=np.float32),
                ),
            )

    def _paint_2d(self, sinks, layers: dict) -> None:
        for name, color in LAYERS:
            _, px = layers[name]
            path = f"frame/pts2d/{name}"
            if len(px) == 0:
                clear_all(sinks, path)
                continue
            radius = PX_SUBSET_RADIUS if name in ("robbed", "births") else PX_RADIUS
            log_all(
                sinks,
                path,
                rr.Points2D(
                    px,
                    colors=np.tile(color, (len(px), 1)).astype(np.uint8),
                    radii=np.full(len(px), radius, dtype=np.float32),
                    draw_order=2.0,
                ),
            )

    def _clear_layers(self, sinks: Sequence[RerunSink]) -> None:
        for name, _ in LAYERS:
            clear_all(sinks, f"tracking/{name}")
            clear_all(sinks, f"frame/pts2d/{name}")


class SignalsPainter:
    """One curve per 3D layer, in the layer's colour, so a plot reads back to points you can see.

    Counts come from the layers themselves (not from separate tallies), so the graphs cannot drift
    out of sync with the 3D view. Derivatives are raw — no smoothing, so a single-KF spike survives
    — and divide by the real frame gap, which is `segment_every` (only KFs emit).
    """

    def __init__(self):
        self._prev_counts: dict[str, float] = {}
        self._prev_frame: int | None = None
        self._prev_pos = None
        self._prev_rot = None
        self._prev_speed: dict[str, float] = {}

    def setup(self, sinks: Sequence[RerunSink]) -> None:
        for name, color in LAYERS:
            log_all(sinks, f"signals/count/{name}", rr.SeriesLines(colors=[color], names=[name]), static=True)
            log_all(
                sinks,
                f"signals/deriv/d_{name}",
                rr.SeriesLines(colors=[color], names=[f"d_{name}"]),
                static=True,
            )
        # Rerun has no axis labels, so the units live in the series name (i.e. the legend).
        for axis, color, unit in CAM_SERIES:
            log_all(
                sinks,
                f"signals/cam/v_{axis}",
                rr.SeriesLines(colors=[color], names=[f"v_{axis} [{unit}/frame]"]),
                static=True,
            )
            log_all(
                sinks,
                f"signals/accel/a_{axis}",
                rr.SeriesLines(colors=[color], names=[f"a_{axis} [{unit}/frame²]"]),
                static=True,
            )

    def paint(self, sinks: Sequence[RerunSink], frame: Frame, layers: dict | None) -> None:
        if layers is None:
            return

        counts = {name: float(len(layers[name][0])) for name, _ in LAYERS}
        for name, value in counts.items():
            log_all(sinks, f"signals/count/{name}", rr.Scalars(value))

        d_frame = (frame.frame_id - self._prev_frame) if self._prev_frame is not None else 0
        if d_frame > 0:
            for name, value in counts.items():
                if name in self._prev_counts:
                    log_all(
                        sinks,
                        f"signals/deriv/d_{name}",
                        rr.Scalars((value - self._prev_counts[name]) / d_frame),
                    )
            self._paint_camera(sinks, frame, d_frame)

        self._prev_counts = counts
        self._prev_frame = frame.frame_id
        self._prev_pos, self._prev_rot = frame.c2w[:3, 3], frame.c2w[:3, :3]

    def _paint_camera(self, sinks: Sequence[RerunSink], frame: Frame, d_frame: int) -> None:
        """Camera speed and its derivative. Acceleration is the one that predicts a broken match:
        a jerk between KFs moves the map further than `match_distance_th` tolerates."""
        pos, rot = frame.c2w[:3, 3], frame.c2w[:3, :3]
        if self._prev_pos is None:
            return

        cos = np.clip((np.trace(rot @ self._prev_rot.T) - 1) / 2, -1.0, 1.0)
        speed = {
            "lin": float(np.linalg.norm(pos - self._prev_pos) / d_frame),  # m / frame
            "ang": float(np.degrees(np.arccos(cos)) / d_frame),  # deg / frame
        }
        for axis, value in speed.items():
            log_all(sinks, f"signals/cam/v_{axis}", rr.Scalars(value))
            if axis in self._prev_speed:
                log_all(
                    sinks,
                    f"signals/accel/a_{axis}",
                    rr.Scalars((value - self._prev_speed[axis]) / d_frame),
                )
        self._prev_speed = speed
