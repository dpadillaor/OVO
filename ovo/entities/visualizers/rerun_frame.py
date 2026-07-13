"""Decoding of queue messages into a normalized Frame. Only place that knows the raw layout."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .rerun_contracts import is_stream_frame_message, is_stream_message
from .rerun_utils import ceiling_mask, resolve_instance_ids


@dataclass
class Frame:
    """A stream frame, already decoded: float32, ceiling-cut, instance ids resolved."""

    frame_id: int
    points: np.ndarray
    instance_ids: np.ndarray
    c2w: np.ndarray
    point_ids: np.ndarray | None = None  # permanent point ids, aligned with `points` (tracking only)
    rgb: np.ndarray | None = None
    sam_map: np.ndarray | None = None
    ins_map: np.ndarray | None = None
    assigned_map: np.ndarray | None = None
    kf_id: int | None = None
    corrected_trajectory: list | None = None
    signals: dict | None = None


def decode(data: Any, fallback_step: int) -> Frame | None:
    """Message -> Frame. Returns None when the message is not a frame."""
    if is_stream_frame_message(data):
        raw = dict(data)
        frame_id = int(data["frame_id"])
    elif is_stream_message(data):
        points, obj_ids, _colors, c2w = data
        raw = {"points": points, "obj_ids": obj_ids, "c2w": c2w}
        frame_id = fallback_step
    else:
        return None

    points = np.asarray(raw["points"], dtype=np.float32)
    instance_ids = resolve_instance_ids(raw["obj_ids"], points.shape[0])
    point_ids = raw.get("point_ids")

    keep = ceiling_mask(points)

    return Frame(
        frame_id=frame_id,
        points=points[keep],
        instance_ids=instance_ids[keep],
        c2w=np.asarray(raw["c2w"], dtype=np.float32),
        point_ids=np.asarray(point_ids)[keep] if point_ids is not None else None,
        rgb=raw.get("rgb"),
        sam_map=raw.get("sam_map"),
        ins_map=raw.get("ins_map"),
        assigned_map=raw.get("assigned_ins_map"),
        kf_id=raw.get("kf_id"),
        corrected_trajectory=raw.get("corrected_trajectory"),
        signals=raw.get("track_signals"),
    )
