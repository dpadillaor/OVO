from typing import Any, Literal, TypeGuard, TypedDict

import numpy as np


class StreamMessage(TypedDict):
    points: np.ndarray
    obj_ids: np.ndarray
    colors: Any
    c2w: np.ndarray


class TrackSignals(TypedDict):
    """Per-KF tracking diagnostics, snapshotted before the mask loop mutates the instance ids.

    `matched_*` describe the reprojected points that survived the depth match, aligned row-wise:
    `matched_pre` says which of them already belonged to an instance. robbed/birth ids are subsets.
    `frame_id` is what makes them safe to consume: update_map re-sends the frame afterwards.
    """

    frame_id: int
    n_matched: int
    n_pre_assign: int
    n_orphans: int
    n_births: int
    n_robos: int
    matched_ids: np.ndarray  # (M,) permanent point ids
    matched_px: np.ndarray  # (M,2) pixel (x, y) each point projects to
    matched_pre: np.ndarray  # (M,) bool: had an instance on entry
    robbed_ids: np.ndarray  # points that changed owner this KF
    birth_ids: np.ndarray  # points seeding an instance created this KF


class StreamFrameMessage(TypedDict):
    type: Literal["stream_frame"]
    frame_id: int
    points: np.ndarray
    obj_ids: np.ndarray
    colors: Any
    normals: np.ndarray | None
    c2w: np.ndarray
    rgb: np.ndarray | None
    ins_map: np.ndarray | None
    sam_map: np.ndarray | None
    corrected_trajectory: list | None
    # Tracking mode only (None in plain stream): permanent point ids aligned with `points`,
    # and the per-frame diagnostics signals (counts + robbed/new point ids).
    point_ids: np.ndarray | None
    track_signals: dict | None


def is_stream_message(data: Any) -> TypeGuard[tuple[np.ndarray, np.ndarray, Any, np.ndarray]]:
    if not isinstance(data, (tuple, list)):
        return False
    if len(data) != 4:
        return False
    points, obj_ids, colors, c2w = data
    return isinstance(points, np.ndarray) and isinstance(obj_ids, np.ndarray) and isinstance(c2w, np.ndarray)


class UpdateMapMessage(TypedDict):
    type: Literal["update_map"]
    frame_id: int
    c2w: np.ndarray
    n_fused: int
    decisions: list


class JumpEventMessage(TypedDict):
    type: Literal["jump_event"]
    frame_id: int
    c2w: np.ndarray
    kf_index: int
    translation_magnitude: float
    rotation_magnitude: float


def is_jump_event_message(data: Any) -> TypeGuard[JumpEventMessage]:
    if not isinstance(data, dict):
        return False
    return data.get("type") == "jump_event" and all(
        key in data for key in ("frame_id", "c2w", "kf_index", "translation_magnitude", "rotation_magnitude")
    )


def is_update_map_message(data: Any) -> TypeGuard[UpdateMapMessage]:
    if not isinstance(data, dict):
        return False
    return data.get("type") == "update_map" and all(
        key in data for key in ("frame_id", "c2w", "n_fused", "decisions")
    )


def is_stream_frame_message(data: Any) -> TypeGuard[StreamFrameMessage]:
    if not isinstance(data, dict):
        return False
    required_keys = ("type", "frame_id", "points", "obj_ids", "colors", "c2w", "rgb", "ins_map")
    if not all(key in data for key in required_keys):
        return False
    if data.get("type") != "stream_frame":
        return False
    if not isinstance(data.get("frame_id"), int):
        return False
    if not isinstance(data.get("points"), np.ndarray):
        return False
    if not isinstance(data.get("obj_ids"), np.ndarray):
        return False
    if not isinstance(data.get("c2w"), np.ndarray):
        return False
    rgb = data.get("rgb")
    ins_map = data.get("ins_map")
    sam_map = data.get("sam_map")
    return (
        (rgb is None or isinstance(rgb, np.ndarray))
        and (ins_map is None or isinstance(ins_map, np.ndarray))
        and (sam_map is None or isinstance(sam_map, np.ndarray))
    )
