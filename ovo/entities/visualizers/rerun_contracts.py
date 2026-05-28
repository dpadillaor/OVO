from typing import Any, Literal, TypeGuard, TypedDict

import numpy as np


class FusionMessage(TypedDict):
    type: Literal["fusion"]
    points: np.ndarray
    before_ids: np.ndarray
    after_ids: np.ndarray
    frame_id: int


class LoopClosureMessage(TypedDict):
    type: Literal["loop_closure"]
    pcd_before: np.ndarray
    pcd_after: np.ndarray
    ids: np.ndarray
    traj_before: dict[int, np.ndarray]
    traj_after: dict[int, np.ndarray]
    frame_id: int


class StreamMessage(TypedDict):
    points: np.ndarray
    obj_ids: np.ndarray
    colors: Any
    c2w: np.ndarray


class StreamFrameMessage(TypedDict):
    type: Literal["stream_frame"]
    frame_id: int
    points: np.ndarray
    obj_ids: np.ndarray
    colors: Any
    c2w: np.ndarray
    rgb: np.ndarray | None
    ins_map: np.ndarray | None
    sam_map: np.ndarray | None
    corrected_trajectory: list | None


def is_fusion_message(data: Any) -> TypeGuard[FusionMessage]:
    if not isinstance(data, dict):
        return False
    return data.get("type") == "fusion" and all(
        key in data for key in ("points", "before_ids", "after_ids", "frame_id")
    )


def is_loop_closure_message(data: Any) -> TypeGuard[LoopClosureMessage]:
    if not isinstance(data, dict):
        return False
    return data.get("type") == "loop_closure" and all(
        key in data for key in ("pcd_before", "pcd_after", "ids", "traj_before", "traj_after")
    )


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
