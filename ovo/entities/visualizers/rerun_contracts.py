from typing import Any, Literal, TypeGuard, TypedDict

import numpy as np


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
