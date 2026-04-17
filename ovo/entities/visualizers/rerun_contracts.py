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
