"""I/O for pair inspection: scene mesh, GT instance ids, prediction masks.

Vertex ``k`` is the same point across mesh ply, GT ``{scene}.txt`` and every
prediction mask. A prediction is identified by its column index (= line order).
GT id encoding: ``class = id // 1000``, ``instance = id % 1000``; ``0`` = void.
"""
from __future__ import annotations

import csv
import sys
import pathlib
from dataclasses import dataclass

import numpy as np
import open3d as o3d
import torch

# Make the repo root importable so we can reuse the production prediction loader.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ovo.utils.io_utils import load_instance_preds  # noqa: E402


@dataclass
class SceneData:
    """Per-vertex scene data. xyz (N,3), rgb (N,3)|None, gt_ids (N,), pred_masks (N,M) bool."""

    xyz: np.ndarray
    rgb: np.ndarray | None
    gt_ids: np.ndarray
    pred_masks: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def n_pred_instances(self) -> int:
        return int(self.pred_masks.shape[1])


@dataclass
class FusionDecision:
    """One fusion merge/split decision on a pair. merged = fusion united them."""

    i1: int
    i2: int
    merged: bool
    reason: str | None       # Rejection cause; None when merged
    accept_mode: str | None  # Accept branch A/B/AB; None when not merged or column absent


@dataclass
class MapData:
    """Raw OVO map (pre-projection). xyz (P,3), rgb (P,3), obj_ids (P,); -1 = unassigned."""

    xyz: np.ndarray
    rgb: np.ndarray
    obj_ids: np.ndarray

    @property
    def n_points(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def instance_ids(self) -> np.ndarray:
        """Sorted instance ids present in the map (excludes -1)."""
        ids = np.unique(self.obj_ids)
        return ids[ids >= 0]


def load_ovo_map(exp_path: str | pathlib.Path, scene: str) -> MapData:
    """Load the raw OVO map (``{exp_path}/{scene}/ovo_map.ckpt``): point cloud + obj_ids."""
    ckpt_path = pathlib.Path(exp_path) / scene / "ovo_map.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"OVO map checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    mp = ckpt["map_params"]
    xyz = np.asarray(mp["xyz"], dtype=np.float64)
    rgb = np.asarray(mp["color"], dtype=np.float64) / 255.0  # stored as uint8 0-255
    obj_ids = np.asarray(mp["obj_ids"], dtype=np.int64).reshape(-1)
    return MapData(xyz=xyz, rgb=rgb, obj_ids=obj_ids)


def reproject_ids_to_gt(
    obj_ids: np.ndarray, ovo_xyz: np.ndarray, gt_xyz: np.ndarray
) -> set[int]:
    """Project per-OVO-point ``obj_ids`` onto the GT mesh; return ids winning >=1 vertex.

    Same KDTree majority vote production uses to evaluate (match_labels_to_vtx):
    an instance whose points reach no GT vertex is absent from the result.
    """
    from ovo.utils import eval_utils  # local import: pulls torch/sklearn

    _, _, matched = eval_utils.match_labels_to_vtx(
        torch.as_tensor(obj_ids.reshape(-1)),
        torch.as_tensor(np.asarray(ovo_xyz, dtype=np.float64)),
        torch.as_tensor(np.asarray(gt_xyz, dtype=np.float64)),
    )
    ids = np.unique(matched.numpy())
    return {int(i) for i in ids if i >= 0}


def load_scene_mesh(
    mesh_root: str | pathlib.Path, scene: str
) -> tuple[np.ndarray, np.ndarray | None]:
    """Load vertex positions (and colors, if any) from ``{scene}_mesh.ply``."""
    mesh_path = pathlib.Path(mesh_root) / f"{scene}_mesh.ply"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Scene mesh not found: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    xyz = np.asarray(mesh.vertices, dtype=np.float64)
    rgb = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None
    return xyz, rgb


def load_gt_instance_ids(gt_root: str | pathlib.Path, scene: str) -> np.ndarray:
    """Load the per-vertex ground-truth instance ids from ``{scene}.txt``."""
    gt_path = pathlib.Path(gt_root) / f"{scene}.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground-truth file not found: {gt_path}")

    with open(gt_path) as f:
        return np.array(f.read().splitlines(), dtype=np.int64)


def load_scene_data(
    scene: str,
    *,
    mesh_root: str | pathlib.Path,
    gt_root: str | pathlib.Path,
    exp_path: str | pathlib.Path,
) -> SceneData:
    """Load mesh, GT and predictions for ``scene``; cross-check vertex counts."""
    xyz, rgb = load_scene_mesh(mesh_root, scene)
    gt_ids = load_gt_instance_ids(gt_root, scene)
    preds = load_instance_preds(str(exp_path), scene)
    pred_masks = np.asarray(preds["pred_masks"], dtype=bool)

    n_points = xyz.shape[0]
    if gt_ids.shape[0] != n_points:
        raise ValueError(
            f"Vertex/GT mismatch for '{scene}': mesh has {n_points} vertices "
            f"but GT file has {gt_ids.shape[0]} entries."
        )
    # An empty prediction file yields a (0, 0) array; only validate a populated one.
    if pred_masks.size and pred_masks.shape[0] != n_points:
        raise ValueError(
            f"Vertex/prediction mismatch for '{scene}': mesh has {n_points} "
            f"vertices but masks have {pred_masks.shape[0]} rows."
        )

    return SceneData(xyz=xyz, rgb=rgb, gt_ids=gt_ids, pred_masks=pred_masks)


def load_pre_fusion_scene(
    ckpt_path: str | pathlib.Path,
    scene: str,
    *,
    mesh_root: str | pathlib.Path,
    gt_root: str | pathlib.Path,
) -> tuple[SceneData, np.ndarray, int]:
    """Pre-fusion instances from ``pre_fusion.ckpt`` projected onto the GT mesh.

    Returns ``(SceneData, col_obj_ids, n_instances)`` where ``col_obj_ids[j]``
    is the map obj_id of prediction column ``j`` and ``n_instances`` is the true
    pre-fusion instance count (unique obj_ids >= 0 in the map, pre-projection).
    One KDTree projection yields both the GT-aligned masks and the obj_id->column map.
    """
    from ovo.utils import eval_utils  # local import: pulls torch/sklearn

    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pre-fusion checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    map_params = ckpt["map_params"]

    obj_ids_np = np.asarray(map_params["obj_ids"], dtype=np.int64).reshape(-1)
    n_instances = int(np.unique(obj_ids_np[obj_ids_np >= 0]).size)
    obj_ids = torch.as_tensor(obj_ids_np)
    ovo_xyz = torch.as_tensor(np.asarray(map_params["xyz"], dtype=np.float64))

    gt_xyz, gt_rgb = load_scene_mesh(mesh_root, scene)
    gt_ids = load_gt_instance_ids(gt_root, scene)
    if gt_ids.shape[0] != gt_xyz.shape[0]:
        raise ValueError(
            f"Vertex/GT mismatch for '{scene}': mesh has {gt_xyz.shape[0]} vertices "
            f"but GT file has {gt_ids.shape[0]} entries."
        )

    # Project the pre-fusion point partition onto the GT mesh: each GT vertex
    # inherits the majority obj_id of its nearest OVO points.
    _, masks, matched_ids = eval_utils.match_labels_to_vtx(
        obj_ids, ovo_xyz, torch.from_numpy(gt_xyz)
    )
    pred_masks = masks.numpy().T.astype(bool)  # (M, V) -> (V, M)

    scene_data = SceneData(xyz=gt_xyz, rgb=gt_rgb, gt_ids=gt_ids, pred_masks=pred_masks)
    return scene_data, matched_ids.numpy().astype(np.int64), n_instances


def load_pre_fusion_map(ckpt_path: str | pathlib.Path) -> MapData:
    """Raw pre-fusion OVO point cloud (xyz, rgb, obj_ids) from ``pre_fusion.ckpt``.

    These are the instances' original (un-projected) points, in the map's own
    frame -- which, under drift, differs from the GT mesh frame.
    """
    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pre-fusion checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    mp = ckpt["map_params"]
    xyz = np.asarray(mp["xyz"], dtype=np.float64)
    rgb = np.asarray(mp["color"], dtype=np.float64)
    if rgb.size and rgb.max() > 1.0:  # stored as uint8 0-255
        rgb = rgb / 255.0
    obj_ids = np.asarray(mp["obj_ids"], dtype=np.int64).reshape(-1)
    return MapData(xyz=xyz, rgb=rgb, obj_ids=obj_ids)


@dataclass
class DriftInputs:
    """Raw drift-epoch inputs from ``pre_fusion.ckpt``: creation frame per obj_id + jump frame."""

    created_at_frame: dict[int, int]  # obj_id -> frame id where the instance was created
    jump_frame: int                   # frame id of the (single) jump drift


def load_drift_inputs(ckpt_path: str | pathlib.Path) -> DriftInputs:
    """Per-instance creation frame and the jump frame from ``pre_fusion.ckpt`` (one torch.load).

    Raises if the checkpoint holds anything other than exactly one jump or if any
    instance lacks ``created_at_frame``.
    """
    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pre-fusion checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    jumps = ckpt.get("jump_events", [])
    if len(jumps) != 1:
        raise ValueError(
            f"Expected exactly one jump in {ckpt_path.name}, found {len(jumps)}. "
            f"Drift-epoch classification assumes a single jump."
        )
    jump_frame = int(jumps[0]["frame_id"])

    ovo = ckpt["ovo_params"]
    created_at_frame: dict[int, int] = {}
    for oid in np.asarray(ovo["ins_3d_ids"]).reshape(-1).tolist():
        val = ovo.get(f"ins3d_{int(oid)}_created_at_frame")
        if val is None:
            raise ValueError(
                f"Instance {int(oid)} has no created_at_frame in {ckpt_path.name}; "
                f"cannot assign a drift epoch."
            )
        created_at_frame[int(oid)] = int(val)

    return DriftInputs(created_at_frame=created_at_frame, jump_frame=jump_frame)


def instance_mask(scene: SceneData, index: int) -> np.ndarray:
    """Return the boolean mask of predicted instance ``index`` (column ``index``)."""
    n = scene.n_pred_instances
    if not 0 <= index < n:
        raise IndexError(f"Instance index {index} out of range [0, {n}).")
    return scene.pred_masks[:, index]


def column_obj_ids(
    exp_path: str | pathlib.Path,
    scene: str,
    *,
    mesh_root: str | pathlib.Path,
) -> np.ndarray:
    """obj_id per prediction column: result[j] = obj_id of column j.

    Reproduces the write-time GT-mesh projection (match_labels_to_vtx), so it
    re-runs the KDTree over the full map (a few seconds). Robust to pruned ids.
    """
    from ovo.utils import eval_utils  # local import: pulls torch/sklearn

    map_data = load_ovo_map(exp_path, scene)
    gt_xyz, _ = load_scene_mesh(mesh_root, scene)

    obj_ids = torch.from_numpy(map_data.obj_ids)
    points = torch.from_numpy(map_data.xyz)
    mesh_vtx = torch.from_numpy(gt_xyz)
    _, _, matched = eval_utils.match_labels_to_vtx(obj_ids, points, mesh_vtx)
    return matched.numpy().astype(np.int64)


def obj_id_to_column(matched: np.ndarray, obj_id: int) -> int:
    """Prediction column index for ``obj_id`` (``matched`` from column_obj_ids)."""
    hits = np.flatnonzero(matched == obj_id)
    if hits.size == 0:
        raise ValueError(
            f"obj_id {obj_id} not among prediction instances. "
            f"Available: {matched.tolist()}"
        )
    return int(hits[0])


_ACCEPTED = "ACCEPTED"


def load_fusion_decisions(
    csv_path: str | pathlib.Path,
) -> tuple[list[dict[str, str]], int]:
    """Raw rows of ``fusion_decisions.csv`` + the single frame_id they share.

    Raises if the file is empty or spans more than one fusion event (frame_id),
    since a pre-fusion checkpoint snapshots exactly one event.
    """
    csv_path = pathlib.Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Fusion decisions file not found: {csv_path}")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No fusion decisions in {csv_path}.")

    frame_ids = {int(r["frame_id"]) for r in rows}
    if len(frame_ids) != 1:
        raise ValueError(
            f"Expected a single fusion event in {csv_path}, found frame_ids "
            f"{sorted(frame_ids)}. Multi-frame evaluation is not supported."
        )
    return rows, frame_ids.pop()


def parse_fusion_decision(row: dict[str, str]) -> FusionDecision:
    """One CSV row -> typed decision. merged = (result == 'ACCEPTED')."""
    merged = row["result"] == _ACCEPTED
    return FusionDecision(
        i1=int(row["i1"]),
        i2=int(row["i2"]),
        merged=merged,
        reason=None if merged else (row.get("reason") or None),
        accept_mode=(row.get("accept_mode") or None) if merged else None,
    )


def gt_ids_under_mask(scene: SceneData, mask: np.ndarray) -> dict[int, int]:
    """{gt_id: vertex_count} under ``mask``, sorted desc. Void ids (<=0) excluded."""
    ids = scene.gt_ids[mask]
    ids = ids[ids > 0]
    if ids.size == 0:
        return {}
    uniq, counts = np.unique(ids, return_counts=True)
    order = np.argsort(counts)[::-1]
    return {int(uniq[i]): int(counts[i]) for i in order}
