"""Open3D visualization of a predicted-instance pair.

Colors: grey context, red A, blue B; GT footprint green if A/B share dominant
GT, else yellow (GT of A) + orange (GT of B). Paint order context->GT->B->A.
"""
from __future__ import annotations

import colorsys

import numpy as np
import open3d as o3d

from core.agnostic_impact.loaders import SceneData
from core.agnostic_impact.merge_decision_eval import analyze_pair

# RGB triples in [0, 1].
COLOR_CONTEXT = (0.80, 0.80, 0.80)
COLOR_A = (0.90, 0.15, 0.15)       # predicted instance A
COLOR_B = (0.15, 0.35, 0.90)       # predicted instance B
COLOR_SAME_GT = (0.20, 0.75, 0.30)  # green: shared dominant GT
COLOR_GT_A = (0.95, 0.85, 0.10)    # yellow: dominant GT of A
COLOR_GT_B = (0.95, 0.55, 0.10)    # orange: dominant GT of B


def _point_cloud(xyz: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def build_pair_cloud(
    scene: SceneData, index_a: int, index_b: int, *, z_max: float | None = None
) -> tuple[o3d.geometry.PointCloud, dict]:
    """Colored point cloud + summary dict for the pair. ``z_max`` clips the
    rendered cloud (z <= z_max) without affecting the summary."""
    pair = analyze_pair(scene, index_a, index_b)

    # Start from a uniform grey context, then paint GT footprint, then the
    # predictions on top.
    colors = np.tile(np.array(COLOR_CONTEXT), (scene.n_points, 1))

    if pair.same_object:
        # Both predictions land on the same GT object -> paint it green.
        colors[scene.gt_ids == pair.dominant_gt_a] = COLOR_SAME_GT
    else:
        # Different GT objects -> yellow for A's GT, orange for B's GT.
        if pair.dominant_gt_b is not None:
            colors[scene.gt_ids == pair.dominant_gt_b] = COLOR_GT_B
        if pair.dominant_gt_a is not None:
            colors[scene.gt_ids == pair.dominant_gt_a] = COLOR_GT_A

    colors[pair.mask_b] = COLOR_B
    colors[pair.mask_a] = COLOR_A

    summary = {
        "index_a": index_a,
        "index_b": index_b,
        "verts_a": int(pair.mask_a.sum()),
        "verts_b": int(pair.mask_b.sum()),
        "same_object": pair.same_object,
        "gt_under_a": pair.gt_under_a,
        "gt_under_b": pair.gt_under_b,
        "dominant_gt_a": pair.dominant_gt_a,
        "dominant_gt_b": pair.dominant_gt_b,
    }

    xyz = scene.xyz
    if z_max is not None:
        keep = scene.xyz[:, 2] <= z_max
        xyz, colors = xyz[keep], colors[keep]

    return _point_cloud(xyz, colors), summary


COLOR_GT_FOOTPRINT = (0.85, 0.78, 0.45)  # khaki: the selected GT object
COLOR_MATCHED = (0.20, 0.80, 0.25)       # green: prediction that won the GT

# Spurious (over-split) fragments share one hue so they read as one category;
# saturation+value vary per fragment so individual pieces stay distinguishable.
_SPURIOUS_HUE = 0.0  # red


def _spurious_shades(n: int) -> list[tuple[float, float, float]]:
    """n shades of ``_SPURIOUS_HUE``: uniform family, distinct per fragment."""
    if n <= 0:
        return []
    sats = np.linspace(1.0, 0.45, n)
    vals = np.linspace(0.65, 1.0, n)
    return [colorsys.hsv_to_rgb(_SPURIOUS_HUE, s, v) for s, v in zip(sats, vals)]


def build_gt_inspection_cloud(
    scene: SceneData,
    gt_id: int,
    matched_col: int | None,
    spurious_cols: list[int],
    *,
    z_max: float | None = None,
) -> o3d.geometry.PointCloud:
    """Grey scene with the GT footprint (khaki), its matched prediction (green)
    and each spurious fragment in a distinct color. Paint order: GT -> spurious -> matched."""
    colors = np.tile(np.array(COLOR_CONTEXT), (scene.n_points, 1))
    colors[scene.gt_ids == gt_id] = COLOR_GT_FOOTPRINT

    shades = _spurious_shades(len(spurious_cols))
    for col, shade in zip(spurious_cols, shades):
        colors[scene.pred_masks[:, col]] = shade
    if matched_col is not None:
        colors[scene.pred_masks[:, matched_col]] = COLOR_MATCHED

    xyz = scene.xyz
    if z_max is not None:
        keep = scene.xyz[:, 2] <= z_max
        xyz, colors = xyz[keep], colors[keep]
    return _point_cloud(xyz, colors)


def build_instance_overlay_cloud(
    gt_xyz: np.ndarray,
    map_xyz: np.ndarray,
    map_obj_ids: np.ndarray,
    obj_id: int,
    *,
    z_max: float | None = None,
) -> tuple[o3d.geometry.PointCloud, int]:
    """GT mesh (grey) + one instance's original OVO points (red), in raw frames.

    Overlays the un-projected checkpoint points on the GT mesh so drift is
    visible. Returns the cloud and the instance's point count.
    """
    inst = map_xyz[map_obj_ids == obj_id]

    xyz = np.vstack([gt_xyz, inst]) if inst.size else gt_xyz
    colors = np.vstack([
        np.tile(np.array(COLOR_CONTEXT), (len(gt_xyz), 1)),
        np.tile(np.array(COLOR_A), (len(inst), 1)),
    ]) if inst.size else np.tile(np.array(COLOR_CONTEXT), (len(gt_xyz), 1))

    if z_max is not None:
        keep = xyz[:, 2] <= z_max
        xyz, colors = xyz[keep], colors[keep]

    return _point_cloud(xyz, colors), int(len(inst))


def visualize_pair(
    scene: SceneData,
    index_a: int,
    index_b: int,
    *,
    point_size: float = 3.0,
    z_max: float | None = None,
    window_name: str | None = None,
) -> dict:
    """Open a blocking Open3D window for the pair; returns the summary dict."""
    pcd, summary = build_pair_cloud(scene, index_a, index_b, z_max=z_max)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name or f"pair {index_a} vs {index_b}")
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()
    return summary


class LiveViewer:
    """Non-blocking Open3D window: show() swaps the cloud, tick() pumps a frame."""

    def __init__(self, *, point_size: float = 3.0, window_name: str = "pair debug"):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=window_name)
        self.vis.get_render_option().point_size = point_size
        self._geom: o3d.geometry.PointCloud | None = None

    def show(self, pcd: o3d.geometry.PointCloud) -> None:
        """Replace the cloud, keeping the camera after the first show."""
        first = self._geom is None
        if self._geom is not None:
            self.vis.remove_geometry(self._geom, reset_bounding_box=False)
        self.vis.add_geometry(pcd, reset_bounding_box=first)
        self._geom = pcd

    def tick(self) -> bool:
        """Pump one render frame. Returns False if the window was closed."""
        alive = self.vis.poll_events()
        self.vis.update_renderer()
        return alive

    def close(self) -> None:
        self.vis.destroy_window()
