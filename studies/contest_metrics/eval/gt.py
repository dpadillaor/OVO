"""GT por punto: proyecta la etiqueta de instancia GT (malla) sobre los puntos del mapa.

Sin drift (datos limpios) el frame del mapa OVO coincide con el de la malla GT, así que
un vecino más cercano basta: cada punto OVO hereda la etiqueta del vértice GT más próximo.
Con eso se califica cualquier subconjunto de puntos —merge (instancia entera) o split
(subconjunto)— contra la verdad, que es lo que el eval por-instancia de fusion_metrics no
podía hacer.

Codificación GT: id = clase*1000 + instancia; id <= 0 = void (fondo, se ignora).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from ..common import paths
from ..common.mapstate import MapState


@dataclass
class SceneGT:
    """Etiqueta GT de instancia por punto del mapa, alineada con el orden de MapState."""

    point_ids: np.ndarray                # (N,) id permanente, mismo orden que map_state
    gt_labels: np.ndarray                # (N,) id GT del vértice más cercano (0 = void)
    _row_of_id: dict[int, int]           # point_id -> fila, para subconjuntos por id

    def dominant_over_rows(self, rows: np.ndarray) -> int | None:
        """GT mayoritario (void excluido) sobre un conjunto de FILAS. None si todo void/vacío."""
        labels = self.gt_labels[rows]
        labels = labels[labels > 0]
        if labels.size == 0:
            return None
        vals, counts = np.unique(labels, return_counts=True)
        return int(vals[counts.argmax()])

    def dominant_over_ids(self, point_ids) -> int | None:
        """GT mayoritario sobre un conjunto de POINT_IDS (p.ej. split_points). Ids ausentes se ignoran."""
        rows = [self._row_of_id[int(p)] for p in point_ids if int(p) in self._row_of_id]
        if not rows:
            return None
        return self.dominant_over_rows(np.asarray(rows, dtype=np.int64))


def load_scene_gt(
    scene: str,
    map_state: MapState,
    mesh_root: str | pathlib.Path = paths.MESH_ROOT,
    gt_root: str | pathlib.Path = paths.GT_ROOT,
) -> SceneGT:
    """Proyecta el GT de la malla sobre los puntos del mapa por vecino más cercano (KDTree)."""
    mesh_path = pathlib.Path(mesh_root) / f"{scene}_mesh.ply"
    gt_path = pathlib.Path(gt_root) / f"{scene}.txt"
    if not mesh_path.exists():
        raise FileNotFoundError(f"malla GT no encontrada: {mesh_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"etiquetas GT no encontradas: {gt_path}")

    gt_xyz = np.asarray(o3d.io.read_triangle_mesh(str(mesh_path)).vertices, dtype=np.float64)
    gt_ids = np.array(gt_path.read_text().splitlines(), dtype=np.int64)     # (V,) por vértice
    if gt_xyz.shape[0] != gt_ids.shape[0]:  # etiqueta[i] es del vértice i -> deben cuadrar
        raise ValueError(f"{scene}: malla {gt_xyz.shape[0]} vértices != GT {gt_ids.shape[0]} etiquetas")

    pts = map_state.xyz.numpy().astype(np.float64)                          # (N,3) frame OVO
    _, nearest = cKDTree(gt_xyz).query(pts, k=1)                            # vértice GT más cercano
    gt_labels = gt_ids[nearest]                                            # etiqueta por punto

    pid = map_state.point_ids.numpy().astype(np.int64)
    return SceneGT(pid, gt_labels, {int(p): i for i, p in enumerate(pid)})
