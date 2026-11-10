"""Instance masks on the GT mesh: each mesh vertex takes the instance of its nearest map points.

The projection is the one `run_eval.py` uses to write instance predictions
(`ovo.utils.eval_utils.match_labels_to_vtx`: mode of the 5 nearest points, unassigned points
left out), so a mask here is the prediction that run would have evaluated.

    masks, ids = project(xyz, obj_ids, mesh_vertices)
    masks.shape    # (len(ids), len(mesh_vertices)), one row per instance that reached the mesh

An instance none of whose points wins a vertex has no mask.
"""

import numpy as np
import torch

from ovo.utils.eval_utils import match_labels_to_vtx


def project(points: np.ndarray, ins_ids: np.ndarray, vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """points (P, 3), ins_ids (P,) with -1 for unassigned, vertices (V, 3) ->
    (masks (N, V) bool, ids (N,) int), ids ascending."""
    _, masks, ids = match_labels_to_vtx(torch.as_tensor(ins_ids), torch.as_tensor(points), torch.as_tensor(vertices))
    return masks.numpy().astype(bool), ids.numpy().astype(np.int64)
