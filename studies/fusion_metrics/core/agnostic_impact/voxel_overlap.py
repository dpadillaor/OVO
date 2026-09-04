"""Voxel-overlap pair filter.

Decides which instance pairs *physically superpose* from the pre-fusion
checkpoint, using the real fusion ``VoxelIndex`` (same quantization as the
``overlap_voxel`` criterion). A cell->instances broadphase enumerates only the
pairs that share a cell, then overlap = ``shared / min(|cells_a|, |cells_b|)``
(max-containment: a fragment fully inside a bigger instance scores ~1). The
fusion-metrics eval uses this to score decisions only on pairs that actually
overlap, instead of the whole (mostly trivially-separate) universe.
"""
from __future__ import annotations

import pathlib
from itertools import combinations

import numpy as np
import torch

from ovo.entities.fusion.voxel import VoxelIndex


def overlapping_pairs(ckpt_path: str | pathlib.Path, voxel_size: float = 0.05,
                      th: float = 0.5, origin: float = 0.0) -> set[frozenset[int]]:
    """Instance-id pairs whose pre-fusion clouds overlap by voxel max-containment >= ``th``.

    Returns a set of ``frozenset({i, j})`` so lookup is order-independent.
    """
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    mp = ck["map_params"]
    xyz = torch.as_tensor(np.asarray(mp["xyz"], dtype=np.float32))
    obj = torch.as_tensor(np.asarray(mp["obj_ids"], dtype=np.int64).reshape(-1))

    index: dict[int, VoxelIndex] = {}
    cell_to_ins: dict[int, set[int]] = {}
    for i in torch.unique(obj).tolist():
        if i < 0:
            continue
        vi = VoxelIndex.build(xyz[obj == i], voxel_size, origin)
        index[i] = vi
        for c in vi.cells.tolist():
            cell_to_ins.setdefault(c, set()).add(i)

    candidates: set[tuple[int, int]] = set()
    for ins in cell_to_ins.values():
        if len(ins) > 1:
            candidates.update(combinations(sorted(ins), 2))

    out: set[frozenset[int]] = set()
    for a, b in candidates:
        denom = min(len(index[a]), len(index[b]))
        if denom and index[a].shared(index[b]) / denom >= th:
            out.add(frozenset((a, b)))
    return out
