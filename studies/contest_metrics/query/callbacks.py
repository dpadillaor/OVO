"""Reconstruye los callbacks geométricos sim/seam/color del contest (espejo de ovo.update_map)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from ovo.utils import instance_utils

from ..common.mapstate import MapState


@dataclass(frozen=True)
class ContestCallbacks:
    """Las tres señales caras que el discriminator invoca en las bandas ambiguas."""

    sim: Callable[[int, int], float | None]              # cos-sim de descriptores CLIP
    seam: Callable[[int, int, tuple], float | None]      # giro de la normal en la costura (grados)
    color: Callable[[int, int, tuple], tuple]            # (ΔE hacia W, ΔE hacia el resto de L)


def build_callbacks(m: MapState) -> ContestCallbacks:
    """Cierra los callbacks sobre el mapa dado; idénticos a los de ovo.py salvo la fuente de datos."""
    ids_flat = m.point_ids.flatten()
    ins = m.points_ins_ids

    def sim(a: int, w: int) -> float | None:
        fa, fw = m.clip_features.get(a), m.clip_features.get(w)
        if fa is None or fw is None:
            return None
        return float(torch.nn.functional.cosine_similarity(fa[0], fw[0], dim=0))

    def _chunk_mask(chunk_ids: tuple) -> torch.Tensor:
        chunk_t = torch.as_tensor(list(chunk_ids), dtype=ids_flat.dtype, device=ids_flat.device)
        return torch.isin(ids_flat, chunk_t)

    def seam(loser: int, winner: int, chunk_ids: tuple) -> float | None:
        if not chunk_ids or m.normals is None or m.xyz is None:
            return None
        chunk, w = _chunk_mask(chunk_ids), ins == winner
        if chunk.sum() == 0 or w.sum() == 0:
            return None
        return instance_utils.seam_normal_angle(m.xyz[chunk], m.normals[chunk], m.xyz[w], m.normals[w])

    def color(loser: int, winner: int, chunk_ids: tuple) -> tuple:
        if not chunk_ids or m.colors is None or m.xyz is None:
            return None, None
        chunk, w = _chunk_mask(chunk_ids), ins == winner
        l = (ins == loser) & ~chunk
        if chunk.sum() == 0 or w.sum() == 0 or l.sum() == 0:
            return None, None
        return instance_utils.seam_color_two_sided(
            m.xyz[chunk], m.colors[chunk], m.xyz[w], m.colors[w], m.xyz[l], m.colors[l]
        )

    return ContestCallbacks(sim=sim, seam=seam, color=color)
