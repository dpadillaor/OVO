"""Mapa semántico materializado desde un checkpoint: lo que el contest necesita para agregar y decidir."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MapState:
    """Estado vivo del mapa (todo en CPU) que alimenta el aggregator y los callbacks geométricos."""

    point_ids: torch.Tensor                  # (N,) id permanente por punto
    points_ins_ids: torch.Tensor             # (N,) instancia dueña por punto (-1 = fondo)
    xyz: torch.Tensor                        # (N, 3)
    normals: torch.Tensor                    # (N, 3)
    colors: torch.Tensor                     # (N, 3)
    clip_features: dict[int, torch.Tensor]   # id de instancia -> descriptor CLIP (1, D)

    @property
    def sizes(self) -> dict[int, int]:
        """nº de puntos por instancia viva (denominador de containment)."""
        vals, counts = self.points_ins_ids.unique(return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}
