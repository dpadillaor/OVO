"""Deriva las features normalizadas por par a partir del store y el estado VIVO
del mapa (points_ids, points_ins_ids). Es puro: no toca el mapa, no decide nada,
y no guarda nada (se recalcula bajo demanda, porque caduca en cada merge/poda).

Cada par es (defender, challenger): el defender es el `owner` agregado de los puntos
disputados; el challenger es el `grabber` que se los roba.

Coste: O(nº de puntos en disputa). Se llama a la cadencia de la fusión, no por KF.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from .types import InsId, PointId, PairFeatures
from .store import ContestStore


class ContestAggregator:
    def __init__(self, min_count: int = 1) -> None:
        # un robo cuenta como "firme" si el punto vio al challenger en >= min_count KFs
        self.min_count = min_count

    def _owner_lookup(
        self,
        point_ids: torch.Tensor,
        points_ins_ids: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """Dueño actual de cada id en `query` (-2 si el punto ya no existe).

        Usa searchsorted sobre los ids ordenados -> O(M log N), sin construir un
        diccionario de millones de entradas.
        """
        sorted_ids, order = torch.sort(point_ids)
        pos = torch.searchsorted(sorted_ids, query).clamp(max=sorted_ids.numel() - 1)
        found = sorted_ids[pos] == query
        owners = torch.full_like(query, -2, dtype=points_ins_ids.dtype)
        owners[found] = points_ins_ids[order[pos[found]]]
        return owners

    def pairs(
        self,
        store: ContestStore,
        point_ids: torch.Tensor,
        points_ins_ids: torch.Tensor,
        point_obs: torch.Tensor | None = None,
    ) -> List[PairFeatures]:
        if len(store) == 0:
            return []

        # tamaño (nº de puntos) de cada instancia viva -> denominador de la contención
        vals, counts = points_ins_ids.unique(return_counts=True)
        size: Dict[InsId, int] = {int(v): int(c) for v, c in zip(vals, counts)}

        contested = torch.tensor(list(store.points()), device=point_ids.device)
        owners = self._owner_lookup(point_ids, points_ins_ids, contested).tolist()
        ids = contested.tolist()

        # NOTA: point_obs (visibilidad geométrica del SLAM) ya NO es el denominador de
        # persistence. Iba en otro reloj (mapping, gateado por movimiento) que ni acota
        # el cociente en [0,1] (podía dar >1). El denominador correcto es store.claims_of(p):
        # nº de reclamos del punto bajo cualquier máscara, mismo reloj semántico que los robos.

        firm: Dict[Tuple[InsId, InsId], int] = defaultdict(int)
        total_grabs: Dict[Tuple[InsId, InsId], int] = defaultdict(int)
        persistence_sum: Dict[Tuple[InsId, InsId], float] = defaultdict(float)
        persistence_cnt: Dict[Tuple[InsId, InsId], int] = defaultdict(int)
        disputed_total: Dict[InsId, int] = defaultdict(int)
        split_points: Dict[Tuple[InsId, InsId], List[PointId]] = defaultdict(list)
        for p, d in zip(ids, owners):
            if d < 0:  # punto podado o sin dueño -> se ignora (y se podará del store)
                continue
            denom = store.claims_of(p)  # total de reclamos -> c <= denom -> persistence en [0,1]
            any_grab = False
            for ch, c in store.grabbers_of(p).items():
                if ch == d:
                    continue
                any_grab = True
                total_grabs[(d, ch)] += c
                if c >= self.min_count:
                    firm[(d, ch)] += 1
                    split_points[(d, ch)].append(p)
                if denom > 0:
                    persistence_sum[(d, ch)] += c / denom
                    persistence_cnt[(d, ch)] += 1
            if any_grab:
                disputed_total[d] += 1

        out: List[PairFeatures] = []
        for (d, ch), fp in firm.items():
            nd = size.get(d, 0)
            if nd == 0:
                continue
            nch = size.get(ch, 0)
            rev = (firm.get((ch, d), 0) / nch) if nch else 0.0
            pcnt = persistence_cnt.get((d, ch), 0)
            pers = persistence_sum.get((d, ch), 0.0) / pcnt if pcnt > 0 else 0.0
            dt = disputed_total.get(d, 0)
            focus_val = fp / dt if dt > 0 else 0.0
            out.append(
                PairFeatures(
                    defender=d,
                    challenger=ch,
                    containment=fp / nd,
                    reverse_containment=rev,
                    firm_points=fp,
                    total_grabs=total_grabs[(d, ch)],
                    persistence=pers,
                    focus=focus_val,
                    split_points=tuple(split_points.get((d, ch), ())),
                )
            )
        return out
