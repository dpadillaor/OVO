"""Deriva las features normalizadas por par a partir del store y el estado VIVO
del mapa (points_ids, points_ins_ids). Es puro: no toca el mapa, no decide nada,
y no guarda nada (se recalcula bajo demanda, porque caduca en cada merge/poda).

Coste: O(nº de puntos en disputa). Se llama a la cadencia de la fusión, no por KF.
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from .types import InsId, PairFeatures
from .store import ContestStore


class ContestAggregator:
    def __init__(self, min_count: int = 1) -> None:
        # un voto cuenta como "fuerte" si el punto vio al ganador en >= min_count KFs
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

        # lookup per-point total observations
        if point_obs is not None:
            total_obs = self._owner_lookup(point_ids, point_obs, contested).tolist()
        else:
            total_obs = [0] * len(ids)

        strong: Dict[Tuple[InsId, InsId], int] = defaultdict(int)
        mass: Dict[Tuple[InsId, InsId], int] = defaultdict(int)
        persistence_sum: Dict[Tuple[InsId, InsId], float] = defaultdict(float)
        persistence_cnt: Dict[Tuple[InsId, InsId], int] = defaultdict(int)
        contested_total: Dict[InsId, int] = defaultdict(int)
        split_points: Dict[Tuple[InsId, InsId], List[PointId]] = defaultdict(list)
        for p, a, tobs in zip(ids, owners, total_obs):
            if a < 0:  # punto podado o sin dueño -> se ignora (y se podará del store)
                continue
            any_contest = False
            for w, c in store.winners_of(p).items():
                if w == a:
                    continue
                any_contest = True
                mass[(a, w)] += c
                if c >= self.min_count:
                    strong[(a, w)] += 1
                    split_points[(a, w)].append(p)
                if tobs > 0:
                    persistence_sum[(a, w)] += c / tobs
                    persistence_cnt[(a, w)] += 1
            if any_contest:
                contested_total[a] += 1

        out: List[PairFeatures] = []
        for (a, w), sp in strong.items():
            na = size.get(a, 0)
            if na == 0:
                continue
            nw = size.get(w, 0)
            rev = (strong.get((w, a), 0) / nw) if nw else 0.0
            pcnt = persistence_cnt.get((a, w), 0)
            pers = persistence_sum.get((a, w), 0.0) / pcnt if pcnt > 0 else 0.0
            ct = contested_total.get(a, 0)
            focus_val = sp / ct if ct > 0 else 0.0
            out.append(
                PairFeatures(
                    loser=a,
                    winner=w,
                    containment=sp / na,
                    reverse_containment=rev,
                    strong_points=sp,
                    mass=mass[(a, w)],
                    persistence=pers,
                    focus=focus_val,
                    split_points=tuple(split_points.get((a, w), ())),
                )
            )
        return out
