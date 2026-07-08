"""Deriva las features normalizadas por par a partir del store y el estado VIVO
del mapa (points_ids, points_ins_ids). Es puro: no toca el mapa, no decide nada,
y no guarda nada (se recalcula bajo demanda, porque caduca en cada merge/poda).

Cada par es (defender, challenger): el defender es el `owner` agregado de los puntos
disputados; el challenger es el `grabber` que se los roba.

Coste: O(nº de puntos en disputa). Se llama a la cadencia de la fusión, no por KF.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch

from .types import InsId, PointId, PairFeatures
from .store import ContestStore


@dataclass
class _PairEvidence:
    """Evidencia cruda acumulada de un par (defender, challenger). Sin ratios: eso es _finalize.
    Dueño de sus propios contadores: _accumulate orquesta, la aritmética de guardado vive aquí."""
    firm_points: int = 0
    total_grabs: int = 0
    persistence_sum: float = 0.0
    persistence_count: int = 0
    exclusive_points: int = 0                                # puntos firmes sin otro challenger
    split_points: List[PointId] = field(default_factory=list)

    def add_grab(self, grabs: int) -> None:
        """Un robo (firme o no): suma los KFs robados al bruto del par."""
        self.total_grabs += grabs

    def add_firm(self, point: PointId, exclusive: bool) -> None:
        """El robo fue firme: el punto cuenta (y es exclusivo si nadie más lo disputa)."""
        self.firm_points += 1
        self.split_points.append(point)
        if exclusive:
            self.exclusive_points += 1

    def add_persist(self, persistence: float) -> None:
        """Persistencia por punto (grabs/claims): alimenta la media del par."""
        self.persistence_sum += persistence
        self.persistence_count += 1


class ContestAggregator:
    def __init__(self, min_grabs: int = 5, firm_tau: float = 0.30) -> None:
        # firmeza de un robo por punto: cuenta si el challenger lo robó >= min_grabs KFs
        # (evidencia) Y esos robos son >= firm_tau de los claims del punto (compromiso).
        # La fracción no infla containment con parpadeos ni castiga fragmentos de vida corta.
        self.min_grabs = min_grabs
        self.firm_tau = firm_tau

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
        """Orquesta: tamaños -> puntos disputados -> acumular evidencia -> normalizar a PairFeatures."""
        if len(store) == 0:
            return []
        size = self._sizes(points_ins_ids)
        ids, owners = self._contested(store, point_ids, points_ins_ids)
        evidence_by_pair, disputed_by_defender = self._accumulate(ids, owners, store)
        return self._finalize(evidence_by_pair, size, disputed_by_defender)

    # ---- pasos -----------------------------------------------------------
    def _sizes(self, points_ins_ids: torch.Tensor) -> Dict[InsId, int]:
        """Nº de puntos de cada instancia viva -> denominador de la contención."""
        ins_ids, point_counts = points_ins_ids.unique(return_counts=True)
        return {int(ins): int(n) for ins, n in zip(ins_ids, point_counts)}

    def _contested(
        self, store: ContestStore, point_ids: torch.Tensor, points_ins_ids: torch.Tensor,
    ) -> Tuple[List[PointId], List[InsId]]:
        """Ids de los puntos en disputa y su dueño actual (según el mapa vivo)."""
        contested = torch.tensor(list(store.points()), device=point_ids.device)
        owners = self._owner_lookup(point_ids, points_ins_ids, contested).tolist()
        return contested.tolist(), owners

    def _is_firm(self, grabs: int, persistence: float) -> bool:
        """Firme si el challenger robó el punto suficientes veces Y lo tuvo una fracción real de su vida:
          - evidencia:   grabs >= min_grabs         (no es ruido de pocos KFs).
          - compromiso:  persistence >= firm_tau    (no es un roce de refilón).
        (firm_tau=0 -> persistence>=0 siempre cierto -> se reduce a solo evidencia.)
        """
        return grabs >= self.min_grabs and persistence >= self.firm_tau

    def _accumulate(
        self, ids: List[PointId], owners: List[InsId], store: ContestStore,
    ) -> Tuple[Dict[Tuple[InsId, InsId], _PairEvidence], Dict[InsId, int]]:
        """Recorre los puntos disputados y llena la evidencia cruda por par. Sin ratios.

        NOTA: el denominador de persistence es store.claims_of(point) (reclamos del punto bajo
        cualquier máscara, mismo reloj semántico que los robos) -> grabs/claims en [0,1]. No es
        point_obs (visibilidad del SLAM, otro reloj, no acotado).
        """
        evidence_by_pair: Dict[Tuple[InsId, InsId], _PairEvidence] = defaultdict(_PairEvidence)
        disputed_by_defender: Dict[InsId, int] = defaultdict(int)
        for point, defender in zip(ids, owners):
            claims = store.claims_of(point)
            if defender < 0 or claims <= 0:  # sin dueño (podado) o sin reclamos -> se ignora
                continue
            grab_counts = store.grabbers_of(point)
            n_challengers = sum(1 for inst in grab_counts if inst != defender)
            any_grab = False
            for challenger, grabs in grab_counts.items():
                if challenger == defender:
                    continue
                any_grab = True
                persistence = grabs / claims
                evidence = evidence_by_pair[(defender, challenger)]
                evidence.add_grab(grabs)
                if self._is_firm(grabs, persistence):
                    evidence.add_firm(point, exclusive=(n_challengers == 1))
                evidence.add_persist(persistence)
            if any_grab:
                disputed_by_defender[defender] += 1
        return evidence_by_pair, disputed_by_defender

    def _finalize(
        self,
        evidence_by_pair: Dict[Tuple[InsId, InsId], _PairEvidence],
        size: Dict[InsId, int],
        disputed_by_defender: Dict[InsId, int],
    ) -> List[PairFeatures]:
        """Normaliza la evidencia cruda a PairFeatures. Solo pares con >=1 punto firme."""
        features: List[PairFeatures] = []
        for (defender, challenger), evidence in evidence_by_pair.items():
            firm_points = evidence.firm_points
            if firm_points == 0:  # sin puntos firmes -> no era un par real (igual que el firm.items() de antes)
                continue
            defender_size = size.get(defender, 0)
            if defender_size == 0:
                continue
            challenger_size = size.get(challenger, 0)
            reverse = evidence_by_pair.get((challenger, defender))
            reverse_firm_points = reverse.firm_points if reverse is not None else 0
            reverse_containment = (reverse_firm_points / challenger_size) if challenger_size else 0.0
            mean_persistence = evidence.persistence_sum / evidence.persistence_count if evidence.persistence_count > 0 else 0.0
            n_disputed = disputed_by_defender.get(defender, 0)
            focus = firm_points / n_disputed if n_disputed > 0 else 0.0
            exclusivity = evidence.exclusive_points / firm_points
            features.append(
                PairFeatures(
                    defender=defender,
                    challenger=challenger,
                    containment=firm_points / defender_size,
                    reverse_containment=reverse_containment,
                    firm_points=firm_points,
                    total_grabs=evidence.total_grabs,
                    persistence=mean_persistence,
                    focus=focus,
                    exclusivity=exclusivity,
                    split_points=tuple(evidence.split_points),
                )
            )
        return features
