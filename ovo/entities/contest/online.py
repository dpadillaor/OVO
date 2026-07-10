"""Agregación online incremental del contest (R1) — modo SOMBRA.

Mantiene las PairFeatures VIVAS por defender, recomputando solo los defenders "sucios" (los tocados
desde el último refresh) en vez de re-derivar todo el store desde cero como el batch (`report()`).

Diseño (ver docs/contest_online_design_2026-07-09.md):
  - `punto` = verdad (el store, ya incremental). `par/instancia` = se RECOMPUTAN por defender sucio
    (no se mantienen con +=/-=, para esquivar la reversión de exclusivity/media).
  - unidad sucia = DEFENDER (el radio de impacto de un robo son los pares de un defender).
  - v1 "reparte, no reduce": `features_for` hace bucket global + finalize de los sucios (O(all
    contested) por refresh). El índice owner->puntos (reduce) se difiere (§11.1).

NO decide ni muta el mapa: solo mantiene el cache de features para que el ShadowValidator lo compare
con el batch. El coordinator recibe los eventos de propiedad (assignment/merge/remove/split) y marca
sucio; el `refresh()` recomputa y actualiza el cache.
"""
from typing import Dict, Iterable, List, Set

import torch

from .types import InsId, PairFeatures, PointId, RefreshStats
from .store import ContestStore
from .aggregator import ContestAggregator


class DirtyDefenders:
    """Set de defenders pendientes de recomputar. Contenedor tonto; lo alimentan los call sites."""

    def __init__(self) -> None:
        self._s: Set[InsId] = set()

    def mark(self, defender: InsId) -> None:
        self._s.add(int(defender))

    def mark_many(self, defenders: Iterable[InsId]) -> None:
        for d in defenders:
            self._s.add(int(d))

    def drop(self, defender: InsId) -> None:
        self._s.discard(int(defender))

    def take(self) -> Set[InsId]:
        """Devuelve los sucios y vacía el set (coalesce: N marcas -> 1 recompute)."""
        s = self._s
        self._s = set()
        return s

    def __len__(self) -> int:
        return len(self._s)


class FeatureCache:
    """PairFeatures vivas por defender: {defender: [PairFeatures]}."""

    def __init__(self) -> None:
        self._d: Dict[InsId, List[PairFeatures]] = {}

    def update(self, defender: InsId, pairs: List[PairFeatures]) -> None:
        self._d[int(defender)] = list(pairs)

    def get(self, defender: InsId) -> List[PairFeatures]:
        return self._d.get(int(defender), [])

    def drop(self, defender: InsId) -> None:
        self._d.pop(int(defender), None)

    def all(self) -> Dict[InsId, List[PairFeatures]]:
        return self._d


class IncrementalAggregator(ContestAggregator):
    """Agregador con alcance: recomputa features solo de los defenders pedidos.

    v1: reusa la maquinaria del batch (`_sizes`/`_contested`/`_accumulate`/`_finalize`) y filtra por
    defender al final -> mismos números que el batch por construcción (el shadow valida la COMPLETITUD
    del dirty set, no la aritmética). Coste O(all contested) por llamada ("reparte, no reduce")."""

    def features_for(
        self,
        store: ContestStore,
        defenders: Iterable[InsId],
        point_ids: torch.Tensor,
        points_ins_ids: torch.Tensor,
        point_obs: torch.Tensor | None = None,
    ) -> Dict[InsId, List[PairFeatures]]:
        want: Set[InsId] = {int(d) for d in defenders}
        if not want:
            return {}
        out: Dict[InsId, List[PairFeatures]] = {d: [] for d in want}
        if len(store) == 0:
            return out
        size = self._sizes(points_ins_ids)
        ids, owners = self._contested(store, point_ids, points_ins_ids)
        evidence_by_pair, disputed_by_defender = self._accumulate(ids, owners, store)
        for pf in self._finalize(evidence_by_pair, size, disputed_by_defender):
            if pf.defender in want:
                out[pf.defender].append(pf)
        return out


class OnlineContestCoordinator:
    """Orquesta el agregado online: recibe eventos de propiedad, marca sucio, recomputa en refresh.

    Contrato-invariante (R2): toda mutación de `points_ins_ids` DEBE notificarse por
    note_assignment/on_merge/on_remove/on_split, o el dirty set se queda viejo en silencio. El shadow
    test caza los incumplimientos.
    """

    def __init__(self, store: ContestStore, min_grabs: int = 5, firm_tau: float = 0.30) -> None:
        self.store = store
        self.aggregator = IncrementalAggregator(min_grabs=min_grabs, firm_tau=firm_tau)
        self.cache = FeatureCache()
        self.dirty = DirtyDefenders()
        # puntos afectados por eventos estructurales (merge/remove/split); sus owners se resuelven en
        # refresh (que tiene points_ins_ids) y se marcan sucios entonces.
        self._pending_points: Set[PointId] = set()

    # ---- eventos de propiedad (la costura con OVO) -----------------------
    def note_assignment(self, owner_ids: torch.Tensor) -> None:
        """Puntos asignados bajo una máscara este KF -> sus owners cambian features (grab/claim/growth)."""
        if owner_ids is None or owner_ids.numel() == 0:
            return
        for o in owner_ids.flatten().unique().cpu().tolist():
            if o >= 0:
                self.dirty.mark(o)

    def on_merge(self, target: InsId, source: InsId) -> None:
        """LLAMAR ANTES de store.on_merge: captura los puntos de `source` mientras siguen en el store."""
        self._pending_points.update(self.store.grabber_points(source))
        self.dirty.mark(target)
        self.cache.drop(source)
        self.dirty.drop(source)

    def on_remove(self, ins: InsId) -> None:
        """LLAMAR ANTES de store.on_remove: los puntos que `ins` robaba pierden un challenger."""
        self._pending_points.update(self.store.grabber_points(ins))
        self.cache.drop(ins)
        self.dirty.drop(ins)

    def on_split(self, defender: InsId, challenger: InsId, split_points: Iterable[PointId]) -> None:
        if split_points:
            self._pending_points.update(int(p) for p in split_points)
        self.dirty.mark(defender)
        self.dirty.mark(challenger)

    # ---- recompute -------------------------------------------------------
    def refresh(
        self,
        point_ids: torch.Tensor,
        points_ins_ids: torch.Tensor,
        point_obs: torch.Tensor | None = None,
    ) -> RefreshStats:
        point_ids = point_ids.flatten()
        # resolver owners de los puntos pendientes (eventos estructurales) y marcarlos sucios
        if self._pending_points:
            q = torch.tensor(sorted(self._pending_points), device=point_ids.device)
            owners = self.aggregator._owner_lookup(point_ids, points_ins_ids, q)
            self.dirty.mark_many(o for o in owners.cpu().tolist() if o >= 0)
            self._pending_points = set()
        dirty = self.dirty.take()
        feats = self.aggregator.features_for(self.store, dirty, point_ids, points_ins_ids, point_obs)
        n_pairs = 0
        for d, pairs in feats.items():
            self.cache.update(d, pairs)
            n_pairs += len(pairs)
        return RefreshStats(n_dirty=len(dirty), n_pairs=n_pairs)

    def all_features(self) -> Dict[InsId, List[PairFeatures]]:
        """Features vivas, sin los defenders sin pares (para casar con la ausencia en el batch)."""
        return {d: v for d, v in self.cache.all().items() if v}
