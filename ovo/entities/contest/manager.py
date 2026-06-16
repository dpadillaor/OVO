"""Fachada del mecanismo. OVO tiene un `ContestManager` y lo llama en 4 sitios:

  1. record(...)    en _track_objects            (camino caliente, cada KF, barato)
  2. on_merge(...)  en _fuse_overlapping_instances (al fusionar, junto a cooccurrence)
  3. on_remove(...) en _remove_missing_instances   (al borrar, junto a cooccurrence)
  4. report(...)    en update_map                  (a la cadencia de la fusión)

ESTA FASE NO CAMBIA COMPORTAMIENTO. `report` solo deriva, clasifica y devuelve
los veredictos para loguear/medir. El Actuator (aplicar merge/split al mapa) se
enchufa en la siguiente fase, detrás de su propio flag.
"""
import csv
import json
from typing import Dict, List, Optional, Set

import torch

from .types import Decision, PointId, PairFeatures, Verdict
from .store import ContestStore
from .aggregator import ContestAggregator
from .discriminator import ContestDiscriminator


class ContestManager:
    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("contest", {})
        self.enabled: bool = cfg.get("enabled", True)
        self.store = ContestStore()
        self.aggregator = ContestAggregator(min_count=cfg.get("min_count", 1))
        self.discriminator = ContestDiscriminator(
            high=cfg.get("high", 0.7),
            low=cfg.get("low", 0.5),
            min_mass=cfg.get("min_mass", 50),
            min_split_cont=cfg.get("min_split_cont", 0.0),
            max_rev_split=cfg.get("max_rev_split", 0.5),
            min_persist_split=cfg.get("min_persist_split", 0.1),
            sim_merge=cfg.get("sim_merge", 0.81),
            max_seam_angle=cfg.get("max_seam_angle", 15.0),
        )
        # fase 2 opcional: reevaluar "frontera real" con el union-find de los
        # merges decididos en el MISMO batch (root real, no identidad). Resuelve
        # el caso 95: varios IDs ganadores que en realidad son un solo objeto.
        self.reeval_frontier: bool = cfg.get("reeval_frontier", False)
        # podar el store cada N llamadas a report (la poda recorre los vivos)
        self._prune_every: int = cfg.get("prune_every", 10)
        self._reports: int = 0
        self._verdicts_log: List[dict] = []
        self._output_dir: str | None = None

    def set_output_dir(self, path) -> None:
        self._output_dir = str(path)

    # ---- 1. camino caliente (una llamada por máscara con puntos en disputa) ----
    def record(self, contested_point_ids: torch.Tensor, winner: int) -> None:
        if not self.enabled or contested_point_ids.numel() == 0:
            return
        self.store.record(contested_point_ids.cpu().flatten().tolist(), int(winner))

    # ---- 2 y 3. ciclo de vida (enganchados junto a self.cooccurrence) ----
    def on_merge(self, target: int, source: int) -> None:
        self.store.on_merge(int(target), int(source))

    def on_remove(self, ins: int) -> None:
        self.store.on_remove(int(ins))

    # ---- 4. cadencia de fusión: features -> clasificación -> veredictos ----
    def report(
        self,
        point_ids: torch.Tensor,
        points_ins_ids: torch.Tensor,
        point_obs: torch.Tensor | None = None,
        sim=None,
        seam=None,
        color=None,
    ) -> List[Verdict]:
        if not self.enabled or len(self.store) == 0:
            return []

        self._reports += 1
        point_ids = point_ids.flatten()
        if self._prune_every > 0 and self._reports % self._prune_every == 0:
            live: Set[PointId] = set(point_ids.cpu().tolist())
            self.store.prune_to_live(live)

        pairs = self.aggregator.pairs(self.store, point_ids, points_ins_ids, point_obs)
        by_loser: Dict[int, list] = {}
        for f in pairs:
            by_loser.setdefault(f.loser, []).append(f)

        verdicts: List[Verdict] = []
        for loser, fs in by_loser.items():
            verdicts.append(self.discriminator.classify(loser, fs, sim=sim, seam=seam, color=color))

        verdicts = self._resolve_pairs(verdicts, by_loser)

        if self.reeval_frontier:
            verdicts = self._reeval_frontier(verdicts, by_loser)

        for v in verdicts:
            fs = by_loser.get(v.loser, [])
            best = self._find_pair(v.winner, fs)
            self._verdicts_log.append({
                "report": self._reports,
                "loser": v.loser,
                "winner": v.winner,
                "decision": v.decision.name,
                "reason": v.reason,
                "containment": best.containment if best is not None else None,
                "reverse_containment": best.reverse_containment if best is not None else None,
                "strong_points": best.strong_points if best is not None else None,
                "mass": best.mass if best is not None else None,
                "persistence": best.persistence if best is not None else None,
                "focus": best.focus if best is not None else None,
            })
        if self._output_dir:
            self.dump_verdicts(self._output_dir + "/contest_verdicts.csv")
        return verdicts

    def _resolve_pairs(self, verdicts: List[Verdict], by_loser: Dict[int, list]) -> List[Verdict]:
        """Decide UNA vez por par no-ordenado {A,B}, con los dos containments.

        `classify` mira cada perdedor por separado (A->B y B->A son veredictos
        distintos sobre la misma relación física). Aquí los juntamos: la decisión
        es una propiedad del PAR, no de un lado. Esto elimina de raíz las
        contradicciones (MERGE+SPLIT, SPLIT+SPLIT) sin parchear caso a caso.

        Reemplaza al antiguo `_reconcile` (que solo cubría MERGE-vs-SPLIT y dejaba
        escapar el SPLIT-vs-SPLIT, p.ej. dos trozos de una misma pared).
        """
        # veredictos sin ganador (frontera real, ruido, borde) pasan tal cual
        passthrough = [v for v in verdicts if v.winner is None]
        directed = {(v.loser, v.winner): v for v in verdicts if v.winner is not None}

        def cont(a: int, w: int) -> float:
            f = self._find_pair(w, by_loser.get(a, []))
            return f.containment if f is not None and f.winner == w else 0.0

        resolved: List[Verdict] = []
        seen: Set[tuple] = set()
        for (a, w), vab in directed.items():
            key = (a, w) if a < w else (w, a)
            if key in seen:
                continue
            seen.add(key)
            vba = directed.get((w, a))
            if vba is None:
                resolved.append(vab)  # solo un sentido opina -> se respeta
            else:
                resolved.append(self._join(vab, vba, cont(a, w), cont(w, a), self.discriminator.low))
        return passthrough + resolved

    def _reeval_frontier(self, verdicts: List[Verdict], by_loser: Dict[int, list]) -> List[Verdict]:
        """FASE 2 (opt-in): revisita las "frontera real" con el root del batch.

        La rama strong marca NO_ACTION "frontera real" cuando un perdedor está muy
        contenido en >=2 ganadores con IDs distintos. Pero esos IDs pueden ser el
        MISMO objeto si se fusionan en este mismo batch (caso 95: 78/33/81). Aquí:
          1. construimos un union-find con los MERGE_CONTAINMENT ya resueltos,
          2. recomputamos las raíces de los ganadores fuertes con ese find,
          3. si colapsan a 1 raíz -> el "muro" era ficticio -> MERGE.
        No re-agrega geometría: solo reordena las decisiones del propio batch.
        """
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            parent[find(a)] = find(b)

        # 1. union-find con los merges decididos en este batch (perdedor -> ganador)
        for v in verdicts:
            if v.decision is Decision.MERGE_CONTAINMENT and v.winner is not None:
                union(v.loser, v.winner)

        high = self.discriminator.high
        out: List[Verdict] = []
        for v in verdicts:
            # solo las frontera-real de la rama strong (no la frontera simétrica)
            if v.decision is Decision.NO_ACTION and "frontera real" in v.reason:
                strong = [p for p in by_loser.get(v.loser, []) if p.containment >= high]
                roots = {find(p.winner) for p in strong}
                if strong and len(roots) == 1:
                    best = max(strong, key=lambda p: p.containment)
                    out.append(Verdict(
                        Decision.MERGE_CONTAINMENT, v.loser, winner=best.winner,
                        reason=f"frontera resuelta por root real (cont={best.containment:.2f})",
                    ))
                    continue
            out.append(v)
        return out

    @staticmethod
    def _join(vab: Verdict, vba: Verdict, cab: float, cba: float, low: float) -> Verdict:
        """Resuelve los dos sentidos de un par en un solo veredicto.

        Dirección de mayor contención = candidato a fragmento (el que está más
        metido en el otro). Reglas:
          - hay un MERGE          -> MERGE en la dirección de mayor contención
          - ambos SPLIT (simétrico)-> si la mayor contención >= low, es un casi-merge
                                      (la pared) -> MERGE; si no, es frontera -> NO_ACTION
          - un solo SPLIT          -> se respeta (trozo direccional real)
        """
        hi, hi_c, lo = (vab, cab, vba) if cab >= cba else (vba, cba, vab)
        decs = {vab.decision, vba.decision}

        if Decision.MERGE_CONTAINMENT in decs:
            return Verdict(
                Decision.MERGE_CONTAINMENT, hi.loser, winner=hi.winner,
                reason=f"merge (resuelto por par, cont={hi_c:.2f})",
            )

        if vab.decision is Decision.SPLIT and vba.decision is Decision.SPLIT:
            if hi_c >= low:
                return Verdict(
                    Decision.MERGE_CONTAINMENT, hi.loser, winner=hi.winner,
                    reason=f"split simétrico resuelto -> merge (cont={hi_c:.2f})",
                )
            return Verdict(
                Decision.NO_ACTION, hi.loser,
                reason=f"split simétrico -> frontera (cont={hi_c:.2f}<{low})",
            )

        if hi.decision is Decision.SPLIT:
            return hi
        if lo.decision is Decision.SPLIT:
            return lo
        return hi

    @staticmethod
    def _find_pair(winner: Optional[int], pairs: List[PairFeatures]) -> Optional[PairFeatures]:
        if winner is None:
            return pairs[0] if pairs else None
        for f in pairs:
            if f.winner == winner:
                return f
        return pairs[0] if pairs else None

    @staticmethod
    def summarize(verdicts: List[Verdict]) -> Dict[str, int]:
        """Conteo por tipo de decisión, para loguear sin ruido."""
        out = {d.name: 0 for d in Decision}
        for v in verdicts:
            out[v.decision.name] += 1
        return out

    # ---- serialización con la escena ----
    def to_dict(self) -> dict:
        return {"store": self.store.to_dict()}

    def load_dict(self, data: dict) -> None:
        if data and "store" in data:
            self.store = ContestStore.from_dict(data["store"])

    def dump(self, path: str) -> None:
        """Vuelca el registro crudo por punto a JSON, al final de la escena."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def dump_verdicts(self, path: str) -> None:
        """Vuelca el log acumulado de veredictos a CSV."""
        if not self._verdicts_log:
            return
        fields = [
            "report", "loser", "winner", "decision", "reason",
            "containment", "reverse_containment", "strong_points", "mass", "persistence", "focus",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._verdicts_log)
