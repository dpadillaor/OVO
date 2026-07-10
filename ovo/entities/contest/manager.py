"""Fachada del mecanismo. OVO tiene un `ContestManager` y lo llama en varios sitios:

  1. record_grab/record_claim/record_sighting  en _track_objects  (camino caliente, cada KF, barato)
  2. on_merge(...)  en _fuse_overlapping_instances (al fusionar, junto a cooccurrence)
  3. on_remove(...) en _remove_missing_instances   (al borrar, junto a cooccurrence)
  4. report(...)    en update_map                  (a la cadencia de la fusión)

ESTA FASE NO CAMBIA COMPORTAMIENTO. `report` solo deriva, clasifica y devuelve
los veredictos para loguear/medir. El Actuator (aplicar merge/split al mapa) se
enchufa en la siguiente fase, detrás de su propio flag.
"""
import csv
import json
import os
import time
from typing import Dict, List, Optional, Set

import torch

from .types import Decision, FeatureTolerance, PointId, PairFeatures, Verdict
from .store import ContestStore
from .aggregator import ContestAggregator
from .discriminator import ContestDiscriminator, ContestThresholds
from .online import OnlineContestCoordinator
from .shadow import ShadowValidator


class ContestManager:
    def __init__(self, config: dict | None = None) -> None:
        cfg = (config or {}).get("contest", {})
        self.enabled: bool = cfg.get("enabled", True)
        self.store = ContestStore()
        self.aggregator = ContestAggregator(min_grabs=cfg.get("min_grabs", 5), firm_tau=cfg.get("firm_tau", 0.30))
        self.discriminator = ContestDiscriminator(ContestThresholds.from_cfg(cfg))
        # fase 2 opcional: reevaluar "frontera real" con el union-find de los
        # merges decididos en el MISMO batch (root real, no identidad). Resuelve
        # el caso 95: varios challengers que en realidad son un solo objeto.
        self.reeval_frontier: bool = cfg.get("reeval_frontier", False)
        # podar el store cada N llamadas a report (la poda recorre los vivos)
        self._prune_every: int = cfg.get("prune_every", 10)
        self._reports: int = 0
        self._verdicts_log: List[dict] = []
        self._output_dir: str | None = None
        # sub-tiempos de la última report(): aggregate/classify/resolve/frontier/dump.
        # ovo.py los lee tras report() y los mergea en contest_times.
        self.last_timings: Dict[str, float] = {}

        # ---- R1: agregador online en modo SOMBRA ----
        # agg_mode: batch (default, cero cambio) | online_shadow. El online calcula en paralelo y se
        # compara con el batch (oráculo) en cada report; el batch sigue decidiendo/actuando.
        self.agg_mode: str = cfg.get("agg_mode", "batch")
        self._tol = FeatureTolerance(atol=cfg.get("atol", 1e-6))
        self._coordinator: OnlineContestCoordinator | None = None
        self._validator: ShadowValidator | None = None
        self._shadow_log: List[dict] = []
        if self.agg_mode != "batch":
            self._coordinator = OnlineContestCoordinator(
                self.store, min_grabs=cfg.get("min_grabs", 5), firm_tau=cfg.get("firm_tau", 0.30))
            self._validator = ShadowValidator()

    def set_output_dir(self, path) -> None:
        self._output_dir = str(path)
        os.makedirs(self._output_dir, exist_ok=True)

    # ---- 1. camino caliente (una llamada por máscara con puntos en disputa) ----
    def record_grab(self, grabbed_point_ids: torch.Tensor, grabber: int) -> None:
        if not self.enabled or grabbed_point_ids.numel() == 0:
            return
        self.store.record_grab(grabbed_point_ids.cpu().flatten().tolist(), int(grabber))

    def record_claim(self, claimed_point_ids: torch.Tensor) -> None:
        """+1 al total de reclamos de cada punto asignado bajo una máscara (leal o robo).
        Denominador de persistence, en el reloj semántico (mismo que record_grab)."""
        if not self.enabled or claimed_point_ids.numel() == 0:
            return
        self.store.record_claim(claimed_point_ids.cpu().flatten().tolist())

    def record_sighting(self, seen_point_ids: torch.Tensor) -> None:
        """+1 a cada punto visto este KF, con o sin máscara. Para fiabilidad/P4."""
        if not self.enabled or seen_point_ids.numel() == 0:
            return
        self.store.record_sighting(seen_point_ids.cpu().flatten().tolist())

    def note_assignment(self, owner_ids: torch.Tensor) -> None:
        """Owners de los puntos asignados bajo máscara este KF -> marca sucios (online). No-op en batch."""
        if self._coordinator is not None:
            self._coordinator.note_assignment(owner_ids)

    # ---- 2 y 3. ciclo de vida (enganchados junto a self.cooccurrence) ----
    def on_merge(self, target: int, source: int) -> None:
        # coordinator ANTES del store: captura los puntos de source mientras siguen en el índice inverso.
        if self._coordinator is not None:
            self._coordinator.on_merge(int(target), int(source))
        self.store.on_merge(int(target), int(source))

    def on_remove(self, ins: int) -> None:
        if self._coordinator is not None:
            self._coordinator.on_remove(int(ins))
        self.store.on_remove(int(ins))

    def on_split(self, defender: int, challenger: int, split_points) -> None:
        """SPLIT aplicado: el trozo pasa de defender a challenger -> ambos sucios (online). No-op en batch."""
        if self._coordinator is not None:
            self._coordinator.on_split(int(defender), int(challenger), split_points)

    # ---- 4. cadencia de fusión: features -> clasificación -> veredictos ----
    def report(
        self,
        point_ids: torch.Tensor,
        points_ins_ids: torch.Tensor,
        point_obs: torch.Tensor | None = None,
        sim=None,
    ) -> List[Verdict]:
        if not self.enabled or len(self.store) == 0:
            return []

        self._reports += 1
        point_ids = point_ids.flatten()
        t = {}
        t0 = time.time()
        if self._prune_every > 0 and self._reports % self._prune_every == 0:
            live: Set[PointId] = set(point_ids.cpu().tolist())
            self.store.prune_to_live(live)
        t["prune"] = round(time.time() - t0, 4)

        t0 = time.time()
        pairs = self.aggregator.pairs(self.store, point_ids, points_ins_ids, point_obs)
        by_defender: Dict[int, list] = {}
        for f in pairs:
            by_defender.setdefault(f.defender, []).append(f)
        t["aggregate"] = round(time.time() - t0, 4)

        t0 = time.time()
        verdicts: List[Verdict] = []
        for defender, fs in by_defender.items():
            verdicts.extend(self.discriminator.classify(defender, fs, sim=sim))
        t["classify"] = round(time.time() - t0, 4)

        # ---- R1 sombra: refrescar el online y comparar contra el batch (oráculo) ----
        if self._coordinator is not None and self._validator is not None:
            stats = self._coordinator.refresh(point_ids, points_ins_ids, point_obs)
            rep = self._validator.compare(by_defender, self._coordinator.all_features(), self._tol, self._reports)
            self._record_shadow(rep, stats)

        t0 = time.time()
        verdicts = self._resolve_pairs(verdicts, by_defender)
        if self.reeval_frontier:
            verdicts = self._reeval_frontier(verdicts, by_defender)
        t["resolve"] = round(time.time() - t0, 4)

        t0 = time.time()
        for v in verdicts:
            fs = by_defender.get(v.defender, [])
            if v.challenger is not None:
                best = self._find_pair(v.challenger, fs)
                challenger = v.challenger
            else:
                # NO_ACTION sin challenger designado (borde/frontera/ruido): registra el par
                # de mayor containment para que la fila sea legible. Las features son de ESE
                # par -> se ve contra quién casi actuó, aunque la decisión sea no tocar.
                best = max(fs, key=lambda f: f.containment) if fs else None
                challenger = best.challenger if best is not None else None
            self._verdicts_log.append({
                "report": self._reports,
                "defender": v.defender,
                "challenger": challenger,
                "decision": v.decision.name,
                "reason": v.reason,
                "containment": best.containment if best is not None else None,
                "reverse_containment": best.reverse_containment if best is not None else None,
                "firm_points": best.firm_points if best is not None else None,
                "total_grabs": best.total_grabs if best is not None else None,
                "persistence": best.persistence if best is not None else None,
                "focus": best.focus if best is not None else None,
                "exclusivity": best.exclusivity if best is not None else None,
                "sim": v.sim,
                "seam_angle": v.seam_angle,
                "de_ch": v.de_ch,
                "de_def": v.de_def,
            })
        if self._output_dir:
            self.dump_verdicts(self._output_dir + "/contest_verdicts.csv")
        t["dump"] = round(time.time() - t0, 4)

        self.last_timings = t
        return verdicts

    def _resolve_pairs(self, verdicts: List[Verdict], by_defender: Dict[int, list]) -> List[Verdict]:
        """Decide UNA vez por par no-ordenado {A,B}, con los dos containments.

        `classify` mira cada defender por separado (A->B y B->A son veredictos
        distintos sobre la misma relación física). Aquí los juntamos: la decisión
        es una propiedad del PAR, no de un lado. Esto elimina de raíz las
        contradicciones (MERGE+SPLIT, SPLIT+SPLIT) sin parchear caso a caso.

        Reemplaza al antiguo `_reconcile` (que solo cubría MERGE-vs-SPLIT y dejaba
        escapar el SPLIT-vs-SPLIT, p.ej. dos trozos de una misma pared).
        """
        # veredictos sin challenger (frontera real, ruido, borde) pasan tal cual
        passthrough = [v for v in verdicts if v.challenger is None]
        directed = {(v.defender, v.challenger): v for v in verdicts if v.challenger is not None}

        def cont(a: int, ch: int) -> float:
            f = self._find_pair(ch, by_defender.get(a, []))
            return f.containment if f is not None and f.challenger == ch else 0.0

        resolved: List[Verdict] = []
        seen: Set[tuple] = set()
        for (a, ch), vab in directed.items():
            key = (a, ch) if a < ch else (ch, a)
            if key in seen:
                continue
            seen.add(key)
            vba = directed.get((ch, a))
            if vba is None:
                resolved.append(vab)  # solo un sentido opina -> se respeta
            else:
                resolved.append(self._join(vab, vba, cont(a, ch), cont(ch, a), self.discriminator.th.low))
        return passthrough + resolved

    def _reeval_frontier(self, verdicts: List[Verdict], by_defender: Dict[int, list]) -> List[Verdict]:
        """FASE 2 (opt-in): revisita los empates de contención con el root del batch.

        La rama strong marca NO_ACTION "mostly contained in several, tie" cuando un defender
        está muy contenido en >=2 challengers con IDs distintos. Pero esos IDs pueden ser el
        MISMO objeto si se fusionan en este mismo batch (caso 95: 78/33/81). Aquí:
          1. construimos un union-find con los MERGE_CONTAINMENT ya resueltos,
          2. recomputamos las raíces de los challengers fuertes con ese find,
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

        # 1. union-find con los merges decididos en este batch (defender -> challenger)
        for v in verdicts:
            if v.decision is Decision.MERGE_CONTAINMENT and v.challenger is not None:
                union(v.defender, v.challenger)

        high = self.discriminator.th.high
        out: List[Verdict] = []
        for v in verdicts:
            # solo los empates de contención de la rama strong
            if v.decision is Decision.NO_ACTION and "mostly contained in several" in v.reason:
                strong = [p for p in by_defender.get(v.defender, []) if p.containment >= high]
                roots = {find(p.challenger) for p in strong}
                if strong and len(roots) == 1:
                    best = max(strong, key=lambda p: p.containment)
                    out.append(Verdict(
                        Decision.MERGE_CONTAINMENT, v.defender, challenger=best.challenger,
                        reason=f"tie resolved by batch root (cont {best.containment:.2f}) -> merge",
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
                Decision.MERGE_CONTAINMENT, hi.defender, challenger=hi.challenger,
                reason=f"merge (resuelto por par, cont={hi_c:.2f})",
            )

        if vab.decision is Decision.SPLIT and vba.decision is Decision.SPLIT:
            if hi_c >= low:
                return Verdict(
                    Decision.MERGE_CONTAINMENT, hi.defender, challenger=hi.challenger,
                    reason=f"split simétrico resuelto -> merge (cont={hi_c:.2f})",
                )
            return Verdict(
                Decision.NO_ACTION, hi.defender,
                reason=f"split simétrico -> frontera (cont={hi_c:.2f}<{low})",
            )

        if hi.decision is Decision.SPLIT:
            return hi
        if lo.decision is Decision.SPLIT:
            return lo
        return hi

    @staticmethod
    def _find_pair(challenger: Optional[int], pairs: List[PairFeatures]) -> Optional[PairFeatures]:
        if challenger is None:
            return pairs[0] if pairs else None
        for f in pairs:
            if f.challenger == challenger:
                return f
        return pairs[0] if pairs else None

    def _record_shadow(self, rep, stats) -> None:
        """Registra el resultado de la comparación sombra. Vuelca el CSV de mismatches si hay output_dir."""
        for m in rep.mismatches:
            self._shadow_log.append({
                "report": rep.report_idx, "defender": m.defender, "challenger": m.challenger,
                "field": m.field, "batch": m.batch, "online": m.online, "kind": m.kind,
            })
        if rep.ok:
            print(f"[shadow] report {rep.report_idx}: OK ({rep.n_pairs_batch} pairs, n_dirty={stats.n_dirty})")
        else:
            print(f"[shadow] report {rep.report_idx}: {len(rep.mismatches)} MISMATCHES "
                  f"(batch {rep.n_pairs_batch} / online {rep.n_pairs_online})")
            for m in rep.mismatches[:10]:
                print(f"    def {m.defender} ch {m.challenger} {m.field}: batch={m.batch} online={m.online} {m.kind}")
        if self._output_dir:
            self._dump_shadow(self._output_dir + "/shadow_report.csv")

    def _dump_shadow(self, path: str) -> None:
        fields = ["report", "defender", "challenger", "field", "batch", "online", "kind"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._shadow_log)

    @property
    def shadow_ok(self) -> bool:
        """True si NINGÚN report tuvo mismatches (para tests/aserciones)."""
        return not self._shadow_log

    @staticmethod
    def summarize(verdicts: List[Verdict]) -> Dict[str, int]:
        """Conteo por tipo de decisión, para loguear sin ruido."""
        out = {d.name: 0 for d in Decision}
        for v in verdicts:
            out[v.decision.name] += 1
        return out

    # ---- serialización con la escena ----
    def to_dict(self) -> dict:
        # "claims"/"sightings" van como claves hermanas de "grabs" (no anidadas) para
        # mantener el formato {punto:{grabber:count}} que consumen los lectores externos.
        return {
            "grabs": self.store.to_dict(),
            "claims": self.store.claims_to_dict(),
            "sightings": self.store.sightings_to_dict(),
        }

    def load_dict(self, data: dict) -> None:
        if data and "grabs" in data:
            self.store = ContestStore.from_dict(data["grabs"])
            self.store.load_claims(data.get("claims", {}))
            self.store.load_sightings(data.get("sightings", {}))

    def dump(self, path: str) -> None:
        """Vuelca el registro crudo por punto a JSON, al final de la escena."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    def dump_verdicts(self, path: str) -> None:
        """Vuelca el log acumulado de veredictos a CSV."""
        if not self._verdicts_log:
            return
        fields = [
            "report", "defender", "challenger", "decision", "reason",
            "containment", "reverse_containment", "firm_points", "total_grabs", "persistence", "focus",
            "exclusivity", "sim", "seam_angle", "de_ch", "de_def",
        ]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self._verdicts_log)
