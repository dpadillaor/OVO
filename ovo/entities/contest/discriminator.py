"""Política: convierte las features de un defender (y sus challengers) en un Verdict.

Pura: no toca el mapa. Abierta a extensión (añadir un patrón = añadir una rama,
sin reescribir las demás). Los umbrales son empíricos -> van en config.

Mapa de decisiones (lo mismo que la tabla de casos, en código):
  - masa baja                             -> NO_ACTION  (ruido)
  - muy contenido en 1 objeto             -> MERGE_CONTAINMENT  (fragmento)
  - muy contenido en varios objetos       -> NO_ACTION  (sin dueño claro)
  - parcial + firme + exclusivo + 1 raíz  -> MERGE_CONTAINMENT  (fragmento, sin descriptor ni split)
  - contención baja: por CADA challenger con trozo exclusivo+persistente -> SPLIT (0..N)
  - contención baja, ningún trozo cualifica -> NO_ACTION  (borde)
"""
from dataclasses import dataclass, fields
from typing import Callable, List

from .types import Decision, InsId, PairFeatures, Verdict


@dataclass(frozen=True)
class ContestThresholds:
    """Umbrales empíricos del discriminador, agrupados por banda. Default = config vigente."""
    # gate global
    min_mass: int = 50                    # total_grabs por debajo -> ruido
    # bandas por containment
    high: float = 0.6                     # cont >= high -> strong (fragmento contenido)
    low: float = 0.4                      # cont < low -> borde/focused; [low, high) -> partial
    # banda strong
    min_persist_strong: float = 0.5       # dueño fiable: cada punto robado de media >= esto (mayoría)
    # banda partial (merge de fragmento, sin descriptor ni split)
    partial_merge_persist: float = 0.6    # lo agarran de media >= esto (dueño real, no roce)
    partial_merge_excl: float = 0.8       # y el trozo es de un solo pretendiente (zona no disputada)
    # banda SPLIT única (0..N): firmeza local del trozo, sin focus ni gate de tamaño
    min_split_excl: float = 0.9           # el trozo lo disputa un solo challenger (eje discriminante)
    min_split_persist: float = 0.6        # lo roba sostenido, no de refilón

    @classmethod
    def from_cfg(cls, cfg: dict) -> "ContestThresholds":
        """Construye desde un dict de config, ignorando claves desconocidas."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (cfg or {}).items() if k in known})


class ContestDiscriminator:
    def __init__(
        self,
        th: ContestThresholds = ContestThresholds(),
        root: Callable[[InsId], InsId] = lambda x: x,  # union-find de la fusión
    ) -> None:
        self.th = th
        self.root = root

    def classify(
        self,
        defender: InsId,
        pairs: List[PairFeatures],
        sim: Callable[[InsId, InsId], float | None] | None = None,
    ) -> List[Verdict]:
        """Veredictos de un defender. MERGE es terminal (lista de 1); SPLIT es por-par
        (lista de 0..N: un objeto puede soltar varios trozos a varios vecinos)."""
        pairs = [p for p in pairs if p.total_grabs >= self.th.min_mass]
        if not pairs:
            return [Verdict(Decision.NO_ACTION, defender, reason="ruido (masa baja)")]

        # cascada por containment: contenido en un dueño -> merge; si no, retiene trozos -> split.
        strong = [p for p in pairs if p.containment >= self.th.high]
        partial = [p for p in pairs if self.th.low <= p.containment < self.th.high]
        if strong:
            return self._strong_verdict(defender, strong)
        if partial:
            return self._partial_verdict(defender, partial)
        return self._split_verdicts(defender, pairs, sim)

    def _strong_verdict(self, defender: InsId, strong: List[PairFeatures]) -> List[Verdict]:
        # strong = contenido en un dueño. Merge si un dueño fiable (pers, mayoría) único lo contiene.
        firm = [p for p in strong if p.persistence >= self.th.min_persist_strong]
        if not firm:
            return [Verdict(Decision.NO_ACTION, defender,
                            reason="mostly contained but no firm owner -> no action")]
        if len({self.root(p.challenger) for p in firm}) == 1:
            best = max(firm, key=lambda p: p.containment)
            return [Verdict(Decision.MERGE_CONTAINMENT, defender, challenger=best.challenger,
                            reason=f"mostly contained in 1 object (cont {best.containment:.2f}) -> merge")]
        return [Verdict(Decision.NO_ACTION, defender,
                        reason="mostly contained in several, no clear owner -> no action")]

    def _partial_verdict(self, defender: InsId, partial: List[PairFeatures]) -> List[Verdict]:
        # fragmento -> merge si dueño firme (pers) Y trozo no disputado (excl) Y 1 raíz.
        qual = [p for p in partial
                if p.persistence >= self.th.partial_merge_persist
                and p.exclusivity >= self.th.partial_merge_excl]
        if len({self.root(p.challenger) for p in qual}) == 1:
            best = max(qual, key=lambda p: p.containment)
            return [Verdict(Decision.MERGE_CONTAINMENT, defender, challenger=best.challenger,
                            reason=f"fragment (cont {best.containment:.2f}, pers {best.persistence:.2f}, excl {best.exclusivity:.2f}) -> merge")]
        return [Verdict(Decision.NO_ACTION, defender,
                        reason="partial, no firm exclusive owner -> no action")]

    def _split_verdicts(self, defender: InsId, pairs: List[PairFeatures],
                        sim: Callable[[InsId, InsId], float | None] | None) -> List[Verdict]:
        # SPLIT (0..N): cada par persistente Y exclusivo suelta su trozo. excl -> trozos disjuntos.
        verdicts = [self._split(defender, p, sim) for p in pairs
                    if p.persistence >= self.th.min_split_persist
                    and p.exclusivity >= self.th.min_split_excl]
        return verdicts or [Verdict(Decision.NO_ACTION, defender, reason="borde")]

    def _split(self, defender: InsId, p: PairFeatures,
               sim: Callable[[InsId, InsId], float | None] | None) -> Verdict:
        s = sim(defender, p.challenger) if sim is not None else None
        sim_txt = f", sim={s:.2f}" if s is not None else ""
        return Verdict(
            Decision.SPLIT, defender, challenger=p.challenger,
            split_points=list(p.split_points) if p.split_points else None,
            reason=f"split (firm={p.firm_points}, excl={p.exclusivity:.2f}, persist={p.persistence:.2f}{sim_txt}) -> split",
            sim=s,
        )
