"""Política: convierte las features de un defender (y sus challengers) en un Verdict.

Pura: no toca el mapa. Abierta a extensión (añadir un patrón = añadir una rama,
sin reescribir las demás). Los umbrales son empíricos -> van en config.

Mapa de decisiones (lo mismo que la tabla de casos, en código):
  - masa baja                             -> NO_ACTION  (ruido)
  - muy contenido en 1 objeto             -> MERGE_CONTAINMENT  (fragmento)
  - muy contenido en varios objetos       -> NO_ACTION  (sin dueño claro)
  - parcial + firme + exclusivo + 1 raíz  -> MERGE_CONTAINMENT  (fragmento, sin descriptor ni split)
  - contención baja: por CADA challenger con trozo grande+exclusivo+persistente -> SPLIT (0..N)
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
    # banda FOCUS (sniper alta precisión: un challenger domina lo disputado del defender;
    # admite <=1 par, pero cualquier tamaño -> rescata transferencias limpias pequeñas)
    min_split_focus: float = 0.7          # el challenger domina lo disputado
    min_focus_excl: float = 0.9           # trozo muy exclusivo (split destructivo -> estricto)
    min_focus_persist: float = 0.6        # y muy persistente
    # banda POR-PAR (las gordas que focus omite por concurrencia; cada trozo se juzga solo -> 0..N)
    min_split_firm: int = 50              # tamaño del trozo: puntos firmes (corta migajas)
    min_split_excl: float = 0.8           # el trozo lo disputa un solo challenger (eje discriminante)
    min_split_persist: float = 0.5        # suelo: lo roba sostenido, no de refilón

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

        # strong = el defender está MAYORMENTE contenido en un challenger (cont >= high).
        # La pertenencia es SOLO por containment: si cabe dentro, es asunto de strong y aquí
        # se resuelve (no se fuga a focused/split, que no tendría sentido para algo contenido).
        strong = [p for p in pairs if p.containment >= self.th.high]
        partial = [p for p in pairs if self.th.low <= p.containment < self.th.high]

        if strong:
            # dueños FIRMES: lo agarran con firmeza de mayoría (pers >= min_persist_strong).
            # Un challenger que contiene al defender pero lo agarra <mitad de las veces no es
            # dueño fiable (p.ej. silla mal segmentada que solapa una mesa).
            firm = [p for p in strong if p.persistence >= self.th.min_persist_strong]
            if firm:
                best = max(firm, key=lambda p: p.containment)
                roots = {self.root(p.challenger) for p in firm}
                if len(roots) == 1:
                    return [Verdict(
                        Decision.MERGE_CONTAINMENT, defender, challenger=best.challenger,
                        reason=f"mostly contained in 1 object (cont {best.containment:.2f}) -> merge",
                    )]
                # varios objetos firmes lo contienen -> no se puede decidir
                return [Verdict(
                    Decision.NO_ACTION, defender,
                    reason="mostly contained in several, no clear owner -> no action",
                )]
            # contenido pero ningún dueño lo agarra con firmeza -> no fiable, no se toca
            # (NO cae a focused: algo mayormente contenido no es un trozo que transferir).
            return [Verdict(
                Decision.NO_ACTION, defender,
                reason="mostly contained but no firm owner -> no action",
            )]

        if partial:
            # BANDA PARCIAL [low, high): un fragmento (contención media) se fusiona con su
            # dueño si lo agarra con FIRMEZA de dueño (persistence >= partial_merge_persist,
            # no roce) y el trozo es de UN SOLO pretendiente (exclusivity >= partial_merge_excl,
            # zona no disputada). Y todos los dueños cualificados deben ser el MISMO objeto
            # (1 raíz), si no es una frontera. Sin descriptor (poco fiable aquí) y sin split.
            qual = [p for p in partial
                    if p.persistence >= self.th.partial_merge_persist
                    and p.exclusivity >= self.th.partial_merge_excl]
            if qual:
                roots = {self.root(p.challenger) for p in qual}
                if len(roots) == 1:
                    best = max(qual, key=lambda p: p.containment)
                    return [Verdict(
                        Decision.MERGE_CONTAINMENT, defender, challenger=best.challenger,
                        reason=f"fragment (cont {best.containment:.2f}, pers {best.persistence:.2f}, excl {best.exclusivity:.2f}) -> merge",
                    )]
            # sin dueño firme+exclusivo, o varios objetos -> no se toca
            return [Verdict(
                Decision.NO_ACTION, defender,
                reason="partial, no firm exclusive owner -> no action",
            )]

        # SPLIT: el defender no se mergea pero puede RETENER trozos ajenos. DOS fuentes
        # complementarias, ambas emiten a la lista (dedup por challenger):
        #   1. FOCUS  — sniper de alta precisión (un challenger domina lo disputado); cualquier
        #               tamaño, pero <=1 par -> rescata transferencias limpias pequeñas.
        #   2. POR-PAR — cada trozo grande+exclusivo+persistente por su cuenta (0..N) -> rescata
        #               las gordas que focus omite cuando el defender está muy disputado.
        return self._split_verdicts(defender, pairs, sim)

    def _split_verdicts(
        self,
        defender: InsId,
        pairs: List[PairFeatures],
        sim: Callable[[InsId, InsId], float | None] | None,
    ) -> List[Verdict]:
        # dos bandas de split (mismo estilo que strong/partial): defínelas, luego combina.
        focus = [p for p in pairs
                 if p.focus >= self.th.min_split_focus
                 and p.exclusivity >= self.th.min_focus_excl
                 and p.persistence >= self.th.min_focus_persist]
        perpair = [p for p in pairs
                   if p.firm_points >= self.th.min_split_firm
                   and p.exclusivity >= self.th.min_split_excl
                   and p.persistence >= self.th.min_split_persist]

        # focus tiene prioridad; por-par añade los OTROS challengers (dedup por challenger).
        taken = {p.challenger for p in focus}
        verdicts = [self._split(defender, p, "focus", sim) for p in focus]
        verdicts += [self._split(defender, p, "por-par", sim) for p in perpair if p.challenger not in taken]
        return verdicts or [Verdict(Decision.NO_ACTION, defender, reason="borde")]

    def _split(self, defender: InsId, p: PairFeatures, kind: str,
               sim: Callable[[InsId, InsId], float | None] | None) -> Verdict:
        s = sim(defender, p.challenger) if sim is not None else None
        sim_txt = f", sim={s:.2f}" if s is not None else ""
        return Verdict(
            Decision.SPLIT, defender, challenger=p.challenger,
            split_points=list(p.split_points) if p.split_points else None,
            reason=f"{kind} (firm={p.firm_points}, focus={p.focus:.2f}, excl={p.exclusivity:.2f}, persist={p.persistence:.2f}{sim_txt}) -> split",
            sim=s,
        )
