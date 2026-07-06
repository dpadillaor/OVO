"""Política: convierte las features de un defender (y sus challengers) en un Verdict.

Pura: no toca el mapa. Abierta a extensión (añadir un patrón = añadir una rama,
sin reescribir las demás). Los umbrales son empíricos -> van en config.

Mapa de decisiones (lo mismo que la tabla de casos, en código):
  - masa baja                             -> NO_ACTION  (ruido)
  - contención alta + 1 raíz              -> MERGE_CONTAINMENT  (fragmento)
  - contención alta + simétrico           -> DEFER_TO_FUSION    (que decida geometría)
  - contención alta + >=2 raíces          -> NO_ACTION  (frontera real)
  - contención parcial                    -> SPLIT (candidato)
  - contención baja + focus alta (>0.7)   -> SPLIT (pocos puntos disputados pero un solo challenger los domina)
  - contención baja + focus baja          -> NO_ACTION  (borde)
"""
from typing import Callable, List

from .types import Decision, InsId, PairFeatures, Verdict


class ContestDiscriminator:
    def __init__(
        self,
        high: float = 0.7,                    # contención alta -> fragmento
        low: float = 0.5,                     # por debajo -> borde
        min_mass: int = 50,                   # total_grabs por debajo -> ruido
        min_split_cont: float = 0.0,          # contención mínima para SPLIT por focus (0 = sin corte; sesga contra defenders grandes, filtra focus/mass/persist/normal)
        max_rev_split: float = 0.5,           # rev por encima -> bidireccional, NO split parcial
        min_persist_split: float = 0.1,       # persistencia por debajo -> parpadeo, NO split parcial (solo partial)
        sim_merge: float = 0.81,              # cos-sim descriptor por encima -> mismo objeto -> MERGE (franja parcial)
        max_seam_angle: float = 15.0,         # giro de normal en la costura por encima -> objeto en contacto, NO split (dominancia)
        root: Callable[[InsId], InsId] = lambda x: x,  # union-find de la fusión
    ) -> None:
        self.high = high
        self.low = low
        self.min_mass = min_mass
        self.min_split_cont = min_split_cont
        self.max_rev_split = max_rev_split
        self.min_persist_split = min_persist_split
        self.sim_merge = sim_merge
        self.max_seam_angle = max_seam_angle
        self.root = root

    def classify(
        self,
        defender: InsId,
        pairs: List[PairFeatures],
        sim: Callable[[InsId, InsId], float | None] | None = None,
        seam: Callable[[InsId, InsId, tuple], float | None] | None = None,
        color: Callable[[InsId, InsId, tuple], tuple] | None = None,
    ) -> Verdict:
        pairs = [p for p in pairs if p.total_grabs >= self.min_mass]
        if not pairs:
            return Verdict(Decision.NO_ACTION, defender, reason="ruido (masa baja)")

        strong = [p for p in pairs if p.containment >= self.high]
        partial = [p for p in pairs if self.low <= p.containment < self.high]

        if strong:
            best = max(strong, key=lambda p: p.containment)

            # simétrico: el challenger también está muy contenido en el defender.
            # No es un fragmento direccional -> que lo resuelva la fusión existente.
            if len(strong) == 1 and best.reverse_containment >= self.high:
                return Verdict(
                    Decision.DEFER_TO_FUSION, defender, challenger=best.challenger,
                    reason="simétrico -> fusión (geometría/descriptor)",
                )

            # unicidad de challenger POR RAÍZ: ¿todo va a parar al mismo objeto?
            roots = {self.root(p.challenger) for p in strong}
            if len(roots) == 1:
                return Verdict(
                    Decision.MERGE_CONTAINMENT, defender, challenger=best.challenger,
                    reason="contención alta, 1 raíz",
                )
            # varios challengers que NO se fusionan entre sí -> frontera real
            return Verdict(
                Decision.NO_ACTION, defender,
                reason="frontera real (>=2 raíces distintas)",
            )

        if partial:
            best = max(partial, key=lambda p: p.containment)
            # FRANJA AMBIGUA (0.5-0.70): la geometría no separa fragmento de trozo.
            # El descriptor desempata: mismo objeto -> MERGE entero; distinto -> SPLIT.
            s = sim(defender, best.challenger) if sim is not None else None
            if s is not None and s >= self.sim_merge:
                return Verdict(
                    Decision.MERGE_CONTAINMENT, defender, challenger=best.challenger,
                    reason=f"parcial + descriptor igual (sim={s:.2f}) -> merge",
                )
            # descriptor distinto (o ausente) -> es split o frontera; aplican guards
            # guarda 1: bidireccional -> no es un split direccional limpio -> frontera
            if best.reverse_containment >= self.max_rev_split:
                return Verdict(
                    Decision.NO_ACTION, defender, challenger=best.challenger,
                    reason=f"parcial bidireccional (rev={best.reverse_containment:.2f}) -> no split",
                )
            # guarda 2: disputa inestable (parpadeo de pocos KFs) -> ruido temporal
            if best.persistence < self.min_persist_split:
                return Verdict(
                    Decision.NO_ACTION, defender, challenger=best.challenger,
                    reason=f"parcial inestable (persist={best.persistence:.2f}) -> no split",
                )
            sim_txt = f", sim={s:.2f}" if s is not None else ""
            return Verdict(
                Decision.SPLIT, defender, challenger=best.challenger,
                split_points=list(best.split_points) if best.split_points else None,
                reason=f"parcial (containment={best.containment:.2f}{sim_txt}) -> candidato split",
            )

        # contención baja (< low) pero un solo challenger domina los puntos disputados
        focused = [p for p in pairs if p.focus >= 0.7 and p.containment >= self.min_split_cont and p.total_grabs >= self.min_mass]
        if focused:
            best = max(focused, key=lambda p: p.focus)
            # contención baja + un solo challenger domina = objeto adyacente (se rozan).
            # NO se fusiona: el descriptor (CLIP) no distingue INSTANCIAS de la misma
            # clase (p.ej. 3 sillas pegadas -> sim alto -> over-merge real visto en
            # office3: 102 absorbió 88+105, hub 35 tragó 10). Solo SPLIT (candidato).
            s = sim(defender, best.challenger) if sim is not None else None
            sim_txt = f", sim={s:.2f}" if s is not None else ""
            # guarda: el challenger ve los puntos solo de refilón (parpadeo). Las sillas
            # adyacentes (hub 35, 102) caen TODAS aquí: persist<0.06 (el challenger roza
            # los puntos de la silla vecina). El fragmento real (mesa 89->103) tiene
            # persist alta (0.23). Umbral 0.1 mata las sillas, conserva el fragmento.
            if best.persistence < self.min_persist_split:
                return Verdict(
                    Decision.NO_ACTION, defender, challenger=best.challenger,
                    reason=f"dominancia inestable (persist={best.persistence:.2f}{sim_txt}) -> no split",
                )
            # guarda geométrica: giro de la normal de superficie a través de la costura
            # chunk<->challenger. Fragmento real (trozo de mesa) -> normales paralelas (~8°);
            # objeto en contacto (cojín sobre sofá, cosa sobre mesa) -> quiebre (>20°).
            # Es la única señal que separa estos cuando máscara 2D y descriptor no pueden.
            ang = seam(defender, best.challenger, best.split_points) if seam is not None else None
            if ang is not None and ang > self.max_seam_angle:
                return Verdict(
                    Decision.NO_ACTION, defender, challenger=best.challenger,
                    reason=f"dominancia: objeto en contacto (giro normal={ang:.1f}°{sim_txt}) -> no split",
                )
            # guarda de color de DOS sentidos: el chunk debe fundirse (color) mejor con
            # el challenger que con el resto del defender. Si se parece más al defender, es
            # suyo -> no transferir. Caza el caso que el normal no ve: superficies coplanares
            # de distinto color (póster en pared, trozo oscuro propio pegado a objeto grande).
            de_ch = de_d = None
            if color is not None:
                de_ch, de_d = color(defender, best.challenger, best.split_points)
            if de_ch is not None and de_d is not None and de_ch > de_d:
                return Verdict(
                    Decision.NO_ACTION, defender, challenger=best.challenger,
                    reason=f"dominancia: color del defender (ΔE_ch={de_ch:.1f} > ΔE_def={de_d:.1f}{sim_txt}) -> no split",
                )
            ang_txt = f", normal={ang:.1f}°" if ang is not None else ""
            col_txt = f", ΔE_ch={de_ch:.1f}/def={de_d:.1f}" if de_ch is not None and de_d is not None else ""
            return Verdict(
                Decision.SPLIT, defender, challenger=best.challenger,
                split_points=list(best.split_points) if best.split_points else None,
                reason=f"dominancia (focus={best.focus:.2f}, cont={best.containment:.3f}{sim_txt}{ang_txt}{col_txt}) -> split",
            )

        return Verdict(Decision.NO_ACTION, defender, reason="borde (contención baja)")
