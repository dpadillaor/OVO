from dataclasses import dataclass
from enum import Enum, auto
from typing import ClassVar, List, Optional, Tuple

InsId = int
PointId = int

class Decision(Enum):
    MERGE_CONTAINMENT = auto()  # nuestro: el defender es un fragmento del challenger
    SPLIT = auto()              # nuestro: el defender son dos objetos; soltar un trozo
    DEFER_TO_FUSION = auto()    # simétrico/geométrico -> lo decide la fusión existente
    NO_ACTION = auto()          # borde, frontera real o ruido

@dataclass(frozen=True)
class PairFeatures:
    """Lo que el Aggregator deriva para un par (defender vs challenger).

    defender  = instancia que TIENE los puntos disputados (agregado del `owner` del hot path).
    challenger = instancia que se los DISPUTA grab tras grab (agregado del `grabber`).
    """
    defender: InsId             # instancia titular de los puntos
    challenger: InsId           # instancia que le disputa los puntos
    containment: float          # firm_points / |defender|: fracción del defender que el challenger le roba, en [0, 1]
    reverse_containment: float  # lo mismo en el sentido contrario (challenger -> defender)
    firm_points: int            # nº de PUNTOS del defender robados con firmeza (grabs >= min_grabs)
    total_grabs: int            # nº de ROBOS crudos acumulados (suma de grabs, sin umbral)
    persistence: float = 0.0    # persistencia media: media de grabs_challenger / claims por punto
    focus: float = 0.0          # firm_points / total de puntos disputados del defender
    exclusivity: float = 0.0    # fracción de firm_points que SOLO este challenger disputa (nadie más)
    split_points: tuple = ()    # ids de los puntos a reasignar en un SPLIT

@dataclass(frozen=True)
class Verdict:
    """Lo que el Discriminator decide para un defender."""
    decision: Decision
    defender: InsId
    challenger: Optional[InsId] = None
    split_points: Optional[List[PointId]] = None  # para SPLIT: puntos a soltar (siguiente fase)
    reason: str = ""                               # traza legible, para logs/depuración
    # señales geométricas de la banda ambigua, cuando se calcularon (para logs/análisis; no
    # afectan la decisión, que ya está tomada por la rama que las rellena).
    sim: Optional[float] = None                    # cos-sim descriptor
    seam_angle: Optional[float] = None             # giro de la normal en la costura (grados)
    de_ch: Optional[float] = None                  # ΔE color hacia el challenger
    de_def: Optional[float] = None                 # ΔE color hacia el defender


# ---- online / shadow (R1) -------------------------------------------------

@dataclass(frozen=True)
class FeatureTolerance:
    """Comparación batch↔online (R1). Ints exactos; floats con atol (persistence es una media -> el
    orden de suma puede diferir en el último ULP, ver docs/contest_online_design_2026-07-09.md §11.1)."""
    atol: float = 1e-6
    FLOAT_FIELDS: ClassVar[Tuple[str, ...]] = (
        "containment", "reverse_containment", "persistence", "focus", "exclusivity",
    )
    INT_FIELDS: ClassVar[Tuple[str, ...]] = ("firm_points", "total_grabs")


@dataclass(frozen=True)
class Mismatch:
    """Una discrepancia de un campo entre las features batch y online de un par."""
    defender: InsId
    challenger: InsId
    field: str                       # nombre del campo de PairFeatures
    batch: float
    online: float
    kind: str = "value"              # "value" | "missing_online" | "missing_batch"


@dataclass(frozen=True)
class ComparisonReport:
    """Resultado de comparar batch vs online en un report (shadow)."""
    report_idx: int
    n_pairs_batch: int
    n_pairs_online: int
    mismatches: Tuple[Mismatch, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.mismatches


@dataclass(frozen=True)
class RefreshStats:
    """Stats de un refresh online (para logs/telemetría)."""
    n_dirty: int                     # defenders recomputados este refresh
    n_pairs: int                     # pares producidos
