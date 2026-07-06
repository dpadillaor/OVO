from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

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
    firm_points: int            # nº de PUNTOS del defender robados con firmeza (grabs >= min_count)
    total_grabs: int            # nº de ROBOS crudos acumulados (suma de grabs, sin umbral)
    persistence: float = 0.0    # persistencia media: media de grabs_challenger / claims por punto
    focus: float = 0.0          # firm_points / total de puntos disputados del defender
    split_points: tuple = ()    # ids de los puntos a reasignar en un SPLIT

@dataclass(frozen=True)
class Verdict:
    """Lo que el Discriminator decide para un defender."""
    decision: Decision
    defender: InsId
    challenger: Optional[InsId] = None
    split_points: Optional[List[PointId]] = None  # para SPLIT: puntos a soltar (siguiente fase)
    reason: str = ""                               # traza legible, para logs/depuración
