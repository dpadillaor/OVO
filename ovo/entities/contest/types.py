from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

InsId = int
PointId = int

class Decision(Enum):
    MERGE_CONTAINMENT = auto()  # nuestro: el perdedor es un fragmento del ganador
    SPLIT = auto()              # nuestro: el perdedor son dos objetos; soltar un trozo
    DEFER_TO_FUSION = auto()    # simétrico/geométrico -> lo decide la fusión existente
    NO_ACTION = auto()          # borde, frontera real o ruido

@dataclass(frozen=True)
class PairFeatures:
    """Lo que el Aggregator deriva para un par (perdedor -> ganador)."""
    loser: InsId                # Instancia que tiene la pertenencia del punto
    winner: InsId               # Instancia dominante de la máscara donde cayó el punto
    containment: float          # |puntos de A vistos bajo W| / |A|, fracción de los puntos de A que se ven bajo W, en [0, 1].
    reverse_containment: float  # Lo mismo en el sentido contrario (W -> A)
    strong_points: int          # Numero de puntos de A que vieron a W (por encima del umbral)
    mass: int                   # Suma cruda de observaciones acumuladas
    persistence: float = 0.0    # Persistencia media: media de obs_winner / total_obs por punto
    focus: float = 0.0          # strong_points / total_puntos_disputados_de_A
    split_points: tuple = ()    # IDs de los puntos a reasignar en un SPLIT

@dataclass(frozen=True)
class Verdict:
    """Lo que el Discriminator decide para un perdedor."""
    decision: Decision
    loser: InsId
    winner: Optional[InsId] = None
    subset: Optional[List[PointId]] = None  # para SPLIT: puntos a soltar (siguiente fase)
    reason: str = ""                         # traza legible, para logs/depuración
