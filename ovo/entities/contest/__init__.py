"""Mecanismo de puntos en disputa (contest).

Recupera la señal 2D que `_track_objects` descartaba (co-pertenencia a máscara)
para hacer lo que la fusión geométrica no puede: merge direccional por contención
y, más adelante, split. Vive AL LADO de la fusión, no dentro.

Capa de datos (esta fase, sin cambio de comportamiento):
    ContestManager  -> fachada que OVO engancha en 4 puntos
      ContestStore        -> registro por punto + ciclo de vida
      ContestAggregator   -> features normalizadas (contención, dirección)
      ContestDiscriminator-> features -> Verdict (la casuística)

Siguiente fase: Actuator (aplica MERGE_CONTAINMENT / SPLIT), detrás de un flag.
"""
from .types import Decision, Verdict, PairFeatures, InsId, PointId
from .store import ContestStore
from .aggregator import ContestAggregator
from .discriminator import ContestDiscriminator
from .manager import ContestManager

__all__ = [
    "ContestManager",
    "ContestStore",
    "ContestAggregator",
    "ContestDiscriminator",
    "Decision",
    "Verdict",
    "PairFeatures",
    "InsId",
    "PointId",
]
