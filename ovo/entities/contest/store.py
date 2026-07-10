"""Registro por punto de los puntos en disputa.

Único trabajo: por cada punto problemático, acumular cuántas veces (en cuántos KFs)
cayó bajo la máscara de cada instancia "grabber", y mantenerse coherente con el
mapa (merge / remove / poda). NO normaliza nada y NO decide nada.

Como cada KF suma a lo sumo +1 por (punto, grabber), el valor del contador ES el
número de KFs distintos en que ese punto se vio bajo ese grabber. La persistencia
sale gratis.

Vocabulario (hot path, por punto/KF):
  - owner   = instancia dueña del punto ahora (estado del mapa).
  - grabber = instancia de la máscara que lo agarra este KF.
  - grab    = evento grabber != owner (un robo).
  - sightings ⊇ claims ⊇ Σ grabs. Lealtad = claims − Σ grabs.

Memoria: solo se guardan los puntos en disputa (una minoría), no todos. La clave
es el id permanente del punto, así que sobrevive a reordenado/poda del SLAM.
"""
from collections import defaultdict
from typing import Dict, Iterable, Set

from .types import InsId, PointId


class ContestStore:
    def __init__(self) -> None:
        # point_id -> { grabber_ins_id : nº de KFs robado por ese grabber }
        self._grabs: Dict[PointId, Dict[InsId, int]] = defaultdict(lambda: defaultdict(int))
        # índice inverso para limpieza y agregación baratas: grabber -> {point_id}
        self._by_grabber: Dict[InsId, Set[PointId]] = defaultdict(set)
        # point_id -> nº de KFs en que el punto cayó bajo ALGUNA máscara usada (leal o robo).
        # Es el denominador honesto de persistence: como cada robo hacia un grabber es también
        # un reclamo, grabs(grabber) <= claims(punto), así persistence queda acotada en [0, 1].
        self._claims: Dict[PointId, int] = defaultdict(int)
        # point_id -> nº de KFs en que el punto se vio (reproyectó bien), con o sin máscara.
        # fiabilidad = claims/sightings; huérfano-asignado (P4) = sightings - claims.
        self._sightings: Dict[PointId, int] = defaultdict(int)

    # ---- camino caliente -------------------------------------------------
    def record_grab(self, point_ids: Iterable[PointId], grabber: InsId) -> None:
        """Suma +1 a cada punto robado por `grabber`. Una llamada por máscara y KF."""
        bucket = self._by_grabber[grabber]
        for p in point_ids:
            self._grabs[p][grabber] += 1
            bucket.add(p)

    def record_claim(self, point_ids: Iterable[PointId]) -> None:
        """Suma +1 al total de reclamos de cada punto (bajo cualquier máscara, leal o robo).

        Se llama para TODOS los puntos asignados que caen bajo una máscara, una vez por KF.
        Es el denominador de persistence, en el MISMO reloj (semántico) que los robos.
        """
        for p in point_ids:
            self._claims[p] += 1

    def record_sighting(self, point_ids: Iterable[PointId]) -> None:
        """Suma +1 a cada punto visto este KF (reproyectó bien), con o sin máscara. Una vez por KF."""
        for p in point_ids:
            self._sightings[p] += 1

    # ---- consulta --------------------------------------------------------
    def grabbers_of(self, p: PointId) -> Dict[InsId, int]:
        return self._grabs.get(p, {})

    def claims_of(self, p: PointId) -> int:
        """Total de reclamos del punto (denominador de persistence)."""
        return self._claims.get(p, 0)

    def sightings_of(self, p: PointId) -> int:
        """Veces visto. Para fiabilidad = claims/sightings y P4 = sightings - claims."""
        return self._sightings.get(p, 0)

    def points(self) -> Iterable[PointId]:
        return self._grabs.keys()

    def grabber_points(self, grabber: InsId) -> Set[PointId]:
        """Copia de los puntos que `grabber` roba (índice inverso). Para invalidación online:
        al fusionar/borrar `grabber`, sus puntos cambian de challenger -> hay que recomputar sus owners."""
        return set(self._by_grabber.get(grabber, set()))

    def __len__(self) -> int:
        return len(self._grabs)

    # ---- ciclo de vida (espejo de CooccurrenceGraph) ---------------------
    def on_merge(self, target: InsId, source: InsId) -> None:
        """`source` se fusiona en `target`: los robos hacia source pasan a target."""
        if source == target:
            return
        moved = self._by_grabber.pop(source, set())
        tgt = self._by_grabber[target]
        for p in moved:
            c = self._grabs[p].pop(source, 0)
            if c:
                self._grabs[p][target] += c
            tgt.add(p)

    def on_remove(self, ins: InsId) -> None:
        """`ins` desaparece: se borra como grabber de todos los puntos."""
        for p in self._by_grabber.pop(ins, set()):
            self._grabs[p].pop(ins, None)
            if not self._grabs[p]:
                del self._grabs[p]

    def prune_to_live(self, live_point_ids: Set[PointId]) -> int:
        """Elimina puntos que ya no están en el mapa (podados por el SLAM).

        `live_point_ids` debe ser un set para que la pertenencia sea O(1).
        Conviene llamarlo de tanto en tanto (no en cada KF): es O(nº disputados).
        Devuelve cuántos puntos se eliminaron.
        """
        stale = [p for p in self._grabs if p not in live_point_ids]
        for p in stale:
            for g in self._grabs[p]:
                s = self._by_grabber.get(g)
                if s is not None:
                    s.discard(p)
            del self._grabs[p]
        # claims y sightings cubren más puntos que _grabs (incluyen leales/huérfanos
        # nunca disputados); se podan todos los muertos, no solo los de _grabs.
        for p in [p for p in self._claims if p not in live_point_ids]:
            del self._claims[p]
        for p in [p for p in self._sightings if p not in live_point_ids]:
            del self._sightings[p]
        return len(stale)

    # ---- serialización (guardar/restaurar con la escena) -----------------
    def to_dict(self) -> dict:
        return {str(p): dict(g) for p, g in self._grabs.items()}

    def claims_to_dict(self) -> dict:
        """Reclamos por punto (denominador). Serializado aparte de los grabs."""
        return {str(p): int(c) for p, c in self._claims.items()}

    def load_claims(self, data: dict) -> None:
        """Restaura los reclamos. Los checkpoints antiguos no los tienen -> quedan vacíos."""
        for p_str, c in (data or {}).items():
            self._claims[int(p_str)] = int(c)

    def sightings_to_dict(self) -> dict:
        return {str(p): int(c) for p, c in self._sightings.items()}

    def load_sightings(self, data: dict) -> None:
        for p_str, c in (data or {}).items():
            self._sightings[int(p_str)] = int(c)

    @classmethod
    def from_dict(cls, data: dict) -> "ContestStore":
        store = cls()
        for p_str, grabbers in data.items():
            p = int(p_str)
            for g_str, c in grabbers.items():
                g = int(g_str)
                store._grabs[p][g] = int(c)
                store._by_grabber[g].add(p)
        return store
