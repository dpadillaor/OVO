"""Registro por punto de los puntos en disputa.

Único trabajo: por cada punto problemático, acumular cuántas veces (en cuántos KFs)
cayó bajo la máscara de cada instancia "ganadora", y mantenerse coherente con el
mapa (merge / remove / poda). NO normaliza nada y NO decide nada.

Como cada KF suma a lo sumo +1 por (punto, ganador), el valor del contador ES el
número de KFs distintos en que ese punto se vio bajo ese ganador. La persistencia
sale gratis.

Memoria: solo se guardan los puntos en disputa (una minoría), no todos. La clave
es el id permanente del punto, así que sobrevive a reordenado/poda del SLAM.
"""
from collections import defaultdict
from typing import Dict, Iterable, Set

from .types import InsId, PointId


class ContestStore:
    def __init__(self) -> None:
        # point_id -> { winner_ins_id : nº de KFs visto bajo esa instancia }
        self._counts: Dict[PointId, Dict[InsId, int]] = defaultdict(lambda: defaultdict(int))
        # índice inverso para limpieza y agregación baratas: winner -> {point_id}
        self._by_winner: Dict[InsId, Set[PointId]] = defaultdict(set)

    # ---- camino caliente -------------------------------------------------
    def record(self, point_ids: Iterable[PointId], winner: InsId) -> None:
        """Suma +1 a cada punto bajo `winner`. Una llamada por máscara y KF."""
        bucket = self._by_winner[winner]
        for p in point_ids:
            self._counts[p][winner] += 1
            bucket.add(p)

    # ---- consulta --------------------------------------------------------
    def winners_of(self, p: PointId) -> Dict[InsId, int]:
        return self._counts.get(p, {})

    def points(self) -> Iterable[PointId]:
        return self._counts.keys()

    def __len__(self) -> int:
        return len(self._counts)

    # ---- ciclo de vida (espejo de CooccurrenceGraph) ---------------------
    def on_merge(self, target: InsId, source: InsId) -> None:
        """`source` se fusiona en `target`: los votos hacia source pasan a target."""
        if source == target:
            return
        moved = self._by_winner.pop(source, set())
        tgt = self._by_winner[target]
        for p in moved:
            c = self._counts[p].pop(source, 0)
            if c:
                self._counts[p][target] += c
            tgt.add(p)

    def on_remove(self, ins: InsId) -> None:
        """`ins` desaparece: se borra como ganador de todos los puntos."""
        for p in self._by_winner.pop(ins, set()):
            self._counts[p].pop(ins, None)
            if not self._counts[p]:
                del self._counts[p]

    def prune_to_live(self, live_point_ids: Set[PointId]) -> int:
        """Elimina puntos que ya no están en el mapa (podados por el SLAM).

        `live_point_ids` debe ser un set para que la pertenencia sea O(1).
        Conviene llamarlo de tanto en tanto (no en cada KF): es O(nº disputados).
        Devuelve cuántos puntos se eliminaron.
        """
        stale = [p for p in self._counts if p not in live_point_ids]
        for p in stale:
            for w in self._counts[p]:
                s = self._by_winner.get(w)
                if s is not None:
                    s.discard(p)
            del self._counts[p]
        return len(stale)

    # ---- serialización (guardar/restaurar con la escena) -----------------
    def to_dict(self) -> dict:
        return {str(p): dict(w) for p, w in self._counts.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "ContestStore":
        store = cls()
        for p_str, winners in data.items():
            p = int(p_str)
            for w_str, c in winners.items():
                w = int(w_str)
                store._counts[p][w] = int(c)
                store._by_winner[w].add(p)
        return store
