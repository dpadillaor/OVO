"""Motor de consulta: agrega los pares una vez y clasifica bajo demanda con la heurística REAL del contest.

Reutiliza ContestAggregator + ContestDiscriminator de producción -> lo que decide aquí es
idéntico a lo que decidió el run, por construcción. La herramienta solo consulta y presenta.
"""

from __future__ import annotations

from collections import defaultdict

from dataclasses import asdict

from ovo.entities.contest.aggregator import ContestAggregator
from ovo.entities.contest.discriminator import ContestDiscriminator, ContestThresholds
from ovo.entities.contest.manager import ContestManager
from ovo.entities.contest.store import ContestStore
from ovo.entities.contest.types import PairFeatures, Verdict

from ..common.mapstate import MapState
from ..common.resolve import resolve
from .callbacks import build_callbacks
from .loader import Substrate, load_substrate
from .report import PointReport, ProbeReport

_SORT_KEYS = {
    "total_grabs": lambda f: f.total_grabs,
    "containment": lambda f: f.containment,
    "firm_points": lambda f: f.firm_points,
    "focus": lambda f: f.focus,
    "persistence": lambda f: f.persistence,
    "exclusivity": lambda f: f.exclusivity,
}


def _discriminator(cfg: dict) -> ContestDiscriminator:
    """Discriminator con los mismos umbrales que usó el run (defaults si no hay overrides)."""
    return ContestDiscriminator(ContestThresholds.from_cfg(cfg))


def _thresholds(d: ContestDiscriminator) -> dict:
    return asdict(d.th)


class ContestProbe:
    """Replica el classify sobre un mapa guardado. Agrega una vez; cada consulta clasifica con geometría real."""

    def __init__(self, map_state: MapState, store: ContestStore, contest_cfg: dict,
                 overrides: dict | None = None) -> None:
        cfg = {**(contest_cfg or {}).get("contest", {}), **(overrides or {})}
        self._map = map_state
        self._store = store
        self._disc = _discriminator(cfg)
        self._cb = build_callbacks(map_state)
        agg = ContestAggregator(min_grabs=cfg.get("min_grabs", 5), firm_tau=cfg.get("firm_tau", 0.30))
        self._pairs = agg.pairs(store, map_state.point_ids, map_state.points_ins_ids)
        self._by_def: dict[int, list[PairFeatures]] = defaultdict(list)
        for f in self._pairs:
            self._by_def[f.defender].append(f)

    # ---- construcción --------------------------------------------------
    @classmethod
    def from_substrate(cls, sub: Substrate, overrides: dict | None = None) -> "ContestProbe":
        return cls(sub.map_state, sub.store, sub.contest_cfg, overrides)

    @classmethod
    def from_experiment(cls, exp: str, scene: str, overrides: dict | None = None) -> "ContestProbe":
        ctx = resolve(exp, scene)
        if not ctx.ok:
            raise FileNotFoundError("; ".join(ctx.errors) or "no analizable")
        return cls.from_substrate(load_substrate(ctx), overrides)

    # ---- consultas -----------------------------------------------------
    def directed(self, defender: int, challenger: int) -> PairFeatures | None:
        """Features del sentido defender<-challenger (None si ese sentido no tuvo disputa)."""
        return next((f for f in self._by_def.get(defender, []) if f.challenger == challenger), None)

    def classify_all(self, defender: int) -> list[Verdict]:
        """Veredictos reales del discriminator para ese defender (MERGE: 1; SPLIT: 0..N)."""
        fs = self._by_def.get(defender)
        if not fs:
            return []
        return self._disc.classify(defender, fs, sim=self._cb.sim)

    def _verdict_toward(self, defender: int, challenger: int) -> Verdict | None:
        """El veredicto de `defender` dirigido a ese challenger concreto (None si no opina sobre él)."""
        return next((v for v in self.classify_all(defender) if v.challenger == challenger), None)

    def explain(self, a: int, b: int) -> ProbeReport:
        """Interroga el par {a, b}: ambos sentidos, veredicto de cada lado, y el resuelto."""
        fa, fb = self.directed(a, b), self.directed(b, a)
        va, vb = self._verdict_toward(a, b), self._verdict_toward(b, a)
        return ProbeReport(
            a=a, b=b,
            size_a=self._map.sizes.get(a, 0), size_b=self._map.sizes.get(b, 0),
            ab=fa, ba=fb, verdict_a=va, verdict_b=vb,
            resolved=self._resolve(a, b, va, vb, fa, fb),
            thresholds=_thresholds(self._disc),
        )

    def top_pairs(self, n: int = 20, by: str = "total_grabs") -> list[PairFeatures]:
        """Pares más disputados según un criterio, para descubrir qué mirar."""
        if by not in _SORT_KEYS:
            raise ValueError(f"criterio '{by}' no válido: {list(_SORT_KEYS)}")
        return sorted(self._pairs, key=_SORT_KEYS[by], reverse=True)[:n]

    def branch(self, decision: str | None = None) -> dict[str, list[Verdict]]:
        """Clasifica TODOS los defenders y agrupa por rama de decisión (en vivo, sobre el sustrato).

        decision=None -> todas las ramas. decision='SPLIT' -> solo esa. Bucketea el veredicto
        real de cada defender, no el par resuelto (queremos ver qué produce classify por rama).
        """
        buckets: dict[str, list[Verdict]] = defaultdict(list)
        for defender in self._by_def:
            for v in self.classify_all(defender):
                buckets[v.decision.name].append(v)
        if decision is not None:
            key = decision.upper()
            return {key: buckets.get(key, [])}
        return dict(buckets)

    def point(self, p: int) -> PointReport:
        """Counters crudos de un punto desde el store: claims, sightings, grabs, P4, persistencia/grabber.

        No necesita el mapa (solo el store) -> barato. P4 = sightings - claims = lealtad invisible
        (KFs en que el punto cayó bajo una máscara que no salió adelante).
        """
        claims = self._store.claims_of(p)
        sightings = self._store.sightings_of(p)
        grabbers = dict(self._store.grabbers_of(p))
        persist = {ch: (c / claims if claims else 0.0) for ch, c in grabbers.items()}
        return PointReport(
            point=p, claims=claims, sightings=sightings, grabbers=grabbers,
            persistence_per_grabber=persist,
            p4=sightings - claims,
            loyalty=claims - sum(grabbers.values()),
        )

    # ---- interno -------------------------------------------------------
    def _resolve(self, a: int, b: int, va: Verdict | None, vb: Verdict | None,
                 fa: PairFeatures | None, fb: PairFeatures | None) -> Verdict | None:
        """Junta los dos sentidos como en manager._resolve_pairs, pero solo para este par."""
        a_drives = va is not None and va.challenger == b
        b_drives = vb is not None and vb.challenger == a
        if a_drives and b_drives:
            cab = fa.containment if fa else 0.0
            cba = fb.containment if fb else 0.0
            return ContestManager._join(va, vb, cab, cba, self._disc.th.low)
        if a_drives:
            return va
        if b_drives:
            return vb
        return None
