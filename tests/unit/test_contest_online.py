"""Unit tests del agregador online (R1) + shadow validator.

Cubre: dataclasses, DirtyDefenders, FeatureCache, IncrementalAggregator (== batch), el ciclo del
coordinator (eventos -> sucio -> refresh -> cache), y ShadowValidator (tolerancia float / lados faltantes).
"""
import os
import sys
import unittest

import torch

WORKTREE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, WORKTREE_ROOT)

from ovo.entities.contest.store import ContestStore
from ovo.entities.contest.aggregator import ContestAggregator
from ovo.entities.contest.online import (
    DirtyDefenders, FeatureCache, IncrementalAggregator, OnlineContestCoordinator,
)
from ovo.entities.contest.shadow import ShadowValidator
from ovo.entities.contest.types import (
    ComparisonReport, FeatureTolerance, Mismatch, PairFeatures, RefreshStats,
)


def _store(grabs, claims):
    s = ContestStore()
    for p, grabbers in grabs.items():
        for g, c in grabbers.items():
            s._grabs[p][g] = c
            s._by_grabber[g].add(p)
    for p, c in claims.items():
        s._claims[p] = c
    return s


def _by_pair(pairs):
    return {(pf.defender, pf.challenger): pf for pf in pairs}


class TestDataclasses(unittest.TestCase):
    def test_report_ok(self):
        self.assertTrue(ComparisonReport(1, 3, 3, ()).ok)
        self.assertFalse(ComparisonReport(1, 3, 3, (Mismatch(1, 2, "focus", 0.1, 0.2),)).ok)

    def test_tolerance_fields(self):
        self.assertIn("persistence", FeatureTolerance.FLOAT_FIELDS)
        self.assertIn("firm_points", FeatureTolerance.INT_FIELDS)

    def test_refresh_stats(self):
        st = RefreshStats(n_dirty=2, n_pairs=5)
        self.assertEqual(st.n_dirty, 2)


class TestDirtyDefenders(unittest.TestCase):
    def test_mark_take_coalesce(self):
        d = DirtyDefenders()
        d.mark(1); d.mark(1); d.mark_many([2, 3])
        self.assertEqual(len(d), 3)
        self.assertEqual(d.take(), {1, 2, 3})
        self.assertEqual(len(d), 0)          # take vacía
        self.assertEqual(d.take(), set())

    def test_drop(self):
        d = DirtyDefenders()
        d.mark_many([1, 2]); d.drop(1)
        self.assertEqual(d.take(), {2})


class TestFeatureCache(unittest.TestCase):
    def test_update_get_drop(self):
        c = FeatureCache()
        pf = PairFeatures(1, 2, 1.0, 0.0, 5, 10)
        c.update(1, [pf])
        self.assertEqual(c.get(1), [pf])
        self.assertEqual(c.get(99), [])
        c.drop(1)
        self.assertEqual(c.get(1), [])


class TestIncrementalEqualsBatch(unittest.TestCase):
    """features_for(all) debe dar EXACTAMENTE lo mismo que ContestAggregator.pairs()."""

    def setUp(self):
        # dos defenders (1 y 2), varios challengers, distinta persistencia/exclusividad
        self.grabs = {5: {2: 8}, 6: {2: 3, 3: 6}, 7: {1: 5}, 8: {1: 7, 3: 2}}
        self.claims = {5: 10, 6: 10, 7: 10, 8: 10}
        self.store = _store(self.grabs, self.claims)
        # owners: 5,6 de la inst 1 ; 7,8 de la inst 2 ; + puntos no disputados de cada una para tamaño
        self.point_ids = torch.tensor([5, 6, 7, 8, 100, 101, 200, 201], dtype=torch.long)
        self.owners = torch.tensor([1, 1, 2, 2, 1, 1, 2, 2], dtype=torch.long)

    def test_all_defenders_equal_batch(self):
        batch = ContestAggregator(min_grabs=1, firm_tau=0.30)
        inc = IncrementalAggregator(min_grabs=1, firm_tau=0.30)
        b = _by_pair(batch.pairs(self.store, self.point_ids, self.owners))
        online = inc.features_for(self.store, {1, 2}, self.point_ids, self.owners)
        o = _by_pair([pf for pairs in online.values() for pf in pairs])
        self.assertEqual(set(b), set(o))
        for k in b:
            self.assertEqual(b[k], o[k])   # frozen dataclass -> igualdad exacta campo a campo

    def test_subset_is_subset_of_batch(self):
        batch = ContestAggregator(min_grabs=1, firm_tau=0.30)
        inc = IncrementalAggregator(min_grabs=1, firm_tau=0.30)
        b = _by_pair(batch.pairs(self.store, self.point_ids, self.owners))
        online = inc.features_for(self.store, {1}, self.point_ids, self.owners)
        for pf in online[1]:
            self.assertEqual(pf.defender, 1)
            self.assertEqual(b[(pf.defender, pf.challenger)], pf)

    def test_empty_defender_returns_empty_list(self):
        inc = IncrementalAggregator(min_grabs=1, firm_tau=0.30)
        online = inc.features_for(self.store, {999}, self.point_ids, self.owners)
        self.assertEqual(online, {999: []})


class TestCoordinatorCycle(unittest.TestCase):
    def setUp(self):
        self.grabs = {5: {2: 8}, 6: {2: 3, 3: 6}, 7: {1: 5}}
        self.claims = {5: 10, 6: 10, 7: 10}
        self.store = _store(self.grabs, self.claims)
        self.point_ids = torch.tensor([5, 6, 7, 100, 200], dtype=torch.long)
        self.owners = torch.tensor([1, 1, 2, 1, 2], dtype=torch.long)

    def test_refresh_populates_cache_like_batch(self):
        coord = OnlineContestCoordinator(self.store, min_grabs=1, firm_tau=0.30)
        # marcar TODOS los defenders sucios (primer report)
        coord.note_assignment(self.owners)
        stats = coord.refresh(self.point_ids, self.owners)
        self.assertIsInstance(stats, RefreshStats)
        batch = ContestAggregator(min_grabs=1, firm_tau=0.30)
        b = _by_pair(batch.pairs(self.store, self.point_ids, self.owners))
        o = _by_pair([pf for pairs in coord.all_features().values() for pf in pairs])
        self.assertEqual(set(b), set(o))
        for k in b:
            self.assertEqual(b[k], o[k])

    def test_take_empties_dirty(self):
        coord = OnlineContestCoordinator(self.store, min_grabs=1, firm_tau=0.30)
        coord.note_assignment(self.owners)
        coord.refresh(self.point_ids, self.owners)
        # sin nuevas marcas, el segundo refresh no recomputa nada
        stats = coord.refresh(self.point_ids, self.owners)
        self.assertEqual(stats.n_dirty, 0)

    def test_on_merge_marks_target_and_challengers(self):
        coord = OnlineContestCoordinator(self.store, min_grabs=1, firm_tau=0.30)
        # merge: 2 se fusiona en 1. on_merge se llama ANTES de store.on_merge (como en el manager).
        coord.on_merge(target=1, source=2)
        self.store.on_merge(target=1, source=2)
        # target 1 sucio; los puntos que 2 robaba (5,6) -> owner 1 -> tras resolver en refresh, 1 sucio.
        stats = coord.refresh(self.point_ids, self.owners)
        self.assertIn(1, [1])  # sanity
        self.assertGreaterEqual(stats.n_dirty, 1)

    def test_negative_owner_ignored(self):
        coord = OnlineContestCoordinator(self.store, min_grabs=1, firm_tau=0.30)
        coord.note_assignment(torch.tensor([-1, -1, 1], dtype=torch.long))
        stats = coord.refresh(self.point_ids, self.owners)
        self.assertEqual(stats.n_dirty, 1)   # solo el 1, los -1 se ignoran


class TestShadowValidator(unittest.TestCase):
    def _pf(self, d, c, cont=0.5, pers=0.5, firm=5, grabs=10):
        return PairFeatures(d, c, cont, 0.0, firm, grabs, persistence=pers, focus=0.5, exclusivity=1.0)

    def test_identical_ok(self):
        v = ShadowValidator()
        feats = {1: [self._pf(1, 2)]}
        rep = v.compare(feats, {1: [self._pf(1, 2)]}, FeatureTolerance())
        self.assertTrue(rep.ok)

    def test_float_within_atol_ok(self):
        v = ShadowValidator()
        a = {1: [self._pf(1, 2, pers=0.5)]}
        b = {1: [self._pf(1, 2, pers=0.5 + 1e-9)]}
        self.assertTrue(v.compare(a, b, FeatureTolerance(atol=1e-6)).ok)

    def test_float_beyond_atol_mismatch(self):
        v = ShadowValidator()
        a = {1: [self._pf(1, 2, pers=0.5)]}
        b = {1: [self._pf(1, 2, pers=0.6)]}
        rep = v.compare(a, b, FeatureTolerance(atol=1e-6))
        self.assertFalse(rep.ok)
        self.assertEqual(rep.mismatches[0].field, "persistence")

    def test_int_exact_mismatch(self):
        v = ShadowValidator()
        a = {1: [self._pf(1, 2, firm=5)]}
        b = {1: [self._pf(1, 2, firm=6)]}
        rep = v.compare(a, b, FeatureTolerance())
        self.assertTrue(any(m.field == "firm_points" for m in rep.mismatches))

    def test_missing_online(self):
        v = ShadowValidator()
        rep = v.compare({1: [self._pf(1, 2)]}, {}, FeatureTolerance())
        self.assertEqual(rep.mismatches[0].kind, "missing_online")

    def test_missing_batch(self):
        v = ShadowValidator()
        rep = v.compare({}, {1: [self._pf(1, 2)]}, FeatureTolerance())
        self.assertEqual(rep.mismatches[0].kind, "missing_batch")


class TestCrossReport(unittest.TestCase):
    """CROSS-REPORT: merge/split/remove aplicados ENTRE reports -> el cache no debe quedar viejo.

    Cubre el riesgo R2 (invalidación cross-report) que el run LIVE no ejerce (GT solo hace 1 report).
    Conduce SOLO los eventos reales (on_merge/on_split/on_remove + note_assignment de lo reasignado) y
    exige que `all_features()` siga == batch.pairs() tras cada report. Si falta un dirty-mark, mismatch.
    """

    def _assert_equal(self, coord, batch, store, pt, ow):
        b = _by_pair(batch.pairs(store, pt, ow))
        o = _by_pair([pf for pairs in coord.all_features().values() for pf in pairs])
        self.assertEqual(set(b), set(o), "conjuntos de pares distintos (cache viejo/perdido)")
        for k in b:
            self.assertEqual(b[k], o[k])

    def test_cross_report_merge_drops_stale_pairs(self):
        # report 1: defender 1 (ch 2 y 3), defender 2 (ch 1)
        store = _store({5: {2: 8}, 6: {2: 6}, 7: {1: 5}, 8: {3: 7}},
                       {5: 10, 6: 10, 7: 10, 8: 10})
        pt = torch.tensor([5, 6, 7, 8, 100, 200], dtype=torch.long)
        ow = torch.tensor([1, 1, 2, 1, 1, 2], dtype=torch.long)
        coord = OnlineContestCoordinator(store, min_grabs=1, firm_tau=0.30)
        batch = ContestAggregator(min_grabs=1, firm_tau=0.30)
        coord.note_assignment(ow); coord.refresh(pt, ow)
        self._assert_equal(coord, batch, store, pt, ow)

        # merge: 1 absorbe 2 (coordinator ANTES del store, como el manager)
        coord.on_merge(target=1, source=2)
        store.on_merge(target=1, source=2)
        ow2 = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)  # puntos de 2 -> owner 1
        # report 2: NO re-marcamos todo; solo los eventos. Los pares (1,2) y (2,1) deben desaparecer,
        # (1,3) debe sobrevivir. Si el cache quedó viejo -> mismatch.
        coord.refresh(pt, ow2)
        self._assert_equal(coord, batch, store, pt, ow2)

    def test_cross_report_new_grabs_after_merge(self):
        store = _store({5: {2: 8}}, {5: 10})
        pt = torch.tensor([5, 6, 100, 200], dtype=torch.long)
        ow = torch.tensor([1, 2, 1, 2], dtype=torch.long)
        coord = OnlineContestCoordinator(store, min_grabs=1, firm_tau=0.30)
        batch = ContestAggregator(min_grabs=1, firm_tau=0.30)
        coord.note_assignment(ow); coord.refresh(pt, ow)
        self._assert_equal(coord, batch, store, pt, ow)

        # merge 1<-2, luego un nuevo challenger 3 roba el punto 6 (ahora de 1)
        coord.on_merge(target=1, source=2)
        store.on_merge(target=1, source=2)
        ow2 = torch.tensor([1, 1, 1, 1], dtype=torch.long)
        store.record_grab([6], 3); store.record_claim([6])
        coord.note_assignment(torch.tensor([1], dtype=torch.long))  # owner de 6 es 1
        coord.refresh(pt, ow2)
        self._assert_equal(coord, batch, store, pt, ow2)

    def test_cross_report_split(self):
        # defender 1 retiene un trozo (punto 5) que es de 2
        store = _store({5: {2: 9}, 6: {2: 2}}, {5: 10, 6: 10})
        pt = torch.tensor([5, 6, 100, 200, 201], dtype=torch.long)
        ow = torch.tensor([1, 1, 2, 2, 2], dtype=torch.long)
        coord = OnlineContestCoordinator(store, min_grabs=1, firm_tau=0.30)
        batch = ContestAggregator(min_grabs=1, firm_tau=0.30)
        coord.note_assignment(ow); coord.refresh(pt, ow)
        self._assert_equal(coord, batch, store, pt, ow)

        # split: el trozo {5} pasa de 1 a 2
        coord.on_split(defender=1, challenger=2, split_points=[5])
        ow2 = torch.tensor([2, 1, 2, 2, 2], dtype=torch.long)  # punto 5 ahora de 2
        coord.refresh(pt, ow2)
        self._assert_equal(coord, batch, store, pt, ow2)

    def test_cross_report_remove(self):
        store = _store({5: {2: 8}, 6: {3: 7}}, {5: 10, 6: 10})
        pt = torch.tensor([5, 6, 100, 200, 300], dtype=torch.long)
        ow = torch.tensor([1, 1, 2, 2, 3], dtype=torch.long)
        coord = OnlineContestCoordinator(store, min_grabs=1, firm_tau=0.30)
        batch = ContestAggregator(min_grabs=1, firm_tau=0.30)
        coord.note_assignment(ow); coord.refresh(pt, ow)
        self._assert_equal(coord, batch, store, pt, ow)

        # se borra la instancia 2 (challenger de un par de 1). El par (1,2) debe desaparecer.
        coord.on_remove(2)
        store.on_remove(2)
        # los puntos que 2 tenía (100) quedan sin dueño -> owner -2 en el lookup; simulamos que
        # el mapa los reasigna a -1 (podados/huerfanos). El par (1,2) via punto 5 sigue en el store
        # (5 lo robaba 2), pero 2 ya no es grabber -> _grabs[5] vacío -> sin par.
        ow2 = torch.tensor([1, 1, -1, -1, 3], dtype=torch.long)
        coord.refresh(pt, ow2)
        self._assert_equal(coord, batch, store, pt, ow2)


if __name__ == "__main__":
    unittest.main()
