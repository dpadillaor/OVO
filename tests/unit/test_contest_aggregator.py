import unittest
import os
import sys

import torch

WORKTREE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, WORKTREE_ROOT)

from ovo.entities.contest.store import ContestStore
from ovo.entities.contest.aggregator import ContestAggregator


class TestAggregatorPersistence(unittest.TestCase):
    """persistence = robos(W) / reclamos_totales(punto), acotada en [0, 1]."""

    def _run(self, grabs, claims, point_ids, owners):
        """Helper: monta un store y corre aggregator.pairs()."""
        s = ContestStore()
        for p, grabbers in grabs.items():
            for g, c in grabbers.items():
                s._grabs[p][g] = c
                s._by_grabber[g].add(p)
        for p, c in claims.items():
            s._claims[p] = c
        agg = ContestAggregator(min_grabs=1, firm_tau=0.0)
        pairs = agg.pairs(
            s,
            torch.tensor(point_ids, dtype=torch.long),
            torch.tensor(owners, dtype=torch.long),
        )
        return {(pf.defender, pf.challenger): pf for pf in pairs}

    def test_persistence_basic_fraction(self):
        # punto 5 (de la instancia 1) robado 8 veces por la instancia 2; reclamado 10 en total
        pairs = self._run(
            grabs={5: {2: 8}},
            claims={5: 10},
            point_ids=[5, 10, 11],
            owners=[1, 2, 2],
        )
        self.assertIn((1, 2), pairs)
        self.assertAlmostEqual(pairs[(1, 2)].persistence, 0.8, places=6)

    def test_persistence_bounded_at_one(self):
        # robado en TODOS sus reclamos -> persistence == 1.0 exacta (nunca > 1)
        pairs = self._run(
            grabs={5: {2: 9}},
            claims={5: 9},
            point_ids=[5, 10, 11],
            owners=[1, 2, 2],
        )
        self.assertLessEqual(pairs[(1, 2)].persistence, 1.0)
        self.assertAlmostEqual(pairs[(1, 2)].persistence, 1.0, places=6)

    def test_regression_no_longer_exceeds_one(self):
        """El bug viejo: robos(9) / pcd_obs(1) = 9.0. Ahora denom=claims>=robos -> <=1."""
        pairs = self._run(
            grabs={5: {2: 9}},
            claims={5: 9},        # claims >= robos por el invariante del hot path
            point_ids=[5, 10, 11],
            owners=[1, 2, 2],
        )
        self.assertLessEqual(pairs[(1, 2)].persistence, 1.0)

    def test_no_pair_when_no_claims(self):
        """Sin reclamos (p.ej. checkpoint legacy) el punto se ignora (denom 0) -> no hay par,
        sin dividir por cero. (`_accumulate`: 'sin dueño o sin reclamos -> se ignora'.)"""
        pairs = self._run(
            grabs={5: {2: 3}},
            claims={},            # denom 0 -> punto ignorado
            point_ids=[5, 10, 11],
            owners=[1, 2, 2],
        )
        self.assertNotIn((1, 2), pairs)

    def test_persistence_averaged_over_points(self):
        # dos puntos de la instancia 1 robados por 2, distinta persistencia -> media
        # p5: 8/10=0.8 ; p6: 2/10=0.2 ; media = 0.5
        pairs = self._run(
            grabs={5: {2: 8}, 6: {2: 2}},
            claims={5: 10, 6: 10},
            point_ids=[5, 6, 10, 11],
            owners=[1, 1, 2, 2],
        )
        self.assertAlmostEqual(pairs[(1, 2)].persistence, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
