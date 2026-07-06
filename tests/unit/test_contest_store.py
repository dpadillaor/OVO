import unittest
import os
import sys

WORKTREE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, WORKTREE_ROOT)

from ovo.entities.contest.store import ContestStore


class TestContestStoreClaims(unittest.TestCase):
    """Contador de reclamos (_claims): denominador de persistence."""

    def test_claims_of_default_zero(self):
        s = ContestStore()
        self.assertEqual(s.claims_of(42), 0)

    def test_record_claim_increments(self):
        s = ContestStore()
        s.record_claim([5, 5, 6])   # una llamada: 5 y 6 reclamados; 5 dos veces en la lista
        self.assertEqual(s.claims_of(5), 2)
        self.assertEqual(s.claims_of(6), 1)
        s.record_claim([5])
        self.assertEqual(s.claims_of(5), 3)

    def test_claims_ge_grabs_invariant(self):
        """Invariante del hot path: cada robo es también un reclamo -> claims >= sum(grabs)."""
        s = ContestStore()
        # simula 5 KFs: el punto 7 siempre bajo alguna máscara (5 reclamos),
        # robado por el grabber=2 en 3 de ellos.
        for kf in range(5):
            s.record_claim([7])                    # denominador
            if kf < 3:
                s.record_grab([7], grabber=2)      # numerador (robo)
        total_grabs = sum(s.grabbers_of(7).values())
        self.assertEqual(total_grabs, 3)
        self.assertEqual(s.claims_of(7), 5)
        self.assertGreaterEqual(s.claims_of(7), total_grabs)

    def test_prune_removes_stale_claims(self):
        s = ContestStore()
        s.record_claim([1, 2, 3])
        s.record_grab([1], grabber=9)
        removed = s.prune_to_live({1})       # 2 y 3 mueren
        self.assertEqual(s.claims_of(1), 1)
        self.assertEqual(s.claims_of(2), 0)
        self.assertEqual(s.claims_of(3), 0)
        self.assertEqual(removed, 0)         # removed cuenta _grabs, no _claims

    def test_serialization_roundtrip(self):
        s = ContestStore()
        s.record_claim([1, 1, 2])
        s.record_grab([1], grabber=9)
        claims = s.claims_to_dict()
        grabs = s.to_dict()
        s2 = ContestStore.from_dict(grabs)
        s2.load_claims(claims)
        self.assertEqual(s2.claims_of(1), 2)
        self.assertEqual(s2.claims_of(2), 1)
        self.assertEqual(s2.grabbers_of(1), {9: 1})

    def test_load_claims_tolerates_missing(self):
        """Checkpoints antiguos sin 'claims' -> reclamos vacíos, sin romper."""
        s = ContestStore()
        s.load_claims({})
        s.load_claims(None)
        self.assertEqual(s.claims_of(1), 0)

    def test_claims_instance_agnostic_on_merge(self):
        """Los reclamos son por punto, no por instancia -> merge/remove no los tocan."""
        s = ContestStore()
        s.record_claim([1])
        s.record_grab([1], grabber=5)
        s.on_merge(target=8, source=5)       # 5 se fusiona en 8
        self.assertEqual(s.claims_of(1), 1)  # el reclamo del punto no cambia
        s.on_remove(8)
        self.assertEqual(s.claims_of(1), 1)

    def test_sightings_record_and_query(self):
        s = ContestStore()
        self.assertEqual(s.sightings_of(3), 0)
        s.record_sighting([3, 3, 4])
        self.assertEqual(s.sightings_of(3), 2)
        self.assertEqual(s.sightings_of(4), 1)

    def test_p4_orphan_derivable(self):
        """P4 (huérfano-asignado) = sightings - claims para un punto siempre asignado."""
        s = ContestStore()
        # 5 KFs visto (sightings=5); en 3 bajo máscara usada (claims=3) -> 2 huérfanos
        for kf in range(5):
            s.record_sighting([7])
            if kf < 3:
                s.record_claim([7])
        self.assertEqual(s.sightings_of(7), 5)
        self.assertEqual(s.claims_of(7), 3)
        self.assertEqual(s.sightings_of(7) - s.claims_of(7), 2)   # P4 derivado

    def test_sightings_prune_and_serialize(self):
        s = ContestStore()
        s.record_sighting([1, 2])
        s.record_claim([1])
        s.prune_to_live({1})
        self.assertEqual(s.sightings_of(1), 1)
        self.assertEqual(s.sightings_of(2), 0)   # 2 podado
        # roundtrip
        s2 = ContestStore.from_dict(s.to_dict())
        s2.load_claims(s.claims_to_dict())
        s2.load_sightings(s.sightings_to_dict())
        self.assertEqual(s2.sightings_of(1), 1)


if __name__ == "__main__":
    unittest.main()
