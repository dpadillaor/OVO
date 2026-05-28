import unittest
import sys
import os

WORKTREE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, WORKTREE_ROOT)

from ovo.utils.cooccurrence_graph import CooccurrenceGraph

class TestCooccurrenceGraph(unittest.TestCase):
    def setUp(self):
        self.graph = CooccurrenceGraph()

    def test_increment_basic(self):
        """U.1: Basic increment and symmetry."""
        self.graph.increment(1, 2, kf_id=10)
        self.assertEqual(self.graph.weight(1, 2), 1)
        self.assertEqual(self.graph.weight(2, 1), 1)
        self.assertEqual(self.graph.get_kfs(1, 2), [10])

    def test_increment_duplicate_kf(self):
        """U.2: Duplicate keyframe not counted twice."""
        self.graph.increment(1, 2, kf_id=10)
        self.graph.increment(1, 2, kf_id=10)
        self.assertEqual(self.graph.weight(1, 2), 1)

    def test_increment_multiple_kfs(self):
        """U.3: Multiple keyframes accumulate."""
        self.graph.increment(1, 2, kf_id=10)
        self.graph.increment(1, 2, kf_id=20)
        self.assertEqual(self.graph.weight(1, 2), 2)
        self.assertEqual(self.graph.get_kfs(1, 2), [10, 20])

    def test_unknown_pair_weight_zero(self):
        """U.4: Unknown pair returns 0."""
        self.assertEqual(self.graph.weight(99, 100), 0)

    def test_remove_cleans_edges(self):
        """U.5: remove() cleans all edges to that node."""
        self.graph.increment(1, 2, kf_id=10)
        self.graph.increment(1, 3, kf_id=11)
        self.graph.remove(1)
        self.assertEqual(self.graph.weight(1, 2), 0)
        self.assertEqual(self.graph.weight(2, 1), 0)
        self.assertEqual(self.graph.weight(1, 3), 0)

    def test_merge_inherits_edges(self):
        """U.6: merge() transfers source edges to target."""
        self.graph.increment(1, 3, kf_id=10)
        self.graph.increment(2, 3, kf_id=20)
        self.graph.merge(target=1, source=2)
        self.assertEqual(self.graph.weight(1, 3), 2)
        self.assertEqual(self.graph.weight(2, 3), 0)

    def test_merge_no_self_loop(self):
        """U.7: merge() does not create self-loops."""
        self.graph.increment(1, 2, kf_id=10)
        self.graph.increment(2, 3, kf_id=20)
        self.graph.merge(target=1, source=2)
        self.assertEqual(self.graph.weight(1, 1), 0)
        self.assertEqual(len(self.graph.get_neighbors(2)), 0)

if __name__ == '__main__':
    unittest.main()
