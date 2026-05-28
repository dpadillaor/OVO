import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import torch

WORKTREE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, WORKTREE_ROOT)

from ovo.entities.fusion import create_fusion_strategy, FusionStrategy
from ovo.utils.cooccurrence_graph import CooccurrenceGraph


def _make_instance(id_):
    inst = MagicMock()
    inst.id = id_
    inst.clip_feature = [torch.ones(512)]
    return inst


class TestFusionVeto(unittest.TestCase):
    def setUp(self):
        self.config = {
            "fusion_method": "clip",
            "th_centroid": 1.5,
            "th_cossim": 0.81,
            "th_points": 0.1,
            "cooccurrence_veto_threshold": 5,
        }

    def test_veto_active_semantic(self):
        """F.1: shared_kfs > threshold → veto (semantic chain)."""
        graph = CooccurrenceGraph()
        for kf in range(10):
            graph.increment(1, 2, kf_id=kf)

        fusion = create_fusion_strategy(self.config, graph)
        inst1 = _make_instance(1)
        inst2 = _make_instance(2)
        pcd = (MagicMock(), torch.zeros(3))

        with patch('ovo.entities.fusion.compute_centroid_distance', return_value=0.0), \
             patch('ovo.entities.fusion.compute_pcd_overlap', return_value=1.0):
            self.assertFalse(fusion.same_instance(inst1, inst2, pcd, pcd))

    def test_veto_inactive_semantic(self):
        """F.2: shared_kfs <= threshold → no veto (semantic chain)."""
        graph = CooccurrenceGraph()  # weight(1,2) == 0

        fusion = create_fusion_strategy(self.config, graph)
        inst1 = _make_instance(1)
        inst2 = _make_instance(2)
        pcd = (MagicMock(), torch.zeros(3))

        with patch('ovo.entities.fusion.compute_centroid_distance', return_value=0.0), \
             patch('ovo.entities.fusion.compute_pcd_overlap', return_value=1.0):
            self.assertTrue(fusion.same_instance(inst1, inst2, pcd, pcd))

    def test_shared_kfs_in_decisions(self):
        """F.4: shared_kfs recorded in all decision types."""
        graph = CooccurrenceGraph()
        graph.increment(1, 2, kf_id=1)

        fusion = create_fusion_strategy(self.config, graph)
        inst1 = _make_instance(1)
        inst2 = _make_instance(2)
        pcd = (MagicMock(), torch.zeros(3))

        with patch('ovo.entities.fusion.compute_centroid_distance', return_value=0.0), \
             patch('ovo.entities.fusion.compute_pcd_overlap', return_value=1.0):
            fusion.same_instance(inst1, inst2, pcd, pcd)

        decisions = fusion.pop_decisions()
        self.assertEqual(len(decisions), 1)
        self.assertIn("shared_kfs", decisions[0])
        self.assertEqual(decisions[0]["shared_kfs"], 1)

    def test_custom_chain_skips_cos_sim(self):
        """F.5: custom fusion_criteria without cos_sim still works."""
        graph = CooccurrenceGraph()
        config = {**self.config, "fusion_criteria": ["centroid", "overlap"]}
        fusion = create_fusion_strategy(config, graph)
        inst1 = _make_instance(1)
        inst2 = _make_instance(2)
        pcd = (MagicMock(), torch.zeros(3))

        with patch('ovo.entities.fusion.compute_centroid_distance', return_value=0.0), \
             patch('ovo.entities.fusion.compute_pcd_overlap', return_value=1.0):
            self.assertTrue(fusion.same_instance(inst1, inst2, pcd, pcd))

    def test_custom_chain_order_matters(self):
        """F.6: centroid rejection fires before overlap even if overlap would accept."""
        graph = CooccurrenceGraph()
        config = {**self.config, "fusion_criteria": ["centroid", "overlap"], "th_centroid": 0.5}
        fusion = create_fusion_strategy(config, graph)
        inst1 = _make_instance(1)
        inst2 = _make_instance(2)
        pcd = (MagicMock(), torch.zeros(3))

        with patch('ovo.entities.fusion.compute_centroid_distance', return_value=1.0), \
             patch('ovo.entities.fusion.compute_pcd_overlap', return_value=1.0):
            self.assertFalse(fusion.same_instance(inst1, inst2, pcd, pcd))

        decisions = fusion.pop_decisions()
        self.assertEqual(decisions[0]["reason"], "centroid")


if __name__ == '__main__':
    unittest.main()
