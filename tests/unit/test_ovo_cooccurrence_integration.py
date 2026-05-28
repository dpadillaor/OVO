import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import torch

WORKTREE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, WORKTREE_ROOT)

from ovo.entities.ovo import OVO
from ovo.utils.cooccurrence_graph import CooccurrenceGraph


class TestOVOCooccurrenceIntegration(unittest.TestCase):
    def test_graph_initialized(self):
        """I.1: OVO has a CooccurrenceGraph after init."""
        config = {
            "track_th": 0.5,
            "sam": {"multi_crop": False, "mask_res": 256},
            "clip": {"embed_type": "vanilla"},
            "fusion_method": "clip",
        }
        with patch('ovo.entities.ovo.create_fusion_strategy'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.MaskGenerator'), \
             patch('ovo.entities.ovo.Logger'):
            ovo = OVO(config, logger=MagicMock(), cam_intrinsics=torch.eye(3))

        self.assertTrue(hasattr(ovo, 'cooccurrence'))
        self.assertIsInstance(ovo.cooccurrence, CooccurrenceGraph)

    def test_graph_passed_to_fusion_strategy(self):
        """I.2: create_fusion_strategy receives the graph instance."""
        config = {
            "track_th": 0.5,
            "sam": {"multi_crop": False, "mask_res": 256},
            "clip": {"embed_type": "vanilla"},
            "fusion_method": "clip",
        }
        with patch('ovo.entities.ovo.create_fusion_strategy') as mock_factory, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.MaskGenerator'), \
             patch('ovo.entities.ovo.Logger'):
            ovo = OVO(config, logger=MagicMock(), cam_intrinsics=torch.eye(3))

        mock_factory.assert_called_once()
        _, kwargs = mock_factory.call_args
        passed_graph = mock_factory.call_args[0][1] if len(mock_factory.call_args[0]) > 1 else kwargs.get('cooccurrence_graph')
        self.assertIsInstance(passed_graph, CooccurrenceGraph)
        self.assertIs(passed_graph, ovo.cooccurrence)


if __name__ == '__main__':
    unittest.main()
