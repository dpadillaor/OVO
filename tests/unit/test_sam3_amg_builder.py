"""Unit tests del builder de AMG compartido (fuente única SAM2/SAM3).

Cubre `build_amg_from_predictor`: que envuelve un predictor arbitrario con la maquinaria del AMG
de SAM2 fijando los atributos que `generate()` usa, sin cargar ningún modelo. El build real de
SAM3 (`build_sam3_image_amg`) es pesado (GPU + pesos) y no se prueba aquí.
"""
import os
import sys
import unittest

WORKTREE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.insert(0, WORKTREE_ROOT)

from ovo.utils.segment_utils import build_amg_from_predictor


class _DummyPredictor:
    """Sustituto barato de un predictor interactivo; solo hace falta la identidad."""


class TestBuildAmgFromPredictor(unittest.TestCase):
    def test_wires_predictor_and_core_attrs(self):
        pred = _DummyPredictor()
        amg = build_amg_from_predictor(pred, points_per_side=16, pred_iou_thresh=0.8,
                                       stability_score_thresh=0.95)
        self.assertIs(amg.predictor, pred)
        self.assertEqual(amg.pred_iou_thresh, 0.8)
        self.assertEqual(amg.stability_score_thresh, 0.95)
        self.assertTrue(amg.multimask_output)
        self.assertEqual(amg.output_mode, "binary_mask")

    def test_point_grid_scales_with_points_per_side(self):
        a16 = build_amg_from_predictor(_DummyPredictor(), points_per_side=16)
        a32 = build_amg_from_predictor(_DummyPredictor(), points_per_side=32)
        self.assertEqual(a16.point_grids[0].shape[0], 16 * 16)
        self.assertEqual(a32.point_grids[0].shape[0], 32 * 32)

    def test_is_a_sam2_amg_instance(self):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        amg = build_amg_from_predictor(_DummyPredictor(), points_per_side=16)
        self.assertIsInstance(amg, SAM2AutomaticMaskGenerator)


if __name__ == "__main__":
    unittest.main()
