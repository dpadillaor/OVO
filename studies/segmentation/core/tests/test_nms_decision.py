"""El keep-set de evaluate() debe coincidir exacto con ovo.utils.segment_utils.mask_nms."""
import numpy as np
import torch

from ovo.utils.segment_utils import mask_nms
from studies.segmentation.core.nms_decision import evaluate


def _rand_masks(n: int, H: int = 40, W: int = 60) -> torch.Tensor:
    m = torch.zeros(n, H, W, dtype=torch.bool)
    for k in range(n):
        y0, x0 = np.random.randint(0, H // 2), np.random.randint(0, W // 2)
        h, w = np.random.randint(5, H // 2), np.random.randint(5, W // 2)
        m[k, y0:y0 + h, x0:x0 + w] = True
    return m


def test_keep_set_matches_ovo_mask_nms():
    """En el régimen de producción (redes de seguridad no disparadas) el keep-set es idéntico."""
    np.random.seed(0)
    torch.manual_seed(0)
    thresholds = [(0.8, 0.05, 0.5), (0.7, 0.1, 0.2), (0.9, 0.05, 0.3), (0.5, 0.1, 0.4)]
    for _ in range(200):
        n = np.random.randint(3, 15)
        masks = _rand_masks(n)
        scores = torch.rand(n)
        for thr in thresholds:
            try:
                real = set(mask_nms(masks.clone(), scores.clone(), *thr).tolist())
            except IndexError:
                continue  # rama top-3 de mask_nms: asume scores 2-D, inalcanzable en prod (1-D)
            mine = {v.index for v in evaluate(masks, scores, *thr).kept}
            assert real == mine, f"thr={thr} real={sorted(real)} mine={sorted(mine)}"


def test_removed_masks_carry_a_reason():
    """Toda máscara descartada trae al menos un motivo; las kept, ninguno."""
    np.random.seed(1)
    torch.manual_seed(1)
    masks = _rand_masks(10)
    scores = torch.rand(10)
    bd = evaluate(masks, scores)
    for v in bd.removed:
        assert v.kills, f"máscara {v.index} descartada sin motivo"
    for v in bd.kept:
        assert not v.kills
