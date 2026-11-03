"""Unit tests for VoxelFusionThresholds and same_instance_by_overlap (ovo.utils.instance_utils)."""

import pytest
import torch

from ovo.utils import instance_utils
from ovo.utils.instance_utils import VoxelFusionThresholds, same_instance_by_overlap

TH = VoxelFusionThresholds()


class _FakeInstance:
    def __init__(self, clip_feature):
        self.clip_feature = clip_feature


def _instance(*values):
    return _FakeInstance(torch.tensor([values], dtype=torch.float32))


@pytest.fixture
def fixed_cos(monkeypatch):
    """Make the cosine similarity return an exact float, to test the threshold borders."""
    def set_cos(value):
        monkeypatch.setattr(instance_utils.torch.nn.functional, "cosine_similarity",
                            lambda *args, **kwargs: torch.tensor(value, dtype=torch.float64))
        return _instance(1.0, 0.0), _instance(1.0, 0.0)
    return set_cos


def test_thresholds_defaults():
    assert TH == VoxelFusionThresholds(min_cos_sim=0.81, strong_overlap=0.5, weak_overlap=0.2)
    assert VoxelFusionThresholds.from_cfg({}) == TH


def test_thresholds_from_cfg():
    th = VoxelFusionThresholds.from_cfg({"th_cossim": 0.7, "voxel_th_geom": 0.6, "voxel_th_sem": 0.1})
    assert th == VoxelFusionThresholds(0.7, 0.6, 0.1)


def test_real_features_identical_and_orthogonal():
    a, b = _instance(1.0, 2.0, 3.0), _instance(2.0, 4.0, 6.0)
    assert same_instance_by_overlap(a, b, 0.6, TH)
    assert not same_instance_by_overlap(_instance(1.0, 0.0), _instance(0.0, 1.0), 1.0, TH)


@pytest.mark.parametrize("cos, overlap, expected", [
    # semantic gate: below min_cos_sim never fuses, whatever the overlap
    (0.80, 1.0, False),
    (0.81, 0.6, True),     # cos exactly at the gate passes it (strict `<` rejects)
    # strong branch: overlap > strong_overlap
    (0.85, 0.51, True),
    (0.85, 0.5, False),    # overlap exactly 0.5 does not accept
    # weak branch: cos > 0.9 and overlap > weak_overlap
    (0.85, 0.3, False),    # weak overlap but cos not above 0.9
    (0.9, 0.3, False),     # cos exactly 0.9 does not take the weak branch
    (0.95, 0.3, True),
    (0.95, 0.2, False),    # overlap exactly at weak_overlap does not accept
    (0.95, 0.0, False),
])
def test_rule_branches_and_borders(fixed_cos, cos, overlap, expected):
    a, b = fixed_cos(cos)
    assert same_instance_by_overlap(a, b, overlap, TH) is expected


def test_rule_uses_given_thresholds(fixed_cos):
    a, b = fixed_cos(0.75)
    loose = VoxelFusionThresholds(min_cos_sim=0.7, strong_overlap=0.3, weak_overlap=0.1)
    assert same_instance_by_overlap(a, b, 0.31, loose)
    assert not same_instance_by_overlap(a, b, 0.31, TH)
