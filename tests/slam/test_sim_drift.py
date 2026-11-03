"""drift: seeded per-frame error."""

import pytest
import torch

from ovo.slam.simulated.drift import Drift, DriftCfg


def steps(cfg, n=5):
    drift = Drift(cfg)
    return [drift.step() for _ in range(n)]


def test_defaults():
    assert DriftCfg() == DriftCfg(translation_std=0.01, rotation_std_deg=0.5, seed=42)


def test_same_seed_same_steps():
    for a, b in zip(steps(DriftCfg(seed=7)), steps(DriftCfg(seed=7))):
        assert torch.equal(a, b)


def test_different_seed_differs():
    a, b = steps(DriftCfg(seed=7), 1)[0], steps(DriftCfg(seed=8), 1)[0]
    assert not torch.allclose(a, b)


def test_successive_steps_differ():
    a, b = steps(DriftCfg(), 2)
    assert not torch.allclose(a, b)


def test_step_is_a_rigid_transform():
    T = steps(DriftCfg(translation_std=0.5, rotation_std_deg=20.0), 1)[0]
    R = T[:3, :3]
    assert T.dtype == torch.float64
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
    assert torch.equal(T[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64))
    assert not torch.allclose(T, torch.eye(4, dtype=torch.float64))


def test_zero_std_is_identity():
    for T in steps(DriftCfg(translation_std=0.0, rotation_std_deg=0.0)):
        assert torch.equal(T, torch.eye(4, dtype=torch.float64))


def test_zero_rotation_std_only_translates():
    T = steps(DriftCfg(translation_std=0.1, rotation_std_deg=0.0), 1)[0]
    assert torch.equal(T[:3, :3], torch.eye(3, dtype=torch.float64))
    assert torch.linalg.norm(T[:3, 3]) > 0


@pytest.mark.parametrize("kwargs", [{"translation_std": -0.01}, {"rotation_std_deg": -1.0}])
def test_negative_std_raises(kwargs):
    with pytest.raises(ValueError, match="std must be >= 0"):
        DriftCfg(**kwargs)
