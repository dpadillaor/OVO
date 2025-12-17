"""
Fixtures for fusion strategy tests
"""

import pytest
import torch
from unittest.mock import MagicMock


@pytest.fixture
def mock_instance():
    """Create a mock Instance3D object with basic attributes."""
    def _create_instance(instance_id: int, clip_feature=None, pe_feature=None, num_points=10):
        instance = MagicMock()
        instance.id = instance_id
        instance.clip_feature = clip_feature if clip_feature is not None else torch.randn(1, 512)
        instance.pe_feature = pe_feature if pe_feature is not None else torch.randn(1, 256)
        instance.kfs_ids = []
        instance.points_ids = list(range(num_points))
        instance.top_kf = []
        return instance
    return _create_instance


@pytest.fixture
def sample_points_centroid():
    """Create sample points and centroid data."""
    def _create_data(center=(0, 0, 0), num_points=100, spread=0.5):
        center = torch.tensor(center, dtype=torch.float32)
        points = center + torch.randn(num_points, 3) * spread
        centroid = points.mean(axis=0)
        return (points, centroid)
    return _create_data


@pytest.fixture
def mock_fusion_strategy():
    """Create a mock fusion strategy for delegation tests."""
    strategy = MagicMock()
    strategy.same_instance = MagicMock(return_value=True)
    return strategy


@pytest.fixture
def fusion_config():
    """Base fusion configuration."""
    return {
        "th_centroid": 1.5,
        "th_cossim": 0.81,
        "th_points": 0.1,
        "fusion_method": "clip"
    }


@pytest.fixture
def minimal_ovo_config():
    """Minimal configuration for OVO initialization tests."""
    return {
        "fusion_method": "clip",
        "th_centroid": 1.5,
        "th_cossim": 0.81,
        "th_points": 0.1,
        "verbose": False,
        "sam": {"multi_crop": False},
        "clip": {"embed_type": "vanilla"},
    }
