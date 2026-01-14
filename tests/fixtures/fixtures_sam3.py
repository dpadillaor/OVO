"""Fixtures for SAM3Generator tests."""

import pytest
import torch
from unittest.mock import MagicMock, patch


@pytest.fixture
def sam3_config_vit_only():
    """Configuration for vit_only mode."""
    return {
        "checkpoint_path": "/mock/path/sam3.pt",
        "components": "vit_only",
        "load_from_hf": False,
    }


@pytest.fixture
def sam3_config_vit_neck():
    """Configuration for vit_neck mode."""
    return {
        "checkpoint_path": "/mock/path/sam3.pt",
        "components": "vit_neck",
        "load_from_hf": False,
    }


@pytest.fixture
def sam3_config_full():
    """Configuration for full mode."""
    return {
        "checkpoint_path": "/mock/path/sam3.pt",
        "components": "full",
        "load_from_hf": False,
    }


@pytest.fixture
def sample_image():
    """Sample image tensor (B, C, H, W) in range [0, 1]."""
    return torch.rand(1, 3, 224, 224)


@pytest.fixture
def sample_image_batch():
    """Batch of sample images."""
    return torch.rand(4, 3, 224, 224)


@pytest.fixture
def sample_masks():
    """Sample binary masks (N, H, W)."""
    masks = torch.zeros(5, 224, 224)
    for i in range(5):
        masks[i, i*40:(i+1)*40, i*40:(i+1)*40] = 1
    return masks


@pytest.fixture
def mock_instance_sam3():
    """Create mock Instance3D with sam3_feature."""
    def _create(instance_id, embed_dim=1024):
        instance = MagicMock()
        instance.id = instance_id
        instance.sam3_feature = torch.randn(1, embed_dim)
        instance.clip_feature = torch.randn(1, 512)
        instance.pe_feature = torch.randn(1, 256)
        instance.kfs_ids = []
        instance.points_ids = list(range(10))
        return instance
    return _create


@pytest.fixture
def minimal_ovo_config_sam3():
    """Minimal OVO config with SAM3 enabled."""
    return {
        "fusion_method": "sam3",
        "th_centroid": 1.5,
        "th_cossim": 0.81,
        "th_points": 0.1,
        "verbose": False,
        "sam": {"multi_crop": False},
        "clip": {"embed_type": "vanilla"},
        "sam3": {
            "checkpoint_path": "/mock/path/sam3.pt",
            "components": "vit_only",
            "load_from_hf": False,
        },
    }

@pytest.fixture
def sample_points_centroid_sam3():
    """Create sample points and centroid data."""
    def _create_data(center=(0, 0, 0), num_points=100, spread=0.5):
        center = torch.tensor(center, dtype=torch.float32)
        points = center + torch.randn(num_points, 3) * spread
        centroid = points.mean(axis=0)
        return (points, centroid)
    return _create_data
