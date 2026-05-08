"""
Fixtures for fusion encoder adapter tests.
"""
import pytest
import torch
from unittest.mock import MagicMock

@pytest.fixture
def mock_pe_generator():
    """Mock PEGenerator."""
    generator = MagicMock()
    # Simulate extraction returning random embeddings
    generator.extract_pe.side_effect = lambda img, masks: torch.randn(len(masks), 768)
    return generator

@pytest.fixture
def mock_sam3_generator():
    """Mock SAM3Generator."""
    generator = MagicMock()
    # Simulate extraction returning random embeddings (1024-dim for vit_only)
    generator.extract_sam3.side_effect = lambda img, masks: torch.randn(len(masks), 1024)
    generator.embed_dim = 1024
    return generator

@pytest.fixture
def mock_keyframes_data():
    """Mock keyframes dictionary structure."""
    return {
        "ins_pe_descriptors": {},
        "ins_sam3_descriptors": {}
    }

@pytest.fixture
def mock_objects_dict():
    """Mock objects dictionary."""
    objects = {}
    for i in range(3):
        obj = MagicMock()
        obj.id = i
        obj.update_pe = MagicMock()
        obj.to_update_pe = True
        obj.update_sam3 = MagicMock()
        obj.to_update_sam3 = True
        objects[i] = obj
    return objects
