"""
Pytest configuration and shared fixtures.

This file automatically loads fixtures from the fixtures directory.
"""

# Import fixtures to make them available to all tests
from tests.fixtures.fixtures_fusion import (
    mock_instance,
    sample_points_centroid,
    mock_fusion_strategy,
    fusion_config,
    minimal_ovo_config,
)

from tests.fixtures.fixtures_sam3 import (
    sam3_config_vit_only,
    sam3_config_vit_neck,
    sam3_config_full,
    sample_image,
    sample_image_batch,
    sample_masks,
    mock_instance_sam3,
    minimal_ovo_config_sam3,
)
