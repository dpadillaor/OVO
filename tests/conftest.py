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
