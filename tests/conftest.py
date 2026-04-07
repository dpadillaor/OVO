"""
Pytest configuration and shared fixtures.

This file automatically loads fixtures from the fixtures directory.
"""
import sys
from pathlib import Path

# Make thirdParty/perception_models importable as a top-level package
_pm_path = str(Path(__file__).parent.parent / "thirdParty" / "perception_models")
if _pm_path not in sys.path:
    sys.path.insert(0, _pm_path)

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

from tests.fixtures.fixtures_encoders import (
    mock_pe_generator,
    mock_dino_generator,
    mock_sam3_generator,
    mock_keyframes_data,
    mock_objects_dict,
)
