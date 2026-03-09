# TDD Tests for SAM3 Perception Model Integration

These tests follow the TDD methodology: **Write tests first → Run and fail (Red) → Implement → Pass (Green) → Refactor**.

All tests should be created in `tests/unit/test_sam3_generator.py` and fixtures in `tests/fixtures/fixtures_sam3.py`.

---

## Test Suite Overview

| Test Class | Purpose |
|------------|---------|
| `TestSAM3GeneratorInit` | Constructor and component loading |
| `TestSAM3GeneratorConfig` | Configuration options |
| `TestSAM3GeneratorEncoding` | Image encoding functionality |
| `TestSAM3GeneratorFusion` | Integration with fusion strategy |
| `TestSAM3GeneratorDeviceHandling` | Device (CPU/CUDA) management |

---

## 1. Constructor and Component Loading Tests

```python
class TestSAM3GeneratorInit:
    """Test SAM3Generator initialization with different component configurations."""

    def test_init_vit_only_mode(self):
        """SAM3Generator should load only ViT when components='vit_only'."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "components": "vit_only"
        }
        generator = SAM3Generator(config, device="cpu")

        assert generator.vit is not None
        assert generator.neck is None
        assert generator.text_encoder is None

    def test_init_vit_neck_mode(self):
        """SAM3Generator should load ViT and Neck when components='vit_neck'."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "components": "vit_neck"
        }
        generator = SAM3Generator(config, device="cpu")

        assert generator.vit is not None
        assert generator.neck is not None
        assert generator.text_encoder is None

    def test_init_full_mode(self):
        """SAM3Generator should load all components when components='full'."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "components": "full"
        }
        generator = SAM3Generator(config, device="cpu")

        assert generator.vit is not None
        assert generator.neck is not None
        assert generator.text_encoder is not None

    def test_default_components_is_vit_only(self):
        """SAM3Generator should default to 'vit_only' mode."""
        config = {"checkpoint_path": "/path/to/sam3.pt"}
        generator = SAM3Generator(config, device="cpu")

        assert generator.components == "vit_only"
        assert generator.vit is not None
        assert generator.neck is None

    def test_invalid_components_raises_error(self):
        """SAM3Generator should raise ValueError for invalid components option."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "components": "invalid_option"
        }

        with pytest.raises(ValueError, match="Invalid components"):
            SAM3Generator(config, device="cpu")

    def test_missing_checkpoint_raises_error(self):
        """SAM3Generator should raise error when checkpoint_path is missing and load_from_hf=False."""
        config = {"components": "vit_only", "load_from_hf": False}

        with pytest.raises(ValueError, match="checkpoint_path"):
            SAM3Generator(config, device="cpu")

    def test_load_from_huggingface(self):
        """SAM3Generator should download from HuggingFace when load_from_hf=True."""
        config = {
            "components": "vit_only",
            "load_from_hf": True
        }
        # This test requires mocking hf_hub_download
        with patch('ovo.entities.sam3_generator.hf_hub_download') as mock_download:
            mock_download.return_value = "/tmp/sam3.pt"
            generator = SAM3Generator(config, device="cpu")
            mock_download.assert_called()
```

---

## 2. Configuration Tests

```python
class TestSAM3GeneratorConfig:
    """Test SAM3Generator configuration handling."""

    def test_embed_dim_vit_only(self):
        """embed_dim should be 1024 in vit_only mode (raw ViT output)."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "components": "vit_only"
        }
        generator = SAM3Generator(config, device="cpu")

        assert generator.embed_dim == 1024

    def test_embed_dim_vit_neck(self):
        """embed_dim should be 256 in vit_neck mode (after neck projection)."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "components": "vit_neck"
        }
        generator = SAM3Generator(config, device="cpu")

        assert generator.embed_dim == 256

    def test_image_size_default(self):
        """Default image_size should be 1008 (SAM3's native resolution)."""
        config = {"checkpoint_path": "/path/to/sam3.pt"}
        generator = SAM3Generator(config, device="cpu")

        assert generator.image_size == 1008

    def test_image_size_override(self):
        """image_size should be configurable via config."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "image_size": 512
        }
        generator = SAM3Generator(config, device="cpu")

        assert generator.image_size == 512

    def test_use_half_precision(self):
        """SAM3Generator should support half precision mode."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "use_half": True
        }
        generator = SAM3Generator(config, device="cuda")

        # Check model is in half precision
        assert next(generator.vit.parameters()).dtype == torch.float16

    def test_stores_config(self):
        """SAM3Generator should store the configuration."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "components": "vit_neck",
            "use_half": False
        }
        generator = SAM3Generator(config, device="cpu")

        assert generator.config == config
```

---

## 3. Image Encoding Tests

```python
class TestSAM3GeneratorEncoding:
    """Test SAM3Generator image encoding functionality."""

    def test_encode_image_returns_correct_shape_vit_only(self, sam3_generator_vit_only, sample_image):
        """encode_image should return (B, 1024) tensor in vit_only mode."""
        result = sam3_generator_vit_only.encode_image(sample_image)

        assert result.shape == (1, 1024)

    def test_encode_image_returns_correct_shape_vit_neck(self, sam3_generator_vit_neck, sample_image):
        """encode_image should return (B, 256) tensor in vit_neck mode."""
        result = sam3_generator_vit_neck.encode_image(sample_image)

        assert result.shape == (1, 256)

    def test_encode_image_batch(self, sam3_generator_vit_only, sample_image_batch):
        """encode_image should handle batched inputs."""
        batch_size = 4
        result = sam3_generator_vit_only.encode_image(sample_image_batch)

        assert result.shape == (batch_size, 1024)

    def test_encode_image_3d_input(self, sam3_generator_vit_only):
        """encode_image should handle 3D input (C, H, W) by adding batch dim."""
        image_3d = torch.randn(3, 224, 224)
        result = sam3_generator_vit_only.encode_image(image_3d)

        assert result.shape == (1, 1024)

    def test_encode_image_normalized_output(self, sam3_generator_vit_only, sample_image):
        """encode_image output should be L2 normalized."""
        result = sam3_generator_vit_only.encode_image(sample_image)
        norms = torch.norm(result, p=2, dim=-1)

        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_extract_sam3_from_masks(self, sam3_generator_vit_only, sample_image, sample_masks):
        """extract_sam3 should compute embeddings for each mask."""
        num_masks = sample_masks.shape[0]
        result = sam3_generator_vit_only.extract_sam3(sample_image, sample_masks)

        assert result.shape == (num_masks, 1024)

    def test_extract_sam3_empty_masks(self, sam3_generator_vit_only, sample_image):
        """extract_sam3 should return empty tensor for no masks."""
        empty_masks = torch.zeros(0, 224, 224)
        result = sam3_generator_vit_only.extract_sam3(sample_image, empty_masks)

        assert result.shape[0] == 0
```

---

## 4. Fusion Integration Tests

```python
class TestSAM3GeneratorFusion:
    """Test SAM3Generator integration with OVO fusion strategy."""

    def test_fusion_factory_creates_sam3_strategy(self):
        """Factory should return SemanticGeometricFusion for 'sam3' config."""
        config = {
            "fusion_method": "sam3",
            "th_centroid": 1.5,
            "th_cossim": 0.81,
            "th_points": 0.1
        }
        strategy = create_fusion_strategy(config)

        assert isinstance(strategy, SemanticGeometricFusion)
        assert strategy.feature_attr == "sam3_feature"

    def test_instance_has_sam3_feature_attribute(self, mock_instance_sam3):
        """Instance3D should have sam3_feature attribute."""
        instance = mock_instance_sam3(1)

        assert hasattr(instance, 'sam3_feature')
        assert instance.sam3_feature.shape[-1] == 1024  # or 256 for neck mode

    def test_ovo_initializes_sam3_generator(self, minimal_ovo_config_sam3):
        """OVO should initialize SAM3Generator when sam3 config is present."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen:
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            MockSAM3Gen.assert_called_once()
            assert ovo.sam3_generator is not None

    def test_ovo_updates_sam3_features(self, minimal_ovo_config_sam3):
        """OVO should call update_objects_sam3 during map update."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            with patch.object(ovo, 'update_objects_sam3') as mock_update:
                # Trigger map update (simplified)
                ovo.keyframes_queue = [1]
                ovo.keyframes = {
                    "ins_descriptors": {1: torch.randn(5, 512)},
                    "ins_sam3_descriptors": {1: torch.randn(5, 1024)},
                    "frame_id": [1],
                    "ins_maps": [torch.zeros(5, 100, 100)],
                }
                ovo.update_map((torch.randn(100, 3), torch.arange(100), torch.ones(100).long()), [])

                mock_update.assert_called()
```

---

## 5. Device Handling Tests

```python
class TestSAM3GeneratorDeviceHandling:
    """Test SAM3Generator device management."""

    def test_to_cuda(self, sam3_generator_cpu):
        """to('cuda') should move model to GPU."""
        sam3_generator_cpu.to("cuda")

        assert sam3_generator_cpu.device == "cuda"
        assert next(sam3_generator_cpu.vit.parameters()).device.type == "cuda"

    def test_to_cpu(self, sam3_generator_cuda):
        """to('cpu') should move model to CPU."""
        sam3_generator_cuda.to("cpu")

        assert sam3_generator_cuda.device == "cpu"
        assert next(sam3_generator_cuda.vit.parameters()).device.type == "cpu"

    def test_cuda_method(self, sam3_generator_cpu):
        """cuda() should move model to GPU."""
        sam3_generator_cpu.cuda()

        assert sam3_generator_cpu.device == "cuda"

    def test_cpu_method(self, sam3_generator_cuda):
        """cpu() should move model to CPU."""
        sam3_generator_cuda.cpu()

        assert sam3_generator_cpu.device == "cpu"

    def test_output_on_same_device(self, sam3_generator_cuda, sample_image_cuda):
        """Output tensor should be on same device as input."""
        result = sam3_generator_cuda.encode_image(sample_image_cuda)

        assert result.device.type == "cuda"
```

---

## 6. Required Fixtures

Add to `tests/fixtures/fixtures_sam3.py`:

```python
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
```

---

---

## 7. Instance3D Integration Tests

**File:** `tests/unit/test_instance3d_sam3.py`

Tests for SAM3 feature support in Instance3D class.

```python
"""Tests for Instance3D SAM3 feature integration."""

import pytest
import torch
from ovo.entities.instance3d import Instance3D


class TestInstance3DSAM3Attributes:
    """Test Instance3D has SAM3-related attributes."""

    def test_instance_has_sam3_feature_attribute(self):
        """Instance3D should have sam3_feature attribute initialized to None."""
        instance = Instance3D(id=1)

        assert hasattr(instance, 'sam3_feature')
        assert instance.sam3_feature is None

    def test_instance_has_sam3_feature_kf_attribute(self):
        """Instance3D should have sam3_feature_kf attribute initialized to None."""
        instance = Instance3D(id=1)

        assert hasattr(instance, 'sam3_feature_kf')
        assert instance.sam3_feature_kf is None

    def test_instance_has_to_update_sam3_flag(self):
        """Instance3D should have to_update_sam3 flag initialized to False."""
        instance = Instance3D(id=1)

        assert hasattr(instance, 'to_update_sam3')
        assert instance.to_update_sam3 == False

    def test_instance_update_sets_to_update_sam3_flag(self):
        """Instance3D.update() should set to_update_sam3 to True."""
        instance = Instance3D(id=1)
        instance.update(points_ids=[1, 2, 3], kf_id=0, area=100)

        assert instance.to_update_sam3 == True


class TestInstance3DSAM3Update:
    """Test Instance3D.update_sam3() method."""

    def test_update_sam3_computes_feature(self):
        """update_sam3 should compute sam3_feature from keyframe descriptors."""
        instance = Instance3D(id=1)
        instance.update(points_ids=[1, 2, 3], kf_id=0, area=100)

        keyframes_sam3 = {
            0: {1: torch.randn(1, 1024)}
        }

        instance.update_sam3(keyframes_sam3)

        assert instance.sam3_feature is not None
        assert instance.sam3_feature.shape == (1, 1024)

    def test_update_sam3_sets_feature_kf(self):
        """update_sam3 should set sam3_feature_kf to the selected keyframe."""
        instance = Instance3D(id=1)
        instance.update(points_ids=[1, 2, 3], kf_id=0, area=100)

        keyframes_sam3 = {
            0: {1: torch.randn(1, 1024)}
        }

        instance.update_sam3(keyframes_sam3)

        assert instance.sam3_feature_kf is not None

    def test_update_sam3_clears_to_update_flag(self):
        """update_sam3 should set to_update_sam3 to False after update."""
        instance = Instance3D(id=1)
        instance.update(points_ids=[1, 2, 3], kf_id=0, area=100)

        keyframes_sam3 = {
            0: {1: torch.randn(1, 1024)}
        }

        instance.update_sam3(keyframes_sam3)

        assert instance.to_update_sam3 == False

    def test_update_sam3_skips_when_flag_false(self):
        """update_sam3 should skip computation when to_update_sam3 is False."""
        instance = Instance3D(id=1)
        instance.to_update_sam3 = False
        instance.sam3_feature = torch.randn(1, 1024)
        original_feature = instance.sam3_feature.clone()

        keyframes_sam3 = {
            0: {1: torch.randn(1, 1024)}  # Different feature
        }

        instance.update_sam3(keyframes_sam3)

        assert torch.equal(instance.sam3_feature, original_feature)

    def test_update_sam3_force_update(self):
        """update_sam3 should recompute when force_update=True."""
        instance = Instance3D(id=1)
        instance.kfs_ids = [0]
        instance.to_update_sam3 = False
        instance.sam3_feature = torch.zeros(1, 1024)

        keyframes_sam3 = {
            0: {1: torch.ones(1, 1024)}  # Different feature
        }

        instance.update_sam3(keyframes_sam3, force_update=True)

        assert torch.allclose(instance.sam3_feature, torch.ones(1, 1024))

    def test_update_sam3_multiple_keyframes_uses_median(self):
        """update_sam3 should select feature minimizing L1 norm (median)."""
        instance = Instance3D(id=1)
        instance.kfs_ids = [0, 1, 2]
        instance.to_update_sam3 = True

        # Create 3 features where the middle one is the median
        keyframes_sam3 = {
            0: {1: torch.tensor([[0.0, 0.0, 0.0, 0.0]])},
            1: {1: torch.tensor([[0.5, 0.5, 0.5, 0.5]])},  # Median
            2: {1: torch.tensor([[1.0, 1.0, 1.0, 1.0]])},
        }

        instance.update_sam3(keyframes_sam3)

        # Should select the median feature (index 1)
        expected = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        assert torch.allclose(instance.sam3_feature, expected)

    def test_update_sam3_empty_keyframes(self):
        """update_sam3 should handle empty keyframes gracefully."""
        instance = Instance3D(id=1)
        instance.kfs_ids = [0]
        instance.to_update_sam3 = True

        keyframes_sam3 = {}  # No keyframes

        instance.update_sam3(keyframes_sam3)

        assert instance.sam3_feature is None


class TestInstance3DSAM3Export:
    """Test Instance3D export/restore with SAM3 features."""

    def test_export_includes_sam3_feature(self):
        """export() should include sam3_feature in the dictionary."""
        instance = Instance3D(id=1)
        instance.sam3_feature = torch.randn(1, 1024)
        instance.sam3_feature_kf = 5

        exported = instance.export()

        assert f"ins3d_1_sam3_feature" in exported
        assert f"ins3d_1_sam3_feature_kf" in exported

    def test_restore_loads_sam3_feature(self):
        """restore() should load sam3_feature from dictionary."""
        instance = Instance3D(id=1)
        sam3_feat = torch.randn(1, 1024)

        obj_dict = {
            "ins3d_1_clip_feature": None,
            "ins3d_1_clip_feature_kf": None,
            "ins3d_1_pe_feature": None,
            "ins3d_1_pe_feature_kf": None,
            "ins3d_1_sam3_feature": sam3_feat,
            "ins3d_1_sam3_feature_kf": 3,
        }

        instance.restore(obj_dict, debug_info=False)

        assert torch.equal(instance.sam3_feature, sam3_feat)
        assert instance.sam3_feature_kf == 3

    def test_restore_sets_to_update_sam3_when_none(self):
        """restore() should set to_update_sam3=True when sam3_feature is None."""
        instance = Instance3D(id=1)

        obj_dict = {
            "ins3d_1_clip_feature": None,
            "ins3d_1_clip_feature_kf": None,
            "ins3d_1_pe_feature": None,
            "ins3d_1_pe_feature_kf": None,
            "ins3d_1_sam3_feature": None,
            "ins3d_1_sam3_feature_kf": None,
        }

        instance.restore(obj_dict, debug_info=False)

        assert instance.to_update_sam3 == True
```

---

## 8. Fusion Strategy Integration Tests

**File:** `tests/unit/test_fusion_sam3.py`

Tests for SAM3 integration in the fusion strategy factory.

```python
"""Tests for SAM3 fusion strategy integration."""

import pytest
import torch
from unittest.mock import MagicMock

from ovo.entities.fusion import (
    FusionStrategy,
    SemanticGeometricFusion,
    create_fusion_strategy
)


class TestFusionFactorySAM3:
    """Test fusion factory creates SAM3 strategy correctly."""

    def test_create_sam3_fusion_strategy(self):
        """Factory should return SemanticGeometricFusion for 'sam3' config."""
        config = {
            "fusion_method": "sam3",
            "th_centroid": 1.5,
            "th_cossim": 0.81,
            "th_points": 0.1
        }
        strategy = create_fusion_strategy(config)

        assert isinstance(strategy, SemanticGeometricFusion)
        assert strategy.feature_attr == "sam3_feature"

    def test_sam3_strategy_stores_thresholds(self):
        """SAM3 strategy should store threshold values from config."""
        config = {
            "fusion_method": "sam3",
            "th_centroid": 2.0,
            "th_cossim": 0.9,
            "th_points": 0.2
        }
        strategy = create_fusion_strategy(config)

        assert strategy.th_centroid == 2.0
        assert strategy.th_cossim == 0.9
        assert strategy.th_points == 0.2


class TestSemanticGeometricFusionSAM3:
    """Test SemanticGeometricFusion works with sam3_feature."""

    def test_same_instance_reads_sam3_feature(self, mock_instance_sam3, sample_points_centroid):
        """same_instance should read sam3_feature from instances."""
        config = {
            "th_centroid": 10.0,  # High threshold to pass
            "th_cossim": 0.5,    # Low threshold to pass
            "th_points": 0.01
        }
        strategy = SemanticGeometricFusion(config, feature_attr="sam3_feature")

        # Create instances with similar SAM3 features
        instance1 = mock_instance_sam3(1)
        instance2 = mock_instance_sam3(2)
        instance1.sam3_feature = torch.tensor([[1.0, 0.0, 0.0]])
        instance2.sam3_feature = torch.tensor([[1.0, 0.0, 0.0]])  # Same

        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.1, 0, 0))  # Close

        result = strategy.same_instance(instance1, instance2, data1, data2)

        # Should fuse because features are identical and points are close
        assert result == True

    def test_same_instance_rejects_dissimilar_sam3_features(self, mock_instance_sam3, sample_points_centroid):
        """same_instance should reject instances with dissimilar sam3_features."""
        config = {
            "th_centroid": 10.0,
            "th_cossim": 0.9,  # High threshold
            "th_points": 0.01
        }
        strategy = SemanticGeometricFusion(config, feature_attr="sam3_feature")

        instance1 = mock_instance_sam3(1)
        instance2 = mock_instance_sam3(2)
        instance1.sam3_feature = torch.tensor([[1.0, 0.0, 0.0]])
        instance2.sam3_feature = torch.tensor([[0.0, 1.0, 0.0]])  # Orthogonal

        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.1, 0, 0))

        result = strategy.same_instance(instance1, instance2, data1, data2)

        assert result == False

    def test_same_instance_handles_1024_dim_features(self, mock_instance_sam3, sample_points_centroid):
        """same_instance should work with 1024-dimensional SAM3 features."""
        config = {
            "th_centroid": 10.0,
            "th_cossim": 0.8,
            "th_points": 0.01
        }
        strategy = SemanticGeometricFusion(config, feature_attr="sam3_feature")

        # Create normalized 1024-dim features
        feat1 = torch.randn(1, 1024)
        feat1 = feat1 / feat1.norm()
        feat2 = feat1 + 0.1 * torch.randn(1, 1024)  # Slightly perturbed
        feat2 = feat2 / feat2.norm()

        instance1 = mock_instance_sam3(1)
        instance2 = mock_instance_sam3(2)
        instance1.sam3_feature = feat1
        instance2.sam3_feature = feat2

        data1 = sample_points_centroid((0, 0, 0))
        data2 = sample_points_centroid((0.1, 0, 0))

        # Should not raise any dimension errors
        result = strategy.same_instance(instance1, instance2, data1, data2)
        assert isinstance(result, bool)
```

---

## 9. OVO Integration Tests

**File:** `tests/unit/test_ovo_sam3.py`

Tests for SAM3 integration in the main OVO class.

```python
"""Tests for OVO SAM3 integration."""

import pytest
import torch
from unittest.mock import MagicMock, patch


class TestOVOSAM3Initialization:
    """Test OVO initializes SAM3Generator correctly."""

    def test_ovo_initializes_sam3_generator_when_config_present(self, minimal_ovo_config_sam3):
        """OVO should create SAM3Generator when 'sam3' key is in config."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            MockSAM3Gen.assert_called_once()
            assert ovo.sam3_generator is not None

    def test_ovo_sam3_generator_none_when_no_config(self, minimal_ovo_config):
        """OVO should have sam3_generator=None when 'sam3' not in config."""
        with patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            # minimal_ovo_config doesn't have 'sam3' key
            ovo = OVO(minimal_ovo_config, mock_logger, eval=True)

            assert ovo.sam3_generator is None

    def test_ovo_creates_sam3_fusion_strategy(self, minimal_ovo_config_sam3):
        """OVO should create SAM3 fusion strategy when fusion_method='sam3'."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger
            from ovo.entities.fusion import SemanticGeometricFusion

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            assert isinstance(ovo.fusion_strategy, SemanticGeometricFusion)
            assert ovo.fusion_strategy.feature_attr == "sam3_feature"


class TestOVOSAM3Keyframes:
    """Test OVO manages SAM3 keyframe descriptors."""

    def test_ovo_has_ins_sam3_descriptors_in_keyframes(self, minimal_ovo_config_sam3):
        """OVO.keyframes should have 'ins_sam3_descriptors' key."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            assert "ins_sam3_descriptors" in ovo.keyframes

    def test_ovo_stores_sam3_descriptors_per_keyframe(self, minimal_ovo_config_sam3):
        """OVO should store SAM3 descriptors indexed by keyframe ID."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            # Setup mock SAM3 generator
            mock_sam3 = MagicMock()
            mock_sam3.extract_sam3.return_value = torch.randn(5, 1024)
            MockSAM3Gen.return_value = mock_sam3

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Simulate adding descriptors for keyframe 0
            kf_id = 0
            descriptors = torch.randn(5, 1024)
            ovo.keyframes["ins_sam3_descriptors"][kf_id] = {
                i: descriptors[i:i+1] for i in range(5)
            }

            assert kf_id in ovo.keyframes["ins_sam3_descriptors"]
            assert len(ovo.keyframes["ins_sam3_descriptors"][kf_id]) == 5


class TestOVOUpdateObjectsSAM3:
    """Test OVO.update_objects_sam3() method."""

    def test_ovo_has_update_objects_sam3_method(self, minimal_ovo_config_sam3):
        """OVO should have update_objects_sam3 method."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            assert hasattr(ovo, 'update_objects_sam3')
            assert callable(ovo.update_objects_sam3)

    def test_update_objects_sam3_calls_instance_update(self, minimal_ovo_config_sam3):
        """update_objects_sam3 should call update_sam3 on each instance."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Create mock instances
            mock_instance1 = MagicMock()
            mock_instance2 = MagicMock()
            ovo.objects = {1: mock_instance1, 2: mock_instance2}

            ovo.update_objects_sam3()

            mock_instance1.update_sam3.assert_called_once()
            mock_instance2.update_sam3.assert_called_once()

    def test_update_objects_sam3_passes_keyframes(self, minimal_ovo_config_sam3):
        """update_objects_sam3 should pass ins_sam3_descriptors to instances."""
        with patch('ovo.entities.ovo.SAM3Generator'), \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Setup keyframes with SAM3 descriptors
            ovo.keyframes["ins_sam3_descriptors"] = {
                0: {1: torch.randn(1, 1024)}
            }

            mock_instance = MagicMock()
            ovo.objects = {1: mock_instance}

            ovo.update_objects_sam3()

            # Verify keyframes dict was passed
            call_args = mock_instance.update_sam3.call_args
            assert call_args[0][0] == ovo.keyframes["ins_sam3_descriptors"]


class TestOVOMapUpdateWithSAM3:
    """Test OVO.update_map integrates SAM3 feature extraction."""

    def test_update_map_extracts_sam3_features(self, minimal_ovo_config_sam3):
        """update_map should extract SAM3 features for new keyframes."""
        with patch('ovo.entities.ovo.SAM3Generator') as MockSAM3Gen, \
             patch('ovo.entities.ovo.CLIPGenerator'), \
             patch('ovo.entities.ovo.PEGenerator'):
            from ovo.entities.ovo import OVO
            from ovo.entities.logger import Logger

            mock_sam3 = MagicMock()
            mock_sam3.extract_sam3.return_value = torch.randn(3, 1024)
            MockSAM3Gen.return_value = mock_sam3

            mock_logger = MagicMock(spec=Logger)
            ovo = OVO(minimal_ovo_config_sam3, mock_logger, eval=True)

            # Setup for update_map
            ovo.keyframes_queue = [0]
            ovo.keyframes["frame_id"] = [0]
            ovo.keyframes["ins_maps"] = [torch.zeros(3, 100, 100)]
            ovo.keyframes["rgb"] = [torch.randn(3, 100, 100)]

            with patch.object(ovo, 'complete_semantic_info'), \
                 patch.object(ovo, 'update_objects_clip'), \
                 patch.object(ovo, 'update_objects_pe'), \
                 patch.object(ovo, 'update_objects_sam3'):

                points_3d = torch.randn(100, 3)
                points_ids = torch.arange(100)
                points_ins_ids = torch.ones(100).long()
                ovo.update_map((points_3d, points_ids, points_ins_ids), [])

                # Verify SAM3 extraction was called
                mock_sam3.extract_sam3.assert_called()
```

---

## 10. Updated Fixtures

Add to `tests/fixtures/fixtures_sam3.py`:

```python
@pytest.fixture
def sample_points_centroid():
    """Create sample points and centroid data."""
    def _create_data(center=(0, 0, 0), num_points=100, spread=0.5):
        center = torch.tensor(center, dtype=torch.float32)
        points = center + torch.randn(num_points, 3) * spread
        centroid = points.mean(axis=0)
        return (points, centroid)
    return _create_data
```

---

## Test Execution Order (TDD)

### Phase 1: SAM3Generator Tests
```bash
pytest tests/unit/test_sam3_generator.py -v
# Expected: All fail (SAM3Generator doesn't exist)
```

### Phase 2: Instance3D Tests
```bash
pytest tests/unit/test_instance3d_sam3.py -v
# Expected: All fail (sam3_feature attribute doesn't exist)
```

### Phase 3: Fusion Tests
```bash
pytest tests/unit/test_fusion_sam3.py -v
# Expected: Fail (sam3 not in strategy_map)
```

### Phase 4: OVO Integration Tests
```bash
pytest tests/unit/test_ovo_sam3.py -v
# Expected: Fail (no SAM3Generator import, no update_objects_sam3)
```

### Full Test Suite
```bash
# Run all SAM3-related tests
pytest tests/unit/test_sam3*.py tests/unit/test_instance3d_sam3.py tests/unit/test_fusion_sam3.py tests/unit/test_ovo_sam3.py -v

# Run with coverage
pytest tests/unit/ --cov=ovo.entities -v
```

---

## Test Files Summary

| File | Tests | Target Code |
|------|-------|-------------|
| `test_sam3_generator.py` | 25+ | `ovo/entities/sam3_generator.py` |
| `test_instance3d_sam3.py` | 12 | `ovo/entities/instance3d.py` |
| `test_fusion_sam3.py` | 5 | `ovo/entities/fusion.py` |
| `test_ovo_sam3.py` | 10 | `ovo/entities/ovo.py` |
| `fixtures_sam3.py` | - | Test fixtures |
