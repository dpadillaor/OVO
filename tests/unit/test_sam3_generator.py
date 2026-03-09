"""Tests for SAM3Generator class."""

import pytest
import torch
import sys
from unittest.mock import MagicMock, patch

# Mock sam3 module before importing SAM3Generator
mock_sam3 = MagicMock()
sys.modules["sam3"] = mock_sam3
sys.modules["sam3.model_builder"] = mock_sam3.model_builder

from ovo.entities.sam3_generator import SAM3Generator
from ovo.entities.fusion import create_fusion_strategy, SemanticGeometricFusion


class TestSAM3GeneratorInit:
    """Test SAM3Generator initialization with different component configurations."""

    @pytest.fixture
    def mock_sam3_model(self):
        """Creates a mock SAM3 model with standard architecture."""
        model = MagicMock()
        model.backbone.vision_backbone.trunk = MagicMock()
        model.backbone.vision_backbone.neck = MagicMock()
        model.backbone.text_backbone = MagicMock()
        return model

    def test_init_vit_only_mode(self, mock_sam3_model):
        """SAM3Generator should load only ViT when components='vit_only'."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "components": "vit_only"
        }
        with patch('sam3.model_builder.build_sam3_image_model', return_value=mock_sam3_model):
            generator = SAM3Generator(config, device="cpu")

            assert generator.vit is not None
            assert generator.neck is None
            assert generator.text_encoder is None

    def test_init_vit_neck_mode(self, mock_sam3_model):
        """SAM3Generator should load ViT and Neck when components='vit_neck'."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "components": "vit_neck"
        }
        with patch('sam3.model_builder.build_sam3_image_model', return_value=mock_sam3_model):
            generator = SAM3Generator(config, device="cpu")

            assert generator.vit is not None
            assert generator.neck is not None
            assert generator.text_encoder is None

    def test_init_full_mode(self, mock_sam3_model):
        """SAM3Generator should load all components when components='full'."""
        config = {
            "checkpoint_path": "/path/to/sam3.pt",
            "components": "full"
        }
        with patch('sam3.model_builder.build_sam3_image_model', return_value=mock_sam3_model):
            generator = SAM3Generator(config, device="cpu")

            assert generator.vit is not None
            assert generator.neck is not None
            assert generator.text_encoder is not None

    def test_default_components_is_vit_only(self, mock_sam3_model):
        """SAM3Generator should default to 'vit_only' mode."""
        config = {"checkpoint_path": "/path/to/sam3.pt"}
        with patch('sam3.model_builder.build_sam3_image_model', return_value=mock_sam3_model):
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


class TestSAM3GeneratorConfig:
    """Test SAM3Generator configuration handling."""

    @pytest.fixture
    def mock_sam3_model(self):
        return MagicMock()

    def test_embed_dim_vit_only(self, mock_sam3_model):
        """embed_dim should be 1024 in vit_only mode."""
        config = {"checkpoint_path": "mock", "components": "vit_only"}
        with patch('sam3.model_builder.build_sam3_image_model', return_value=mock_sam3_model):
            generator = SAM3Generator(config, device="cpu")
            assert generator.embed_dim == 1024

    def test_embed_dim_vit_neck(self, mock_sam3_model):
        """embed_dim should be 256 in vit_neck mode."""
        config = {"checkpoint_path": "mock", "components": "vit_neck"}
        with patch('sam3.model_builder.build_sam3_image_model', return_value=mock_sam3_model):
            generator = SAM3Generator(config, device="cpu")
            assert generator.embed_dim == 256


class TestSAM3GeneratorEncoding:
    """Test SAM3Generator image encoding functionality."""

    @pytest.fixture
    def sam3_generator_vit_only(self):
        mock_model = MagicMock()
        # Use return_value instead of side_effect for simpler mocking of tensor operations
        mock_model.backbone.vision_backbone.trunk.return_value = torch.randn(1, 1024, 64, 64)
        
        config = {"checkpoint_path": "mock", "components": "vit_only"}
        with patch('sam3.model_builder.build_sam3_image_model', return_value=mock_model):
            gen = SAM3Generator(config, device="cpu")
            return gen

    @pytest.fixture
    def sam3_generator_vit_neck(self):
        mock_model = MagicMock()
        mock_model.backbone.vision_backbone.trunk.return_value = torch.randn(1, 1024, 64, 64)
        mock_model.backbone.vision_backbone.neck.return_value = [torch.randn(1, 256, 64, 64)]
        
        config = {"checkpoint_path": "mock", "components": "vit_neck"}
        with patch('sam3.model_builder.build_sam3_image_model', return_value=mock_model):
            gen = SAM3Generator(config, device="cpu")
            return gen

    def test_encode_image_returns_correct_shape_vit_only(self, sam3_generator_vit_only, sample_image):
        """encode_image should return (B, 1024) tensor in vit_only mode."""
        # Ensure trunk return shape matches batch size of input
        B = sample_image.shape[0]
        sam3_generator_vit_only.vit.return_value = torch.randn(B, 1024, 64, 64)
        
        result = sam3_generator_vit_only.encode_image(sample_image)
        assert result.shape == (B, 1024)

    def test_encode_image_returns_correct_shape_vit_neck(self, sam3_generator_vit_neck, sample_image):
        """encode_image should return (B, 256) tensor in vit_neck mode."""
        B = sample_image.shape[0]
        sam3_generator_vit_neck.vit.return_value = torch.randn(B, 1024, 64, 64)
        sam3_generator_vit_neck.neck.return_value = [torch.randn(B, 256, 64, 64)]
        
        result = sam3_generator_vit_neck.encode_image(sample_image)
        assert result.shape == (B, 256)

    def test_extract_sam3_from_masks(self, sam3_generator_vit_only, sample_image, sample_masks):
        """extract_sam3 should compute embeddings for each mask."""
        num_masks = sample_masks.shape[0]
        masks_bool = sample_masks.to(torch.bool)
        
        # When extract_sam3 calls encode_image, it passes num_masks as batch size
        sam3_generator_vit_only.vit.return_value = torch.randn(num_masks, 1024, 64, 64)
        
        result = sam3_generator_vit_only.extract_sam3(sample_image, masks_bool)
        assert result.shape == (num_masks, 1024)


class TestSAM3GeneratorDeviceHandling:
    """Test SAM3Generator device management."""
    
    @pytest.fixture
    def sam3_generator_mock(self):
        mock_model = MagicMock()
        config = {"checkpoint_path": "mock", "components": "vit_only"}
        with patch('sam3.model_builder.build_sam3_image_model', return_value=mock_model):
            gen = SAM3Generator(config, device="cpu")
            return gen

    def test_to_cuda(self, sam3_generator_mock):
        """to('cuda') should move model to GPU."""
        sam3_generator_mock.to("cuda")
        assert sam3_generator_mock.device == "cuda"
        sam3_generator_mock.vit.to.assert_called_with("cuda")
