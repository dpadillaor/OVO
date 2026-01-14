# Tests: Perception Encoders Integration

## Resumen

Suite de tests para validar la integración de PE en OVO. Organizado en:
1. **Unit Tests** - Componentes aislados
2. **Integration Tests** - Flujos completos
3. **Config Tests** - Validación de configuración

---

## 1. Unit Tests: PEGenerator

**Archivo**: `tests/unit/test_pe_generator.py`

### Initialization Tests

```python
class TestPEGeneratorInit:
    """Tests for PEGenerator initialization."""

    def test_init_creates_model(self, pe_config, device):
        """Verify PEGenerator initializes PE model correctly."""
        gen = PEGenerator(pe_config, device)
        assert gen.model is not None
        assert gen.device == device

    def test_init_creates_tokenizer(self, pe_config, device):
        """Verify tokenizer is initialized for text encoding."""
        gen = PEGenerator(pe_config, device)
        assert gen.tokenizer is not None

    def test_init_creates_preprocessing(self, pe_config, device):
        """Verify preprocessing transforms are set up."""
        gen = PEGenerator(pe_config, device)
        assert gen.preprocess is not None

    def test_get_pe_dim_returns_correct_dimension(self, pe_generator):
        """Verify get_pe_dim returns model's embedding dimension."""
        dim = pe_generator.get_pe_dim
        assert isinstance(dim, int)
        assert dim > 0  # Typically 768 or 1024
```

### Image Encoding Tests

```python
class TestPEGeneratorImageEncoding:
    """Tests for PE image encoding."""

    def test_encode_image_single_input_shape(self, pe_generator, sample_image_tensor):
        """Single image encoding produces correct shape."""
        # Input: (3, H, W) tensor in [0, 1]
        output = pe_generator.encode_image(sample_image_tensor)
        assert output.shape == (1, pe_generator.get_pe_dim)

    def test_encode_image_batch_input_shape(self, pe_generator, sample_images_batch):
        """Batch encoding produces correct shape."""
        # Input: (N, 3, H, W) tensor
        N = sample_images_batch.shape[0]
        output = pe_generator.encode_image(sample_images_batch)
        assert output.shape == (N, pe_generator.get_pe_dim)

    def test_encode_image_output_normalized(self, pe_generator, sample_image_tensor):
        """Verify output embeddings are L2-normalized."""
        output = pe_generator.encode_image(sample_image_tensor)
        norms = torch.norm(output, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_encode_image_deterministic(self, pe_generator, sample_image_tensor):
        """Same input produces same output."""
        output1 = pe_generator.encode_image(sample_image_tensor)
        output2 = pe_generator.encode_image(sample_image_tensor)
        assert torch.allclose(output1, output2)
```

### Extract PE Tests

```python
class TestPEGeneratorExtractPE:
    """Tests for extract_pe method."""

    def test_extract_pe_output_shape(self, pe_generator, sample_image_hwc, sample_masks):
        """extract_pe produces embeddings for each mask."""
        # Input: image (H,W,3) in [0,255], masks (N,H,W) binary
        N = sample_masks.shape[0]
        output = pe_generator.extract_pe(sample_image_hwc, sample_masks)
        assert output.shape == (N, pe_generator.get_pe_dim)

    def test_extract_pe_handles_input_range(self, pe_generator):
        """Verify extract_pe handles [0,255] input correctly."""
        image = torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)
        masks = torch.ones((2, 480, 640), dtype=torch.bool)
        # Should not raise - normalizes internally
        output = pe_generator.extract_pe(image, masks)
        assert output.shape[0] == 2

    def test_extract_pe_different_masks_different_embeddings(self, pe_generator, sample_image_hwc):
        """Different masks produce different embeddings."""
        mask1 = torch.zeros((1, 480, 640), dtype=torch.bool)
        mask1[0, :240, :320] = True  # Top-left quadrant
        mask2 = torch.zeros((1, 480, 640), dtype=torch.bool)
        mask2[0, 240:, 320:] = True  # Bottom-right quadrant

        emb1 = pe_generator.extract_pe(sample_image_hwc, mask1)
        emb2 = pe_generator.extract_pe(sample_image_hwc, mask2)

        # Should be different (unless image is uniform)
        assert not torch.allclose(emb1, emb2, atol=0.1)
```

### Text Encoding Tests

```python
class TestPEGeneratorTextEncoding:
    """Tests for PE text encoding."""

    def test_get_txt_embedding_single(self, pe_generator):
        """Single text encoding produces correct shape."""
        output = pe_generator.get_txt_embedding(["a chair"])
        assert output.shape == (1, pe_generator.get_pe_dim)

    def test_get_txt_embedding_batch(self, pe_generator):
        """Batch text encoding works correctly."""
        texts = ["a chair", "a table", "a sofa"]
        output = pe_generator.get_txt_embedding(texts)
        assert output.shape == (3, pe_generator.get_pe_dim)

    def test_get_txt_embedding_normalized(self, pe_generator):
        """Text embeddings are L2-normalized."""
        output = pe_generator.get_txt_embedding(["a chair"])
        norm = torch.norm(output, p=2, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
```

### Similarity Tests

```python
class TestPEGeneratorSimilarity:
    """Tests for similarity computation."""

    def test_get_embed_txt_similarity_shape(self, pe_generator, sample_embedding):
        """Similarity computation returns correct shape."""
        texts = ["chair", "table", "sofa"]
        similarities = pe_generator.get_embed_txt_similarity(sample_embedding, texts)
        assert similarities.shape == (len(texts),)

    def test_get_embed_txt_similarity_range(self, pe_generator, sample_embedding):
        """Similarity values are in valid range."""
        texts = ["chair", "table"]
        similarities = pe_generator.get_embed_txt_similarity(sample_embedding, texts)
        assert (similarities >= -1).all() and (similarities <= 1).all()
```

---

## 2. Unit Tests: Instance3D PE Methods

**Archivo**: `tests/unit/test_instance3d_pe.py`

### Storage Tests

```python
class TestInstance3DPEStorage:
    """Tests for Instance3D PE storage fields."""

    def test_pe_feature_initially_none(self):
        """New instance has pe_feature = None."""
        instance = Instance3D(id=1, ...)
        assert instance.pe_feature is None
        assert instance.pe_feature_kf is None

    def test_to_update_pe_initially_false(self):
        """New instance has to_update_pe = False."""
        instance = Instance3D(id=1, ...)
        assert instance.to_update_pe is False

    def test_add_top_kf_sets_update_flag(self, instance):
        """add_top_kf sets to_update_pe flag."""
        instance.add_top_kf(kf_id=5, score=0.9)
        assert instance.to_update_pe is True
```

### update_pe Tests

```python
class TestInstance3DUpdatePE:
    """Tests for Instance3D.update_pe() method."""

    def test_update_pe_single_embedding(self):
        """update_pe with single keyframe sets pe_feature directly."""
        instance = create_instance_with_kfs([1])
        pe_keyframes = {1: {instance.id: torch.randn(1, 768)}}

        instance.to_update_pe = True
        instance.update_pe(pe_keyframes)

        assert instance.pe_feature is not None
        assert instance.pe_feature.shape == (1, 768)
        assert instance.pe_feature_kf == 1

    def test_update_pe_l1_median_selection(self):
        """update_pe selects L1 median from multiple embeddings."""
        instance = create_instance_with_kfs([1, 2, 3])

        # Create embeddings where middle one is the median
        emb_a = torch.tensor([[1.0, 0.0, 0.0]])  # Outlier
        emb_b = torch.tensor([[0.5, 0.5, 0.0]])  # Should be selected (median)
        emb_c = torch.tensor([[0.0, 1.0, 0.0]])  # Outlier

        pe_keyframes = {
            1: {instance.id: emb_a},
            2: {instance.id: emb_b},
            3: {instance.id: emb_c},
        }

        instance.to_update_pe = True
        instance.update_pe(pe_keyframes)

        # Should select emb_b (kf=2) as it has minimum L1 distance to others
        assert instance.pe_feature_kf == 2
        assert torch.allclose(instance.pe_feature, emb_b)

    def test_update_pe_respects_n_top_kf(self):
        """update_pe uses only top_kf embeddings when n_top_kf > 0."""
        instance = create_instance_with_kfs([1, 2, 3, 4, 5])
        instance.n_top_kf = 2
        instance.top_kf = [5, 4]  # Only these should be used

        pe_keyframes = {i: {instance.id: torch.randn(1, 768)} for i in range(1, 6)}

        instance.to_update_pe = True
        instance.update_pe(pe_keyframes)

        # pe_feature_kf should be from top_kf (4 or 5)
        assert instance.pe_feature_kf in [4, 5]

    def test_update_pe_uses_all_kfs_when_n_top_kf_zero(self):
        """update_pe uses all kfs_ids when n_top_kf = 0."""
        instance = create_instance_with_kfs([1, 2, 3])
        instance.n_top_kf = 0

        pe_keyframes = {i: {instance.id: torch.randn(1, 768)} for i in [1, 2, 3]}

        instance.to_update_pe = True
        instance.update_pe(pe_keyframes)

        # pe_feature_kf can be any of the kfs
        assert instance.pe_feature_kf in [1, 2, 3]

    def test_update_pe_clears_flag(self):
        """update_pe sets to_update_pe = False."""
        instance = create_instance_with_kfs([1])
        pe_keyframes = {1: {instance.id: torch.randn(1, 768)}}

        instance.to_update_pe = True
        instance.update_pe(pe_keyframes)

        assert instance.to_update_pe is False

    def test_update_pe_skips_when_flag_false(self):
        """update_pe does nothing when to_update_pe is False."""
        instance = create_instance_with_kfs([1])
        pe_keyframes = {1: {instance.id: torch.randn(1, 768)}}

        instance.to_update_pe = False
        instance.pe_feature = None
        instance.update_pe(pe_keyframes)

        assert instance.pe_feature is None  # Unchanged

    def test_update_pe_force_update(self):
        """force_update=True recomputes even if to_update_pe is False."""
        instance = create_instance_with_kfs([1])
        old_feature = torch.randn(1, 768)
        new_feature = torch.randn(1, 768)

        instance.pe_feature = old_feature
        instance.to_update_pe = False

        pe_keyframes = {1: {instance.id: new_feature}}
        instance.update_pe(pe_keyframes, force_update=True)

        assert torch.allclose(instance.pe_feature, new_feature)
```

### Export/Restore Tests

```python
class TestInstance3DPEPersistence:
    """Tests for PE export/restore."""

    def test_export_includes_pe_fields(self):
        """export() includes pe_feature and pe_feature_kf."""
        instance = create_instance_with_pe()
        exported = instance.export()

        assert f"ins3d_{instance.id}_pe_feature" in exported
        assert f"ins3d_{instance.id}_pe_feature_kf" in exported

    def test_export_pe_none_handled(self):
        """export() handles None pe_feature."""
        instance = Instance3D(id=1, ...)
        exported = instance.export()

        assert exported[f"ins3d_{instance.id}_pe_feature"] is None

    def test_restore_pe_fields(self):
        """restore() correctly loads pe_feature and pe_feature_kf."""
        original = create_instance_with_pe()
        exported = original.export()

        restored = Instance3D(id=original.id, ...)
        restored.restore(exported)

        assert torch.allclose(restored.pe_feature, original.pe_feature)
        assert restored.pe_feature_kf == original.pe_feature_kf

    def test_restore_sets_update_flag_when_missing(self):
        """restore() sets to_update_pe if pe_feature missing from saved data."""
        instance = Instance3D(id=1, ...)
        # Saved data without PE fields (old format)
        old_data = {"ins3d_1_clip_feature": torch.randn(1, 512)}

        instance.restore(old_data)

        assert instance.to_update_pe is True  # Needs recomputation
```

---

## 3. Unit Tests: Fusion Strategy con PE

**Archivo**: `tests/unit/test_fusion_pe.py`

### Strategy Creation Tests

```python
class TestPEFusionStrategyCreation:
    """Tests for creating fusion strategy with PE."""

    def test_create_fusion_strategy_pe(self):
        """Factory creates SemanticGeometricFusion for fusion_method='pe'."""
        config = {"fusion_method": "pe", "th_centroid": 1.5, "th_cossim": 0.81}
        strategy = create_fusion_strategy(config)

        assert isinstance(strategy, SemanticGeometricFusion)
        assert strategy.feature_attr == "pe_feature"

    def test_strategy_uses_correct_thresholds(self):
        """Strategy uses thresholds from config."""
        config = {"fusion_method": "pe", "th_centroid": 2.0, "th_cossim": 0.85}
        strategy = create_fusion_strategy(config)

        assert strategy.th_centroid == 2.0
        assert strategy.th_cossim == 0.85
```

### Fusion Decision Tests

```python
class TestPEFusionDecisions:
    """Tests for PE-based fusion decisions."""

    def test_same_instance_high_similarity_fuses(self, pe_strategy):
        """Instances with similar PE features and geometry are fused."""
        inst1, inst2 = create_similar_instances_pe(cos_sim=0.95, overlap=0.6)

        result = pe_strategy.same_instance(inst1, inst2)

        assert result is True

    def test_same_instance_low_similarity_no_fuse(self, pe_strategy):
        """Instances with different PE features are not fused."""
        inst1, inst2 = create_different_instances_pe(cos_sim=0.3, overlap=0.6)

        result = pe_strategy.same_instance(inst1, inst2)

        assert result is False

    def test_same_instance_high_similarity_low_overlap_no_fuse(self, pe_strategy):
        """High PE similarity but no geometric overlap → no fuse."""
        inst1, inst2 = create_similar_instances_pe(cos_sim=0.95, overlap=0.05)

        result = pe_strategy.same_instance(inst1, inst2)

        assert result is False

    def test_same_instance_geometric_only_high_overlap(self, pe_strategy):
        """Very high geometric overlap fuses even with lower similarity."""
        inst1, inst2 = create_instances_pe(cos_sim=0.7, p_dist=0.6)

        result = pe_strategy.same_instance(inst1, inst2)

        # p_dist > 0.5 → fuses regardless of cos_sim
        assert result is True

    def test_same_instance_pe_feature_none_raises(self, pe_strategy):
        """Raises or handles gracefully when pe_feature is None."""
        inst1 = create_instance_with_pe()
        inst2 = Instance3D(id=2, ...)  # No pe_feature

        with pytest.raises((AttributeError, TypeError)):
            pe_strategy.same_instance(inst1, inst2)
```

---

## 4. Unit Tests: Config Validation

**Archivo**: `tests/unit/test_config_validation.py`

```python
class TestConfigValidation:
    """Tests for configuration validation."""

    def test_fusion_method_pe_without_generator_raises(self, device):
        """fusion_method='pe' without PE config raises ValueError."""
        config = {"fusion_method": "pe"}  # No "pe" key

        with pytest.raises(ValueError, match="requires PE generator"):
            OVO(config, device)

    def test_fusion_method_pe_with_generator_succeeds(self, device, pe_config):
        """fusion_method='pe' with PE config initializes correctly."""
        config = {"fusion_method": "pe", "pe": pe_config}

        ovo = OVO(config, device)

        assert ovo.pe_generator is not None
        assert ovo.fusion_method.lower() == "pe"

    def test_fusion_method_clip_without_pe_succeeds(self, device):
        """fusion_method='clip' works without PE config."""
        config = {"fusion_method": "clip"}

        ovo = OVO(config, device)

        assert ovo.fusion_method.lower() == "clip"
        assert ovo.pe_generator is None  # Not needed

    def test_fusion_method_geometric_always_valid(self, device):
        """fusion_method='geometric' works with minimal config."""
        config = {"fusion_method": "geometric"}

        ovo = OVO(config, device)

        assert ovo.fusion_method.lower() == "geometric"

    def test_invalid_fusion_method_raises(self, device):
        """Unknown fusion_method raises ValueError."""
        config = {"fusion_method": "unknown_method"}

        with pytest.raises(ValueError):
            OVO(config, device)
```

---

## 5. Integration Tests: Workflow Completo

**Archivo**: `tests/integration/test_pe_workflow.py`

### Descriptor Flow Tests

```python
class TestPEDescriptorFlow:
    """Integration tests for PE descriptor flow."""

    def test_pe_extraction_stores_in_keyframes(self, ovo_pe, sample_frame):
        """PE descriptors stored in keyframes after processing."""
        kf_id = ovo_pe.process_frame(sample_frame)

        assert kf_id in ovo_pe.keyframes["ins_pe_descriptors"]
        assert len(ovo_pe.keyframes["ins_pe_descriptors"][kf_id]) > 0

    def test_pe_flows_to_instance3d(self, ovo_pe, sample_frames):
        """PE descriptors flow from keyframes to Instance3D objects."""
        for frame in sample_frames:
            ovo_pe.process_frame(frame)

        ovo_pe.update_objects_pe()

        for obj in ovo_pe.objects.values():
            assert obj.pe_feature is not None
            assert obj.pe_feature.shape[1] == ovo_pe.pe_generator.get_pe_dim
```

### Selective Computation Tests

```python
class TestSelectiveComputation:
    """Tests for selective descriptor computation."""

    def test_clip_always_computed_with_pe_fusion(self, ovo_pe, sample_frame):
        """CLIP descriptors computed even when fusion_method='pe'."""
        kf_id = ovo_pe.process_frame(sample_frame)

        # CLIP should still be computed
        assert kf_id in ovo_pe.keyframes["ins_clip_descriptors"]
        assert len(ovo_pe.keyframes["ins_clip_descriptors"][kf_id]) > 0

    def test_pe_not_computed_when_clip_fusion(self, ovo_clip, sample_frame):
        """PE NOT computed when fusion_method='clip'."""
        kf_id = ovo_clip.process_frame(sample_frame)

        # PE should NOT be computed
        assert kf_id not in ovo_clip.keyframes.get("ins_pe_descriptors", {})

    def test_clip_and_pe_both_computed_when_pe_fusion(self, ovo_pe, sample_frame):
        """Both CLIP and PE computed when fusion_method='pe'."""
        kf_id = ovo_pe.process_frame(sample_frame)

        assert kf_id in ovo_pe.keyframes["ins_clip_descriptors"]
        assert kf_id in ovo_pe.keyframes["ins_pe_descriptors"]
```

### Fusion Decision Tests

```python
class TestPEFusionIntegration:
    """Tests for PE-based fusion in map optimization."""

    def test_similar_pe_instances_merge(self, ovo_pe, overlapping_detections):
        """Instances with similar PE features merge during optimization."""
        # Setup: Create two instances with overlapping geometry
        # and assign similar PE features
        ...

        ovo_pe.update_map()

        # Should have merged into one instance
        assert len(ovo_pe.objects) == 1

    def test_different_pe_instances_no_merge(self, ovo_pe, overlapping_detections):
        """Instances with different PE features don't merge."""
        # Setup: Create two instances with overlapping geometry
        # but very different PE features
        ...

        ovo_pe.update_map()

        # Should remain as two instances
        assert len(ovo_pe.objects) == 2
```

### Persistence Tests

```python
class TestPEPersistence:
    """Tests for PE descriptor persistence."""

    def test_pe_survives_capture_restore(self, ovo_pe, populated_map):
        """PE descriptors persist through capture_dict/restore_dict."""
        original_features = {
            obj_id: obj.pe_feature.clone()
            for obj_id, obj in ovo_pe.objects.items()
        }

        saved = ovo_pe.capture_dict()
        ovo_pe.restore_dict(saved)

        for obj_id, obj in ovo_pe.objects.items():
            assert torch.allclose(obj.pe_feature, original_features[obj_id])

    def test_pe_keyframes_persist(self, ovo_pe, populated_map):
        """Keyframe PE descriptors persist through save/load."""
        original_kf_pe = {
            kf_id: dict(descriptors)
            for kf_id, descriptors in ovo_pe.keyframes["ins_pe_descriptors"].items()
        }

        saved = ovo_pe.capture_dict()
        ovo_pe.restore_dict(saved)

        for kf_id, descriptors in ovo_pe.keyframes["ins_pe_descriptors"].items():
            for ins_id, pe in descriptors.items():
                assert torch.allclose(pe, original_kf_pe[kf_id][ins_id])
```

### Map Optimization Tests

```python
class TestPEMapOptimization:
    """Tests for PE during map optimization operations."""

    def test_pe_transfer_on_merge(self, ovo_pe, mergeable_instances):
        """PE descriptors transferred during instance merge."""
        source_id, target_id = get_merge_ids(mergeable_instances)
        source_pe_kfs = get_pe_keyframes_for_instance(ovo_pe, source_id)

        ovo_pe.update_map()  # Triggers merge

        # Source PE descriptors should be transferred to target
        for kf_id in source_pe_kfs:
            assert target_id in ovo_pe.keyframes["ins_pe_descriptors"][kf_id]

    def test_pe_cleanup_on_keyframe_removal(self, ovo_pe, populated_map):
        """PE descriptors cleaned up when keyframe removed."""
        kf_to_remove = list(ovo_pe.keyframes["ins_pe_descriptors"].keys())[0]

        ovo_pe.remove_keyframe(kf_to_remove)

        assert kf_to_remove not in ovo_pe.keyframes["ins_pe_descriptors"]
```

---

## 6. Fixtures (conftest.py)

**Archivo**: `tests/conftest.py`

```python
import pytest
import torch

@pytest.fixture
def device():
    """Test device (CPU for CI, CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def pe_config():
    """Minimal PE configuration for tests."""
    return {
        "model_card": "PE-Core-L14-336",
        "mask_res": 336,
        "use_half": False,
    }

@pytest.fixture
def pe_generator(pe_config, device):
    """Real PEGenerator instance for integration tests."""
    from ovo.entities.pe_generator import PEGenerator
    return PEGenerator(pe_config, device)

@pytest.fixture
def mock_pe_generator():
    """Mock PEGenerator for unit tests (no model loading)."""
    from unittest.mock import MagicMock
    mock = MagicMock()
    mock.get_pe_dim = 768
    mock.extract_pe.return_value = torch.randn(2, 768)
    return mock

@pytest.fixture
def sample_image_hwc():
    """Sample image in (H, W, 3) format, [0, 255] range."""
    return torch.randint(0, 256, (480, 640, 3), dtype=torch.uint8)

@pytest.fixture
def sample_image_tensor():
    """Sample image in (3, H, W) format, [0, 1] range."""
    return torch.rand(3, 336, 336)

@pytest.fixture
def sample_masks():
    """Sample binary masks (N, H, W)."""
    masks = torch.zeros(3, 480, 640, dtype=torch.bool)
    masks[0, 100:200, 100:200] = True
    masks[1, 200:300, 300:400] = True
    masks[2, 300:400, 400:500] = True
    return masks

@pytest.fixture
def ovo_pe(pe_config, device):
    """OVO instance configured for PE fusion."""
    from ovo.entities.ovo import OVO
    config = {
        "fusion_method": "pe",
        "pe": pe_config,
        "th_centroid": 1.5,
        "th_cossim": 0.81,
        "th_points": 0.1,
    }
    return OVO(config, device)

@pytest.fixture
def ovo_clip(device):
    """OVO instance configured for CLIP fusion (no PE)."""
    from ovo.entities.ovo import OVO
    config = {
        "fusion_method": "clip",
        "th_centroid": 1.5,
        "th_cossim": 0.81,
    }
    return OVO(config, device)

@pytest.fixture
def pe_strategy():
    """SemanticGeometricFusion strategy with PE feature."""
    from ovo.entities.fusion import create_fusion_strategy
    config = {"fusion_method": "pe", "th_centroid": 1.5, "th_cossim": 0.81}
    return create_fusion_strategy(config)


# Helper functions for creating test instances
def create_instance_with_kfs(kf_ids):
    """Create Instance3D with specified keyframe IDs."""
    from ovo.entities.instance3d import Instance3D
    instance = Instance3D(id=1, ...)
    instance.kfs_ids = kf_ids
    return instance

def create_instance_with_pe(pe_dim=768):
    """Create Instance3D with PE feature set."""
    instance = create_instance_with_kfs([1])
    instance.pe_feature = torch.randn(1, pe_dim)
    instance.pe_feature_kf = 1
    return instance

def create_similar_instances_pe(cos_sim=0.95, overlap=0.5):
    """Create two instances with specified PE similarity and geometric overlap."""
    inst1 = create_instance_with_pe()
    inst2 = create_instance_with_pe()

    # Set PE features to achieve desired cosine similarity
    inst1.pe_feature = torch.randn(1, 768)
    inst1.pe_feature = inst1.pe_feature / inst1.pe_feature.norm()

    # Create inst2 feature with desired similarity
    noise = torch.randn(1, 768)
    noise = noise / noise.norm()
    inst2.pe_feature = cos_sim * inst1.pe_feature + (1 - cos_sim) * noise
    inst2.pe_feature = inst2.pe_feature / inst2.pe_feature.norm()

    # Set geometry for desired overlap
    # ... (depends on Instance3D geometry representation)

    return inst1, inst2
```

---

## 7. Orden de Implementación de Tests

1. **Fase 1**: `conftest.py` - Fixtures base
2. **Fase 2**: `test_pe_generator.py` - Validar generator aislado
3. **Fase 3**: `test_config_validation.py` - Validar configs
4. **Fase 4**: `test_instance3d_pe.py` - Validar storage/update
5. **Fase 5**: `test_fusion_pe.py` - Validar decisiones fusion
6. **Fase 6**: `test_pe_workflow.py` - E2E integration
