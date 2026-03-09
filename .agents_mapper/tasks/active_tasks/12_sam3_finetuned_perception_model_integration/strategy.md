# Strategy: SAM3 Finetuned Perception Model Integration

## Overview

Integrate SAM3's finetuned Vision Transformer (ViT) into OVO as a new semantic feature generator, following the existing pattern established by `PEGenerator` and `CLIPGenerator`.

---

## Architecture Decision

**Approach:** Create a configurable `SAM3Generator` class that loads only the required components based on user configuration.

```
┌─────────────────────────────────────────────────────┐
│                  SAM3Generator                       │
│  ┌─────────────────────────────────────────────────┐│
│  │  components="vit_only"  → Load ViT (1024-dim)   ││
│  │  components="vit_neck"  → Load ViT + Neck (256) ││
│  │  components="full"      → Load all + Text Enc   ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

This allows:
- Memory-efficient loading (only load what's needed)
- Future expansion (add segmentation later)
- Flexible feature dimensions (1024 or 256)

---

## Implementation Phases

### Phase 1: Create SAM3Generator Class (TDD Red → Green)

**File:** `ovo/entities/sam3_generator.py`

**Tasks:**
1. Create `SAM3Generator` class following `PEGenerator` pattern
2. Implement configurable component loading (`vit_only`, `vit_neck`, `full`)
3. Implement checkpoint loading (local and HuggingFace)
4. Implement `encode_image()` method
5. Implement `extract_sam3()` method (like `extract_pe()`)
6. Implement device management (`to()`, `cpu()`, `cuda()`)

**Key Methods:**
```python
class SAM3Generator:
    def __init__(self, config: Dict, device: str = "cuda"):
        # Load components based on config["components"]

    def encode_image(self, input: torch.Tensor) -> torch.Tensor:
        # Returns (B, embed_dim) features

    def extract_sam3(self, image: torch.Tensor, binary_maps: torch.Tensor) -> torch.Tensor:
        # Extract features for each mask

    def to(self, device: str) -> None:
    def cpu(self) -> None:
    def cuda(self) -> None:
```

---

### Phase 2: Integrate with Fusion Strategy

**Files to modify:**
- `ovo/entities/fusion.py` - Add "sam3" to factory
- `ovo/entities/ovo.py` - Initialize SAM3Generator

**Tasks:**
1. Add `"sam3"` case to `create_fusion_strategy()` factory
2. Map `"sam3"` → `feature_attr = "sam3_feature"`
3. Add `sam3_generator` initialization in `OVO.__init__()`
4. Add `update_objects_sam3()` method (similar to `update_objects_pe()`)
5. Update `keyframes` dict to include `ins_sam3_descriptors`

**Factory Update:**
```python
def create_fusion_strategy(config: Dict) -> FusionStrategy:
    method = config.get("fusion_method", "clip")

    if method == "sam3":
        return SemanticGeometricFusion(config, feature_attr="sam3_feature")
    # ... existing cases
```

---

### Phase 3: Add Instance3D Support

**File:** `ovo/entities/instance.py` (or wherever Instance3D is defined)

**Tasks:**
1. Add `sam3_feature` attribute to Instance3D
2. Initialize as `None` or empty tensor
3. Add update logic in instance merging

---

### Phase 4: Configuration Integration

**File:** `data/configs/` (example configs)

**Tasks:**
1. Add `sam3` section to config schema
2. Document configuration options
3. Create example config for SAM3 fusion

**Example Config:**
```yaml
sam3:
  checkpoint_path: "path/to/sam3_finetuned.pt"  # or null for HuggingFace
  components: "vit_only"  # "vit_only" | "vit_neck" | "full"
  load_from_hf: true
  use_half: false
  image_size: 1008  # optional override

fusion_method: "sam3"
th_centroid: 1.5
th_cossim: 0.81
th_points: 0.1
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `ovo/entities/sam3_generator.py` | **CREATE** | New SAM3Generator class |
| `ovo/entities/fusion.py` | MODIFY | Add "sam3" to factory |
| `ovo/entities/ovo.py` | MODIFY | Add sam3_generator init + update method |
| `ovo/entities/instance3d.py` | MODIFY | Add sam3_feature, sam3_feature_kf, to_update_sam3, update_sam3() |
| `tests/unit/test_sam3_generator.py` | **CREATE** | SAM3Generator unit tests (25+ tests) |
| `tests/unit/test_instance3d_sam3.py` | **CREATE** | Instance3D SAM3 integration tests (14 tests) |
| `tests/unit/test_fusion_sam3.py` | **CREATE** | Fusion strategy SAM3 tests (5 tests) |
| `tests/unit/test_ovo_sam3.py` | **CREATE** | OVO SAM3 integration tests (9 tests) |
| `tests/fixtures/fixtures_sam3.py` | **CREATE** | Test fixtures |
| `tests/conftest.py` | MODIFY | Import SAM3 fixtures |

---

## Implementation Order (TDD Workflow)

### Step 1: Create All Test Files (Red Phase)

```bash
# Create test files
tests/fixtures/fixtures_sam3.py       # Fixtures for all SAM3 tests
tests/unit/test_sam3_generator.py     # SAM3Generator class tests
tests/unit/test_instance3d_sam3.py    # Instance3D integration tests
tests/unit/test_fusion_sam3.py        # Fusion strategy tests
tests/unit/test_ovo_sam3.py           # OVO integration tests

# Update conftest.py to import fixtures
tests/conftest.py
```

```bash
# Run all tests → All fail
pytest tests/unit/test_sam3*.py tests/unit/test_instance3d_sam3.py tests/unit/test_fusion_sam3.py tests/unit/test_ovo_sam3.py -v
```

### Step 2: Implement SAM3Generator (Green Phase - Part 1)

**Target tests:** `test_sam3_generator.py`

```bash
# Create SAM3Generator class
ovo/entities/sam3_generator.py

# Implement in order:
1. __init__ with component loading (vit_only, vit_neck, full)
2. Checkpoint loading (_load_checkpoint)
3. encode_image() method
4. extract_sam3() method
5. Device management (to, cpu, cuda)

# Run tests → SAM3Generator tests pass
pytest tests/unit/test_sam3_generator.py -v
```

### Step 3: Implement Instance3D Changes (Green Phase - Part 2)

**Target tests:** `test_instance3d_sam3.py`

```bash
# Modify Instance3D class
ovo/entities/instance3d.py

# Add:
1. sam3_feature = None (attribute)
2. sam3_feature_kf = None (attribute)
3. to_update_sam3 = False (flag)
4. update_sam3() method (following update_pe pattern)
5. Update export() to include sam3_feature
6. Update restore() to load sam3_feature
7. Update add_top_kf() to set to_update_sam3 = True

# Run tests → Instance3D SAM3 tests pass
pytest tests/unit/test_instance3d_sam3.py -v
```

### Step 4: Implement Fusion Strategy Changes (Green Phase - Part 3)

**Target tests:** `test_fusion_sam3.py`

```bash
# Modify fusion factory
ovo/entities/fusion.py

# Add:
1. "sam3" case in create_fusion_strategy()
2. Map to feature_attr="sam3_feature"

# Run tests → Fusion tests pass
pytest tests/unit/test_fusion_sam3.py -v
```

### Step 5: Implement OVO Integration (Green Phase - Part 4)

**Target tests:** `test_ovo_sam3.py`

```bash
# Modify OVO class
ovo/entities/ovo.py

# Add:
1. Import SAM3Generator
2. self.sam3_generator initialization in __init__
3. "ins_sam3_descriptors" key in self.keyframes
4. update_objects_sam3() method
5. SAM3 extraction call in semantic processing

# Run tests → OVO tests pass
pytest tests/unit/test_ovo_sam3.py -v
```

### Step 6: Full Integration Test (Green Phase - Complete)

```bash
# Run all SAM3-related tests
pytest tests/unit/test_sam3*.py tests/unit/test_instance3d_sam3.py tests/unit/test_fusion_sam3.py tests/unit/test_ovo_sam3.py -v

# Run full test suite to ensure no regressions
pytest tests/ -v

# Run with coverage
pytest tests/unit/ --cov=ovo.entities -v
```

### Step 7: Refactor (Keep Tests Green)

```bash
# Clean up code while keeping all tests passing
# Add config examples
# Update documentation if needed
```

---

## Checkpoint Weight Extraction Logic

The SAM3 checkpoint stores weights with `detector.` prefix. Extraction logic:

```python
def _load_checkpoint(self, checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_state = ckpt.get("model", ckpt)

    if self.components == "vit_only":
        # Extract only ViT trunk weights
        prefix = "detector.backbone.vision_backbone.trunk."
        vit_weights = {
            k.replace(prefix, ""): v
            for k, v in model_state.items()
            if k.startswith(prefix)
        }
        self.vit.load_state_dict(vit_weights, strict=False)

    elif self.components == "vit_neck":
        # Extract ViT + Neck weights
        vit_prefix = "detector.backbone.vision_backbone.trunk."
        neck_prefix = "detector.backbone.vision_backbone."
        # ... similar extraction

    elif self.components == "full":
        # Use SAM3's build function directly
        pass
```

---

## Output Dimensions

| Mode | Output Dimension | Notes |
|------|-----------------|-------|
| `vit_only` | 1024 | Raw ViT output (global average pooled) |
| `vit_neck` | 256 | After neck projection |
| `full` | 256 | Same as vit_neck (text encoder separate) |

**Fusion compatibility note:** Current OVO uses 512-dim features. The fusion logic uses cosine similarity, which is dimension-agnostic. No changes needed to fusion math, only the `feature_attr` name.

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Large model memory | Configurable components - load only what's needed |
| Checkpoint format changes | Robust key mapping with fallbacks |
| Integration breaking existing flow | TDD ensures existing tests pass |
| Performance regression | Benchmark before/after integration |

---

## Success Criteria

### Test Suites Must Pass

| Test File | Tests | Target |
|-----------|-------|--------|
| `test_sam3_generator.py` | 25+ | SAM3Generator class |
| `test_instance3d_sam3.py` | 14 | Instance3D integration |
| `test_fusion_sam3.py` | 5 | Fusion strategy factory |
| `test_ovo_sam3.py` | 9 | OVO integration |

### Functional Requirements

1. All ~53 SAM3-related tests pass
2. Existing tests (fusion, instance3d) still pass (no regressions)
3. Can run OVO with `fusion_method: "sam3"` in config
4. SAM3 features correctly extracted and stored per keyframe
5. Instance3D correctly aggregates sam3_feature using median selection
6. Fusion strategy correctly compares instances using sam3_feature
7. Memory usage matches expected for configured components mode

### Commands to Verify

```bash
# All SAM3 tests pass
pytest tests/unit/test_sam3*.py tests/unit/test_instance3d_sam3.py tests/unit/test_fusion_sam3.py tests/unit/test_ovo_sam3.py -v

# No regressions in existing tests
pytest tests/ -v

# Coverage report
pytest tests/unit/ --cov=ovo.entities --cov-report=term-missing
```
