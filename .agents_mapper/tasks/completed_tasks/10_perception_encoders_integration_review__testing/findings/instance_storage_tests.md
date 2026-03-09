# Research: Instance Storage & Tests Analysis for PE Integration

**Task**: Perception Encoders Integration Review & Testing - Instance Storage & Tests Analysis
**Files Analyzed**:
- `/home/padidavid/repos/OVO/ovo/entities/instance3d.py`
- `/home/padidavid/repos/OVO/ovo/utils/instance_utils.py`
- `/home/padidavid/repos/OVO/tests/unit/test_fusion_selector.py`

**Date**: 2025-12-17

## Summary
Instance3D already has full PE descriptor storage support with parallel structure to CLIP. The PE descriptors use identical median-based fusion logic (`update_pe`) and separate update flags (`to_update_pe`). Tests cover the fusion strategy pattern but lack specific PE descriptor tests. The data model is complete for PE storage and export/restore, but test coverage for PE-specific behavior is missing.

---

## File: /home/padidavid/repos/OVO/ovo/entities/instance3d.py

### Relevant Components

#### PE Descriptor Storage Fields (`instance3d.py:33-34`)
- **Type**: Instance attributes
- **Purpose**: Store PE descriptors and track which keyframe contributed the final descriptor
- **Relevance**: Core data model for PE integration
- **Key Details**:
  - `self.pe_feature = None` - Stores the fused PE descriptor (tensor)
  - `self.pe_feature_kf = None` - Stores the keyframe index that contributed the PE descriptor
  - Initialized to None, indicating no descriptor yet computed

#### PE Update Flag (`instance3d.py:41`)
- **Type**: Boolean flag
- **Purpose**: Track whether PE descriptor needs recomputation
- **Relevance**: Critical for efficient PE descriptor updates
- **Key Details**:
  - `self.to_update_pe = False` - Separate from `self.to_update` (CLIP flag)
  - Set to True when new observations are added via `add_top_kf` (lines 87, 100, 105)
  - Reset to False after `update_pe` completes (line 194)

#### CLIP Descriptor Storage (Comparison) (`instance3d.py:31-32`)
- **Type**: Instance attributes
- **Purpose**: Store CLIP descriptors using same pattern as PE
- **Relevance**: Reference implementation for descriptor storage pattern
- **Key Details**:
  - `self.clip_feature = None` - Fused CLIP descriptor
  - `self.clip_feature_kf = None` - Contributing keyframe index
  - Identical structure to PE fields, providing parallel implementation

#### DINO Descriptor Storage (`instance3d.py:35-36`)
- **Type**: Instance attributes
- **Purpose**: Store DINO descriptors (third descriptor type supported)
- **Relevance**: Shows extensibility of descriptor model
- **Key Details**:
  - `self.dino_feature = None` - Fused DINO descriptor
  - `self.dino_feature_kf = None` - Contributing keyframe index
  - No update method implemented (missing `update_dino`)

#### `update_pe()` Method (`instance3d.py:162-194`)
- **Type**: Instance method
- **Purpose**: Compute fused PE descriptor from keyframe observations using L1 median selection
- **Relevance**: Core PE descriptor fusion logic
- **Key Details**:
  - **Signature**: `update_pe(keyframes_pe: Dict[int, Dict[int, torch.Tensor]], force_update: bool = False)`
  - **Input**: `keyframes_pe` maps keyframe_id → {instance_id → PE_tensor}
  - **Logic**: Identical to `update_clip` (lines 128-160)
    1. Collects PE embeddings from top_kf (if n_top_kf > 0) or all kfs_ids
    2. Stacks embeddings and computes pairwise L1 distances
    3. Selects embedding with minimum total L1 distance (geometric median)
  - **Updates**: `self.pe_feature`, `self.pe_feature_kf`, `self.to_update_pe = False`
  - **Early Exit**: Returns without update if no PE embeddings found

#### `update_clip()` Method (Comparison) (`instance3d.py:128-160`)
- **Type**: Instance method
- **Purpose**: Compute fused CLIP descriptor (reference implementation)
- **Relevance**: Shows expected pattern for descriptor fusion
- **Key Details**:
  - **Signature**: `update_clip(keyframes_clips: Dict[int, Dict[int, torch.Tensor]], force_update: bool = False)`
  - **Logic**: L1 median selection (minimizes sum of L1 distances)
  - **Pattern**: Identical to PE update except for field names

#### `add_top_kf()` Method (`instance3d.py:73-105`)
- **Type**: Instance method
- **Purpose**: Maintain heap of best keyframe observations, trigger update flags
- **Relevance**: Triggers both CLIP and PE updates when instance is observed
- **Key Details**:
  - Sets **both** `self.to_update = True` and `self.to_update_pe = True` (lines 86-87, 99-100, 104-105)
  - Ensures parallel tracking for all descriptor types
  - Uses heap structure for efficient top-N keyframe management

#### `export()` Method (`instance3d.py:196-218`)
- **Type**: Serialization method
- **Purpose**: Export instance state to dictionary for saving
- **Relevance**: Shows how PE descriptors are persisted
- **Key Details**:
  - Exports PE fields: `ins3d_{id}_pe_feature`, `ins3d_{id}_pe_feature_kf` (lines 207-208)
  - Exports CLIP fields: `ins3d_{id}_clip_feature`, `ins3d_{id}_clip_feature_kf` (lines 205-206)
  - Debug mode adds `kfs_ids`, `points_ids`, `top_kf` (lines 213-216)
  - Uses f-string naming convention with instance ID

#### `restore()` Method (`instance3d.py:220-237`)
- **Type**: Deserialization method
- **Purpose**: Restore instance state from saved dictionary
- **Relevance**: Shows PE descriptor restoration and update flag initialization
- **Key Details**:
  - Restores PE fields with `obj_dict.get()` (defaults to None) (lines 230-231)
  - Sets `self.to_update_pe = self.pe_feature is None` (line 232)
  - Sets `self.to_update = self.clip_feature is None` (line 229)
  - Ensures instances without descriptors are flagged for update

#### `old_restore()` Method (`instance3d.py:239-252`)
- **Type**: Legacy deserialization method
- **Purpose**: Restore from old format (uses "default_" prefix instead of "ins3d_")
- **Relevance**: Shows backward compatibility concern
- **Key Details**:
  - **Missing PE support**: Only restores CLIP features (line 246-247)
  - Uses `default_{id}_clip_feature` naming (different from current format)
  - **Gap**: Old save files cannot restore PE descriptors

### Patterns & Observations
1. **Parallel Implementation**: PE and CLIP descriptors use identical patterns:
   - Storage fields: `{type}_feature`, `{type}_feature_kf`
   - Update flags: `to_update`, `to_update_pe`
   - Update methods: `update_clip()`, `update_pe()` with identical L1 median logic

2. **Fusion Algorithm**: L1 median selection (geometric median approximation)
   - Computes pairwise L1 distances: `torch.abs(embeds - embeds.permute(1,0,2)).sum((1,2))`
   - Selects embedding with minimum total distance to all others
   - Robust to outliers, favors "central" descriptor in feature space

3. **Top-K Keyframe Selection**:
   - If `n_top_kf > 0`: Uses heap to select best N keyframes by mask area
   - If `n_top_kf <= 0`: Uses all keyframes in `kfs_ids`
   - Same logic for both CLIP and PE updates

4. **Update Triggering**: Both flags set simultaneously when new observations arrive
   - Ensures CLIP and PE stay synchronized
   - Separate flags allow independent force updates

5. **DINO Partial Support**: Storage fields exist but no `update_dino()` method implemented

### Gaps / Not Found
1. **No PE-specific configuration**: `n_top_kf` is shared across all descriptor types (could be different for PE vs CLIP)
2. **No cross-descriptor validation**: No checks to ensure PE and CLIP are computed from same keyframes
3. **Old restore format lacks PE**: `old_restore()` cannot load PE descriptors from legacy saves
4. **Missing DINO update**: `dino_feature` fields exist but no update method

---

## File: /home/padidavid/repos/OVO/ovo/utils/instance_utils.py

### Relevant Components

#### File Purpose (`instance_utils.py:1-6`)
- **Type**: Module documentation
- **Purpose**: Pure functions for geometric computations, fusion logic migrating to `ovo.entities.fusion`
- **Relevance**: Shows architectural migration away from this file for fusion decisions
- **Key Details**:
  - "Pure Mathematical/Geometric Functions"
  - "The fusion decision logic is being migrated to ovo.entities.fusion"
  - This file becoming geometry-only utilities

#### `same_instance()` Function (`instance_utils.py:52-75`)
- **Type**: Legacy function
- **Purpose**: Determine if two instances should be fused (uses CLIP descriptors only)
- **Relevance**: Shows existing fusion logic does NOT support PE descriptors
- **Key Details**:
  - **Uses CLIP only**: `instance1.clip_feature[0]` (line 64)
  - **Three-stage check**:
    1. Centroid distance < `th_centroid` (line 60-62)
    2. CLIP cosine similarity > `th_cossim` (line 64-66)
    3. Point cloud overlap > 50% OR (cossim > 0.9 AND overlap > 20%) (line 74-75)
  - **Marked deprecated**: "Legacy function - kept for backward compatibility during migration" (line 53-55)
  - **No PE support**: Would need PE-specific version for PE-based fusion

#### `fuse_instances()` Function (`instance_utils.py:78-98`)
- **Type**: Data structure merging function
- **Purpose**: Merge two instances' data (points, keyframes, etc.)
- **Relevance**: Descriptor-agnostic, works with PE/CLIP/DINO equally
- **Key Details**:
  - **Merges structural data**: `points_ids`, `kfs_ids`, `top_kf` (lines 92-96)
  - **Does not touch descriptors**: Descriptors remain in instance1, instance2's are discarded
  - **Updates map**: Changes all instance2 points to instance1 ID (line 97)
  - **Assumes later update**: Expects `update_clip()`/`update_pe()` to be called after fusion

#### Geometric Utility Functions (`instance_utils.py:13-49`)
- **Type**: Pure functions
- **Purpose**: Geometric computations for fusion decisions
- **Relevance**: PE-agnostic, can be used by any fusion strategy
- **Key Details**:
  - `compute_centroid_distance()`: Euclidean distance between centroids (lines 13-24)
  - `compute_pcd_overlap()`: Point cloud overlap ratio using Open3D (lines 27-49)
  - Both used by `same_instance()` for geometric checks

### Patterns & Observations
1. **Architecture Migration**: Comments indicate fusion decision logic moving to `ovo.entities.fusion`
2. **Descriptor Decoupling**: Geometric functions are descriptor-agnostic (PE/CLIP/DINO irrelevant)
3. **Fusion is Two-Phase**:
   - Phase 1: `same_instance()` decides if fusion should happen (descriptor-dependent)
   - Phase 2: `fuse_instances()` merges data structures (descriptor-agnostic)
4. **PE Not Integrated**: `same_instance()` hardcoded to CLIP, needs refactoring for PE support

### Gaps / Not Found
1. **No PE fusion logic**: `same_instance()` only uses CLIP descriptors
2. **No multi-descriptor fusion**: Cannot combine PE + CLIP decisions
3. **No PE utility functions**: No PE-specific equivalents of geometric utilities
4. **Legacy code**: Marked for migration, unclear if PE support will be added here or in new location

---

## File: /home/padidavid/repos/OVO/tests/unit/test_fusion_selector.py

### Relevant Components

#### Test File Purpose (`test_fusion_selector.py:1-8`)
- **Type**: Test suite documentation
- **Purpose**: TDD tests for Fusion Strategy Pattern infrastructure
- **Relevance**: Tests factory pattern and strategy delegation, not PE-specific
- **Key Details**:
  - Tests verify: Factory creates correct strategy, OVO delegates to strategy, strategies comply with interface
  - **Focus on architecture**, not descriptor-specific behavior

#### PE Strategy Factory Test (`test_fusion_selector.py:43-49`)
- **Type**: Factory test
- **Purpose**: Verify factory creates SemanticGeometricFusion for 'pe' config
- **Relevance**: Shows PE is recognized as a fusion method
- **Key Details**:
  - **Config**: `{"fusion_method": "pe", "th_centroid": 1.5, "th_cossim": 0.81, "th_points": 0.1}`
  - **Assertion**: `strategy.feature_attr == "pe_feature"` (line 49)
  - **Shows**: PE is first-class fusion method alongside CLIP/DINO

#### SemanticGeometricFusion Test Coverage (`test_fusion_selector.py:152-170`)
- **Type**: Configuration tests
- **Purpose**: Verify strategy stores thresholds and feature attribute correctly
- **Relevance**: Tests apply to PE strategy (feature_attr="pe_feature")
- **Key Details**:
  - `test_stores_thresholds_from_config`: Verifies `th_centroid`, `th_cossim`, `th_points` (lines 155-162)
  - `test_stores_feature_attribute`: Verifies `feature_attr` storage with "dino_feature" example (lines 164-169)
  - **Generic tests**: Work for any feature type (CLIP/DINO/PE)

#### OVO Integration Tests (`test_fusion_selector.py:192-294`)
- **Type**: Integration tests
- **Purpose**: Verify OVO creates and uses fusion strategy
- **Relevance**: Tests show PE strategy can be created but not that it works correctly
- **Key Details**:
  - `test_ovo_has_fusion_strategy_attribute`: OVO has `fusion_strategy` attribute (lines 197-209)
  - `test_ovo_creates_correct_strategy_from_config`: OVO creates correct strategy from config (lines 211-224)
  - `test_ovo_uses_strategy_for_fusion_decision`: OVO delegates to `strategy.same_instance()` (lines 239-293)
  - **Mock-heavy**: Uses mocks to avoid loading heavy models (lines 200-202, 213-214)

#### Missing PE-Specific Tests (Gap Analysis)
- **Type**: Test gap
- **Purpose**: Identify what's NOT tested for PE descriptors
- **Relevance**: Critical gaps in test coverage
- **Key Details**:
  - **No PE descriptor fusion tests**: No tests verify `update_pe()` logic works
  - **No PE feature extraction tests**: No tests verify PE embeddings are computed/stored correctly
  - **No PE same_instance tests**: No tests verify PE-based fusion decisions
  - **No PE export/restore tests**: No tests verify PE descriptors persist correctly
  - **No PE + CLIP comparison tests**: No tests verify PE and CLIP give different results
  - **No PE keyframe selection tests**: No tests verify top_kf selection works for PE

### Patterns & Observations
1. **Architecture-Focused Testing**: Tests verify the strategy pattern infrastructure, not descriptor behavior
2. **Generic Strategy Tests**: Tests work for any descriptor type (CLIP/DINO/PE)
3. **PE as First-Class Citizen**: Factory test shows PE is treated equally with CLIP/DINO
4. **Heavy Mocking**: Tests use mocks to avoid model loading, may miss integration issues
5. **TDD Approach**: Comments indicate "TDD Red phase" expected initially (line 15)

### Gaps / Not Found
1. **No PE Descriptor Tests**: Zero tests for `Instance3D.update_pe()` behavior
2. **No PE Fusion Logic Tests**: No tests verify PE-based `same_instance()` decisions
3. **No PE Data Pipeline Tests**: No tests verify PE flow from encoder → keyframe → instance
4. **No PE Serialization Tests**: No tests for PE export/restore in `Instance3D`
5. **No Cross-Descriptor Tests**: No tests comparing PE vs CLIP fusion results
6. **No PE Edge Cases**: No tests for PE with missing keyframes, empty observations, etc.

---

## Cross-File Relationships

### Instance3D ↔ instance_utils
- `instance_utils.fuse_instances()` operates on `Instance3D` objects (lines 78-98)
- `instance_utils.same_instance()` reads `Instance3D.clip_feature` (line 64)
- After `fuse_instances()`, caller must invoke `Instance3D.update_clip()`/`update_pe()` to recompute descriptors
- **Gap**: `same_instance()` cannot use PE descriptors, only CLIP

### Instance3D ↔ test_fusion_selector
- Tests verify `Instance3D` instances can be passed to fusion strategies (lines 79-113)
- Tests mock `Instance3D` attributes: `id`, `clip_feature`, `points_ids`, `kfs_ids`, `top_kf` (lines 256-267)
- **Gap**: Tests don't verify `Instance3D.pe_feature` is used by PE fusion strategy

### test_fusion_selector ↔ ovo.entities.fusion
- Tests import `FusionStrategy`, `SemanticGeometricFusion`, `GeometricOnlyFusion`, `create_fusion_strategy` (lines 16-21)
- Tests verify factory creates strategies with correct `feature_attr` (e.g., "pe_feature") (line 49)
- Tests verify OVO delegates fusion decisions to strategy (lines 239-293)
- **Gap**: No tests verify strategies actually read `instance.pe_feature` correctly

---

## Quick Reference Table

| Component | Location | Purpose |
|-----------|----------|---------|
| `pe_feature` | instance3d.py:33 | Store fused PE descriptor tensor |
| `pe_feature_kf` | instance3d.py:34 | Store keyframe index for PE descriptor |
| `to_update_pe` | instance3d.py:41 | Flag indicating PE descriptor needs update |
| `update_pe()` | instance3d.py:162-194 | Compute fused PE descriptor via L1 median |
| `add_top_kf()` | instance3d.py:73-105 | Update top keyframes, set PE/CLIP update flags |
| `export()` PE fields | instance3d.py:207-208 | Export PE descriptors for persistence |
| `restore()` PE fields | instance3d.py:230-232 | Restore PE descriptors from saved state |
| `same_instance()` | instance_utils.py:52-75 | Legacy fusion decision (CLIP only, no PE) |
| `fuse_instances()` | instance_utils.py:78-98 | Merge instance data structures (descriptor-agnostic) |
| PE factory test | test_fusion_selector.py:43-49 | Verify PE strategy creation |
| Strategy interface tests | test_fusion_selector.py:115-149 | Verify all strategies implement same_instance |
| OVO integration tests | test_fusion_selector.py:192-294 | Verify OVO uses fusion strategy |

---

## Recommendations

### 1. Test Coverage Gaps (High Priority)
Create PE-specific tests to verify:
- **`Instance3D.update_pe()` correctness**: Test L1 median selection with known PE embeddings
- **PE keyframe selection**: Test top_kf selection for PE updates (n_top_kf > 0 vs all keyframes)
- **PE export/restore**: Test PE descriptors survive save/load cycle
- **PE fusion decisions**: Test `SemanticGeometricFusion` with `feature_attr="pe_feature"` makes correct decisions
- **PE + CLIP differences**: Test that PE and CLIP give different fusion results for same instances

### 2. Legacy Code Migration (Medium Priority)
- **Update or remove `instance_utils.same_instance()`**: Either add PE support or finalize migration to `ovo.entities.fusion`
- **Update `old_restore()`**: Add PE descriptor restoration for backward compatibility with legacy saves
- **Document migration status**: Clarify which functions in `instance_utils.py` are deprecated vs stable

### 3. Data Model Enhancements (Low Priority)
- **Add cross-descriptor validation**: Verify PE and CLIP computed from same keyframe set
- **Consider per-descriptor n_top_kf**: Allow different top-K settings for PE vs CLIP (if needed)
- **Implement `update_dino()`**: Complete DINO support or remove unused fields

### 4. Documentation (Low Priority)
- **Document PE descriptor lifecycle**: Describe flow from encoder → keyframe → instance → fusion
- **Document L1 median rationale**: Explain why L1 distance used instead of L2 or other aggregation
- **Document force_update usage**: Clarify when `force_update=True` is needed vs automatic updates

---

## Files Not Analyzed (Out of Scope)

The following files might contain relevant information but were not in the permitted list:

1. **`ovo/entities/fusion.py`**: Contains `FusionStrategy`, `SemanticGeometricFusion` implementations
   - Critical for understanding PE fusion logic
   - Tests import from this file but we cannot verify implementation

2. **`ovo/entities/ovo.py`**: Main OVO class that uses fusion strategies
   - Shows how PE descriptors flow through the system
   - Contains `update_map()` which calls fusion logic

3. **`tests/fixtures/fixtures_fusion.py`**: Test fixtures for fusion tests
   - Provides `mock_instance`, `sample_points_centroid` fixtures
   - Shows how test instances are constructed

4. **PE encoder implementation**: Unknown location, not in permitted files
   - Need to understand PE embedding dimension, normalization, etc.
   - Critical for writing realistic PE descriptor tests

5. **Integration test files**: Tests that run full pipeline with real PE encoders
   - Would show if PE descriptors work end-to-end
   - Current file only has unit tests with mocks

6. **Configuration files**: Dataset configs that specify `fusion_method: "pe"`
   - Shows real-world PE usage patterns
   - Could inform test case design

---

## Data Model Summary

### Descriptor Storage Pattern (Instance3D)
```python
# Each descriptor type follows this pattern:
self.{type}_feature = None           # Fused descriptor tensor (e.g., torch.Tensor of shape [1, D])
self.{type}_feature_kf = None        # Keyframe index that contributed the descriptor (int)
self.to_update_{type} = False        # Flag indicating descriptor needs recomputation (bool)

# Supported types: clip, pe, dino (dino lacks update method)
```

### Keyframe Descriptor Storage
```python
# Keyframe-level descriptors stored in external dictionaries:
keyframes_clips: Dict[int, Dict[int, torch.Tensor]]  # kf_id → {instance_id → CLIP_tensor}
keyframes_pe: Dict[int, Dict[int, torch.Tensor]]     # kf_id → {instance_id → PE_tensor}

# Passed to update methods:
instance.update_clip(keyframes_clips)
instance.update_pe(keyframes_pe)
```

### Export/Import Format
```python
# Export produces:
{
    "ins3d_{id}_clip_feature": torch.Tensor or None,
    "ins3d_{id}_clip_feature_kf": int or None,
    "ins3d_{id}_pe_feature": torch.Tensor or None,
    "ins3d_{id}_pe_feature_kf": int or None,
    # If debug_info=True:
    "ins3d_{id}_keyframes_ids": np.ndarray,
    "ins3d_{id}_points_ids": np.ndarray,
    "ins3d_{id}_top_kfs": np.ndarray,  # [(area, kf_id), ...]
}
```

### PE Support Status
✅ **Fully Supported**: Storage fields, update method, export/restore, factory creation
⚠️ **Partially Supported**: Update flags set correctly, no PE-specific tests
❌ **Not Supported**: Legacy restore format, PE fusion in `instance_utils.same_instance()`

### Gaps in PE Support
1. **Testing**: Zero PE-specific tests for descriptor behavior
2. **Legacy Compatibility**: `old_restore()` cannot load PE descriptors
3. **Fusion Logic**: `instance_utils.same_instance()` hardcoded to CLIP
4. **Documentation**: No examples of PE usage, lifecycle, or expected behavior
