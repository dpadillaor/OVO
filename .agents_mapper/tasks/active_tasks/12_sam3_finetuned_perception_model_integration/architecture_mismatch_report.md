# SAM3 Integration Architecture Mismatch Report

**Date**: 2026-01-14
**Branch**: `feature/sam3_PE_integration`
**Comparison**: Current branch vs `main`

## Executive Summary

The SAM3 integration was implemented following an **outdated architecture pattern** that predates the fusion strategy refactor (Task 11). The main branch now uses a **FusionEncoderAdapter** pattern to manage all fusion encoders (PE, DINO, SAM3), while the current SAM3 implementation uses the old direct method approach. This creates code duplication, inconsistency, and violates the new architectural principles.

**Impact**: The SAM3 implementation is functional but does not integrate with the new fusion architecture, resulting in parallel code paths and technical debt.

---

## Architecture Comparison

### Main Branch Architecture (New Pattern)

The main branch implements a clean **Adapter Pattern** for fusion encoders:

```python
# ovo/entities/fusion_encoders.py
class FusionEncoderAdapter(ABC):
    """Abstract adapter for fusion encoders (PE, DINO, SAM3)"""

    @abstractmethod
    def compute_and_update(self, image, binary_maps, matched_ins_ids, kf_id, keyframes, objects):
        """Extract embeddings and update instances"""

    @abstractmethod
    def update_objects(self, objects, keyframes):
        """Batch update all objects"""

    @abstractmethod
    def transfer_on_merge(self, source_ids, target_id, keyframes):
        """Transfer descriptors during instance merge"""

    @abstractmethod
    def cleanup_keyframe(self, kf_id, keyframes):
        """Remove descriptors for deleted keyframe"""
```

**Key Benefits**:
- **Separation of Concerns**: Each encoder manages its own lifecycle
- **Unified Interface**: All encoders expose the same methods
- **Delegated Responsibility**: `OVO` class delegates to adapters instead of implementing logic directly
- **No Code Duplication**: PE, DINO, SAM3 all share the same pattern

### SAM3 Branch Architecture (Old Pattern)

The SAM3 implementation follows the **pre-refactor pattern**:

```python
# ovo/entities/ovo.py (SAM3 branch)
class OVO:
    def __init__(self, ...):
        # Direct generator initialization
        self.sam3_generator = SAM3Generator(config["sam3"], device=device) if "sam3" in config else None
        self.keyframes["ins_sam3_descriptors"] = dict()

    # Direct method implementations (duplicated from PE pattern)
    def _extract_sam3(self, image, binary_maps):
        """Direct extraction method"""

    def _update_matched_objects_sam3(self, sam3_embeds, matched_ins_ids, kf_id):
        """Direct update method"""

    def update_objects_sam3(self, force_update=False):
        """Direct batch update method"""
```

**Problems**:
- **Code Duplication**: Methods mirror PE implementation (94 lines duplicated)
- **Direct Coupling**: OVO class directly manages SAM3 lifecycle
- **Manual Integration**: SAM3 cleanup/merge logic manually added to OVO methods
- **Violates DRY**: Same pattern repeated for each encoder type

---

## Detailed Differences

### 1. OVO Class (`ovo/entities/ovo.py`)

#### Main Branch (Clean)
```python
def __init__(self, ...):
    # Single fusion encoder adapter
    self.fusion_encoder = self._get_fusion_encoder()

def _get_fusion_encoder(self) -> FusionEncoderAdapter | None:
    method = self.fusion_method.lower()
    if method == "pe":
        return PEFusionAdapter(self.pe_generator)
    elif method == "dino":
        return DINOFusionAdapter(self.dino_generator)
    # SAM3 should be added here
    return None

def complete_semantic_info(self):
    # Single delegation call
    if self.fusion_encoder is not None:
        self._compute_fusion_info(image, binary_maps, matched_ins_ids, kf_id)

def update_map(self, map_data, kfs):
    # Single delegation call
    if self.fusion_encoder is not None:
        self.fusion_encoder.update_objects(self.objects, self.keyframes)
```

#### SAM3 Branch (Duplicated)
```python
def __init__(self, ...):
    # Direct generator initialization
    self.sam3_generator = SAM3Generator(config["sam3"], device=device) if "sam3" in config else None
    self.keyframes["ins_sam3_descriptors"] = dict()

def complete_semantic_info(self):
    # PE handled by fusion_encoder
    if self.fusion_encoder is not None:
        self._compute_fusion_info(image, binary_maps, matched_ins_ids, kf_id)

    # SAM3 handled separately (duplication!)
    if self.sam3_generator is not None:
        sam3_embeds = self._extract_sam3(image, binary_maps).cpu()
        self._update_matched_objects_sam3(sam3_embeds, matched_ins_ids, kf_id)

def update_map(self, map_data, kfs):
    # PE handled by fusion_encoder
    if self.fusion_encoder is not None:
        self.fusion_encoder.update_objects(self.objects, self.keyframes)

    # SAM3 handled separately (duplication!)
    if self.sam3_generator is not None:
        self.update_objects_sam3()
```

**Duplication Count**: 94 lines of SAM3-specific code that should be in an adapter

---

### 2. Keyframe Cleanup (`ovo/entities/ovo.py`)

#### Main Branch (Clean)
```python
def _remove_deleted_keyframes(self, kfs: List[int]):
    for i, kf in enumerate(self.keyframes["frame_id"]):
        if kf not in kfs:
            if kf in self.keyframes["ins_descriptors"]:
                self.keyframes["ins_descriptors"].pop(kf)

            # Delegate cleanup to fusion encoder
            if self.fusion_encoder is not None:
                self.fusion_encoder.cleanup_keyframe(kf, self.keyframes)
```

#### SAM3 Branch (Manual)
```python
def _remove_deleted_keyframes(self, kfs: List[int]):
    for i, kf in enumerate(self.keyframes["frame_id"]):
        if kf not in kfs:
            if kf in self.keyframes["ins_descriptors"]:
                self.keyframes["ins_descriptors"].pop(kf)

            # Delegate cleanup to fusion encoder (PE)
            if self.fusion_encoder is not None:
                self.fusion_encoder.cleanup_keyframe(kf, self.keyframes)

            # SAM3 cleanup manually added (should be in adapter!)
            if kf in self.keyframes["ins_sam3_descriptors"]:
                self.keyframes["ins_sam3_descriptors"].pop(kf)
```

---

### 3. Instance Merge Transfer (`ovo/entities/ovo.py`)

#### Main Branch (Clean)
```python
def _update_descriptors_after_fusion(self, fused_objects: Dict[int, int]):
    for id2, id1 in fused_objects.items():
        for kf in self.objects[id2].kfs_ids:
            # ... handle CLIP descriptors ...

            # Delegate to fusion encoder
            if self.fusion_encoder is not None:
                self.fusion_encoder.transfer_on_merge([id2], id1, self.keyframes)
```

#### SAM3 Branch (Manual)
```python
def _update_descriptors_after_fusion(self, fused_objects: Dict[int, int]):
    for id2, id1 in fused_objects.items():
        for kf in self.objects[id2].kfs_ids:
            # ... handle CLIP descriptors ...

            # Delegate to fusion encoder (PE)
            if self.fusion_encoder is not None:
                self.fusion_encoder.transfer_on_merge([id2], id1, self.keyframes)

            # Fallback Handle PE descriptors (redundant!)
            if kf in self.keyframes.get("ins_pe_descriptors", {}) and id2 in self.keyframes["ins_pe_descriptors"][kf]:
                ins_pe_descriptor2 = self.keyframes["ins_pe_descriptors"][kf].pop(id2)
                if id1 not in self.keyframes["ins_pe_descriptors"][kf] or True:
                    self.keyframes["ins_pe_descriptors"][kf][id1] = ins_pe_descriptor2

            # SAM3 manually added (should be in adapter!)
            if kf in self.keyframes.get("ins_sam3_descriptors", {}) and id2 in self.keyframes["ins_sam3_descriptors"][kf]:
                ins_sam3_descriptor2 = self.keyframes["ins_sam3_descriptors"][kf].pop(id2)
                if id1 not in self.keyframes["ins_sam3_descriptors"][kf] or True:
                    self.keyframes["ins_sam3_descriptors"][kf][id1] = ins_sam3_descriptor2
```

**Additional Issue**: SAM3 branch also has redundant PE descriptor transfer code that duplicates what `PEFusionAdapter.transfer_on_merge()` already does!

---

### 4. Fusion Strategy Factory (`ovo/entities/fusion.py`)

#### Main Branch
```python
def create_fusion_strategy(config: Dict[str, Any]) -> FusionStrategy:
    strategy_map = {
        "clip": ("clip_feature", SemanticGeometricFusion),
        "dino": ("dino_feature", SemanticGeometricFusion),
        "pe": ("pe_feature", SemanticGeometricFusion),
        "geometric": (None, GeometricOnlyFusion),
    }
```

#### SAM3 Branch (Correct Addition)
```python
def create_fusion_strategy(config: Dict[str, Any]) -> FusionStrategy:
    strategy_map = {
        "clip": ("clip_feature", SemanticGeometricFusion),
        "dino": ("dino_feature", SemanticGeometricFusion),
        "pe": ("pe_feature", SemanticGeometricFusion),
        "sam3": ("sam3_feature", SemanticGeometricFusion),  # ✓ Correct
        "geometric": (None, GeometricOnlyFusion),
    }
```

**Note**: This is the **only correct integration** in the SAM3 branch that follows the new pattern.

---

### 5. Instance3D (`ovo/entities/instance3d.py`)

#### Main Branch
```python
class Instance3D:
    def __init__(self, ...):
        self.clip_feature = None
        self.clip_feature_kf = None
        self.pe_feature = None
        self.pe_feature_kf = None
        self.dino_feature = None
        self.dino_feature_kf = None
        self.to_update = False
        self.to_update_pe = False
```

#### SAM3 Branch (Added)
```python
class Instance3D:
    def __init__(self, ...):
        self.clip_feature = None
        self.clip_feature_kf = None
        self.pe_feature = None
        self.pe_feature_kf = None
        self.sam3_feature = None         # ✓ Added
        self.sam3_feature_kf = None      # ✓ Added
        self.dino_feature = None
        self.dino_feature_kf = None
        self.to_update = False
        self.to_update_pe = False
        self.to_update_sam3 = False      # ✓ Added

    def update_sam3(self, keyframes_sam3, force_update=False):
        """40 lines of update logic (mirrors update_pe)"""
```

**Analysis**: Correctly follows the existing pattern for Instance3D. However, the `update_sam3()` method is called directly by OVO instead of through an adapter.

---

## Files Modified in SAM3 Branch

| File | Lines Changed | Status | Notes |
|------|--------------|--------|-------|
| `ovo/entities/fusion.py` | +1 | ✓ **Correct** | Added `sam3` to fusion strategy map |
| `ovo/entities/instance3d.py` | +47 | ✓ **Correct** | Added sam3_feature, update_sam3() |
| `ovo/entities/sam3_generator.py` | +171 | ✓ **New File** | Generator implementation (good) |
| `ovo/entities/ovo.py` | +161 | ⚠️ **Outdated Pattern** | Should use adapter instead |
| `tests/fixtures/fixtures_sam3.py` | New | ✓ **Good** | Test fixtures |
| `tests/unit/test_sam3_generator.py` | New | ✓ **Good** | Generator tests |
| `tests/unit/test_instance3d_sam3.py` | New | ✓ **Good** | Instance3D tests |
| `tests/unit/test_fusion_sam3.py` | New | ✓ **Good** | Fusion tests |
| `tests/unit/test_ovo_sam3.py` | New | ⚠️ **Needs Update** | Tests the old pattern |

---

## Missing Components

### 1. SAM3FusionAdapter (Does Not Exist)

Should be in `ovo/entities/fusion_encoders.py`:

```python
class SAM3FusionAdapter(FusionEncoderAdapter):
    """Adapter for SAM3 fusion."""

    def __init__(self, sam3_generator: SAM3Generator, storage_key: str = "ins_sam3_descriptors"):
        self.generator = sam3_generator
        self.storage_key = storage_key

    def compute_and_update(self, image, binary_maps, matched_ins_ids, kf_id, keyframes, objects):
        # Logic currently in OVO._extract_sam3 + OVO._update_matched_objects_sam3

    def update_objects(self, objects, keyframes):
        # Logic currently in OVO.update_objects_sam3

    def transfer_on_merge(self, source_ids, target_id, keyframes):
        # Logic currently manually in OVO._update_descriptors_after_fusion

    def cleanup_keyframe(self, kf_id, keyframes):
        # Logic currently manually in OVO._remove_deleted_keyframes
```

### 2. Updated `_get_fusion_encoder()`

Should be updated in `ovo/entities/ovo.py`:

```python
def _get_fusion_encoder(self) -> FusionEncoderAdapter | None:
    method = self.fusion_method.lower()

    if method == "pe":
        return PEFusionAdapter(self.pe_generator)
    elif method == "dino":
        return DINOFusionAdapter(self.dino_generator)
    elif method == "sam3":
        return SAM3FusionAdapter(self.sam3_generator)  # Missing!

    return None
```

### 3. Validation Update

Should be updated in `ovo/entities/ovo.py`:

```python
def _validate_fusion_config(self):
    method = self.fusion_method.lower()

    validation_map = {
        "pe": (self.pe_generator, "PE generator"),
        "sam3": (self.sam3_generator, "SAM3 generator"),  # Missing!
    }

    if method in validation_map:
        generator, name = validation_map[method]
        if generator is None:
            raise ValueError(f"fusion_method='{method}' requires {name}")
```

---

## Code That Should Be Removed

Once `SAM3FusionAdapter` is implemented, the following code in `ovo/entities/ovo.py` becomes redundant:

### Lines to Remove

1. **Direct generator initialization** (lines ~48)
```python
# REMOVE THIS:
self.sam3_generator = SAM3Generator(config["sam3"], device=device) if "sam3" in config else None
```
Generator should be initialized only if needed by adapter factory.

2. **Direct keyframe storage initialization** (lines ~56)
```python
# REMOVE THIS:
self.keyframes["ins_sam3_descriptors"] = dict()
```
Adapter should manage its own storage.

3. **Device management** (lines ~133-136, ~146-149)
```python
# REMOVE THIS:
if self.sam3_generator is not None:
    self.sam3_generator.cpu()

if self.sam3_generator is not None:
    self.sam3_generator.cuda()
```
Adapter should manage device movement.

4. **Extraction and update in complete_semantic_info()** (lines ~407-418)
```python
# REMOVE THIS:
if self.sam3_generator is not None:
    sam3_embeds = self._extract_sam3(image, binary_maps).cpu()
    self._update_matched_objects_sam3(sam3_embeds, matched_ins_ids, kf_id)
```
Should be handled by adapter via `_compute_fusion_info()`.

5. **Manual cleanup** (lines ~459-461)
```python
# REMOVE THIS:
if kf in self.keyframes["ins_sam3_descriptors"]:
    self.keyframes["ins_sam3_descriptors"].pop(kf)
```
Adapter handles cleanup.

6. **Manual update in update_map()** (lines ~514-516)
```python
# REMOVE THIS:
if self.sam3_generator is not None:
    self.update_objects_sam3()
```
Adapter handles updates.

7. **Manual transfer in _update_descriptors_after_fusion()** (lines ~584-587)
```python
# REMOVE THIS:
if kf in self.keyframes.get("ins_sam3_descriptors", {}) and id2 in self.keyframes["ins_sam3_descriptors"][kf]:
    ins_sam3_descriptor2 = self.keyframes["ins_sam3_descriptors"][kf].pop(id2)
    if id1 not in self.keyframes["ins_sam3_descriptors"][kf] or True:
        self.keyframes["ins_sam3_descriptors"][kf][id1] = ins_sam3_descriptor2
```
Adapter handles merge transfers.

8. **All three SAM3 methods** (lines ~635-725)
```python
# REMOVE ALL THREE METHODS:
def _extract_sam3(self, image, binary_maps):
    """94 lines total"""
def _update_matched_objects_sam3(self, sam3_embeds, matched_ins_ids, kf_id):
def update_objects_sam3(self, force_update=False):
```
All logic moved to adapter.

9. **Export/restore in save/load** (lines ~808-809, ~846-848)
```python
# REMOVE THIS:
for kf_id, ins_sam3_descriptors in self.keyframes["ins_sam3_descriptors"].items():
    for ins_id, descriptors in ins_sam3_descriptors.items():
        scene_dict[f"kf_{kf_id}_ins3d_{ins_id}_sam3"] = descriptors.cpu().numpy()

# And this:
sam3_descriptor = scene_dict.get(f"kf_{i}_ins3d_{ins_id}_sam3", None)
if sam3_descriptor is not None:
    self.keyframes["ins_sam3_descriptors"][i][ins_id] = torch.tensor(sam3_descriptor, device=self.device)
```
Adapter should handle persistence (or extend base system).

**Total Lines to Remove**: ~161 lines from `ovo/entities/ovo.py`

---

## Additional Issues Found

### 1. Redundant PE Code

The SAM3 branch added **redundant PE descriptor transfer code**:

```python
# Lines 577-580 in ovo.py (SAM3 branch)
# Fallback Handle PE descriptors (if not using fusion encoder or for redundancy)
if kf in self.keyframes.get("ins_pe_descriptors", {}) and id2 in self.keyframes["ins_pe_descriptors"][kf]:
    ins_pe_descriptor2 = self.keyframes["ins_pe_descriptors"][kf].pop(id2)
    if id1 not in self.keyframes["ins_pe_descriptors"][kf] or True:
        self.keyframes["ins_pe_descriptors"][kf][id1] = ins_pe_descriptor2
```

This duplicates what `PEFusionAdapter.transfer_on_merge()` already does and should be removed.

### 2. Inconsistent Logging

SAM3 branch adds custom logging for SAM3 times but doesn't follow the adapter pattern:

```python
# Lines 420-425
if self.sam3_generator is not None:
    idx = 2 if self.fusion_encoder is None else 4
    if len(self._time_cache) > idx + 1:
        log_stats["t_sam3"] = round(self._time_cache[idx],2)
        log_stats["t_up_sam3"] = round(self._time_cache[idx+1],3)
```

This creates fragile index-based time tracking. The logging system should be refactored to support multiple fusion encoders.

### 3. PE Methods Still Present

The SAM3 branch includes PE extraction methods directly in OVO:

```python
# Lines 635-671 (SAM3 branch)
def _extract_pe(self, image, binary_maps):
    """Should be removed - handled by PEFusionAdapter"""

def _update_matched_objects_pe(self, pe_embeds, matched_ins_ids, kf_id):
    """Should be removed - handled by PEFusionAdapter"""

def update_objects_pe(self, force_update=False):
    """Should be removed - handled by PEFusionAdapter"""
```

These exist in the main branch but should eventually be removed once full adapter migration is complete.

---

## Refactoring Roadmap

### Phase 1: Create SAM3FusionAdapter
1. Create `SAM3FusionAdapter` class in `ovo/entities/fusion_encoders.py`
2. Move logic from `OVO._extract_sam3` → `SAM3FusionAdapter.compute_and_update`
3. Move logic from `OVO.update_objects_sam3` → `SAM3FusionAdapter.update_objects`
4. Implement `transfer_on_merge()` and `cleanup_keyframe()`

### Phase 2: Update OVO Integration
1. Update `OVO._get_fusion_encoder()` to return `SAM3FusionAdapter`
2. Update `OVO._validate_fusion_config()` to validate SAM3
3. Remove direct `self.sam3_generator` initialization from `__init__`
4. Remove `ins_sam3_descriptors` keyframe initialization

### Phase 3: Remove Redundant Code
1. Remove `OVO._extract_sam3()`
2. Remove `OVO._update_matched_objects_sam3()`
3. Remove `OVO.update_objects_sam3()`
4. Remove manual SAM3 cleanup in `_remove_deleted_keyframes()`
5. Remove manual SAM3 transfer in `_update_descriptors_after_fusion()`
6. Remove SAM3 device management code
7. Remove redundant PE transfer code (lines 577-580)

### Phase 4: Update Tests
1. Update `test_ovo_sam3.py` to test adapter pattern
2. Create `test_sam3_fusion_adapter.py`
3. Ensure all existing tests still pass

### Phase 5: Documentation
1. Update CLAUDE.md to document SAM3 integration
2. Update task strategy.md to reflect new architecture
3. Add inline documentation for SAM3FusionAdapter

---

## Benefits of Refactoring

| Metric | Before (SAM3 Branch) | After (Adapted) |
|--------|---------------------|-----------------|
| Lines in `ovo.py` | +161 | ~0 (delegated) |
| Code duplication | 94 lines SAM3 ≈ PE | 0 (shared adapter interface) |
| Integration points | 9 manual locations | 1 (factory) |
| Maintainability | Low (scattered logic) | High (encapsulated) |
| Extensibility | Hard (more duplication) | Easy (add adapter) |
| Consistency | PE uses adapter, SAM3 doesn't | All use adapters |

---

## Conclusion

The SAM3 implementation is **functionally correct** but **architecturally outdated**. It follows the pre-Task-11 pattern where each fusion encoder required direct integration into the `OVO` class. The main branch has since adopted the **FusionEncoderAdapter pattern**, which provides:

- **Clean separation of concerns**
- **Unified interface for all encoders**
- **Elimination of code duplication**
- **Easy extensibility for future encoders**

**Recommendation**: Refactor SAM3 to use `SAM3FusionAdapter` following the established pattern. This will align the codebase with the new architecture and eliminate ~161 lines of redundant code from `ovo.py`.

---

## References

- **Task 11**: Fusion Strategy Refactor (`.agents_mapper/tasks/completed_tasks/11_fusion_strategy_refactor/`)
- **Main Branch**: `ovo/entities/fusion_encoders.py` (FusionEncoderAdapter pattern)
- **Main Branch**: `ovo/entities/ovo.py` (Clean integration via adapters)
- **SAM3 Branch**: `ovo/entities/ovo.py` (Duplicated direct integration)
