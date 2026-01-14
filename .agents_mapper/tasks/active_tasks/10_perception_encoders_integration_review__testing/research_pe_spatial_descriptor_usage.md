# Research: PE-Spatial Descriptor Usage Flow

## Question
When using PE-Spatial (vision-only model), are PE descriptors used only for fusion or also for semantic queries/classification? This could explain weird results.

## Summary
**PE descriptors are used ONLY for fusion/merging, NOT for semantic classification.** Semantic queries always use CLIP regardless of `fusion_method` config.

---

## Findings

### 1. PE Descriptor Storage (`instance3d.py`)

```python
# Lines 33-34: PE features stored as instance attributes
self.pe_feature = None
self.pe_feature_kf = None
```

- `update_pe()` method (lines 162-194) aggregates PE descriptors
- Persisted in exports/restores (lines 207-232)

### 2. PE Used ONLY for Fusion (`fusion.py`)

**Strategy factory** (lines 145-150):
```python
strategy_map = {
    "clip": ("clip_feature", SemanticGeometricFusion),
    "dino": ("dino_feature", SemanticGeometricFusion),
    "pe": ("pe_feature", SemanticGeometricFusion),  # PE for fusion only
    "geometric": (None, GeometricOnlyFusion),
}
```

**Fusion similarity** (lines 63-88):
- `SemanticGeometricFusion.same_instance()` computes cosine similarity between instance features
- This is the **ONLY place PE features are used semantically**
- Safe for vision-only models (no text encoder needed)

### 3. Semantic Classification Uses ONLY CLIP (`ovo.py`)

#### `query()` method (lines 616-631)
```python
# Line 629-630: Hardcoded to CLIP
clips = self.get_objs_clips()  # Gets CLIP features only
sim_map = self.clip_generator.get_embed_txt_similarity(clips, queries, templates)
```

#### `classify_instances()` method (lines 594-613)
- Calls `self.query()` internally
- Also uses CLIP features exclusively

#### `get_objs_clips()` method (lines 634-648)
```python
# Lines 641-642: Only retrieves clip_feature, never pe_feature
clips[i] = obj.clip_feature
```

### 4. PE Text Similarity Blocked for Vision-Only (`pe_generator.py`)

```python
# Lines 156-158
def get_embed_txt_similarity(self, ins_descriptors, txt_queries, templates=['{}']) -> torch.Tensor:
    if self.is_vision_only:
        raise NotImplementedError(f"Text similarity not supported for vision-only model: {self.model_card}")
```

PE-Spatial has `is_vision_only=True` (line 27), so text queries are explicitly blocked.

---

## Usage Summary Table

| Operation | Descriptor Used | PE-Spatial Safe? |
|-----------|-----------------|------------------|
| Fusion/Merging | PE (when `fusion_method="pe"`) | ✅ Yes |
| `query()` | CLIP only | ❌ Requires CLIP text encoder |
| `classify_instances()` | CLIP only | ❌ Requires CLIP text encoder |
| Storage/Export | Both stored | ✅ Yes |

---

## Root Cause of Weird Results

### Semantic Mismatch Problem

When `fusion_method="pe"` with PE-Spatial:

1. **Fusion decisions** use PE similarity (vision embeddings)
2. **Query/classification** uses CLIP similarity (different embedding space)

This creates inconsistency:
- Two instances might look **similar to PE** → get merged
- But look **different to CLIP** → confusing query results
- The semantic "identity" used for merging differs from the one used for labeling

### Example Scenario
```
Instance A (chair) + Instance B (chair)
  → PE says: similar (merge them)
  → Result: Instance A contains both

Later: query("chair")
  → CLIP says: Instance A has mixed features, low similarity
  → Weird result: chair not found or low confidence
```

---

## Recommendations

### If using PE-Spatial (vision-only):

1. ✅ **Safe**: Use for fusion (`fusion_method="pe"`)
2. ❌ **Avoid**: Don't rely on `query()` or `classify_instances()` - these always use CLIP
3. ⚠️ **Consider**: Keep CLIP encoder alongside PE-Spatial if text-based queries are needed

### Architecture Suggestion

For consistent behavior with PE-Spatial:
- Use PE for fusion (geometric + visual similarity)
- Use CLIP only for final labeling (post-fusion)
- Or implement PE-based nearest-neighbor classification using pre-computed category embeddings

---

## Files Investigated
- `ovo/entities/instance3d.py`
- `ovo/entities/ovo.py`
- `ovo/entities/fusion.py`
- `ovo/entities/pe_generator.py`
- `ovo/entities/fusion_encoders.py`
